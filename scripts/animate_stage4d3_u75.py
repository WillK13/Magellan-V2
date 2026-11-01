#!/usr/bin/env python3
"""Render a calibrated Stage 4D.3 u75 migration replay.

This is a visualization of recorded simulator output, not a live execution.
The default summer scenario is the representative season identified by the
Stage 4D.3 methodology and contains nine tasks and seven accepted migrations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

POLICY = "magellan_capacity_lowest_score"
EXPECTED_SITES = {
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
}
CLASS_STYLE = {
    "benchmark-json-medium": ("Benchmark", "#2f80ed"),
    "llm-distilgpt2": ("LLM (DistilGPT-2)", "#9b51e0"),
    "dendro-r9-t1p0": ("Dendro-GR", "#f2994a"),
}
SITE_LABEL_STYLE = {
    "boston": (5.0, 7.0, "left", "bottom"),
    "virginia": (5.0, -7.0, "left", "top"),
}
SITE_TASK_START_ANGLE = {
    "boston": math.pi / 2,
    "virginia": -math.pi / 2,
}
FALLBACK_CONTINENTS = (
    ((-168, 13), (-130, 8), (-105, 20), (-82, 24), (-60, 48), (-75, 72), (-125, 73), (-168, 58)),
    ((-82, 12), (-68, 8), (-48, -5), (-36, -25), (-55, -55), (-76, -42)),
    ((-18, 35), (5, 44), (35, 36), (52, 12), (42, -35), (18, -35), (-5, 5)),
    ((-10, 36), (5, 58), (40, 72), (95, 76), (170, 60), (150, 45), (110, 8), (70, 8), (42, 30)),
    ((112, -11), (154, -10), (153, -40), (115, -44)),
    ((-55, 60), (-20, 63), (-22, 82), (-52, 83)),
)


@dataclass(frozen=True)
class Site:
    node_id: str
    name: str
    longitude: float
    latitude: float
    cpu_cores: float


@dataclass(frozen=True)
class Task:
    task_id: str
    class_id: str
    initial_node_id: str


@dataclass(frozen=True)
class Migration:
    task_id: str
    class_id: str
    source: str
    destination: str
    started: datetime
    finished: datetime
    round_index: int


@dataclass(frozen=True)
class MigrationGroup:
    started: datetime
    finished: datetime
    migrations: tuple[Migration, ...]


@dataclass(frozen=True)
class Frame:
    at: datetime
    stage: int
    active_group: int | None = None


@dataclass
class Replay:
    bundle: Path
    scenario_id: str
    season: str
    arrival: datetime
    finished: datetime
    sites: dict[str, Site]
    tasks: tuple[Task, ...]
    groups: tuple[MigrationGroup, ...]
    stage_locations: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate a checksum-verified Stage 4D.3 u75 calibrated replay."
    )
    parser.add_argument("--bundle", required=True, help="Canonical Stage 4D.3 bundle directory")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--season", choices=("winter", "spring", "summer", "fall"), default="summer")
    parser.add_argument("--output", help="MP4 path (default: experiments/visualizations/...)")
    parser.add_argument(
        "--gif",
        nargs="?",
        const="auto",
        help="Also write a GIF; optionally provide its path",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--frames-between-events", type=int, default=10)
    parser.add_argument("--frames-per-migration", type=int, default=14)
    parser.add_argument("--arc-fade-frames", type=int, default=18)
    parser.add_argument("--no-cartopy", action="store_true", help="Use the dependency-free map fallback")
    parser.add_argument("--hide-capacity", action="store_true", help="Hide subtle per-site CPU labels")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and print the replay plan")
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_checksums(bundle: Path) -> None:
    checksum_path = bundle / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    errors = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = bundle / relative.strip()
        if not target.is_file():
            errors.append(f"missing {relative.strip()}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch for {relative.strip()}")
    if errors:
        raise ValueError("Bundle integrity check failed: " + "; ".join(errors))


def load_sites(cluster_path: Path, occupancy_rows: list[dict[str, str]], scenario_id: str) -> dict[str, Site]:
    config = json.loads(cluster_path.read_text(encoding="utf-8"))
    capacity_by_node: dict[str, float] = {}
    for row in occupancy_rows:
        if row["scenario_id"] == scenario_id and row["policy"] == POLICY:
            capacity_by_node.setdefault(row["node_id"], float(row["capacity_cpu_cores"]))
    sites = {}
    for node in config["nodes"]:
        node_id = node["id"]
        if node_id not in EXPECTED_SITES:
            continue
        configured_capacity = float(node["resources"]["cpu_cores"])
        measured_capacity = capacity_by_node.get(node_id, configured_capacity)
        if not math.isclose(measured_capacity, configured_capacity):
            raise ValueError(
                f"{node_id} capacity differs: occupancy={measured_capacity}, config={configured_capacity}"
            )
        sites[node_id] = Site(
            node_id=node_id,
            name=node["name"],
            longitude=float(node["longitude"]),
            latitude=float(node["latitude"]),
            cpu_cores=measured_capacity,
        )
    if set(sites) != EXPECTED_SITES:
        raise ValueError(f"Expected seven V2 sites, found {sorted(sites)}")
    return sites


def accepted_auction_keys(rows: Iterable[dict[str, str]], scenario_id: str) -> set[tuple[str, ...]]:
    return {
        (
            row["round_index"],
            row["task_id"],
            row["source_node_id"],
            row["destination_node_id"],
        )
        for row in rows
        if row["scenario_id"] == scenario_id
        and row["policy"] == POLICY
        and row["status"] == "accepted"
    }


def load_replay(bundle: Path, cluster_path: Path, season: str) -> Replay:
    verify_checksums(bundle)
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise ValueError("Stage 4D.3 summary does not have passed=true")
    if summary.get("comparison_id") != bundle.name:
        raise ValueError("Bundle directory name does not match summary comparison_id")

    load_rows = read_csv(bundle / "load_cases.csv")
    matches = [row for row in load_rows if row["season"] == season and row["load_id"] == "u75"]
    if len(matches) != 1:
        raise ValueError(f"Expected one {season} u75 load case, found {len(matches)}")
    load_case = matches[0]
    scenario_id = load_case["scenario_id"]
    if int(load_case["task_count"]) != 9:
        raise ValueError(f"{scenario_id} has {load_case['task_count']} tasks, expected 9")

    layout_rows = [
        row for row in read_csv(bundle / "initial_layout.csv") if row["scenario_id"] == scenario_id
    ]
    tasks = tuple(
        Task(row["task_id"], row["class_id"], row["initial_node_id"])
        for row in sorted(layout_rows, key=lambda item: item["task_id"])
    )
    if len(tasks) != 9 or {task.class_id for task in tasks} != set(CLASS_STYLE):
        raise ValueError("u75 layout must contain nine tasks across the three canonical classes")

    outcome_rows = [
        row
        for row in read_csv(bundle / "scenario_outcomes.csv")
        if row["scenario_id"] == scenario_id and row["policy"] == POLICY
    ]
    if len(outcome_rows) != 1:
        raise ValueError(f"Expected one outcome for {scenario_id}/{POLICY}")
    outcome = outcome_rows[0]
    arrival = parse_time(outcome["arrival_utc"])
    finished = arrival + timedelta(seconds=float(outcome["makespan_seconds"]))

    accepted = accepted_auction_keys(read_csv(bundle / "auction_events.csv"), scenario_id)
    migration_rows = [
        row
        for row in read_csv(bundle / "migration_events.csv")
        if row["scenario_id"] == scenario_id and row["policy"] == POLICY
    ]
    migrations = []
    task_ids = {task.task_id for task in tasks}
    for row in migration_rows:
        key = (
            row["round_index"],
            row["task_id"],
            row["source_node_id"],
            row["destination_node_id"],
        )
        if key not in accepted:
            raise ValueError(f"Migration lacks matching accepted auction event: {key}")
        if row["task_id"] not in task_ids:
            raise ValueError(f"Migration references task outside selected layout: {row['task_id']}")
        migrations.append(
            Migration(
                task_id=row["task_id"],
                class_id=row["class_id"],
                source=row["source_node_id"],
                destination=row["destination_node_id"],
                started=parse_time(row["started_at_utc"]),
                finished=parse_time(row["finished_at_utc"]),
                round_index=int(row["round_index"]),
            )
        )
    migrations.sort(key=lambda event: (event.started, event.finished, event.task_id))
    if len(migrations) != int(outcome["migrations"]):
        raise ValueError("Migration row count does not match scenario_outcomes.csv")

    grouped: dict[datetime, list[Migration]] = defaultdict(list)
    for migration in migrations:
        grouped[migration.started].append(migration)
    groups = tuple(
        MigrationGroup(started, max(item.finished for item in items), tuple(items))
        for started, items in sorted(grouped.items())
    )

    locations = {task.task_id: task.initial_node_id for task in tasks}
    stage_locations = [dict(locations)]
    for group in groups:
        migrating_ids = {event.task_id for event in group.migrations}
        if len(migrating_ids) != len(group.migrations):
            raise ValueError("A task migrates more than once in a simultaneous event group")
        for event in group.migrations:
            if locations[event.task_id] != event.source:
                raise ValueError(
                    f"Source continuity mismatch for {event.task_id}: "
                    f"expected {locations[event.task_id]}, event says {event.source}"
                )
            locations[event.task_id] = event.destination
        stage_locations.append(dict(locations))

    occupancy_rows = read_csv(bundle / "occupancy_timeline.csv")
    sites = load_sites(cluster_path, occupancy_rows, scenario_id)
    return Replay(
        bundle=bundle,
        scenario_id=scenario_id,
        season=season,
        arrival=arrival,
        finished=finished,
        sites=sites,
        tasks=tasks,
        groups=groups,
        stage_locations=tuple(stage_locations),
    )


def interpolate_times(start: datetime, end: datetime, count: int, *, include_end: bool) -> list[datetime]:
    if count <= 0 or end <= start:
        return []
    denominator = count if include_end else count + 1
    return [start + (end - start) * (index / denominator) for index in range(1, count + 1)]


def build_frames(replay: Replay, between: int, per_migration: int) -> list[Frame]:
    if between < 1 or per_migration < 2:
        raise ValueError("Frame counts must be positive; --frames-per-migration must be at least 2")
    frames = [Frame(replay.arrival, 0) for _ in range(between)]
    cursor = replay.arrival
    for group_index, group in enumerate(replay.groups):
        for at in interpolate_times(cursor, group.started, between, include_end=False):
            frames.append(Frame(at, group_index))
        for at in [
            group.started + (group.finished - group.started) * (index / (per_migration - 1))
            for index in range(per_migration)
        ]:
            frames.append(Frame(at, group_index, group_index))
        cursor = group.finished
    for at in interpolate_times(cursor, replay.finished, between, include_end=True):
        frames.append(Frame(at, len(replay.groups)))
    frames.extend(Frame(replay.finished, len(replay.groups)) for _ in range(between))
    return frames


def offset_positions(replay: Replay, locations: dict[str, str]) -> dict[str, tuple[float, float]]:
    by_site: dict[str, list[str]] = defaultdict(list)
    for task_id, site_id in locations.items():
        by_site[site_id].append(task_id)
    output = {}
    for site_id, task_ids in by_site.items():
        site = replay.sites[site_id]
        ordered = sorted(task_ids)
        for index, task_id in enumerate(ordered):
            start_angle = SITE_TASK_START_ANGLE.get(site_id, math.pi / 2)
            angle = start_angle + (2 * math.pi * index / max(len(ordered), 1))
            radius = 3.0 if len(ordered) > 1 else 2.3
            lon_scale = max(math.cos(math.radians(site.latitude)), 0.45)
            output[task_id] = (
                site.longitude + radius * math.cos(angle) / lon_scale,
                site.latitude + radius * math.sin(angle),
            )
    return output


def bezier(start: tuple[float, float], end: tuple[float, float], progress: float) -> tuple[float, float]:
    x0, y0 = start
    x2, y2 = end
    dx = x2 - x0
    if dx > 180:
        x2 -= 360
    elif dx < -180:
        x2 += 360
    dx, dy = x2 - x0, y2 - y0
    distance = math.hypot(dx, dy)
    curvature = min(22.0, max(7.0, distance * 0.13))
    norm = max(distance, 1.0)
    x1 = (x0 + x2) / 2 - dy / norm * curvature
    y1 = (y0 + y2) / 2 + dx / norm * curvature
    one_minus = 1.0 - progress
    x = one_minus * one_minus * x0 + 2 * one_minus * progress * x1 + progress * progress * x2
    y = one_minus * one_minus * y0 + 2 * one_minus * progress * y1 + progress * progress * y2
    return ((x + 180) % 360 - 180, y)


def bezier_path(start: tuple[float, float], end: tuple[float, float], stop: float = 1.0) -> tuple[list[float], list[float]]:
    points = [bezier(start, end, stop * index / 40) for index in range(41)]
    xs, ys = [], []
    previous = None
    for x, y in points:
        if previous is not None and abs(x - previous) > 180:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.append(x)
        ys.append(y)
        previous = x
    return xs, ys


def lerp(start: tuple[float, float], end: tuple[float, float], progress: float) -> tuple[float, float]:
    return (start[0] + (end[0] - start[0]) * progress, start[1] + (end[1] - start[1]) * progress)


def cartopy_projection(disabled: bool):
    if disabled:
        return None, None
    try:
        import cartopy.crs as ccrs

        return ccrs.Robinson(), ccrs.PlateCarree()
    except ImportError:
        return None, None


def create_axes(replay: Replay, use_cartopy: bool):
    projection, data_transform = cartopy_projection(not use_cartopy)
    figure = plt.figure(figsize=(16, 9), facecolor="#08131f")
    grid = figure.add_gridspec(1, 2, width_ratios=(4.2, 1.15), wspace=0.02)
    if projection is not None:
        axis = figure.add_subplot(grid[0, 0], projection=projection)
        axis.set_global()
        axis.set_facecolor("#0e2436")
        axis.coastlines(resolution="110m", color="#91a4b2", linewidth=0.55)
        axis.gridlines(color="#8ba0ae", alpha=0.12, linewidth=0.5)
    else:
        axis = figure.add_subplot(grid[0, 0])
        axis.set_facecolor("#0e2436")
        axis.set_xlim(-180, 180)
        axis.set_ylim(-62, 88)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks(range(-180, 181, 60))
        axis.set_yticks(range(-60, 91, 30))
        axis.grid(color="#8ba0ae", alpha=0.12, linewidth=0.5)
        axis.tick_params(colors="#78909c", labelsize=7, length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for polygon in FALLBACK_CONTINENTS:
            axis.add_patch(
                Polygon(polygon, closed=True, facecolor="#263d4d", edgecolor="#718896", linewidth=0.6)
            )
    panel = figure.add_subplot(grid[0, 1])
    panel.set_facecolor("#0b1b29")
    panel.set_xticks([])
    panel.set_yticks([])
    for spine in panel.spines.values():
        spine.set_visible(False)
    return figure, axis, panel, data_transform


class ReplayAnimator:
    def __init__(
        self,
        replay: Replay,
        frames: list[Frame],
        *,
        use_cartopy: bool,
        show_capacity: bool,
        fade_frames: int,
    ) -> None:
        self.replay = replay
        self.frames = frames
        self.fade_frames = max(fade_frames, 1)
        self.figure, self.axis, self.panel, self.transform = create_axes(replay, use_cartopy)
        self.plot_kwargs = {"transform": self.transform} if self.transform is not None else {}
        self.stage_positions = tuple(offset_positions(replay, stage) for stage in replay.stage_locations)
        self.group_end_frame = self._group_end_frames()
        self.task_tokens = {task.task_id: f"T{index}" for index, task in enumerate(replay.tasks, 1)}
        self.task_artists = {}
        self.task_labels = {}
        self.arc_artists = []
        self._draw_static(show_capacity)
        self._create_dynamic()

    def _group_end_frames(self) -> dict[int, int]:
        result = {}
        for index, frame in enumerate(self.frames):
            if frame.active_group is not None:
                result[frame.active_group] = index
        return result

    def _draw_static(self, show_capacity: bool) -> None:
        self.axis.set_title(
            "Magellan V2 · Stage 4D.3 u75 Calibrated Replay",
            color="white",
            fontsize=18,
            fontweight="bold",
            loc="left",
            pad=18,
        )
        self.subtitle = self.axis.text(
            0.01,
            0.985,
            "RECORDED SIMULATION · NOT LIVE EXECUTION",
            transform=self.axis.transAxes,
            color="#f2c94c",
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        for site in self.replay.sites.values():
            self.axis.scatter(
                [site.longitude],
                [site.latitude],
                s=145,
                facecolor="#102b3e",
                edgecolor="#d6e2ea",
                linewidth=1.2,
                zorder=4,
                **self.plot_kwargs,
            )
            capacity = f" · {site.cpu_cores:g} CPU" if show_capacity else ""
            offset_x, offset_y, horizontal, vertical = SITE_LABEL_STYLE.get(
                site.node_id, (0.0, -5.2, "center", "top")
            )
            self.axis.text(
                site.longitude + offset_x,
                site.latitude + offset_y,
                site.name + capacity,
                color="#e7eef3",
                fontsize=8,
                ha=horizontal,
                va=vertical,
                zorder=8,
                **self.plot_kwargs,
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=9,
                label=label,
            )
            for label, color in CLASS_STYLE.values()
        ]
        self.axis.legend(
            handles=handles,
            loc="lower left",
            frameon=False,
            labelcolor="white",
            fontsize=8,
            ncol=3,
        )

        common_prefix = os.path.commonprefix([task.task_id for task in self.replay.tasks])
        self.panel.text(0.06, 0.95, "TASK ROSTER", color="white", fontsize=12, fontweight="bold")
        self.panel.text(
            0.06,
            0.915,
            "Full IDs use the prefix\n" + common_prefix,
            color="#91a4b2",
            fontsize=7.5,
            va="top",
        )
        y = 0.84
        for task in self.replay.tasks:
            label, color = CLASS_STYLE[task.class_id]
            suffix = task.task_id.removeprefix(common_prefix)
            self.panel.text(0.06, y, self.task_tokens[task.task_id], color=color, fontsize=9, fontweight="bold")
            self.panel.text(0.17, y, suffix, color="#e7eef3", fontsize=7.5)
            self.panel.text(0.17, y - 0.023, label, color="#91a4b2", fontsize=6.8)
            y -= 0.073
        self.panel.text(0.06, 0.14, "SIMULATED TIME (UTC)", color="#91a4b2", fontsize=8)
        self.time_text = self.panel.text(0.06, 0.105, "", color="white", fontsize=11, fontweight="bold")
        self.event_text = self.panel.text(
            0.06, 0.075, "", color="#f2c94c", fontsize=7.5, va="top", wrap=True
        )
        self.panel.text(
            0.06,
            0.005,
            "Quiet intervals are compressed; migration clocks use recorded start/finish times.",
            color="#718896",
            fontsize=6.2,
            va="bottom",
            wrap=True,
        )

    def _create_dynamic(self) -> None:
        initial = self.stage_positions[0]
        for task in self.replay.tasks:
            _, color = CLASS_STYLE[task.class_id]
            x, y = initial[task.task_id]
            artist = self.axis.scatter(
                [x],
                [y],
                s=220,
                facecolor=color,
                edgecolor="white",
                linewidth=1.1,
                zorder=12,
                **self.plot_kwargs,
            )
            text = self.axis.text(
                x,
                y,
                self.task_tokens[task.task_id],
                color="white",
                fontsize=6.5,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=13,
                **self.plot_kwargs,
            )
            self.task_artists[task.task_id] = artist
            self.task_labels[task.task_id] = text
        for group in self.replay.groups:
            lines = []
            for _ in group.migrations:
                (line,) = self.axis.plot([], [], color="#56ccf2", linewidth=1.8, alpha=0, zorder=9, **self.plot_kwargs)
                lines.append(line)
            self.arc_artists.append(lines)

    def _active_progress(self, event: Migration, at: datetime) -> float:
        duration = max((event.finished - event.started).total_seconds(), 1e-9)
        return min(1.0, max(0.0, (at - event.started).total_seconds() / duration))

    def update(self, frame_index: int):
        frame = self.frames[frame_index]
        positions = dict(self.stage_positions[frame.stage])
        event_message = ""
        if frame.active_group is not None:
            group_index = frame.active_group
            group = self.replay.groups[group_index]
            before = self.stage_positions[group_index]
            after = self.stage_positions[group_index + 1]
            group_duration = max((group.finished - group.started).total_seconds(), 1e-9)
            group_progress = min(1.0, max(0.0, (frame.at - group.started).total_seconds() / group_duration))
            migrating = {event.task_id: event for event in group.migrations}
            for task in self.replay.tasks:
                if task.task_id in migrating:
                    event = migrating[task.task_id]
                    progress = self._active_progress(event, frame.at)
                    positions[task.task_id] = bezier(before[task.task_id], after[task.task_id], progress)
                else:
                    positions[task.task_id] = lerp(before[task.task_id], after[task.task_id], group_progress)
            descriptions = [
                f"{self.task_tokens[event.task_id]}: {self.replay.sites[event.source].name} → "
                f"{self.replay.sites[event.destination].name}"
                for event in group.migrations
            ]
            event_message = "ACCEPTED MIGRATION\n" + "\n".join(descriptions)

        for task_id, (x, y) in positions.items():
            self.task_artists[task_id].set_offsets([[x, y]])
            self.task_labels[task_id].set_position((x, y))

        for group_index, group in enumerate(self.replay.groups):
            if group_index == frame.active_group:
                alpha = 0.9
            else:
                age = frame_index - self.group_end_frame[group_index]
                alpha = max(0.0, 0.62 * (1.0 - age / self.fade_frames)) if age >= 0 else 0.0
            before = self.stage_positions[group_index]
            after = self.stage_positions[group_index + 1]
            for event, line in zip(group.migrations, self.arc_artists[group_index], strict=True):
                stop = self._active_progress(event, frame.at) if group_index == frame.active_group else 1.0
                xs, ys = bezier_path(before[event.task_id], after[event.task_id], stop)
                line.set_data(xs, ys)
                line.set_alpha(alpha)

        self.time_text.set_text(frame.at.strftime("%Y-%m-%d %H:%M:%S"))
        self.event_text.set_text(event_message)
        return [
            *self.task_artists.values(),
            *self.task_labels.values(),
            *(line for lines in self.arc_artists for line in lines),
            self.time_text,
            self.event_text,
        ]


def print_plan(replay: Replay) -> None:
    migration_count = sum(len(group.migrations) for group in replay.groups)
    print("Stage 4D.3 u75 calibrated replay validated")
    print(f"bundle={replay.bundle}")
    print(f"scenario_id={replay.scenario_id}")
    print(f"policy={POLICY}")
    print(f"tasks={len(replay.tasks)} sites={len(replay.sites)} accepted_migrations={migration_count}")
    for group in replay.groups:
        for event in group.migrations:
            print(
                f"  {event.started.isoformat()}  {event.task_id}: "
                f"{event.source} -> {event.destination}"
            )


def save_animation(animator: ReplayAnimator, frames: list[Frame], args: argparse.Namespace, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        animator.figure,
        animator.update,
        frames=len(frames),
        interval=1000 / args.fps,
        blit=False,
    )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("MP4 export requires ffmpeg on PATH (for example: brew install ffmpeg)")
    movie.save(
        output,
        writer=animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=3200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=args.dpi,
    )
    print(f"wrote_mp4={output}")
    if args.gif:
        gif_path = output.with_suffix(".gif") if args.gif == "auto" else Path(args.gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        movie.save(gif_path, writer=animation.PillowWriter(fps=args.fps), dpi=min(args.dpi, 110))
        print(f"wrote_gif={gif_path}")


def main() -> int:
    args = parse_args()
    bundle = Path(args.bundle).resolve()
    cluster = Path(args.cluster).resolve()
    replay = load_replay(bundle, cluster, args.season)
    print_plan(replay)
    if args.validate_only:
        return 0

    output = Path(args.output) if args.output else Path(
        f"experiments/visualizations/stage4d3-u75-{args.season}-calibrated-replay.mp4"
    )
    frames = build_frames(replay, args.frames_between_events, args.frames_per_migration)
    projection, _ = cartopy_projection(args.no_cartopy)
    animator = ReplayAnimator(
        replay,
        frames,
        use_cartopy=projection is not None,
        show_capacity=not args.hide_capacity,
        fade_frames=args.arc_fade_frames,
    )
    try:
        animator.figure.canvas.draw()
    except Exception as exc:
        if projection is None:
            raise
        print(f"Cartopy map unavailable ({exc}); using built-in map fallback")
        plt.close(animator.figure)
        animator = ReplayAnimator(
            replay,
            frames,
            use_cartopy=False,
            show_capacity=not args.hide_capacity,
            fade_frames=args.arc_fade_frames,
        )
    save_animation(animator, frames, args, output.resolve())
    plt.close(animator.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
