#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d2 import CREDIT_FAIR_POLICY, LOWEST_SCORE_POLICY
from magellan.experiments.stage4e1 import (
    SCALE_POLICIES,
    SCALE_SIZES,
    STATIC_SCALE_POLICY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4E.1 scaling bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = [
        "summary.json",
        "metadata.json",
        "population.csv",
        "task_outcomes.csv",
        "scaling_summary.csv",
        "per_class_summary.csv",
        "auction_events.csv",
        "migration_events.csv",
        "migration_matrix.csv",
        "occupancy_timeline.csv",
        "event_trace.csv",
        "checksums.sha256",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    population = read_csv(root / "population.csv")
    tasks = read_csv(root / "task_outcomes.csv")
    scaling = read_csv(root / "scaling_summary.csv")
    auctions = read_csv(root / "auction_events.csv")
    migrations = read_csv(root / "migration_events.csv")
    occupancy = read_csv(root / "occupancy_timeline.csv")
    events = read_csv(root / "event_trace.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("scale_sizes") or []) != list(SCALE_SIZES):
        errors.append("summary scale_sizes mismatch")
    if list(summary.get("policy_names") or []) != list(SCALE_POLICIES):
        errors.append("summary policy_names mismatch")
    if int(summary.get("capacity_violation_count") or 0) != 0:
        errors.append("summary reports capacity violations")
    if summary.get("queueing_observed_at_100") is not True:
        errors.append("summary did not observe queueing at 100 tasks")

    d44 = Path(str(summary.get("source_stage4d4_bundle") or ""))
    if not d44.is_dir():
        errors.append(f"source Stage 4D.4 bundle not found: {d44}")
    elif validate_checksums(d44):
        errors.append("source Stage 4D.4 checksum validation failed")

    expected_population = sum(SCALE_SIZES)
    if len(population) != expected_population:
        errors.append(f"population coverage {len(population)} != {expected_population}")

    pop_by_scenario = defaultdict(list)
    for row in population:
        pop_by_scenario[row["scenario_id"]].append(row)
    for size in SCALE_SIZES:
        scenario_id = f"scale-summer-{size:03d}"
        rows = pop_by_scenario[scenario_id]
        if len(rows) != size:
            errors.append(f"{scenario_id} population {len(rows)} != {size}")
        class_counts = Counter(row["class_id"] for row in rows)
        if max(class_counts.values()) - min(class_counts.values()) > 1:
            errors.append(f"{scenario_id} workload mix is not balanced: {dict(class_counts)}")
        home_counts = Counter(row["home_node_id"] for row in rows)
        if max(home_counts.values()) - min(home_counts.values()) > 1:
            errors.append(f"{scenario_id} home-node assignment is not balanced: {dict(home_counts)}")

    expected_task_rows = expected_population * len(SCALE_POLICIES)
    if len(tasks) != expected_task_rows:
        errors.append(f"task outcome coverage {len(tasks)} != {expected_task_rows}")

    by_case_policy = defaultdict(list)
    for row in tasks:
        by_case_policy[(row["scenario_id"], row["policy"])].append(row)
        if not truthy(row["completed"]):
            errors.append(
                f"incomplete task {row['scenario_id']}/{row['policy']}/{row['task_id']}"
            )
        if float(row["queue_wait_seconds"]) < 0:
            errors.append(f"negative queue wait for {row['task_id']}")
        if int(row["bid_attempts"]) != int(row["bid_accepts"]) + int(row["bid_rejections"]):
            errors.append(f"bid accounting mismatch for {row['task_id']}/{row['policy']}")

    for size in SCALE_SIZES:
        scenario_id = f"scale-summer-{size:03d}"
        for policy in SCALE_POLICIES:
            rows = by_case_policy[(scenario_id, policy)]
            if len(rows) != size:
                errors.append(f"{scenario_id}/{policy} task coverage {len(rows)} != {size}")

    static_rows = [row for row in tasks if row["policy"] == STATIC_SCALE_POLICY]
    if any(int(row["migrations"]) or int(row["bid_attempts"]) for row in static_rows):
        errors.append("static scaling baseline contains migrations or bids")

    expected_summary_rows = len(SCALE_SIZES) * len(SCALE_POLICIES)
    if len(scaling) != expected_summary_rows:
        errors.append(
            f"scaling_summary coverage {len(scaling)} != {expected_summary_rows}"
        )
    summary_keys = {(int(row["scale_task_count"]), row["policy"]) for row in scaling}
    expected_keys = {
        (size, policy)
        for size in SCALE_SIZES
        for policy in SCALE_POLICIES
    }
    if summary_keys != expected_keys:
        errors.append("scaling_summary size/policy coverage mismatch")

    queued_100 = [
        row
        for row in tasks
        if row["scenario_id"] == "scale-summer-100"
        and float(row["queue_wait_seconds"]) > 0
    ]
    if not queued_100:
        errors.append("100-task population did not produce queueing")

    for row in occupancy:
        if float(row["used_cpu_cores"]) > float(row["capacity_cpu_cores"]) + 1e-9:
            errors.append(
                f"CPU overcommit {row['scenario_id']}/{row['policy']}/"
                f"{row['node_id']}/{row['at_utc']}"
            )
        if int(row["used_memory_mb"]) > int(row["capacity_memory_mb"]):
            errors.append(
                f"memory overcommit {row['scenario_id']}/{row['policy']}/"
                f"{row['node_id']}/{row['at_utc']}"
            )
        if int(row["used_gpu_count"]) > int(row["capacity_gpu_count"]):
            errors.append(
                f"GPU overcommit {row['scenario_id']}/{row['policy']}/"
                f"{row['node_id']}/{row['at_utc']}"
            )

    accepted = [row for row in auctions if row.get("status") == "accepted"]
    rejected = [row for row in auctions if row.get("status") == "rejected"]
    if len(accepted) != len(migrations):
        errors.append(
            f"accepted auction count {len(accepted)} != migration count {len(migrations)}"
        )
    if int(summary.get("auction_event_count") or 0) != len(auctions):
        errors.append("summary auction_event_count mismatch")
    if int(summary.get("migration_event_count") or 0) != len(migrations):
        errors.append("summary migration_event_count mismatch")
    if int(summary.get("animation_event_count") or 0) != len(events):
        errors.append("summary animation_event_count mismatch")

    event_counts = Counter((row["scenario_id"], row["policy"], row["event_type"]) for row in events)
    for size in SCALE_SIZES:
        scenario_id = f"scale-summer-{size:03d}"
        for policy in SCALE_POLICIES:
            for event_type in ("submitted", "admitted", "completed"):
                observed = event_counts[(scenario_id, policy, event_type)]
                if observed != size:
                    errors.append(
                        f"{scenario_id}/{policy}/{event_type} events {observed} != {size}"
                    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4E1_SCALING_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"sizes: {','.join(str(value) for value in SCALE_SIZES)}")
    print(f"policies: {len(SCALE_POLICIES)}/{len(SCALE_POLICIES)}")
    print(f"task_outcomes: {len(tasks)}/{expected_task_rows}")
    print(f"auction_events: {len(auctions)}")
    print(f"accepted_bids: {len(accepted)}")
    print(f"rejected_bids: {len(rejected)}")
    print(f"migration_events: {len(migrations)}")
    print(f"animation_events: {len(events)}")
    print("capacity_violations: 0")
    print(f"queued_tasks_at_100: {len(queued_100)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
