from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from uuid import uuid4

from magellan.experiments.measurement import absolute_percent_error, summarize_samples


def fresh_run_idempotency_key(measurement_id: str) -> str:
    """Return a per-execution task-run key while preserving measurement provenance.

    Stage 4A.2 case names are intentionally stable across campaigns and resume
    attempts. Reusing ``<measurement>-run`` would cause the API to return an
    earlier task run with the same idempotency key, including a completed run.
    A fresh suffix keeps duplicate protection within the single POST while making
    every actual calibration execution distinct.
    """

    return f"{measurement_id}-run-{uuid4().hex}"


@dataclass(frozen=True)
class RepresentativeEdge:
    role: str
    source_node_id: str
    destination_node_id: str
    bandwidth_mbps: float
    rtt_ms: float

    @property
    def edge(self) -> str:
        return f"{self.source_node_id}->{self.destination_node_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_node_id": self.source_node_id,
            "destination_node_id": self.destination_node_id,
            "edge": self.edge,
            "bandwidth_mbps": self.bandwidth_mbps,
            "rtt_ms": self.rtt_ms,
        }


def _load_edge_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Stage 4A.1 edge table is empty")
    return rows


def select_representative_edges(path: str | Path) -> list[RepresentativeEdge]:
    """Select deterministic short/medium/long WAN regimes from Stage 4A.1.

    The selector uses measured bandwidth because checkpoint transport is the
    dominant network term in migration calibration.  "short" is the highest-
    throughput directed edge, "long" the lowest-throughput edge, and "medium"
    the remaining edge closest to the directed-mesh median throughput.
    """

    rows = _load_edge_rows(path)
    parsed = [
        {
            **row,
            "bandwidth": float(row["measured_bandwidth_median_mbps"]),
            "rtt": float(row["measured_rtt_median_ms"]),
        }
        for row in rows
    ]
    ordered = sorted(
        parsed,
        key=lambda row: (
            row["bandwidth"],
            row["source_node_id"],
            row["destination_node_id"],
        ),
    )
    long_row = ordered[0]
    short_row = ordered[-1]
    remaining = [row for row in ordered if row is not long_row and row is not short_row]
    target = median(row["bandwidth"] for row in parsed)
    medium_row = min(
        remaining,
        key=lambda row: (
            abs(row["bandwidth"] - target),
            row["source_node_id"],
            row["destination_node_id"],
        ),
    )

    def build(role: str, row: dict[str, Any]) -> RepresentativeEdge:
        return RepresentativeEdge(
            role=role,
            source_node_id=str(row["source_node_id"]),
            destination_node_id=str(row["destination_node_id"]),
            bandwidth_mbps=float(row["bandwidth"]),
            rtt_ms=float(row["rtt"]),
        )

    return [
        build("short", short_row),
        build("medium", medium_row),
        build("long", long_row),
    ]


def _numeric_samples(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values.append(number)
    return values


def summarize_profile_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(rows)}
    for field in (
        "process_count",
        "cpu_utilization_percent",
        "memory_rss_mb",
        "checkpoint_bytes",
        "progress_rate_units_per_second",
        "estimated_remaining_seconds",
        "measured_power_kw",
    ):
        values = _numeric_samples(rows, field)
        result[field] = summarize_samples(values).as_dict() if values else None
    return result


def summarize_migration_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"sample_count": len(rows)}
    pairs = {
        "checkpoint_absolute_error_percent": (
            "predicted_checkpoint_seconds",
            "actual_checkpoint_seconds",
        ),
        "transfer_absolute_error_percent": (
            "predicted_transfer_seconds",
            "actual_transfer_seconds",
        ),
        "restore_absolute_error_percent": (
            "predicted_restore_seconds",
            "actual_restore_seconds",
        ),
        "downtime_absolute_error_percent": (
            "predicted_downtime_seconds",
            "actual_downtime_seconds",
        ),
    }
    for output, (predicted_field, actual_field) in pairs.items():
        errors: list[float] = []
        for row in rows:
            try:
                predicted = float(row[predicted_field])
                actual = float(row[actual_field])
            except (KeyError, TypeError, ValueError):
                continue
            value = absolute_percent_error(predicted, actual)
            if value is not None:
                errors.append(value)
        result[output] = summarize_samples(errors).as_dict() if errors else None
    return result
