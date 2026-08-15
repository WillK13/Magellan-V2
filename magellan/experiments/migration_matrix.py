from __future__ import annotations

from collections import defaultdict
from math import ceil, floor
from statistics import mean, median, pstdev
from typing import Any, Iterable


CALIBRATED_SOURCES = {"measured_migration_ema"}
CALIBRATED_TRANSFER_MODELS = {
    "affine_migration_transport",
    "end_to_end_measured_bandwidth",
}


def _float_value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_is_calibrated(row: dict[str, Any]) -> bool:
    """Return True when a sample used learned workload + live edge models."""

    return (
        row.get("candidate_calibration_source") in CALIBRATED_SOURCES
        and row.get("candidate_transfer_model") in CALIBRATED_TRANSFER_MODELS
    )


def _error_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_value(row, field)
        if value is not None:
            values.append(abs(value))
    return values


def _actual_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _float_value(row, field)
        if value is not None and value >= 0:
            values.append(value)
    return values


def _percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("At least one sample is required")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_or_none(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    average = mean(values)
    deviation = pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": average,
        "median": median(values),
        "p95": _percentile(values, 95),
        "maximum": max(values),
        "standard_deviation": deviation,
        "coefficient_of_variation": deviation / average if average > 0 else 0.0,
    }


def summarize_migration_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build integrity-neutral descriptive summaries for a migration matrix.

    Accuracy thresholds are deliberately not encoded here. This function reports
    the measurements collected; paper acceptance criteria remain an analysis choice
    rather than a mechanism that can silently discard inconvenient samples.
    """

    completed = [row for row in rows if row.get("final_status") == "completed"]
    calibrated = [row for row in completed if row_is_calibrated(row)]
    cold_or_uncalibrated = [row for row in completed if not row_is_calibrated(row)]

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "sample_count": len(group_rows),
            "transfer_absolute_error_percent": _summary_or_none(
                _error_values(group_rows, "transfer_absolute_error_percent")
            ),
            "downtime_absolute_error_percent": _summary_or_none(
                _error_values(group_rows, "downtime_absolute_error_percent")
            ),
            "checkpoint_absolute_error_percent": _summary_or_none(
                _error_values(group_rows, "checkpoint_absolute_error_percent")
            ),
            "restore_absolute_error_percent": _summary_or_none(
                _error_values(group_rows, "restore_absolute_error_percent")
            ),
            "actual_transfer_seconds": _summary_or_none(
                _actual_values(group_rows, "actual_transfer_seconds")
            ),
            "actual_downtime_seconds": _summary_or_none(
                _actual_values(group_rows, "actual_downtime_seconds")
            ),
        }

    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in calibrated:
        source = str(row.get("source_node_id"))
        destination = str(row.get("destination_node_id"))
        try:
            size = int(row.get("requested_payload_bytes"))
        except (TypeError, ValueError):
            continue
        edge = f"{source}->{destination}"
        by_edge[edge].append(row)
        by_size[str(size)].append(row)
        by_case[(source, destination, size)].append(row)

    case_rows: list[dict[str, Any]] = []
    for (source, destination, size), group in sorted(by_case.items()):
        summary = group_summary(group)
        transfer = summary["transfer_absolute_error_percent"] or {}
        downtime = summary["downtime_absolute_error_percent"] or {}
        actual_transfer = summary["actual_transfer_seconds"] or {}
        actual_downtime = summary["actual_downtime_seconds"] or {}
        case_rows.append(
            {
                "source_node_id": source,
                "destination_node_id": destination,
                "checkpoint_bytes": size,
                "calibrated_sample_count": summary["sample_count"],
                "transfer_ape_median_pct": transfer.get("median"),
                "transfer_ape_p95_pct": transfer.get("p95"),
                "downtime_ape_median_pct": downtime.get("median"),
                "downtime_ape_p95_pct": downtime.get("p95"),
                "actual_transfer_median_seconds": actual_transfer.get("median"),
                "actual_downtime_median_seconds": actual_downtime.get("median"),
            }
        )

    return {
        "total_sample_count": len(rows),
        "completed_sample_count": len(completed),
        "calibrated_sample_count": len(calibrated),
        "cold_or_uncalibrated_sample_count": len(cold_or_uncalibrated),
        "overall_calibrated": group_summary(calibrated),
        "by_edge": {
            edge: group_summary(group) for edge, group in sorted(by_edge.items())
        },
        "by_checkpoint_bytes": {
            size: group_summary(group)
            for size, group in sorted(by_size.items(), key=lambda item: int(item[0]))
        },
        "cases": case_rows,
    }
