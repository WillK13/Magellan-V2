from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from magellan.experiments.measurement import absolute_percent_error, summarize_samples


DEFAULT_VALIDATION_CLASSES = ("benchmark-json-medium", "dendro-r9-t1p0", "llm-distilgpt2")
DEFAULT_VALIDATION_NODES = ("boston", "south-australia", "ethiopia", "virginia")
DEFAULT_MEDIAN_ERROR_GATE_PERCENT = 20.0
DEFAULT_P95_ERROR_GATE_PERCENT = 35.0
MIB = 1024 * 1024


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def runtime_model_tables(
    static_classes_csv: str | Path,
    node_equivalence_csv: str | Path,
) -> tuple[dict[str, float], dict[str, float]]:
    class_runtime = {
        row["class_id"]: float(row["runtime_seconds_median"])
        for row in read_csv(static_classes_csv)
    }
    node_slowdown = {
        row["node_id"]: float(row["slowdown_vs_canonical"])
        for row in read_csv(node_equivalence_csv)
    }
    if not class_runtime or not node_slowdown:
        raise ValueError("Stage 4A.4 runtime-model tables must be non-empty")
    if any(value <= 0 for value in class_runtime.values()):
        raise ValueError("Canonical class runtimes must be positive")
    if any(value <= 0 for value in node_slowdown.values()):
        raise ValueError("Node slowdown factors must be positive")
    return class_runtime, node_slowdown


def predict_runtime_seconds(
    *,
    class_id: str,
    node_id: str,
    class_runtime: dict[str, float],
    node_slowdown: dict[str, float],
) -> float:
    try:
        return class_runtime[class_id] * node_slowdown[node_id]
    except KeyError as exc:
        raise ValueError(f"Missing runtime-model term for {exc.args[0]}") from exc


def runtime_validation_row(
    *,
    class_id: str,
    workload: str,
    node_id: str,
    trial: int,
    run_id: str,
    measurement_id: str,
    actual_seconds: float,
    predicted_seconds: float,
    telemetry_sample_count: int,
) -> dict[str, Any]:
    if actual_seconds <= 0 or predicted_seconds <= 0:
        raise ValueError("Runtime validation values must be positive")
    error = absolute_percent_error(predicted_seconds, actual_seconds)
    assert error is not None
    return {
        "measurement_id": measurement_id,
        "class_id": class_id,
        "workload": workload,
        "node_id": node_id,
        "trial": trial,
        "run_id": run_id,
        "predicted_runtime_seconds": predicted_seconds,
        "actual_runtime_seconds": actual_seconds,
        "absolute_error_seconds": abs(predicted_seconds - actual_seconds),
        "absolute_error_percent": error,
        "telemetry_sample_count": telemetry_sample_count,
    }


def _group_error_summaries(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row["absolute_error_percent"]))
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        stats = summarize_samples(grouped[key]).as_dict()
        output.append({field: key, **stats})
    return output


def summarize_runtime_validation(
    rows: list[dict[str, Any]],
    *,
    median_gate_percent: float = DEFAULT_MEDIAN_ERROR_GATE_PERCENT,
    p95_gate_percent: float = DEFAULT_P95_ERROR_GATE_PERCENT,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Runtime validation rows must be non-empty")
    if median_gate_percent <= 0 or p95_gate_percent <= 0:
        raise ValueError("Runtime validation gates must be positive")
    errors = [float(row["absolute_error_percent"]) for row in rows]
    overall = summarize_samples(errors).as_dict()
    by_class = _group_error_summaries(rows, "class_id")
    by_node = _group_error_summaries(rows, "node_id")

    group_gate_passed = all(
        float(group["median"]) <= median_gate_percent
        and float(group["p95"]) <= p95_gate_percent
        for group in [*by_class, *by_node]
    )
    passed = bool(
        float(overall["median"]) <= median_gate_percent
        and float(overall["p95"]) <= p95_gate_percent
        and group_gate_passed
    )
    return {
        "sample_count": len(rows),
        "median_error_gate_percent": median_gate_percent,
        "p95_error_gate_percent": p95_gate_percent,
        "overall_absolute_error_percent": overall,
        "by_class": by_class,
        "by_node": by_node,
        "runtime_model_transfer_passed": passed,
        "recommended_runtime_model": (
            "single_node_slowdown_factor"
            if passed
            else "collect_workload_family_specific_or_direct_per_node_runtime_factors"
        ),
    }


def checkpoint_scale(checkpoint_bytes: int) -> str:
    if checkpoint_bytes < MIB:
        return "tiny_lt_1MiB"
    if checkpoint_bytes < 500 * MIB:
        return "medium_1MiB_to_500MiB"
    return "large_ge_500MiB"


def summarize_migration_evidence(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "by_checkpoint_scale": []}
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        size = int(float(row.get("actual_checkpoint_bytes") or 0))
        predicted = float(row.get("predicted_downtime_seconds") or 0.0)
        actual = float(row.get("actual_downtime_seconds") or 0.0)
        if actual <= 0:
            continue
        pct = absolute_percent_error(predicted, actual)
        assert pct is not None
        grouped[checkpoint_scale(size)].append(
            {
                "absolute_error_seconds": abs(predicted - actual),
                "absolute_error_percent": pct,
            }
        )
    output = []
    for scale in sorted(grouped):
        items = grouped[scale]
        output.append(
            {
                "checkpoint_scale": scale,
                "sample_count": len(items),
                "absolute_downtime_error_seconds": summarize_samples(
                    item["absolute_error_seconds"] for item in items
                ).as_dict(),
                "absolute_downtime_error_percent": summarize_samples(
                    item["absolute_error_percent"] for item in items
                ).as_dict(),
            }
        )
    return {
        "sample_count": sum(item["sample_count"] for item in output),
        "by_checkpoint_scale": output,
    }
