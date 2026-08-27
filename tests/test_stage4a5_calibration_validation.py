from __future__ import annotations

import csv
from pathlib import Path

import pytest

from magellan.experiments.stage4a5 import (
    checkpoint_scale,
    predict_runtime_seconds,
    runtime_model_tables,
    runtime_validation_row,
    summarize_migration_evidence,
    summarize_runtime_validation,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_runtime_model_tables_and_prediction(tmp_path: Path):
    classes = tmp_path / "classes.csv"
    nodes = tmp_path / "nodes.csv"
    write_csv(classes, [{"class_id": "benchmark-json-medium", "runtime_seconds_median": 120.0}])
    write_csv(nodes, [
        {"node_id": "boston", "slowdown_vs_canonical": 1.0},
        {"node_id": "south-australia", "slowdown_vs_canonical": 0.6},
    ])
    class_runtime, node_slowdown = runtime_model_tables(classes, nodes)
    assert predict_runtime_seconds(
        class_id="benchmark-json-medium",
        node_id="south-australia",
        class_runtime=class_runtime,
        node_slowdown=node_slowdown,
    ) == pytest.approx(72.0)


def test_runtime_validation_gate_passes_and_fails():
    good = []
    for node, actual in (("boston", 100.0), ("virginia", 95.0)):
        for trial in (1, 2):
            good.append(runtime_validation_row(
                class_id="benchmark-json-medium",
                workload="benchmark",
                node_id=node,
                trial=trial,
                run_id=f"r-{node}-{trial}",
                measurement_id=f"m-{node}-{trial}",
                actual_seconds=actual,
                predicted_seconds=actual * 1.05,
                telemetry_sample_count=10,
            ))
    summary = summarize_runtime_validation(good, median_gate_percent=20, p95_gate_percent=35)
    assert summary["runtime_model_transfer_passed"] is True
    bad = list(good)
    bad.append(runtime_validation_row(
        class_id="dendro-r9-t1p0",
        workload="dendro",
        node_id="south-australia",
        trial=1,
        run_id="bad",
        measurement_id="bad",
        actual_seconds=100.0,
        predicted_seconds=160.0,
        telemetry_sample_count=10,
    ))
    summary = summarize_runtime_validation(bad, median_gate_percent=20, p95_gate_percent=35)
    assert summary["runtime_model_transfer_passed"] is False


def test_checkpoint_scale_stratification():
    assert checkpoint_scale(250) == "tiny_lt_1MiB"
    assert checkpoint_scale(100 * 1024 * 1024) == "medium_1MiB_to_500MiB"
    assert checkpoint_scale(900 * 1024 * 1024) == "large_ge_500MiB"
    rows = [
        {"actual_checkpoint_bytes": "250", "predicted_downtime_seconds": "2", "actual_downtime_seconds": "3"},
        {"actual_checkpoint_bytes": str(100 * 1024 * 1024), "predicted_downtime_seconds": "10", "actual_downtime_seconds": "12"},
        {"actual_checkpoint_bytes": str(900 * 1024 * 1024), "predicted_downtime_seconds": "50", "actual_downtime_seconds": "55"},
    ]
    summary = summarize_migration_evidence(rows)
    assert summary["sample_count"] == 3
    assert {item["checkpoint_scale"] for item in summary["by_checkpoint_scale"]} == {
        "tiny_lt_1MiB", "medium_1MiB_to_500MiB", "large_ge_500MiB"
    }
