from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from magellan.experiments.bundle import write_checksums
from scripts.measure_stage4a4_static import CLEANUP_STATUSES, requires_local_model_asset

from magellan.experiments.stage4a4 import (
    build_static_cases,
    llm_training_definition,
    successful_static_bundle,
    summarize_canonical_runs,
    summarize_node_equivalence,
)


def write_profiles(path: Path) -> None:
    rows = []
    for benchmark, base in (("nbody", 600.0), ("json", 750.0), ("matmul", 650.0)):
        for index, size in enumerate(("small", "medium", "large"), start=1):
            rows.append({
                "class_id": f"benchmark-{benchmark}-{size}",
                "workload": benchmark,
                "variant": size,
                "progress_rate_median_units_per_second": str(base / index),
            })
    rows.extend([
        {"class_id": "dendro-r8-t3p0", "workload": "dendro", "variant": "r8-t3", "progress_rate_median_units_per_second": "0.02"},
        {"class_id": "dendro-r9-t1p0", "workload": "dendro", "variant": "r9-t1", "progress_rate_median_units_per_second": ""},
        {"class_id": "dendro-r10-t2p0", "workload": "dendro", "variant": "r10-t2", "progress_rate_median_units_per_second": "0.01"},
        {"class_id": "llm-distilgpt2", "workload": "llm", "variant": "experiment-assets/models/distilgpt2", "progress_rate_median_units_per_second": "0.09"},
    ])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_static_cases_sizes_finite_workloads_from_stage4a3(tmp_path: Path):
    profile = tmp_path / "profile_classes.csv"
    write_profiles(profile)
    cases = build_static_cases(profile, target_seconds=100)
    assert len(cases) == 13
    by_id = {case.class_id: case for case in cases}
    assert by_id["benchmark-nbody-small"].benchmark_iterations == 60000
    assert by_id["benchmark-nbody-large"].benchmark_iterations == 20000
    assert by_id["dendro-r8-t3p0"].resolution == 8
    assert by_id["dendro-r8-t3p0"].time_end == pytest.approx(3.0)
    assert by_id["llm-distilgpt2"].llm_max_steps == 9


def test_llm_static_definition_uses_finite_max_steps():
    definition = llm_training_definition(
        definition_id="static-llm",
        model="experiment-assets/models/distilgpt2",
        node_ids=["boston"],
        max_steps=9,
        checkpoint_every=1,
        sleep_per_step=2.0,
        torch_threads=2,
    )
    args = definition["runtime"]["arguments"]
    assert args[args.index("--max-steps") + 1] == "9"
    assert definition["profile"]["prestaged_node_ids"] == ["boston"]


def test_successful_static_bundle_rejects_migration_and_low_samples(tmp_path: Path):
    summary = {
        "passed": True,
        "status": "completed",
        "telemetry_sample_count": 4,
        "generation": 0,
        "wall_seconds": 100.0,
        "workload": "benchmark",
        "accumulated_migration_seconds": 0.0,
        "accumulated_transfer_cost_usd": 0.0,
        "accumulated_transfer_carbon_grams": 0.0,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_checksums(tmp_path)
    assert successful_static_bundle(tmp_path)
    summary["telemetry_sample_count"] = 2
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_checksums(tmp_path)
    assert not successful_static_bundle(tmp_path)
    summary["telemetry_sample_count"] = 4
    summary["accumulated_migration_seconds"] = 1.0
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_checksums(tmp_path)
    assert not successful_static_bundle(tmp_path)
    summary["telemetry_sample_count"] = 4
    summary["accumulated_migration_seconds"] = 0.0
    summary["workload"] = "dendro"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_checksums(tmp_path)
    assert not successful_static_bundle(tmp_path)
    summary["completion_detection"] = "operator_runtime_reconcile"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_checksums(tmp_path)
    assert successful_static_bundle(tmp_path)


def test_static_aggregates_and_node_slowdown():
    rows = []
    for class_id in ("benchmark-nbody-small", "benchmark-matmul-medium"):
        for trial, runtime in enumerate((90.0, 100.0, 110.0), start=1):
            rows.append({
                "class_id": class_id,
                "workload": "benchmark",
                "variant": class_id.rsplit("-", 1)[-1],
                "trial": trial,
                "wall_seconds": runtime + 1,
                "accumulated_runtime_seconds": runtime,
                "accumulated_cost_usd": runtime / 36000,
                "accumulated_carbon_grams": runtime / 100,
                "telemetry_sample_count": 10,
            })
    classes = summarize_canonical_runs(rows, trials=3)
    assert len(classes) == 2
    assert classes[0]["runtime_seconds_median"] == pytest.approx(101.0)
    assert classes[0]["accounting_runtime_seconds_median_diagnostic"] == pytest.approx(100.0)

    eq = []
    for node, runtimes in (("boston", (90, 100, 110)), ("virginia", (99, 110, 121))):
        for runtime in runtimes:
            eq.append({
                "node_id": node,
                "wall_seconds": runtime,
                "accumulated_runtime_seconds": runtime / 10,
            })
    summary = summarize_node_equivalence(eq, canonical_node_id="boston", trials=3)
    by_node = {row["node_id"]: row for row in summary}
    assert by_node["boston"]["slowdown_vs_canonical"] == pytest.approx(1.0)
    assert by_node["virginia"]["slowdown_vs_canonical"] == pytest.approx(1.1)


def test_static_cleanup_includes_failed_tasks():
    assert "failed" in CLEANUP_STATUSES


def test_static_llm_model_asset_detection():
    assert requires_local_model_asset("experiment-assets/models/distilgpt2")
    assert requires_local_model_asset("./model")
    assert not requires_local_model_asset("distilgpt2")
