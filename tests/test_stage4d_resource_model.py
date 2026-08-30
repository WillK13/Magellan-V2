from __future__ import annotations

import csv
import json
from pathlib import Path

from magellan.experiments.stage4d import (
    CORE_RESOURCE_CLASSES,
    enumerate_maximal_packings,
    homogeneous_capacity_rows,
    load_node_resource_evidence,
    load_workload_resource_evidence,
)


def _write_hardware(root: Path) -> None:
    hardware = {}
    for node_id in (
        "boston",
        "california",
        "south-australia",
        "nepal",
        "ethiopia",
        "france",
        "virginia",
    ):
        hardware[node_id] = {
            "configured": {
                "machine_type": "e2-highmem-2",
                "resources": {
                    "cpu_cores": 2.0,
                    "memory_mb": 16384,
                    "gpu_count": 0,
                    "accelerator_types": [],
                },
            },
            "capabilities": {
                "observed": {
                    "cpu_cores": 2.0,
                    "memory_mb": 16002,
                    "gpu_count": 0,
                    "accelerator_types": [],
                }
            },
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "hardware.json").write_text(json.dumps(hardware), encoding="utf-8")


def _write_profiles(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "class_id": "benchmark-json-medium",
            "cpu_p95_percent": "99.7",
            "memory_p95_mb": "13.0",
        },
        {
            "class_id": "dendro-r9-t1p0",
            "cpu_p95_percent": "177.6",
            "memory_p95_mb": "1378.7",
        },
        {
            "class_id": "llm-distilgpt2",
            "cpu_p95_percent": "76.3",
            "memory_p95_mb": "1571.4",
        },
    ]
    with (root / "profile_classes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "cpu_p95_percent", "memory_p95_mb"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_stage4d_uses_conservative_observed_node_capacity(tmp_path: Path) -> None:
    root = tmp_path / "a1"
    _write_hardware(root)
    capacities = load_node_resource_evidence(root)

    assert set(capacities) == {
        "boston",
        "california",
        "south-australia",
        "nepal",
        "ethiopia",
        "france",
        "virginia",
    }
    boston = capacities["boston"]
    assert boston.effective.cpu_cores == 2.0
    assert boston.effective.memory_mb == 16002
    assert boston.effective.gpu_count == 0


def test_stage4d_resource_requests_come_directly_from_stage4a3_p95(tmp_path: Path) -> None:
    root = tmp_path / "a3"
    _write_profiles(root)
    requests = load_workload_resource_evidence(root)

    assert requests["benchmark-json-medium"].request.cpu_cores == 0.997
    assert requests["benchmark-json-medium"].request.memory_mb == 13
    assert requests["dendro-r9-t1p0"].request.cpu_cores == 1.776
    assert requests["dendro-r9-t1p0"].request.memory_mb == 1379
    assert requests["llm-distilgpt2"].request.cpu_cores == 0.763
    assert requests["llm-distilgpt2"].request.memory_mb == 1572


def test_stage4d_measured_homogeneous_concurrency(tmp_path: Path) -> None:
    a1 = tmp_path / "a1"
    a3 = tmp_path / "a3"
    _write_hardware(a1)
    _write_profiles(a3)
    capacities = load_node_resource_evidence(a1)
    requests = load_workload_resource_evidence(a3)
    rows = homogeneous_capacity_rows(capacities, requests)

    per_class = {
        class_id: {
            int(row["max_concurrent_tasks"])
            for row in rows
            if row["class_id"] == class_id
        }
        for class_id in CORE_RESOURCE_CLASSES
    }
    assert per_class == {
        "benchmark-json-medium": {2},
        "dendro-r9-t1p0": {1},
        "llm-distilgpt2": {2},
    }


def test_stage4d_mixed_packings_are_resource_derived(tmp_path: Path) -> None:
    a1 = tmp_path / "a1"
    a3 = tmp_path / "a3"
    _write_hardware(a1)
    _write_profiles(a3)
    capacity = load_node_resource_evidence(a1)["boston"].effective
    requests = load_workload_resource_evidence(a3)

    packings = enumerate_maximal_packings(capacity, requests)
    counts = {tuple(row["counts"][class_id] for class_id in sorted(requests)) for row in packings}

    # sorted(requests) = benchmark, dendro, llm
    assert counts == {
        (2, 0, 0),
        (1, 0, 1),
        (0, 1, 0),
        (0, 0, 2),
    }
