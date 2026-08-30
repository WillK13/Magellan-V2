#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage4d import (
    CORE_RESOURCE_CLASSES,
    homogeneous_capacity_rows,
    load_node_resource_evidence,
    load_workload_resource_evidence,
    maximal_packing_rows,
    node_capacity_rows,
    workload_request_rows,
)


NODE_FIELDS = [
    "node_id",
    "machine_type",
    "configured_cpu_cores",
    "observed_cpu_cores",
    "effective_cpu_cores",
    "configured_memory_mb",
    "observed_memory_mb",
    "effective_memory_mb",
    "configured_gpu_count",
    "observed_gpu_count",
    "effective_gpu_count",
    "capacity_source",
]
WORKLOAD_FIELDS = [
    "class_id",
    "cpu_p95_percent",
    "cpu_request_cores",
    "memory_p95_mb",
    "memory_request_mb",
    "gpu_request_count",
    "request_source",
]
HOMOGENEOUS_FIELDS = [
    "node_id",
    "class_id",
    "max_concurrent_tasks",
    "individually_feasible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Stage 4D.1 node capacities and workload resource requests "
            "from Stage 4A.1 hardware and Stage 4A.3 p95 profiles."
        )
    )
    parser.add_argument("--stage4b-bundle", required=True)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--model-id")
    return parser.parse_args()


def require_bundle(path: Path, label: str, *, require_passed: bool = True) -> dict:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum validation failed: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if require_passed and summary.get("passed") is not True:
        raise RuntimeError(f"{label} summary passed=false")
    return summary


def source_bundle(stage4b_summary: dict, key: str) -> Path:
    value = stage4b_summary.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Stage 4B summary missing {key}")
    return Path(value)


def main() -> int:
    args = parse_args()
    stage4b = Path(args.stage4b_bundle)
    stage4b_summary = require_bundle(stage4b, "Stage 4B")
    a1 = source_bundle(stage4b_summary, "stage4a1_bundle")
    a3 = source_bundle(stage4b_summary, "stage4a3_bundle")

    a1_summary = require_bundle(a1, "Stage 4A.1", require_passed=False)
    if a1_summary.get("hardware_preflight_passed") is not True:
        raise RuntimeError("Stage 4A.1 hardware preflight did not pass")
    a3_summary = require_bundle(a3, "Stage 4A.3")

    capacities = load_node_resource_evidence(a1)
    requests = load_workload_resource_evidence(a3, class_ids=CORE_RESOURCE_CLASSES)
    node_rows = node_capacity_rows(capacities)
    workload_rows = workload_request_rows(requests)
    homogeneous_rows = homogeneous_capacity_rows(capacities, requests)
    packing_rows = maximal_packing_rows(capacities, requests)

    model_id = args.model_id or (
        f"stage4d1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / model_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    class_ids = sorted(requests)
    packing_fields = [
        "node_id",
        "packing_index",
        "total_tasks",
        "used_cpu_cores",
        "used_memory_mb",
        "used_gpu_count",
        "cpu_fraction",
        "memory_fraction",
        *[f"count_{class_id}" for class_id in class_ids],
    ]

    write_csv(root / "node_capacities.csv", node_rows, NODE_FIELDS)
    write_csv(root / "workload_resource_requests.csv", workload_rows, WORKLOAD_FIELDS)
    write_csv(root / "homogeneous_capacity.csv", homogeneous_rows, HOMOGENEOUS_FIELDS)
    write_csv(root / "maximal_packings.csv", packing_rows, packing_fields)

    homogeneous_by_class = {
        class_id: sorted(
            {
                int(row["max_concurrent_tasks"])
                for row in homogeneous_rows
                if row["class_id"] == class_id
            }
        )
        for class_id in class_ids
    }
    effective_shapes = sorted(
        {
            (
                float(row["effective_cpu_cores"]),
                int(row["effective_memory_mb"]),
                int(row["effective_gpu_count"]),
            )
            for row in node_rows
        }
    )
    all_individually_feasible = all(bool(row["individually_feasible"]) for row in homogeneous_rows)
    passed = (
        len(capacities) == 7
        and len(requests) == len(CORE_RESOURCE_CLASSES)
        and all_individually_feasible
        and all(values and min(values) > 0 for values in homogeneous_by_class.values())
    )

    metadata = {
        "stage": "4D.1",
        "model_id": model_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_stage4b_bundle": str(stage4b),
        "source_stage4a1_bundle": str(a1),
        "source_stage4a3_bundle": str(a3),
        "capacity_rule": "componentwise minimum of Stage 4A.1 configured and observed resources",
        "workload_request_rule": "Stage 4A.3 aggregate p95 CPU and RSS memory; CPU percent / 100 = cores",
        "task_slot_capacity": None,
        "task_slot_capacity_note": "No synthetic task-slot cap; resource vectors are the admission constraint.",
        "workload_classes": class_ids,
    }
    summary = {
        "model_id": model_id,
        "passed": passed,
        "node_count": len(capacities),
        "workload_class_count": len(requests),
        "all_individually_feasible": all_individually_feasible,
        "effective_resource_shapes": [
            {"cpu_cores": cpu, "memory_mb": memory, "gpu_count": gpu}
            for cpu, memory, gpu in effective_shapes
        ],
        "homogeneous_concurrency_by_class": homogeneous_by_class,
        "maximal_packing_count": len(packing_rows),
        "stage4a1_bundle": str(a1),
        "stage4a3_bundle": str(a3),
        "source_stage4b_bundle": str(stage4b),
    }
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    print("== Stage 4D.1 evidence-backed resource capacity model ==")
    print(f"model_id={model_id}")
    print(f"source_stage4b={stage4b}")
    print(f"stage4a1={a1}")
    print(f"stage4a3={a3}")
    print(f"nodes={len(capacities)} workloads={len(requests)}")
    for row in workload_rows:
        class_id = row["class_id"]
        concurrency = homogeneous_by_class[class_id]
        print(
            f"[request] {class_id:28s} "
            f"cpu={float(row['cpu_request_cores']):.3f} "
            f"memory={int(row['memory_request_mb'])}MB "
            f"homogeneous_concurrency={','.join(str(value) for value in concurrency)}"
        )
    if passed:
        print("STAGE_4D1_RESOURCE_MODEL_PASS")
        print(f"bundle: {root}")
        return 0
    print("STAGE_4D1_RESOURCE_MODEL_FAIL")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
