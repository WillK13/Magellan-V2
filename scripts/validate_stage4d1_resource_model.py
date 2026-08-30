#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d import CORE_RESOURCE_CLASSES


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4D.1 resource model bundle.")
    parser.add_argument("bundle")
    args = parser.parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = (
        "metadata.json",
        "summary.json",
        "node_capacities.csv",
        "workload_resource_requests.csv",
        "homogeneous_capacity.csv",
        "maximal_packings.csv",
    )
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    nodes = read_csv(root / "node_capacities.csv")
    workloads = read_csv(root / "workload_resource_requests.csv")
    homogeneous = read_csv(root / "homogeneous_capacity.csv")
    packings = read_csv(root / "maximal_packings.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed=false")
    if metadata.get("stage") != "4D.1":
        errors.append("metadata stage is not 4D.1")
    if metadata.get("task_slot_capacity") is not None:
        errors.append("Stage 4D.1 must not introduce a synthetic task-slot capacity")
    if len(nodes) != 7 or int(summary.get("node_count", -1)) != 7:
        errors.append(f"Expected 7 node capacities, found {len(nodes)}")

    workload_ids = {row["class_id"] for row in workloads}
    if workload_ids != set(CORE_RESOURCE_CLASSES):
        errors.append(f"Unexpected workload classes: {sorted(workload_ids)}")
    expected_homogeneous = len(nodes) * len(CORE_RESOURCE_CLASSES)
    if len(homogeneous) != expected_homogeneous:
        errors.append(
            f"Expected {expected_homogeneous} homogeneous capacity rows, found {len(homogeneous)}"
        )
    for row in nodes:
        configured_cpu = float(row["configured_cpu_cores"])
        observed_cpu = float(row["observed_cpu_cores"])
        effective_cpu = float(row["effective_cpu_cores"])
        configured_mem = int(float(row["configured_memory_mb"]))
        observed_mem = int(float(row["observed_memory_mb"]))
        effective_mem = int(float(row["effective_memory_mb"]))
        if abs(effective_cpu - min(configured_cpu, observed_cpu)) > 1e-9:
            errors.append(f"{row['node_id']}: effective CPU is not min(configured, observed)")
        if effective_mem != min(configured_mem, observed_mem):
            errors.append(f"{row['node_id']}: effective memory is not min(configured, observed)")
    for row in workloads:
        cpu_percent = float(row["cpu_p95_percent"])
        cpu_request = float(row["cpu_request_cores"])
        memory_p95 = float(row["memory_p95_mb"])
        memory_request = int(float(row["memory_request_mb"]))
        if abs(cpu_request - cpu_percent / 100.0) > 1e-9:
            errors.append(f"{row['class_id']}: CPU request does not equal p95/100")
        if memory_request < memory_p95:
            errors.append(f"{row['class_id']}: memory request is below measured p95")
    for row in homogeneous:
        if row["individually_feasible"].lower() != "true":
            errors.append(f"Infeasible core workload: {row['node_id']} {row['class_id']}")
        if int(row["max_concurrent_tasks"]) < 1:
            errors.append(f"Zero homogeneous capacity: {row['node_id']} {row['class_id']}")
    if not packings:
        errors.append("No maximal packings were generated")

    if errors:
        print("STAGE_4D1_RESOURCE_MODEL_BUNDLE_FAILED")
        for error in errors:
            print(f"- {error}")
        return 2

    print("STAGE_4D1_RESOURCE_MODEL_BUNDLE_PASS")
    print(f"model_id: {summary.get('model_id')}")
    print(f"nodes: {len(nodes)}/7")
    print(f"workloads: {len(workloads)}/{len(CORE_RESOURCE_CLASSES)}")
    print(f"maximal_packings: {len(packings)}")
    print("homogeneous_concurrency:")
    for class_id in sorted(CORE_RESOURCE_CLASSES):
        values = sorted({
            int(row["max_concurrent_tasks"])
            for row in homogeneous
            if row["class_id"] == class_id
        })
        print(f"  {class_id}: {','.join(str(value) for value in values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
