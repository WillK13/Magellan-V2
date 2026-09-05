#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d2 import maximal_packing_signatures, read_resource_model
from magellan.experiments.stage5e2 import (
    EXPECTED_CLASS_COUNTS,
    STAGE5E2_LAYOUT,
    stage5e2_passes,
    validate_physical_layout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 5E.2 physical heterogeneous packing bundle."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_source_bundle(path: Path, label: str) -> dict[str, Any]:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum failure: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(f"{label} bundle did not pass")
    return summary


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        print("ERROR: summary passed is not true")
        return 2

    stage5a = Path(str(summary.get("source_stage5a_bundle") or ""))
    stage5e1 = Path(str(summary.get("source_stage5e1_bundle") or ""))
    stage4d1 = Path(str(summary.get("source_stage4d1_bundle") or ""))
    try:
        s5a_summary = require_source_bundle(stage5a, "Stage 5A")
        s5e1_summary = require_source_bundle(stage5e1, "Stage 5E.1")
        require_source_bundle(stage4d1, "Stage 4D.1")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    expected_sha = str(s5a_summary.get("target_git_sha") or "")
    if summary.get("git_sha") != expected_sha:
        print("ERROR: Stage 5E.2 git_sha does not match source Stage 5A")
        return 2
    if int(s5e1_summary.get("passed_case_count") or 0) != 3:
        print("ERROR: source Stage 5E.1 is not 3/3")
        return 2

    task_rows = read_csv(root / "tasks.csv")
    node_rows = read_csv(root / "nodes.csv")
    planned_rows = read_csv(root / "planned_layout.csv")
    task_samples = read_csv(root / "task_profile_samples.csv")
    node_samples = read_csv(root / "node_profile_samples.csv")

    if not stage5e2_passes(task_rows, node_rows):
        print("ERROR: Stage 5E.2 task/node pass invariants failed")
        return 2

    capacities, requests = read_resource_model(stage4d1)
    expected_plan = validate_physical_layout(
        capacities=capacities,
        requests=requests,
        maximal_signatures=maximal_packing_signatures(stage4d1),
    )
    planned_by_node = {row["node_id"]: row for row in planned_rows}
    if set(planned_by_node) != set(STAGE5E2_LAYOUT):
        print("ERROR: planned layout node set mismatch")
        return 2
    for expected in expected_plan:
        actual = planned_by_node[expected["node_id"]]
        if actual.get("packing_signature") != str(expected["packing_signature"]):
            print(f"ERROR: packing signature mismatch on {expected['node_id']}")
            return 2
        if abs(
            float(actual.get("expected_reserved_cpu_cores") or 0.0)
            - float(expected["expected_reserved_cpu_cores"])
        ) > 1e-9:
            print(f"ERROR: reserved CPU plan mismatch on {expected['node_id']}")
            return 2

    if int(summary.get("task_count") or 0) != 11:
        print("ERROR: task_count is not 11")
        return 2
    if int(summary.get("node_count") or 0) != 7:
        print("ERROR: node_count is not 7")
        return 2
    if summary.get("expected_class_counts") != EXPECTED_CLASS_COUNTS:
        print("ERROR: expected class counts changed")
        return 2
    if int(summary.get("frozen_maximal_packing_nodes") or 0) != 7:
        print("ERROR: not all nodes are frozen maximal packings")
        return 2
    if int(summary.get("reservation_match_nodes") or 0) != 7:
        print("ERROR: not all reservation ledgers match the frozen model")
        return 2
    if int(summary.get("capacity_respected_nodes") or 0) != 7:
        print("ERROR: resource capacity was not respected on all nodes")
        return 2
    if int(summary.get("steady_running_tasks") or 0) != 11:
        print("ERROR: not all 11 tasks remained running")
        return 2
    if int(summary.get("cleanup_ok_tasks") or 0) != 11:
        print("ERROR: not all 11 tasks cleaned up")
        return 2
    if len(task_samples) < 22:
        print("ERROR: fewer than two task telemetry samples per task")
        return 2
    if len(node_samples) < 14:
        print("ERROR: fewer than two node resource samples per node")
        return 2

    print("STAGE_5E2_PHYSICAL_HETEROGENEOUS_PACKING_BUNDLE_PASS")
    print(f"comparison_id: {summary['comparison_id']}")
    print(f"git_sha: {summary['git_sha']}")
    print("tasks: 11/11")
    print("mix: benchmark=4 llm=4 dendro=3")
    print("nodes: 7/7 frozen maximal packings")
    print("reservation_matches: 7/7")
    print("capacity_respected: 7/7")
    print(f"profile_sample_rounds: {summary.get('profile_sample_rounds')}")
    print(
        "planned_cpu_fraction: "
        f"{float(summary.get('planned_cpu_fraction') or 0.0) * 100.0:.2f}%"
    )
    observed = summary.get("mean_cluster_task_cpu_percent")
    if observed is not None:
        print(f"mean_observed_task_cpu_percent: {float(observed):.1f}%")
    print("cleanup: 11/11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
