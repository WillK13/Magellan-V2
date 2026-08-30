#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d2 import (
    ALL_POLICIES,
    CANONICAL_TASK_MIX,
    CAPACITY_POLICIES,
    STATIC_POLICY,
    UNLIMITED_POLICY,
    maximal_packing_signatures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4D.2 multi-task auction bundle.")
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
        "initial_layout.csv",
        "task_outcomes.csv",
        "scenario_outcomes.csv",
        "policy_summary.csv",
        "auction_events.csv",
        "migration_events.csv",
        "migration_matrix.csv",
        "occupancy_timeline.csv",
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
    layout = read_csv(root / "initial_layout.csv")
    tasks = read_csv(root / "task_outcomes.csv")
    scenarios = read_csv(root / "scenario_outcomes.csv")
    policy_summary = read_csv(root / "policy_summary.csv")
    auctions = read_csv(root / "auction_events.csv")
    migrations = read_csv(root / "migration_events.csv")
    occupancy = read_csv(root / "occupancy_timeline.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if int(summary.get("seasonal_scenario_count") or 0) != 4:
        errors.append("canonical Stage 4D.2 requires four seasonal scenarios")
    if int(summary.get("tasks_per_scenario") or 0) != 11:
        errors.append("canonical Stage 4D.2 requires 11 tasks per scenario")
    if summary.get("task_mix_per_scenario") != CANONICAL_TASK_MIX:
        errors.append("summary task mix does not match canonical 4/3/4 population")
    if set(summary.get("policy_names") or []) != set(ALL_POLICIES):
        errors.append("summary policy set mismatch")

    d41 = Path(str(summary.get("source_stage4d1_bundle") or ""))
    c4 = Path(str(summary.get("source_stage4c_bundle") or ""))
    if not d41.is_dir():
        errors.append(f"source Stage 4D.1 bundle not found: {d41}")
    if not c4.is_dir():
        errors.append(f"source Stage 4C bundle not found: {c4}")
    if not errors:
        d41_errors = validate_checksums(d41)
        c4_errors = validate_checksums(c4)
        if d41_errors:
            errors.append("source Stage 4D.1 checksum failure: " + "; ".join(d41_errors))
        if c4_errors:
            errors.append("source Stage 4C checksum failure: " + "; ".join(c4_errors))

    scenario_ids = sorted({row["scenario_id"] for row in layout})
    if len(scenario_ids) != 4:
        errors.append(f"initial layout scenario count {len(scenario_ids)} != 4")
    if len(layout) != 44:
        errors.append(f"initial layout row count {len(layout)} != 44")

    maximal = maximal_packing_signatures(d41) if d41.is_dir() else {}
    for scenario_id in scenario_ids:
        subset = [row for row in layout if row["scenario_id"] == scenario_id]
        class_counts = Counter(row["class_id"] for row in subset)
        if dict(class_counts) != CANONICAL_TASK_MIX:
            errors.append(f"{scenario_id} task mix mismatch: {dict(class_counts)}")
        by_node: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in subset:
            by_node[row["initial_node_id"]].append(row)
        if len(by_node) != 7:
            errors.append(f"{scenario_id} initial layout does not cover seven nodes")
        for node_id, node_rows in by_node.items():
            counts = Counter(row["class_id"] for row in node_rows)
            signature = (
                counts["benchmark-json-medium"],
                counts["dendro-r9-t1p0"],
                counts["llm-distilgpt2"],
            )
            if signature not in maximal.get(node_id, set()):
                errors.append(f"{scenario_id}/{node_id} is not a Stage 4D.1 maximal packing: {signature}")

    expected_task_rows = 4 * 11 * len(ALL_POLICIES)
    if len(tasks) != expected_task_rows:
        errors.append(f"task outcome coverage {len(tasks)} != {expected_task_rows}")
    by_scenario_policy: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tasks:
        by_scenario_policy[(row["scenario_id"], row["policy"])].append(row)
        if not truthy(row["completed"]):
            errors.append(f"task did not complete: {row['scenario_id']}/{row['policy']}/{row['task_id']}")
        if int(row["bid_attempts"]) != int(row["bid_accepts"]) + int(row["bid_rejections"]):
            errors.append(f"bid accounting mismatch: {row['task_id']}/{row['policy']}")
    for scenario_id in scenario_ids:
        for policy in ALL_POLICIES:
            if len(by_scenario_policy[(scenario_id, policy)]) != 11:
                errors.append(f"{scenario_id}/{policy} task coverage != 11")

    if len(scenarios) != 4 * len(ALL_POLICIES):
        errors.append(f"scenario outcome coverage {len(scenarios)} != {4 * len(ALL_POLICIES)}")
    for scenario_id in scenario_ids:
        policies = {row["policy"] for row in scenarios if row["scenario_id"] == scenario_id}
        if policies != set(ALL_POLICIES):
            errors.append(f"{scenario_id} scenario policy set mismatch: {sorted(policies)}")

    static_tasks = [row for row in tasks if row["policy"] == STATIC_POLICY]
    if any(int(row["migrations"]) or int(row["bid_attempts"]) for row in static_tasks):
        errors.append("static baseline contains migrations or bids")
    unlimited_tasks = [row for row in tasks if row["policy"] == UNLIMITED_POLICY]
    if any(int(row["bid_rejections"]) for row in unlimited_tasks):
        errors.append("unlimited reference contains rejected bids")

    for row in occupancy:
        if row["policy"] not in CAPACITY_POLICIES:
            errors.append(f"occupancy row has non-capacity policy {row['policy']}")
        if float(row["used_cpu_cores"]) > float(row["capacity_cpu_cores"]) + 1e-9:
            errors.append(f"CPU overcommit at {row['scenario_id']}/{row['policy']}/{row['node_id']}/{row['at_utc']}")
        if int(row["used_memory_mb"]) > int(row["capacity_memory_mb"]):
            errors.append(f"memory overcommit at {row['scenario_id']}/{row['policy']}/{row['node_id']}/{row['at_utc']}")
        if int(row["used_gpu_count"]) > int(row["capacity_gpu_count"]):
            errors.append(f"GPU overcommit at {row['scenario_id']}/{row['policy']}/{row['node_id']}/{row['at_utc']}")

    accepted = [row for row in auctions if row.get("status") == "accepted"]
    rejected = [row for row in auctions if row.get("status") == "rejected"]
    if len(accepted) != len(migrations):
        errors.append(f"accepted auction count {len(accepted)} != migration count {len(migrations)}")
    if int(summary.get("auction_event_count") or 0) != len(auctions):
        errors.append("summary auction_event_count mismatch")
    if int(summary.get("migration_event_count") or 0) != len(migrations):
        errors.append("summary migration_event_count mismatch")
    if int(summary.get("capacity_violation_count") or 0) != 0:
        errors.append("summary reports capacity violations")

    summary_by_policy = {row["policy"]: row for row in policy_summary}
    if set(summary_by_policy) != set(ALL_POLICIES):
        errors.append("policy_summary policy set mismatch")
    for policy in CAPACITY_POLICIES:
        row = summary_by_policy.get(policy)
        if row is not None:
            if int(row["bid_attempts_total"]) != int(row["bid_accepts_total"]) + int(row["bid_rejections_total"]):
                errors.append(f"{policy} policy-summary bid accounting mismatch")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4D2_MULTITASK_AUCTION_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"seasonal_scenarios: {len(scenario_ids)}/4")
    print("tasks_per_scenario: 11")
    print(f"task_outcomes: {len(tasks)}/{expected_task_rows}")
    print(f"auction_events: {len(auctions)}")
    print(f"accepted_bids: {len(accepted)}")
    print(f"rejected_bids: {len(rejected)}")
    print(f"migration_events: {len(migrations)}")
    print("capacity_violations: 0")
    print(f"resource_contention_observed: {bool(summary.get('resource_contention_observed'))}")
    print(f"capacity_movement_observed: {bool(summary.get('capacity_movement_observed'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
