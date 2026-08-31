#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d2 import (
    LOWEST_SCORE_POLICY,
    STATIC_POLICY,
    UNLIMITED_POLICY,
)
from magellan.experiments.stage4d3 import (
    LOAD_ORDER,
    LOAD_QUOTAS,
    SWEEP_POLICIES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4D.3 utilization-sweep bundle.")
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
        "load_cases.csv",
        "initial_layout.csv",
        "task_outcomes.csv",
        "scenario_outcomes.csv",
        "utilization_summary.csv",
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
    load_cases = read_csv(root / "load_cases.csv")
    layout = read_csv(root / "initial_layout.csv")
    tasks = read_csv(root / "task_outcomes.csv")
    scenarios = read_csv(root / "scenario_outcomes.csv")
    utilization = read_csv(root / "utilization_summary.csv")
    auctions = read_csv(root / "auction_events.csv")
    migrations = read_csv(root / "migration_events.csv")
    occupancy = read_csv(root / "occupancy_timeline.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("load_ids") or []) != list(LOAD_ORDER):
        errors.append("summary load_ids mismatch")
    if set(summary.get("policy_names") or []) != set(SWEEP_POLICIES):
        errors.append("summary policy set mismatch")

    d42 = Path(str(summary.get("source_stage4d2_bundle") or ""))
    if not d42.is_dir():
        errors.append(f"source Stage 4D.2 bundle not found: {d42}")
    else:
        d42_errors = validate_checksums(d42)
        if d42_errors:
            errors.append("source Stage 4D.2 checksum validation failed: " + "; ".join(d42_errors))

    season_count = int(summary.get("seasonal_scenario_count") or 0)
    if season_count not in {1, 4}:
        errors.append(f"seasonal_scenario_count must be 1 or 4, got {season_count}")
    expected_case_count = season_count * len(LOAD_ORDER)
    if len(load_cases) != expected_case_count:
        errors.append(f"load case coverage {len(load_cases)} != {expected_case_count}")

    by_season: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_cases:
        by_season[row["season"]].append(row)

    expected_counts = {
        load_id: (
            LOAD_QUOTAS[load_id]["benchmark-json-medium"],
            LOAD_QUOTAS[load_id]["dendro-r9-t1p0"],
            LOAD_QUOTAS[load_id]["llm-distilgpt2"],
        )
        for load_id in LOAD_ORDER
    }

    for season, rows in by_season.items():
        by_load = {row["load_id"]: row for row in rows}
        if set(by_load) != set(LOAD_ORDER):
            errors.append(f"{season} load-id coverage mismatch: {sorted(by_load)}")
            continue
        fractions = [float(by_load[load_id]["achieved_initial_cpu_fraction"]) for load_id in LOAD_ORDER]
        if not all(a < b for a, b in zip(fractions, fractions[1:])):
            errors.append(f"{season} achieved utilization is not strictly increasing: {fractions}")
        for load_id, target in (("u25", 0.25), ("u50", 0.50), ("u75", 0.75)):
            achieved = float(by_load[load_id]["achieved_initial_cpu_fraction"])
            if abs(achieved - target) > 0.02:
                errors.append(f"{season}/{load_id} achieved utilization {achieved:.4f} is too far from {target:.2f}")
        if not 0.75 < float(by_load["umax"]["achieved_initial_cpu_fraction"]) <= 1.0:
            errors.append(f"{season}/umax utilization is outside (0.75, 1.0]")

        task_ids_by_load: dict[str, set[str]] = {}
        for load_id in LOAD_ORDER:
            case = by_load[load_id]
            scenario_id = case["scenario_id"]
            subset = [row for row in layout if row["scenario_id"] == scenario_id]
            counts = Counter(row["class_id"] for row in subset)
            observed = (
                counts["benchmark-json-medium"],
                counts["dendro-r9-t1p0"],
                counts["llm-distilgpt2"],
            )
            if observed != expected_counts[load_id]:
                errors.append(f"{season}/{load_id} class mix mismatch: {observed}")
            task_ids_by_load[load_id] = {row["task_id"] for row in subset}
            if len(subset) != int(case["task_count"]):
                errors.append(f"{season}/{load_id} layout task_count mismatch")

        if not task_ids_by_load["u25"] <= task_ids_by_load["u50"]:
            errors.append(f"{season} u25 is not nested inside u50")
        if not task_ids_by_load["u50"] <= task_ids_by_load["u75"]:
            errors.append(f"{season} u50 is not nested inside u75")
        if not task_ids_by_load["u75"] <= task_ids_by_load["umax"]:
            errors.append(f"{season} u75 is not nested inside umax")

    by_case_policy: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tasks:
        by_case_policy[(row["scenario_id"], row["policy"])].append(row)
        if not truthy(row["completed"]):
            errors.append(f"task did not complete: {row['scenario_id']}/{row['policy']}/{row['task_id']}")
        if int(row["bid_attempts"]) != int(row["bid_accepts"]) + int(row["bid_rejections"]):
            errors.append(f"bid accounting mismatch: {row['scenario_id']}/{row['policy']}/{row['task_id']}")

    for case in load_cases:
        scenario_id = case["scenario_id"]
        task_count = int(case["task_count"])
        for policy in SWEEP_POLICIES:
            if len(by_case_policy[(scenario_id, policy)]) != task_count:
                errors.append(
                    f"{scenario_id}/{policy} task coverage "
                    f"{len(by_case_policy[(scenario_id, policy)])} != {task_count}"
                )

    if len(scenarios) != expected_case_count * len(SWEEP_POLICIES):
        errors.append(
            f"scenario outcome coverage {len(scenarios)} != "
            f"{expected_case_count * len(SWEEP_POLICIES)}"
        )
    for case in load_cases:
        policies = {row["policy"] for row in scenarios if row["scenario_id"] == case["scenario_id"]}
        if policies != set(SWEEP_POLICIES):
            errors.append(f"{case['scenario_id']} scenario policy set mismatch: {sorted(policies)}")

    static_tasks = [row for row in tasks if row["policy"] == STATIC_POLICY]
    if any(int(row["migrations"]) or int(row["bid_attempts"]) for row in static_tasks):
        errors.append("static baseline contains migrations or bids")
    unlimited_tasks = [row for row in tasks if row["policy"] == UNLIMITED_POLICY]
    if any(int(row["bid_rejections"]) for row in unlimited_tasks):
        errors.append("unlimited reference contains rejected bids")

    for row in occupancy:
        if row["policy"] != LOWEST_SCORE_POLICY:
            errors.append(f"occupancy row has unexpected policy {row['policy']}")
        if float(row["used_cpu_cores"]) > float(row["capacity_cpu_cores"]) + 1e-9:
            errors.append(f"CPU overcommit at {row['scenario_id']}/{row['node_id']}/{row['at_utc']}")
        if int(row["used_memory_mb"]) > int(row["capacity_memory_mb"]):
            errors.append(f"memory overcommit at {row['scenario_id']}/{row['node_id']}/{row['at_utc']}")
        if int(row["used_gpu_count"]) > int(row["capacity_gpu_count"]):
            errors.append(f"GPU overcommit at {row['scenario_id']}/{row['node_id']}/{row['at_utc']}")

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

    expected_util_rows = len(LOAD_ORDER) * len(SWEEP_POLICIES)
    if len(utilization) != expected_util_rows:
        errors.append(f"utilization_summary coverage {len(utilization)} != {expected_util_rows}")
    util_keys = {(row["load_id"], row["policy"]) for row in utilization}
    expected_util_keys = {(load_id, policy) for load_id in LOAD_ORDER for policy in SWEEP_POLICIES}
    if util_keys != expected_util_keys:
        errors.append("utilization_summary load/policy coverage mismatch")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4D3_UTILIZATION_SWEEP_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"season_selection: {summary.get('season_selection')}")
    print(f"load_cases: {len(load_cases)}/{expected_case_count}")
    print(f"task_outcomes: {len(tasks)}")
    print(f"auction_events: {len(auctions)}")
    print(f"accepted_bids: {len(accepted)}")
    print(f"rejected_bids: {len(rejected)}")
    print(f"migration_events: {len(migrations)}")
    print("capacity_violations: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
