#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.bidding.models import AuctionStrategy
from magellan.carbon.store import CarbonMetric, CarbonStore, as_utc_timestamp
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage4b import (
    CORE_WORKLOADS,
    load_node_slowdowns,
    load_stage4a1_edges,
    load_workload_calibrations,
)
from magellan.experiments.stage4c import runtime_scales_for_target
from magellan.experiments.stage4d2 import (
    ALL_POLICIES,
    CAPACITY_POLICIES,
    CREDIT_FAIR_POLICY,
    LOWEST_SCORE_POLICY,
    STATIC_POLICY,
    UNLIMITED_POLICY,
    attach_static_ratios,
    build_initial_layout,
    layout_rows,
    maximal_packing_signatures,
    read_resource_model,
    replay_capacity_policy,
    scenario_outcome_row,
    static_task_outcomes,
    summarize_policy_rows,
    unlimited_task_outcomes,
)


TASK_FIELDS = [
    "scenario_id", "policy", "task_id", "class_id", "initial_node_id",
    "final_node_id", "completed", "completion_seconds", "compute_seconds",
    "migration_seconds", "paused_idle_seconds", "pause_overhead_seconds",
    "carbon_grams", "cost_usd", "migrations", "pauses", "decision_count",
    "bid_attempts", "bid_accepts", "bid_rejections", "owner_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4D.2 synchronized multi-task destination-auction replay "
            "using the Stage 4D.1 evidence-backed resource model."
        )
    )
    parser.add_argument("--stage4d1-bundle", required=True)
    parser.add_argument("--stage4c-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    d41 = Path(args.stage4d1_bundle)
    c4 = Path(args.stage4c_bundle)
    d41_summary = require_bundle(d41, "Stage 4D.1")
    c4_summary = require_bundle(c4, "Stage 4C")

    source_b4 = Path(str(d41_summary.get("source_stage4b_bundle") or ""))
    if not str(source_b4):
        raise RuntimeError("Stage 4D.1 summary is missing source_stage4b_bundle")
    if Path(str(c4_summary.get("source_stage4b_bundle") or "")) != source_b4:
        raise RuntimeError("Stage 4D.1 and Stage 4C do not share the same canonical Stage 4B parent")
    b4_summary = require_bundle(source_b4, "Stage 4B")

    a1 = source_bundle(b4_summary, "stage4a1_bundle")
    a2 = source_bundle(b4_summary, "stage4a2_bundle")
    a3 = source_bundle(b4_summary, "stage4a3_bundle")
    a4 = source_bundle(b4_summary, "stage4a4_bundle")
    a5 = source_bundle(b4_summary, "stage4a5_bundle")
    a1_summary = require_bundle(a1, "Stage 4A.1", require_passed=False)
    require_bundle(a2, "Stage 4A.2")
    require_bundle(a3, "Stage 4A.3")
    require_bundle(a4, "Stage 4A.4")
    a5_summary = require_bundle(a5, "Stage 4A.5")
    if a1_summary.get("hardware_preflight_passed") is not True:
        raise RuntimeError("Stage 4A.1 hardware preflight did not pass")
    if a5_summary.get("ready_for_stage4b_runtime_model") is not True:
        raise RuntimeError("Stage 4A.5 did not approve the slowdown runtime model")

    cluster = load_cluster_config(args.cluster)
    policy = load_policy_config(args.policy)
    carbon_store = CarbonStore(cluster, args.datasets, carbon_metric=CarbonMetric.LIFECYCLE)
    capacities, requests = read_resource_model(d41)
    if set(capacities) != {node.id for node in cluster.nodes}:
        raise RuntimeError("Stage 4D.1 node set does not match configured cluster")
    if set(requests) != set(CORE_WORKLOADS):
        raise RuntimeError("Stage 4D.1 workload set does not match Stage 4B/4C core workloads")

    node_slowdowns = load_node_slowdowns(a4)
    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    target_seconds = float(c4_summary.get("target_boston_runtime_seconds") or 0.0)
    if target_seconds <= 0:
        raise RuntimeError("Stage 4C target runtime is missing")
    runtime_scales = runtime_scales_for_target(
        calibrations,
        node_slowdowns=node_slowdowns,
        target_boston_runtime_seconds=target_seconds,
    )
    edge_rows = load_stage4a1_edges(a1, a1_summary)
    selected = read_csv(c4 / "selected_windows.csv")
    if len(selected) != 4:
        raise RuntimeError(f"Stage 4D.2 requires the four canonical Stage 4C seasonal windows, got {len(selected)}")

    season_order = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
    selected.sort(key=lambda row: season_order[row["season"]])
    node_ids = [node.id for node in cluster.nodes]
    maximal = maximal_packing_signatures(d41)

    comparison_id = args.comparison_id or (
        f"stage4d2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4D.2 measured-capacity multi-task auction replay ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4d1={d41}")
    print(f"source_stage4c={c4}")
    print(f"source_stage4b={source_b4}")
    print(f"seasonal_scenarios={len(selected)} tasks_per_scenario=11")
    print(f"task_mix=benchmark:4,dendro:3,llm:4 target_boston_hours={target_seconds / 3600.0:.1f}")
    print("initial_layout=every node receives a Stage-4D.1 maximal measured-resource packing; packing map rotates by season")
    print("capacity_admission=resource vectors only; no synthetic task-slot cap")
    print(f"policies={','.join(ALL_POLICIES)}")

    all_layout_rows: list[dict] = []
    all_task_rows: list[dict] = []
    all_scenario_rows: list[dict] = []
    all_auction_rows: list[dict] = []
    all_migration_rows: list[dict] = []
    all_occupancy_rows: list[dict] = []

    for rotation, selected_row in enumerate(selected):
        season = selected_row["season"]
        arrival = as_utc_timestamp(selected_row["arrival_utc"])
        scenario_id = f"{season}-{arrival.strftime('%Y%m%dT%H%MZ')}-layout{rotation}"
        layout = build_initial_layout(
            scenario_id=scenario_id,
            node_ids=node_ids,
            requests=requests,
            rotation=rotation,
            maximal_signatures=maximal,
        )
        all_layout_rows.extend(layout_rows(scenario_id, layout))
        print(f"\n[scenario] {scenario_id} arrival={arrival.isoformat()} rotation={rotation}")

        static_rows = static_task_outcomes(
            layout=layout,
            calibrations=calibrations,
            runtime_scales=runtime_scales,
            node_slowdowns=node_slowdowns,
            cluster=cluster,
            carbon_store=carbon_store,
            arrival_utc=arrival,
            scenario_id=scenario_id,
        )
        all_task_rows.extend(static_rows)
        all_scenario_rows.append(
            scenario_outcome_row(
                scenario_id=scenario_id,
                season=season,
                arrival_utc=arrival.isoformat(),
                policy=STATIC_POLICY,
                task_rows=static_rows,
            )
        )

        unlimited_rows = unlimited_task_outcomes(
            layout=layout,
            calibrations=calibrations,
            runtime_scales=runtime_scales,
            node_slowdowns=node_slowdowns,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            edge_rows=edge_rows,
            arrival_utc=arrival,
            scenario_id=scenario_id,
        )
        all_task_rows.extend(unlimited_rows)
        unlimited_outcome = scenario_outcome_row(
            scenario_id=scenario_id,
            season=season,
            arrival_utc=arrival.isoformat(),
            policy=UNLIMITED_POLICY,
            task_rows=unlimited_rows,
        )
        all_scenario_rows.append(unlimited_outcome)
        print(
            f"  [unlimited] migrations={unlimited_outcome['migrations']} "
            f"tasks_migrated={unlimited_outcome['tasks_migrated']}"
        )

        for label, strategy in (
            (LOWEST_SCORE_POLICY, AuctionStrategy.LOWEST_SCORE),
            (CREDIT_FAIR_POLICY, AuctionStrategy.CREDIT_FAIR),
        ):
            task_rows, auction_rows, migration_rows, occupancy_rows = replay_capacity_policy(
                policy_label=label,
                auction_strategy=strategy,
                layout=layout,
                capacities=capacities,
                calibrations=calibrations,
                runtime_scales=runtime_scales,
                node_slowdowns=node_slowdowns,
                cluster=cluster,
                policy=policy,
                carbon_store=carbon_store,
                edge_rows=edge_rows,
                arrival_utc=arrival,
                scenario_id=scenario_id,
            )
            all_task_rows.extend(task_rows)
            all_auction_rows.extend(auction_rows)
            all_migration_rows.extend(migration_rows)
            all_occupancy_rows.extend(occupancy_rows)
            outcome = scenario_outcome_row(
                scenario_id=scenario_id,
                season=season,
                arrival_utc=arrival.isoformat(),
                policy=label,
                task_rows=task_rows,
            )
            all_scenario_rows.append(outcome)
            print(
                f"  [{strategy.value}] migrations={outcome['migrations']} "
                f"bids={outcome['bid_attempts']} accepted={outcome['bid_accepts']} "
                f"rejected={outcome['bid_rejections']} nodes_visited={outcome['distinct_nodes_visited']}"
            )

    all_scenario_rows = attach_static_ratios(all_scenario_rows)
    policy_summary = summarize_policy_rows(all_scenario_rows)

    migration_counts: Counter[tuple[str, str, str]] = Counter()
    for row in all_migration_rows:
        migration_counts[(row["policy"], row["source_node_id"], row["destination_node_id"])] += 1
    migration_matrix_rows = [
        {
            "policy": key[0],
            "source_node_id": key[1],
            "destination_node_id": key[2],
            "migration_count": count,
        }
        for key, count in sorted(migration_counts.items())
    ]

    capacity_violations = 0
    for row in all_occupancy_rows:
        if float(row["used_cpu_cores"]) > float(row["capacity_cpu_cores"]) + 1e-9:
            capacity_violations += 1
        if int(row["used_memory_mb"]) > int(row["capacity_memory_mb"]):
            capacity_violations += 1
        if int(row["used_gpu_count"]) > int(row["capacity_gpu_count"]):
            capacity_violations += 1

    task_count_expected = 4 * 11 * len(ALL_POLICIES)
    resource_contention_observed = any(
        int(row["bid_rejections_total"]) > 0
        for row in policy_summary
        if row["policy"] in CAPACITY_POLICIES
    )
    capacity_movement_observed = any(
        int(row["migrations_total"]) > 0
        for row in policy_summary
        if row["policy"] in CAPACITY_POLICIES
    )
    passed = (
        len(all_layout_rows) == 44
        and len(all_task_rows) == task_count_expected
        and len(all_scenario_rows) == 4 * len(ALL_POLICIES)
        and all(bool(row["completed"]) for row in all_task_rows)
        and capacity_violations == 0
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4d1_bundle": str(d41),
        "source_stage4c_bundle": str(c4),
        "source_stage4b_bundle": str(source_b4),
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "seasonal_scenario_count": 4,
        "tasks_per_scenario": 11,
        "task_mix_per_scenario": {
            "benchmark-json-medium": 4,
            "dendro-r9-t1p0": 3,
            "llm-distilgpt2": 4,
        },
        "target_boston_runtime_seconds": target_seconds,
        "policy_names": list(ALL_POLICIES),
        "task_outcome_count": len(all_task_rows),
        "scenario_outcome_count": len(all_scenario_rows),
        "auction_event_count": len(all_auction_rows),
        "migration_event_count": len(all_migration_rows),
        "capacity_violation_count": capacity_violations,
        "resource_contention_observed": resource_contention_observed,
        "capacity_movement_observed": capacity_movement_observed,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4d2_measured_capacity_multitask_auction_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "population": "Eleven simultaneous long-running tasks per seasonal scenario: 4 benchmark, 3 Dendro, 4 DistilGPT2.",
            "initial_layout": "Every node starts with one maximal packing proven feasible by Stage 4D.1. The seven-packing sequence is rotated across nodes for the four Stage 4C seasonal windows so no single region is permanently assigned one workload mix.",
            "capacity": "Stage 4D.1 effective CPU/memory/GPU resources only. No synthetic task-count capacity is added.",
            "runtime": "Each task carries exactly the Stage 4C 72-hour Boston-equivalent useful-work target. Stage 4A.4 slowdown factors determine realized progress and when node resources become free.",
            "decisions": "At each configured scheduler epoch, each runnable task calls production evaluate_task. A task emits only the single destination bid selected by production scoring, matching SchedulerService.",
            "auction": "Bids targeting the same destination in the same synchronized trace epoch are ranked with production rank_bids. Resource admission uses the production ResourceLedger. Outbound tasks are not credited as free capacity until after the round, a conservative no-preemption rule.",
            "rejection": "A rejected migration bid leaves the task running at its current owner until the next scheduler epoch. Capacity-based rejections accrue destination-local credit exactly according to AuctionPolicy; credit_fair uses that state in subsequent rankings.",
            "comparators": "Static initial layout, unlimited independent Magellan reference, measured-capacity lowest_score auction, and measured-capacity credit_fair auction.",
            "non_goals": "No bandwidth sharing, preemption, synthetic queue, or top-k fallback bidding is introduced. Movement and contention are observed outcomes, not pass criteria.",
        },
    }

    write_csv(root / "initial_layout.csv", all_layout_rows, list(all_layout_rows[0].keys()))
    write_csv(root / "task_outcomes.csv", all_task_rows, TASK_FIELDS)
    write_csv(root / "scenario_outcomes.csv", all_scenario_rows, list(all_scenario_rows[0].keys()))
    write_csv(root / "policy_summary.csv", policy_summary, list(policy_summary[0].keys()))
    write_csv(
        root / "auction_events.csv",
        all_auction_rows,
        list(all_auction_rows[0].keys()) if all_auction_rows else ["scenario_id"],
    )
    write_csv(
        root / "migration_events.csv",
        all_migration_rows,
        list(all_migration_rows[0].keys()) if all_migration_rows else ["scenario_id"],
    )
    write_csv(
        root / "migration_matrix.csv",
        migration_matrix_rows,
        ["policy", "source_node_id", "destination_node_id", "migration_count"],
    )
    write_csv(
        root / "occupancy_timeline.csv",
        all_occupancy_rows,
        list(all_occupancy_rows[0].keys()) if all_occupancy_rows else ["scenario_id"],
    )
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4D2_MULTITASK_AUCTION_PASS" if passed else "STAGE_4D2_MULTITASK_AUCTION_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"scenarios: 4")
    print(f"tasks: {len(all_task_rows)}/{task_count_expected} policy-task outcomes")
    print(f"auction_events: {len(all_auction_rows)}")
    print(f"migrations: {len(all_migration_rows)}")
    print(f"capacity_violations: {capacity_violations}")
    print(f"resource_contention_observed: {resource_contention_observed}")
    print(f"capacity_movement_observed: {capacity_movement_observed}")
    for row in policy_summary:
        print(
            f"{row['policy']}: carbon_ratio={float(row['carbon_ratio_mean']):.4f} "
            f"time_ratio={float(row['time_ratio_mean']):.4f} "
            f"cost_ratio={float(row['cost_ratio_mean']):.4f} "
            f"migrations={int(row['migrations_total'])} "
            f"rejections={int(row['bid_rejections_total'])}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
