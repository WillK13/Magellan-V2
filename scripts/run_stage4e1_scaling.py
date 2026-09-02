#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.bidding.models import AuctionStrategy
from magellan.carbon.store import CarbonMetric, as_utc_timestamp
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
    CREDIT_FAIR_POLICY,
    LOWEST_SCORE_POLICY,
    ReplayCarbonStore,
    read_resource_model,
)
from magellan.experiments.stage4e1 import (
    CLASS_SEQUENCE,
    SCALE_POLICIES,
    SCALE_SIZES,
    STATIC_SCALE_POLICY,
    attach_static_ratios,
    build_scale_population,
    class_counts,
    node_counts,
    per_class_summary,
    replay_scale_policy,
)


TASK_FIELDS = [
    "scenario_id",
    "policy",
    "task_id",
    "class_id",
    "home_node_id",
    "arrival_utc",
    "admitted_at_utc",
    "queue_wait_seconds",
    "initial_node_id",
    "final_node_id",
    "completed",
    "completion_latency_seconds",
    "compute_seconds",
    "migration_seconds",
    "paused_idle_seconds",
    "pause_overhead_seconds",
    "carbon_grams",
    "cost_usd",
    "migrations",
    "pauses",
    "decision_count",
    "bid_attempts",
    "bid_accepts",
    "bid_rejections",
    "owner_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4E.1 25/50/100-task measured-capacity scaling evaluation."
    )
    parser.add_argument("--stage4d4-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument(
        "--target-boston-hours",
        type=float,
        default=3.0,
        help=(
            "Boston-equivalent useful work per task. The canonical scale test uses "
            "3 hours because long-duration behavior is already frozen in Stage 4B-4D."
        ),
    )
    parser.add_argument(
        "--arrival-window-hours",
        type=float,
        default=3.0,
        help="Fixed window over which each 25/50/100-task population is submitted.",
    )
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    if args.target_boston_hours <= 0:
        raise ValueError("--target-boston-hours must be positive")
    if args.arrival_window_hours <= 0:
        raise ValueError("--arrival-window-hours must be positive")

    d44 = Path(args.stage4d4_bundle)
    d44_summary = require_bundle(d44, "Stage 4D.4")
    d43 = Path(str(d44_summary.get("source_stage4d3_bundle") or ""))
    d43_summary = require_bundle(d43, "Stage 4D.3")
    d42 = Path(str(d43_summary.get("source_stage4d2_bundle") or ""))
    d42_summary = require_bundle(d42, "Stage 4D.2")
    d41 = Path(str(d42_summary.get("source_stage4d1_bundle") or ""))
    require_bundle(d41, "Stage 4D.1")

    a1 = Path(str(d42_summary.get("stage4a1_bundle") or ""))
    a2 = Path(str(d42_summary.get("stage4a2_bundle") or ""))
    a3 = Path(str(d42_summary.get("stage4a3_bundle") or ""))
    a4 = Path(str(d42_summary.get("stage4a4_bundle") or ""))
    a5 = Path(str(d42_summary.get("stage4a5_bundle") or ""))
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
    capacities, requests = read_resource_model(d41)
    if set(capacities) != {node.id for node in cluster.nodes}:
        raise RuntimeError("Stage 4D.1 node set does not match cluster config")
    if set(requests) != set(CORE_WORKLOADS):
        raise RuntimeError("Stage 4D.1 workload set does not match core workloads")

    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    node_slowdowns = load_node_slowdowns(a4)
    target_seconds = args.target_boston_hours * 3600.0
    runtime_scales = runtime_scales_for_target(
        calibrations,
        node_slowdowns=node_slowdowns,
        target_boston_runtime_seconds=target_seconds,
    )
    edge_rows = load_stage4a1_edges(a1, a1_summary)
    carbon_store = ReplayCarbonStore(
        cluster,
        args.datasets,
        carbon_metric=CarbonMetric.LIFECYCLE,
    )

    source_scenarios = read_csv(d42 / "scenario_outcomes.csv")
    summer_static = [
        row
        for row in source_scenarios
        if row["season"] == "summer" and row["policy"] == "static_initial_layout"
    ]
    if len(summer_static) != 1:
        raise RuntimeError(
            f"Expected one frozen Stage 4D.2 summer static scenario, found {len(summer_static)}"
        )
    start_utc = as_utc_timestamp(summer_static[0]["arrival_utc"])
    arrival_window_seconds = args.arrival_window_hours * 3600.0
    node_ids = [node.id for node in cluster.nodes]

    comparison_id = args.comparison_id or (
        f"stage4e1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4E.1 measured-capacity 25/50/100-task scaling evaluation ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4d4={d44}")
    print(f"source_stage4d3={d43}")
    print(f"source_stage4d2={d42}")
    print(f"source_stage4d1={d41}")
    print(f"trace_window=summer start={start_utc.isoformat()}")
    print(f"task_sizes={','.join(str(value) for value in SCALE_SIZES)}")
    print(f"target_boston_hours={args.target_boston_hours:.2f}")
    print(f"arrival_window_hours={args.arrival_window_hours:.2f}")
    print(f"policies={','.join(SCALE_POLICIES)}")
    print(
        "source_admission=tasks queue at deterministic home regions and consume "
        "resources only after their Stage-4D.1 vector fits"
    )
    print(
        "capacity_admission=measured CPU/memory/GPU only; no synthetic task-slot cap"
    )

    all_population_rows = []
    all_task_rows = []
    all_summary_rows = []
    all_auction_rows = []
    all_migration_rows = []
    all_occupancy_rows = []
    all_event_rows = []

    policy_specs = (
        (STATIC_SCALE_POLICY, None),
        (LOWEST_SCORE_POLICY, AuctionStrategy.LOWEST_SCORE),
        (CREDIT_FAIR_POLICY, AuctionStrategy.CREDIT_FAIR),
    )

    for task_count in SCALE_SIZES:
        scenario_id = f"scale-summer-{task_count:03d}"
        specs = build_scale_population(
            task_count=task_count,
            node_ids=node_ids,
            requests=requests,
            start_utc=start_utc,
            arrival_window_seconds=arrival_window_seconds,
            epoch_seconds=float(cluster.epoch_seconds),
        )
        mix = class_counts(specs)
        homes = node_counts(specs)
        print(
            f"\n[scale] tasks={task_count} "
            f"mix={mix} home_range={min(homes.values())}-{max(homes.values())}"
        )

        for spec in specs:
            all_population_rows.append(
                {
                    "scenario_id": scenario_id,
                    "task_count": task_count,
                    "task_id": spec.task_id,
                    "class_id": spec.class_id,
                    "home_node_id": spec.home_node_id,
                    "arrival_utc": spec.arrival_utc.isoformat(),
                    "cpu_request_cores": spec.resource_request.cpu_cores,
                    "memory_request_mb": spec.resource_request.memory_mb,
                    "gpu_request_count": spec.resource_request.gpu_count,
                }
            )

        for policy_label, strategy in policy_specs:
            print(f"  [{policy_label}] starting", flush=True)
            (
                task_rows,
                auction_rows,
                migration_rows,
                occupancy_rows,
                event_rows,
                run_summary,
            ) = replay_scale_policy(
                policy_label=policy_label,
                auction_strategy=strategy,
                specs=specs,
                capacities=capacities,
                calibrations=calibrations,
                runtime_scales=runtime_scales,
                node_slowdowns=node_slowdowns,
                cluster=cluster,
                policy=policy,
                carbon_store=carbon_store,
                edge_rows=edge_rows,
                scenario_id=scenario_id,
                target_boston_seconds=target_seconds,
                arrival_window_seconds=arrival_window_seconds,
                progress=lambda message, label=policy_label: print(
                    f"    [{label}] {message}",
                    flush=True,
                ),
                progress_every_rounds=24,
            )
            run_summary["scale_task_count"] = task_count
            all_task_rows.extend(task_rows)
            all_summary_rows.append(run_summary)
            all_auction_rows.extend(auction_rows)
            all_migration_rows.extend(migration_rows)
            all_occupancy_rows.extend(occupancy_rows)
            all_event_rows.extend(event_rows)
            print(
                f"  [{policy_label}] drain={run_summary['drain_seconds']/3600.0:.2f}h "
                f"throughput={run_summary['throughput_tasks_per_hour']:.3f}/h "
                f"queue_p95={run_summary['p95_queue_wait_seconds']/3600.0:.2f}h "
                f"carbon={run_summary['carbon_grams']:.2f}g "
                f"migrations={run_summary['migrations']} "
                f"rejections={run_summary['bid_rejections']}",
                flush=True,
            )

    all_summary_rows = attach_static_ratios(all_summary_rows)
    class_rows = per_class_summary(all_task_rows)

    migration_matrix = Counter(
        (
            row["scenario_id"],
            row["policy"],
            row["source_node_id"],
            row["destination_node_id"],
        )
        for row in all_migration_rows
    )
    migration_matrix_rows = [
        {
            "scenario_id": key[0],
            "policy": key[1],
            "source_node_id": key[2],
            "destination_node_id": key[3],
            "migration_count": count,
        }
        for key, count in sorted(migration_matrix.items())
    ]

    capacity_violations = 0
    for row in all_occupancy_rows:
        if float(row["used_cpu_cores"]) > float(row["capacity_cpu_cores"]) + 1e-9:
            capacity_violations += 1
        if int(row["used_memory_mb"]) > int(row["capacity_memory_mb"]):
            capacity_violations += 1
        if int(row["used_gpu_count"]) > int(row["capacity_gpu_count"]):
            capacity_violations += 1

    expected_task_outcomes = sum(SCALE_SIZES) * len(SCALE_POLICIES)
    queueing_observed_at_100 = any(
        row["scenario_id"] == "scale-summer-100"
        and float(row["queue_wait_seconds"]) > 0
        for row in all_task_rows
    )
    passed = (
        len(all_task_rows) == expected_task_outcomes
        and len(all_summary_rows) == len(SCALE_SIZES) * len(SCALE_POLICIES)
        and all(bool(row["completed"]) for row in all_task_rows)
        and capacity_violations == 0
        and queueing_observed_at_100
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4d4_bundle": str(d44),
        "source_stage4d3_bundle": str(d43),
        "source_stage4d2_bundle": str(d42),
        "source_stage4d1_bundle": str(d41),
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "trace_season": "summer",
        "trace_start_utc": start_utc.isoformat(),
        "scale_sizes": list(SCALE_SIZES),
        "policy_names": list(SCALE_POLICIES),
        "target_boston_hours": args.target_boston_hours,
        "arrival_window_hours": args.arrival_window_hours,
        "task_outcome_count": len(all_task_rows),
        "scenario_summary_count": len(all_summary_rows),
        "auction_event_count": len(all_auction_rows),
        "migration_event_count": len(all_migration_rows),
        "animation_event_count": len(all_event_rows),
        "capacity_violation_count": capacity_violations,
        "queueing_observed_at_100": queueing_observed_at_100,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4e1_measured_capacity_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Measure scheduler behavior as the submitted population increases from "
                "25 to 50 to 100 tasks while preserving measured task resource vectors, "
                "node capacities, node slowdowns, carbon traces, migration calibration, "
                "production scoring, and destination-side auctions."
            ),
            "work_duration": (
                "The canonical scaling workload uses 3 Boston-equivalent hours per task. "
                "Long-duration 72-hour behavior is already established by Stage 4C/4D; "
                "Stage 4E.1 isolates population scaling without repeating that horizon."
            ),
            "arrivals": (
                "Each population is deterministically spread across the same fixed "
                "3-hour submission window, snapped to the configured scheduler epoch. "
                "Increasing N therefore increases offered load while holding the arrival "
                "window constant."
            ),
            "source_admission": (
                "Every task has a deterministic home region. Before it runs, it waits in "
                "a source-side queue and consumes no compute resources. A feasibility-"
                "preserving FIFO scan admits it only when its measured Stage-4D.1 resource "
                "vector fits the home node. This prevents initial oversubscription."
            ),
            "policies": (
                "Static tasks remain at their home node after admission. Magellan "
                "lowest_score and credit_fair use the production evaluate_task scorer, "
                "one selected destination bid per task, production rank_bids arbitration, "
                "and measured destination resource ledgers."
            ),
            "capacity": (
                "No synthetic task-slot capacity is introduced. All source admission and "
                "migration admission use the Stage 4D.1 CPU/memory/GPU vectors."
            ),
            "co_location_boundary": (
                "As in Stage 4D.2/4D.3, admitted tasks retain isolated Stage 4A.3 power "
                "and Stage 4A.4 slowdown profiles; unmeasured co-location interference is "
                "not invented."
            ),
            "event_trace": (
                "event_trace.csv records submission, measured-resource admission, "
                "migration start/finish, and completion events for later world-map "
                "visualization. auction_events.csv separately records all bid outcomes."
            ),
            "pass_condition": (
                "PASS requires full 25/50/100 population coverage for all three policies, "
                "all tasks completed, zero resource-capacity violations, and actual queue "
                "contention at the 100-task scale. It is not tied to a preferred policy "
                "or carbon result."
            ),
        },
    }

    write_csv(
        root / "population.csv",
        all_population_rows,
        list(all_population_rows[0].keys()),
    )
    write_csv(root / "task_outcomes.csv", all_task_rows, TASK_FIELDS)
    write_csv(
        root / "scaling_summary.csv",
        all_summary_rows,
        list(all_summary_rows[0].keys()),
    )
    write_csv(
        root / "per_class_summary.csv",
        class_rows,
        list(class_rows[0].keys()),
    )
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
        [
            "scenario_id",
            "policy",
            "source_node_id",
            "destination_node_id",
            "migration_count",
        ],
    )
    write_csv(
        root / "occupancy_timeline.csv",
        all_occupancy_rows,
        list(all_occupancy_rows[0].keys()),
    )
    write_csv(
        root / "event_trace.csv",
        sorted(
            all_event_rows,
            key=lambda row: (
                row["scenario_id"],
                row["policy"],
                row["at_utc"],
                row["task_id"],
                {
                    "submitted": 0,
                    "admitted": 1,
                    "migration_start": 2,
                    "migration_finish": 3,
                    "completed": 4,
                }.get(row["event_type"], 99),
            ),
        ),
        list(all_event_rows[0].keys()),
    )
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    cache = carbon_store.cache_summary()
    print(
        "carbon_cache: "
        f"forecast_hits={cache['forecast_hits']} "
        f"forecast_entries={cache['forecast_entries']} "
        f"average_hits={cache['average_hits']} "
        f"average_entries={cache['average_entries']}"
    )
    marker = "STAGE_4E1_SCALING_PASS" if passed else "STAGE_4E1_SCALING_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"task_outcomes: {len(all_task_rows)}/{expected_task_outcomes}")
    print(f"capacity_violations: {capacity_violations}")
    print(f"queueing_observed_at_100: {queueing_observed_at_100}")
    print("\nScaling summary:")
    for row in all_summary_rows:
        print(
            f"  n={int(row['scale_task_count']):3d} "
            f"{row['policy']:32s} "
            f"drain={float(row['drain_seconds'])/3600.0:7.2f}h "
            f"throughput={float(row['throughput_tasks_per_hour']):6.3f}/h "
            f"queue_p95={float(row['p95_queue_wait_seconds'])/3600.0:7.2f}h "
            f"carbon_ratio={float(row['carbon_ratio_vs_static']):7.4f} "
            f"cost_ratio={float(row['cost_ratio_vs_static']):7.4f} "
            f"migrations={int(row['migrations']):4d} "
            f"rejections={int(row['bid_rejections']):6d}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
