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
    LOWEST_SCORE_POLICY,
    ReplayCarbonStore,
    STATIC_POLICY,
    UNLIMITED_POLICY,
    LayoutTask,
    attach_static_ratios,
    layout_rows,
    read_resource_model,
    replay_capacity_policy,
    scenario_outcome_row,
)
from magellan.experiments.stage4d3 import (
    LOAD_ORDER,
    NOMINAL_TARGET_FRACTIONS,
    SWEEP_POLICIES,
    achieved_cpu_fraction,
    build_nested_utilization_layouts,
    class_counts,
    layout_cpu_cores,
    summarize_utilization_rows,
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
            "Run Stage 4D.3 measured-utilization sweep using the frozen "
            "Stage 4D.2 population and capacity model."
        )
    )
    parser.add_argument("--stage4d2-bundle", required=True)
    parser.add_argument(
        "--season",
        choices=("winter", "spring", "summer", "fall", "all"),
        default="summer",
        help=(
            "Canonical fast sweep defaults to summer, the Stage 4D.2 window "
            "with measured-capacity movement across Ethiopia, South Australia, and France. "
            "Use all for a full four-season replication."
        ),
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def copy_task_subset(
    rows: list[dict[str, str]],
    *,
    source_scenario_id: str,
    new_scenario_id: str,
    policy: str,
    task_ids: set[str],
) -> list[dict]:
    output = []
    for row in rows:
        if (
            row["scenario_id"] == source_scenario_id
            and row["policy"] == policy
            and row["task_id"] in task_ids
        ):
            copied = dict(row)
            copied["scenario_id"] = new_scenario_id
            output.append(copied)
    if len(output) != len(task_ids):
        raise RuntimeError(
            f"Reference subset coverage mismatch for {source_scenario_id}/{policy}: "
            f"{len(output)} != {len(task_ids)}"
        )
    return output


def copy_event_rows(
    rows: list[dict[str, str]],
    *,
    source_scenario_id: str,
    new_scenario_id: str,
    policy: str,
) -> list[dict]:
    output = []
    for row in rows:
        if row["scenario_id"] == source_scenario_id and row["policy"] == policy:
            copied = dict(row)
            copied["scenario_id"] = new_scenario_id
            output.append(copied)
    return output


def reconstruct_layout(
    rows: list[dict[str, str]],
    *,
    scenario_id: str,
    requests,
) -> list[LayoutTask]:
    output = []
    for row in rows:
        if row["scenario_id"] != scenario_id:
            continue
        class_id = row["class_id"]
        output.append(
            LayoutTask(
                task_id=row["task_id"],
                class_id=class_id,
                initial_node_id=row["initial_node_id"],
                resource_request=requests[class_id],
            )
        )
    output.sort(key=lambda task: task.task_id)
    if len(output) != 11:
        raise RuntimeError(f"Source Stage 4D.2 layout {scenario_id} has {len(output)} tasks, expected 11")
    return output


def main() -> int:
    args = parse_args()
    d42 = Path(args.stage4d2_bundle)
    d42_summary = require_bundle(d42, "Stage 4D.2")

    d41 = Path(str(d42_summary.get("source_stage4d1_bundle") or ""))
    c4 = Path(str(d42_summary.get("source_stage4c_bundle") or ""))
    b4 = Path(str(d42_summary.get("source_stage4b_bundle") or ""))
    a1 = Path(str(d42_summary.get("stage4a1_bundle") or ""))
    a2 = Path(str(d42_summary.get("stage4a2_bundle") or ""))
    a3 = Path(str(d42_summary.get("stage4a3_bundle") or ""))
    a4 = Path(str(d42_summary.get("stage4a4_bundle") or ""))
    a5 = Path(str(d42_summary.get("stage4a5_bundle") or ""))

    require_bundle(d41, "Stage 4D.1")
    require_bundle(c4, "Stage 4C")
    require_bundle(b4, "Stage 4B")
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
    carbon_store = ReplayCarbonStore(
        cluster,
        args.datasets,
        carbon_metric=CarbonMetric.LIFECYCLE,
    )
    capacities, requests = read_resource_model(d41)
    if set(capacities) != {node.id for node in cluster.nodes}:
        raise RuntimeError("Stage 4D.1 node set does not match configured cluster")
    if set(requests) != set(CORE_WORKLOADS):
        raise RuntimeError("Stage 4D.1 workload set does not match core workloads")

    node_slowdowns = load_node_slowdowns(a4)
    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    target_seconds = float(d42_summary.get("target_boston_runtime_seconds") or 0.0)
    if target_seconds <= 0:
        raise RuntimeError("Stage 4D.2 target runtime is missing")
    runtime_scales = runtime_scales_for_target(
        calibrations,
        node_slowdowns=node_slowdowns,
        target_boston_runtime_seconds=target_seconds,
    )
    edge_rows = load_stage4a1_edges(a1, a1_summary)

    source_layout_rows = read_csv(d42 / "initial_layout.csv")
    source_task_rows = read_csv(d42 / "task_outcomes.csv")
    source_scenario_rows = read_csv(d42 / "scenario_outcomes.csv")
    source_auction_rows = read_csv(d42 / "auction_events.csv")
    source_migration_rows = read_csv(d42 / "migration_events.csv")
    source_occupancy_rows = read_csv(d42 / "occupancy_timeline.csv")

    source_static_scenarios = [
        row for row in source_scenario_rows if row["policy"] == STATIC_POLICY
    ]
    season_order = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
    source_static_scenarios.sort(key=lambda row: season_order[row["season"]])
    if args.season != "all":
        source_static_scenarios = [
            row for row in source_static_scenarios if row["season"] == args.season
        ]
    if not source_static_scenarios:
        raise RuntimeError(f"No Stage 4D.2 source scenario found for season={args.season}")

    cluster_cpu_cores = sum(float(cap.cpu_cores) for cap in capacities.values())
    comparison_id = args.comparison_id or (
        f"stage4d3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4D.3 measured-utilization capacity sweep ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4d2={d42}")
    print(f"season_selection={args.season}")
    print(f"seasonal_scenarios={len(source_static_scenarios)}")
    print(f"cluster_cpu_cores={cluster_cpu_cores:.3f}")
    print("loads=u25,u50,u75,umax")
    print(f"policies={','.join(SWEEP_POLICIES)}")
    print(
        "reuse=static/unlimited task outcomes are exact subsets of Stage 4D.2; "
        "umax lowest_score is reused exactly; only u25/u50/u75 capacity cases are replayed"
    )

    all_load_rows: list[dict] = []
    all_layout_rows: list[dict] = []
    all_task_rows: list[dict] = []
    all_scenario_rows: list[dict] = []
    all_auction_rows: list[dict] = []
    all_migration_rows: list[dict] = []
    all_occupancy_rows: list[dict] = []

    for source_scenario in source_static_scenarios:
        season = source_scenario["season"]
        source_scenario_id = source_scenario["scenario_id"]
        arrival = as_utc_timestamp(source_scenario["arrival_utc"])
        maximal_layout = reconstruct_layout(
            source_layout_rows,
            scenario_id=source_scenario_id,
            requests=requests,
        )
        nested = build_nested_utilization_layouts(maximal_layout)

        print(f"\n[season] {season} source={source_scenario_id} arrival={arrival.isoformat()}")

        for load_id in LOAD_ORDER:
            layout = nested[load_id]
            scenario_id = f"{season}-{arrival.strftime('%Y%m%dT%H%MZ')}-{load_id}"
            task_ids = {task.task_id for task in layout}
            achieved_cores = layout_cpu_cores(layout)
            achieved_fraction = achieved_cpu_fraction(
                layout,
                cluster_cpu_cores=cluster_cpu_cores,
            )
            counts = class_counts(layout)
            nominal = NOMINAL_TARGET_FRACTIONS[load_id]
            target_fraction = achieved_fraction if nominal is None else nominal

            all_load_rows.append(
                {
                    "scenario_id": scenario_id,
                    "season": season,
                    "source_stage4d2_scenario_id": source_scenario_id,
                    "load_id": load_id,
                    "target_cpu_fraction": target_fraction,
                    "achieved_initial_cpu_cores": achieved_cores,
                    "achieved_initial_cpu_fraction": achieved_fraction,
                    "cluster_cpu_cores": cluster_cpu_cores,
                    "task_count": len(layout),
                    "benchmark_task_count": counts["benchmark-json-medium"],
                    "dendro_task_count": counts["dendro-r9-t1p0"],
                    "llm_task_count": counts["llm-distilgpt2"],
                }
            )
            all_layout_rows.extend(layout_rows(scenario_id, layout))
            print(
                f"  [{load_id}] target={target_fraction * 100:.1f}% "
                f"achieved={achieved_fraction * 100:.2f}% "
                f"cpu={achieved_cores:.3f}/{cluster_cpu_cores:.3f} "
                f"tasks={len(layout)} mix={counts}"
            )

            for reference_policy in (STATIC_POLICY, UNLIMITED_POLICY):
                reference_rows = copy_task_subset(
                    source_task_rows,
                    source_scenario_id=source_scenario_id,
                    new_scenario_id=scenario_id,
                    policy=reference_policy,
                    task_ids=task_ids,
                )
                all_task_rows.extend(reference_rows)
                all_scenario_rows.append(
                    scenario_outcome_row(
                        scenario_id=scenario_id,
                        season=season,
                        arrival_utc=arrival.isoformat(),
                        policy=reference_policy,
                        task_rows=reference_rows,
                    )
                )

            if load_id == "umax":
                capacity_rows = copy_task_subset(
                    source_task_rows,
                    source_scenario_id=source_scenario_id,
                    new_scenario_id=scenario_id,
                    policy=LOWEST_SCORE_POLICY,
                    task_ids=task_ids,
                )
                auction_rows = copy_event_rows(
                    source_auction_rows,
                    source_scenario_id=source_scenario_id,
                    new_scenario_id=scenario_id,
                    policy=LOWEST_SCORE_POLICY,
                )
                migration_rows = copy_event_rows(
                    source_migration_rows,
                    source_scenario_id=source_scenario_id,
                    new_scenario_id=scenario_id,
                    policy=LOWEST_SCORE_POLICY,
                )
                occupancy_rows = copy_event_rows(
                    source_occupancy_rows,
                    source_scenario_id=source_scenario_id,
                    new_scenario_id=scenario_id,
                    policy=LOWEST_SCORE_POLICY,
                )
                print(
                    f"    [lowest_score] reused Stage 4D.2 max-load result "
                    f"migrations={sum(int(row['migrations']) for row in capacity_rows)}"
                )
            else:
                capacity_rows, auction_rows, migration_rows, occupancy_rows = replay_capacity_policy(
                    policy_label=LOWEST_SCORE_POLICY,
                    auction_strategy=AuctionStrategy.LOWEST_SCORE,
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
                    progress=lambda message, load_id=load_id: print(
                        f"    [lowest_score {load_id}] {message}",
                        flush=True,
                    ),
                    progress_every_rounds=24,
                )

            all_task_rows.extend(capacity_rows)
            all_auction_rows.extend(auction_rows)
            all_migration_rows.extend(migration_rows)
            all_occupancy_rows.extend(occupancy_rows)
            capacity_outcome = scenario_outcome_row(
                scenario_id=scenario_id,
                season=season,
                arrival_utc=arrival.isoformat(),
                policy=LOWEST_SCORE_POLICY,
                task_rows=capacity_rows,
            )
            all_scenario_rows.append(capacity_outcome)
            print(
                f"    [lowest_score] migrations={capacity_outcome['migrations']} "
                f"bids={capacity_outcome['bid_attempts']} "
                f"accepted={capacity_outcome['bid_accepts']} "
                f"rejected={capacity_outcome['bid_rejections']}"
            )

    all_scenario_rows = attach_static_ratios(all_scenario_rows)
    utilization_summary = summarize_utilization_rows(
        all_scenario_rows,
        all_load_rows,
    )

    load_by_scenario = {row["scenario_id"]: row["load_id"] for row in all_load_rows}
    migration_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in all_migration_rows:
        migration_counts[
            (
                load_by_scenario[row["scenario_id"]],
                row["policy"],
                row["source_node_id"],
                row["destination_node_id"],
            )
        ] += 1
    migration_matrix_rows = [
        {
            "load_id": key[0],
            "policy": key[1],
            "source_node_id": key[2],
            "destination_node_id": key[3],
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

    expected_tasks_per_season = 3 + 6 + 9 + 11
    expected_task_rows = (
        len(source_static_scenarios) * expected_tasks_per_season * len(SWEEP_POLICIES)
    )
    expected_case_count = len(source_static_scenarios) * len(LOAD_ORDER)
    passed = (
        len(all_load_rows) == expected_case_count
        and len(all_layout_rows) == len(source_static_scenarios) * expected_tasks_per_season
        and len(all_task_rows) == expected_task_rows
        and len(all_scenario_rows) == expected_case_count * len(SWEEP_POLICIES)
        and all(bool(row["completed"]) for row in all_task_rows)
        and capacity_violations == 0
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4d2_bundle": str(d42),
        "source_stage4d1_bundle": str(d41),
        "source_stage4c_bundle": str(c4),
        "source_stage4b_bundle": str(b4),
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "season_selection": args.season,
        "seasonal_scenario_count": len(source_static_scenarios),
        "load_ids": list(LOAD_ORDER),
        "policy_names": list(SWEEP_POLICIES),
        "target_boston_runtime_seconds": target_seconds,
        "cluster_cpu_cores": cluster_cpu_cores,
        "load_case_count": len(all_load_rows),
        "task_outcome_count": len(all_task_rows),
        "scenario_outcome_count": len(all_scenario_rows),
        "auction_event_count": len(all_auction_rows),
        "migration_event_count": len(all_migration_rows),
        "capacity_violation_count": capacity_violations,
        "reference_reuse": {
            "static_and_unlimited": "exact Stage 4D.2 task-outcome subsets",
            "umax_lowest_score": "exact Stage 4D.2 capacity result",
            "new_capacity_replays": ["u25", "u50", "u75"],
        },
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4d3_measured_utilization_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "loads": (
                "Nested balanced workload populations derived from the frozen Stage 4D.2 "
                "maximal 4/3/4 population: 1/1/1, 2/2/2, 3/3/3, then the full 4/3/4 layout. "
                "With Stage 4D.1 p95 CPU requests these correspond to approximately "
                "25%, 50%, 76%, and 88% of the measured 14-core cluster."
            ),
            "nesting": (
                "Subsets are selected downward from the frozen maximal population, maximizing "
                "distinct initial-node coverage with deterministic task-id tie breaking."
            ),
            "season": (
                "The canonical fast run uses the Stage 4D.2 summer window because it already "
                "exhibits capacity-driven movement across Ethiopia, South Australia, and France. "
                "--season all performs the same sweep across all four frozen seasonal windows."
            ),
            "reference_reuse": (
                "Static and unlimited task outcomes are independent of peer occupancy, so their "
                "exact frozen Stage 4D.2 task-level outcomes are subset and re-aggregated. The "
                "maximal measured-capacity lowest_score case is also copied exactly from Stage 4D.2. "
                "Only the 25/50/75-percent measured-capacity cases are newly replayed."
            ),
            "capacity": (
                "Admission uses only Stage 4D.1 measured CPU/memory/GPU requests and capacities. "
                "No synthetic task-slot cap is introduced."
            ),
            "runtime": (
                "Each admitted task retains the isolated Stage 4A.4 slowdown and Stage 4A.3 power "
                "profile, matching the explicit Stage 4D.2 non-interference assumption."
            ),
            "policy": (
                "The sweep isolates capacity pressure using production Magellan with the "
                "lowest_score destination arbiter. Fairness strategies are reserved for Stage 4D.4."
            ),
        },
    }

    write_csv(root / "load_cases.csv", all_load_rows, list(all_load_rows[0].keys()))
    write_csv(root / "initial_layout.csv", all_layout_rows, list(all_layout_rows[0].keys()))
    write_csv(root / "task_outcomes.csv", all_task_rows, TASK_FIELDS)
    write_csv(
        root / "scenario_outcomes.csv",
        all_scenario_rows,
        list(all_scenario_rows[0].keys()),
    )
    write_csv(
        root / "utilization_summary.csv",
        utilization_summary,
        list(utilization_summary[0].keys()),
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
        ["load_id", "policy", "source_node_id", "destination_node_id", "migration_count"],
    )
    write_csv(
        root / "occupancy_timeline.csv",
        all_occupancy_rows,
        list(all_occupancy_rows[0].keys()) if all_occupancy_rows else ["scenario_id"],
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
    marker = "STAGE_4D3_UTILIZATION_SWEEP_PASS" if passed else "STAGE_4D3_UTILIZATION_SWEEP_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"season_selection: {args.season}")
    print(f"load_cases: {len(all_load_rows)}/{expected_case_count}")
    print(f"task_outcomes: {len(all_task_rows)}/{expected_task_rows}")
    print(f"capacity_violations: {capacity_violations}")
    print("\nCapacity curve:")
    for row in utilization_summary:
        if row["policy"] != LOWEST_SCORE_POLICY:
            continue
        print(
            f"  {row['load_id']}: "
            f"util={float(row['achieved_initial_cpu_fraction_mean']) * 100:.2f}% "
            f"carbon_ratio={float(row['carbon_ratio_mean']):.4f} "
            f"time_ratio={float(row['time_ratio_mean']):.4f} "
            f"cost_ratio={float(row['cost_ratio_mean']):.4f} "
            f"migrations={int(row['migrations_total'])} "
            f"rejections={int(row['bid_rejections_total'])}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
