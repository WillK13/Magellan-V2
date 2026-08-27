#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.carbon.store import CarbonMetric, CarbonStore
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.bundle import validate_checksums, write_checksums, write_csv, write_json, write_jsonl
from magellan.experiments.stage4b import (
    CORE_WORKLOADS,
    FrozenCalibrationGraph,
    Scenario,
    annual_scenarios,
    load_node_slowdowns,
    load_stage4a1_edges,
    load_workload_calibrations,
    replay_magellan_causal,
    summarize_policy_rows,
)
from magellan.experiments.stage4c import (
    DYNAMIC_POLICIES,
    LEADERSHIP_QUANTUM_SECONDS,
    MIGRATION_DIAGNOSTIC_FIELDS,
    RESIDENCE_FIELDS,
    aggregate_dynamic_summary,
    boston_static_outcome,
    candidate_window_summary,
    dynamic_outcome_rows,
    dynamic_scenario_summary,
    leadership_timeline,
    leadership_windows,
    migration_diagnostics,
    residence_rows,
    runtime_scales_for_target,
    select_crossover_arrivals,
    selected_dynamic_scenarios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4C 72-hour dynamic carbon crossover evaluation.")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--stage4b-bundle", required=True)
    parser.add_argument("--target-hours", type=float, default=72.0)
    parser.add_argument("--leadership-quantum-seconds", type=float, default=LEADERSHIP_QUANTUM_SECONDS)
    parser.add_argument("--windows-per-season", type=int, default=1)
    parser.add_argument("--months", default=",".join(str(value) for value in range(1, 13)))
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


def _source_bundle(stage4b_summary: dict, key: str) -> Path:
    value = stage4b_summary.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Stage 4B summary missing {key}")
    return Path(value)


def _timeline_for_arrival(
    *,
    arrival,
    cluster,
    node_slowdowns,
    carbon_store,
    target_seconds,
    quantum_seconds,
):
    probe = Scenario(
        scenario_id=f"{arrival.strftime('%Y%m%dT%H%MZ')}-trace-probe",
        class_id=CORE_WORKLOADS[0],
        arrival_utc=arrival,
    )
    return leadership_timeline(
        scenario=probe,
        cluster=cluster,
        node_slowdowns=node_slowdowns,
        carbon_store=carbon_store,
        horizon_seconds=target_seconds,
        quantum_seconds=quantum_seconds,
    )


def main() -> int:
    args = parse_args()
    target_seconds = args.target_hours * 3600.0
    if target_seconds <= 0:
        raise ValueError("target-hours must be positive")
    if args.leadership_quantum_seconds <= 0:
        raise ValueError("leadership-quantum-seconds must be positive")
    if args.windows_per_season <= 0:
        raise ValueError("windows-per-season must be positive")
    months = [int(value) for value in args.months.split(",") if value.strip()]
    if not months:
        raise ValueError("months must be non-empty")

    stage4b = Path(args.stage4b_bundle)
    stage4b_summary = require_bundle(stage4b, "Stage 4B")
    if set(stage4b_summary.get("workload_classes") or []) != set(CORE_WORKLOADS):
        raise RuntimeError("Stage 4B bundle does not contain the frozen Stage 4C workload set")

    a1 = _source_bundle(stage4b_summary, "stage4a1_bundle")
    a2 = _source_bundle(stage4b_summary, "stage4a2_bundle")
    a3 = _source_bundle(stage4b_summary, "stage4a3_bundle")
    a4 = _source_bundle(stage4b_summary, "stage4a4_bundle")
    a5 = _source_bundle(stage4b_summary, "stage4a5_bundle")

    a1_summary = require_bundle(a1, "Stage 4A.1", require_passed=False)
    if a1_summary.get("hardware_preflight_passed") is not True:
        raise RuntimeError("Stage 4A.1 hardware preflight did not pass")
    a2_summary = require_bundle(a2, "Stage 4A.2")
    a3_summary = require_bundle(a3, "Stage 4A.3")
    a4_summary = require_bundle(a4, "Stage 4A.4")
    a5_summary = require_bundle(a5, "Stage 4A.5")
    if a5_summary.get("ready_for_stage4b_runtime_model") is not True:
        raise RuntimeError("Stage 4A.5 did not approve the slowdown-factor runtime model")
    if a5_summary.get("recommended_runtime_model") != "single_node_slowdown_factor":
        raise RuntimeError("Unexpected Stage 4A.5 runtime-model recommendation")

    cluster = load_cluster_config(args.cluster)
    if int(a1_summary.get("node_count") or 0) != len(cluster.nodes):
        raise RuntimeError("Stage 4A.1 node count does not match configured cluster")
    policy = load_policy_config(args.policy)
    carbon_store = CarbonStore(cluster, args.datasets, carbon_metric=CarbonMetric.LIFECYCLE)
    node_slowdowns = load_node_slowdowns(a4)
    if set(node_slowdowns) != {node.id for node in cluster.nodes}:
        raise RuntimeError("Stage 4A.4 slowdown table does not match configured cluster nodes")
    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    runtime_scales = runtime_scales_for_target(
        calibrations,
        node_slowdowns=node_slowdowns,
        target_boston_runtime_seconds=target_seconds,
    )
    edge_rows = load_stage4a1_edges(a1, a1_summary)
    expected_edges = len(cluster.nodes) * (len(cluster.nodes) - 1)
    if len(edge_rows) != expected_edges:
        raise RuntimeError(f"Stage 4A.1 edge count {len(edge_rows)} != expected {expected_edges}")

    # Screen the same deterministic 24 annual arrival windows used by Stage 4B,
    # using only trace volatility—not Magellan outcomes—to select the crossover stress set.
    candidate_probe_scenarios = annual_scenarios(class_ids=(CORE_WORKLOADS[0],), months=months)
    timeline_cache: dict[str, list[dict]] = {}
    candidate_rows: list[dict] = []
    for probe in candidate_probe_scenarios:
        key = probe.arrival_utc.isoformat()
        timeline = _timeline_for_arrival(
            arrival=probe.arrival_utc,
            cluster=cluster,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            target_seconds=target_seconds,
            quantum_seconds=args.leadership_quantum_seconds,
        )
        timeline_cache[key] = timeline
        candidate_rows.append(
            candidate_window_summary(
                arrival_utc=probe.arrival_utc,
                timeline_rows=timeline,
                minimum_sustained_seconds=policy.migration.min_migration_gap_seconds,
                horizon_seconds=target_seconds,
                quantum_seconds=args.leadership_quantum_seconds,
            )
        )
    selected_rows = select_crossover_arrivals(candidate_rows, windows_per_season=args.windows_per_season)
    selected_keys = {row["arrival_utc"] for row in selected_rows}
    for row in candidate_rows:
        if row["arrival_utc"] in selected_keys:
            selected = next(item for item in selected_rows if item["arrival_utc"] == row["arrival_utc"])
            row["selected"] = True
            row["selection_rank_within_season"] = selected["selection_rank_within_season"]
    scenarios = selected_dynamic_scenarios(selected_rows, class_ids=CORE_WORKLOADS)

    comparison_id = args.comparison_id or f"stage4c-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4C 72-hour dynamic carbon crossover evaluation ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4b={stage4b}")
    print(f"candidate_arrivals={len(candidate_rows)} selected_arrivals={len(selected_rows)} workloads={len(CORE_WORKLOADS)} scenarios={len(scenarios)}")
    print(f"policies={','.join(DYNAMIC_POLICIES)} expected_outcomes={len(scenarios) * len(DYNAMIC_POLICIES)}")
    print(f"target_boston_runtime_hours={args.target_hours:g} scheduler_epoch_seconds={cluster.epoch_seconds}")
    print(f"selection=top {args.windows_per_season} trace-only sustained-crossover window(s) per season")
    for row in selected_rows:
        print(
            f"[selected] {row['season']:6s} {row['arrival_utc']} "
            f"sustained_transitions={row['sustained_scheduler_leader_transitions']} "
            f"leaders={row['sustained_scheduler_leader_path'] or '-'}"
        )

    scenario_rows: list[dict] = []
    all_outcome_rows: list[dict] = []
    all_trace_rows: list[dict] = []
    all_leadership_rows: list[dict] = []
    all_window_rows: list[dict] = []
    all_migration_rows: list[dict] = []
    all_residence_rows: list[dict] = []
    dynamic_rows: list[dict] = []

    for index, scenario in enumerate(scenarios, start=1):
        calibration = calibrations[scenario.class_id]
        runtime_scale = runtime_scales[scenario.class_id]
        graph = FrozenCalibrationGraph(cluster=cluster, edge_rows=edge_rows, workload=calibration)
        boston = boston_static_outcome(
            cluster=cluster,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
        )
        magellan = replay_magellan_causal(
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            graph=graph,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
        )
        outcomes = [boston, magellan]
        rows = dynamic_outcome_rows(
            scenario=scenario,
            calibration=calibration,
            boston_static=boston,
            magellan=magellan,
            policy=policy,
        )
        all_outcome_rows.extend(rows)
        magellan_row = next(row for row in rows if row["policy"] == "magellan_causal")

        template_timeline = timeline_cache[scenario.arrival_utc.isoformat()]
        timeline = [
            {**row, "scenario_id": scenario.scenario_id, "class_id": scenario.class_id}
            for row in template_timeline
        ]
        windows = leadership_windows(
            scenario=scenario,
            timeline_rows=timeline,
            magellan=magellan,
            minimum_opportunity_seconds=policy.migration.min_migration_gap_seconds,
            horizon_seconds=target_seconds,
            quantum_seconds=args.leadership_quantum_seconds,
        )
        migrations = migration_diagnostics(
            scenario=scenario,
            magellan=magellan,
            calibration=calibration,
            cluster=cluster,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
        )
        residence = residence_rows(scenario=scenario, magellan=magellan)
        dynamic = dynamic_scenario_summary(
            scenario=scenario,
            magellan=magellan,
            timeline_rows=timeline,
            window_rows=windows,
            migration_rows=migrations,
            residence=residence,
            outcome_row=magellan_row,
        )

        all_leadership_rows.extend(timeline)
        all_window_rows.extend(windows)
        all_migration_rows.extend(migrations)
        all_residence_rows.extend(residence)
        dynamic_rows.append(dynamic)
        scenario_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "class_id": scenario.class_id,
                "workload": calibration.workload,
                "variant": calibration.variant,
                "arrival_utc": scenario.arrival_utc.isoformat(),
                "season": next(row["season"] for row in selected_rows if row["arrival_utc"] == scenario.arrival_utc.isoformat()),
                "canonical_runtime_seconds": calibration.canonical_runtime_seconds,
                "runtime_scale": runtime_scale,
                "target_boston_runtime_seconds": target_seconds,
                "scaled_boston_work_seconds": calibration.scaled_work_seconds(runtime_scale) * node_slowdowns["boston"],
                "power_kw": calibration.power_kw,
                "checkpoint_bytes": calibration.checkpoint_bytes,
            }
        )
        for outcome in outcomes:
            all_trace_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "class_id": scenario.class_id,
                    "arrival_utc": scenario.arrival_utc.isoformat(),
                    "outcome": outcome.model_dump(mode="json"),
                }
            )
        print(
            f"[{index:02d}/{len(scenarios):02d}] {scenario.scenario_id} "
            f"path={'->'.join(magellan.owner_path)} migrations={magellan.migrations} pauses={magellan.pauses} "
            f"leader_changes={dynamic['scheduler_carbon_leader_changes_72h']}"
        )

    expected_outcomes = len(scenarios) * len(DYNAMIC_POLICIES)
    expected_leadership_per_scenario = int(math.ceil(target_seconds / args.leadership_quantum_seconds))
    policy_summary = summarize_policy_rows(all_outcome_rows)
    dynamic_aggregate = aggregate_dynamic_summary(dynamic_rows)
    summary = {
        "comparison_id": comparison_id,
        "source_stage4b_bundle": str(stage4b),
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "target_boston_runtime_seconds": target_seconds,
        "target_boston_runtime_hours": args.target_hours,
        "leadership_quantum_seconds": args.leadership_quantum_seconds,
        "windows_per_season": args.windows_per_season,
        "candidate_arrival_count": len(candidate_rows),
        "selected_arrival_count": len(selected_rows),
        "expected_leadership_samples_per_scenario": expected_leadership_per_scenario,
        "policy_names": list(DYNAMIC_POLICIES),
        "workload_classes": list(CORE_WORKLOADS),
        "months": months,
        "expected_scenario_count": len(selected_rows) * len(CORE_WORKLOADS),
        "observed_scenario_count": len(scenarios),
        "expected_outcome_count": expected_outcomes,
        "observed_outcome_count": len(all_outcome_rows),
        "observed_magellan_migration_event_count": len(all_migration_rows),
        "dynamic_traversal_observed": dynamic_aggregate["dynamic_traversal_observed"],
        "passed": (
            len(all_outcome_rows) == expected_outcomes
            and all(bool(row["completed"]) for row in all_outcome_rows)
            and len(dynamic_rows) == len(scenarios)
            and len(all_leadership_rows) == len(scenarios) * expected_leadership_per_scenario
        ),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4c_72h_dynamic_carbon_crossover",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "horizon": "Each workload is scaled so Boston-static performs exactly 72 hours of useful work. Measured power, checkpoint/restore cost, checkpoint payload, WAN calibration, and regional slowdown factors remain frozen.",
            "candidate_arrivals": "The same 24 deterministic annual arrivals screened by Stage 4B: day 5 at 00:00 UTC and day 20 at 12:00 UTC for every month.",
            "selection": "Within each season, choose the fixed number of arrivals with the most sustained scheduler-carbon leadership transitions over the next 72 hours; ties use number of leaders, raw leadership changes, then earliest UTC. Selection uses only carbon traces/PUE and occurs before Magellan replay.",
            "purpose": "Stress-test whether the production causal Magellan policy revisits placement as carbon leadership changes during a long-running job. Multiple migrations are observed outcomes, never a pass criterion and never forced.",
            "policies": "Stage 4C replays only Boston-static and Magellan. The five-policy representative annual comparison remains frozen in Stage 4B; repeating GAIA/best-static/oracle is unnecessary for the dynamic-path question.",
            "scheduler_carbon_leader": "Lowest lifecycle carbon intensity multiplied by node PUE, matching the equal-duration regional carbon ordering visible to the production scoring model.",
            "realized_work_carbon_leader": "Lowest lifecycle carbon intensity multiplied by PUE and the Stage-4A.4 slowdown factor, an offline diagnostic for carbon per unit of completed Boston-equivalent work.",
            "opportunity_window": "A leadership window is a cross-region opportunity only if the leader differs from the current owner and persists at least the configured minimum migration gap. This is descriptive and does not assert migration should occur.",
            "migration_counterfactual": "For each actual migration, an offline clairvoyant diagnostic compares staying at the source for all remaining work with paying measured migration overhead and then staying at the destination. It is not used by the causal scheduler.",
            "paper_alignment": "The 72-hour horizon captures diurnal/inter-regional variability while remaining short enough that transient carbon volatility is not averaged away, matching the SC26 evaluation rationale.",
        },
    }
    calibration_model = {
        "runtime_scales_by_class": runtime_scales,
        "node_slowdown_factors": node_slowdowns,
        "workloads": {key: value.as_dict() for key, value in calibrations.items()},
        "stage4a1_edge_count": len(edge_rows),
        "stage4a1_calibration_id": a1_summary.get("calibration_id"),
        "stage4a2_calibration_id": a2_summary.get("calibration_id"),
        "stage4a3_calibration_id": a3_summary.get("calibration_id"),
        "stage4a4_calibration_id": a4_summary.get("calibration_id"),
        "stage4a5_calibration_id": a5_summary.get("calibration_id"),
        "stage4b_comparison_id": stage4b_summary.get("comparison_id"),
    }

    write_csv(root / "candidate_windows.csv", candidate_rows, list(candidate_rows[0].keys()))
    write_csv(root / "selected_windows.csv", selected_rows, list(selected_rows[0].keys()))
    write_csv(root / "scenarios.csv", scenario_rows, list(scenario_rows[0].keys()))
    write_csv(root / "outcomes.csv", all_outcome_rows, list(all_outcome_rows[0].keys()))
    write_csv(root / "policy_summary.csv", policy_summary, list(policy_summary[0].keys()))
    write_csv(root / "magellan_dynamic_summary.csv", dynamic_rows, list(dynamic_rows[0].keys()))
    write_csv(root / "leadership_timeline.csv", all_leadership_rows, list(all_leadership_rows[0].keys()))
    write_csv(root / "leadership_windows.csv", all_window_rows, list(all_window_rows[0].keys()) if all_window_rows else ["scenario_id"])
    write_csv(root / "magellan_migrations.csv", all_migration_rows, MIGRATION_DIAGNOSTIC_FIELDS)
    write_csv(root / "magellan_residence.csv", all_residence_rows, RESIDENCE_FIELDS)
    write_jsonl(root / "traces.jsonl", all_trace_rows)
    write_json(root / "dynamic_summary.json", dynamic_aggregate)
    write_json(root / "calibration_model.json", calibration_model)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4C_DYNAMIC_CROSSOVER_PASS" if summary["passed"] else "STAGE_4C_DYNAMIC_CROSSOVER_INCOMPLETE"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"selected_arrivals: {len(selected_rows)}")
    print(f"scenarios: {len(scenarios)}/{summary['expected_scenario_count']}")
    print(f"outcomes: {len(all_outcome_rows)}/{expected_outcomes}")
    print(f"magellan_migrations: {dynamic_aggregate['magellan_migrations_total']}")
    print(f"multi_migration_scenarios: {dynamic_aggregate['scenarios_multi_migration']}/{len(scenarios)}")
    print(f"distinct_owner_paths: {dynamic_aggregate['distinct_owner_paths']}")
    print(f"scheduler_leader_changes_total: {dynamic_aggregate['scheduler_carbon_leader_changes_total']}")
    print(f"scheduler_opportunities_exploited: {dynamic_aggregate['scheduler_opportunities_exploited_total']}/{dynamic_aggregate['scheduler_opportunity_windows_total']}")
    print(f"dynamic_traversal_observed: {dynamic_aggregate['dynamic_traversal_observed']}")
    for row in policy_summary:
        if row["policy"] in DYNAMIC_POLICIES:
            print(
                f"{row['policy']}: carbon_ratio={float(row['carbon_ratio_mean']):.4f} "
                f"time_ratio={float(row['time_ratio_mean']):.4f} cost_ratio={float(row['cost_ratio_mean']):.4f}"
            )
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
