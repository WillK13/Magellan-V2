#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.carbon.store import CarbonMetric, CarbonStore
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.bundle import validate_checksums, write_checksums, write_csv, write_json, write_jsonl
from magellan.experiments.stage4b import (
    CORE_POLICIES,
    CORE_WORKLOADS,
    DEFAULT_GAIA_QUANTUM_SECONDS,
    DEFAULT_ORACLE_WAIT_SECONDS,
    DEFAULT_RUNTIME_SCALE,
    FrozenCalibrationGraph,
    annual_scenarios,
    descriptive_policy_metrics,
    gaia_queue_parameters,
    load_node_slowdowns,
    load_stage4a1_edges,
    load_workload_calibrations,
    outcome_rows,
    scenario_outcomes,
    summarize_policy_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4B calibrated core policy comparison.")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--stage4a1-bundle", required=True)
    parser.add_argument("--stage4a2-bundle", required=True)
    parser.add_argument("--stage4a3-bundle", required=True)
    parser.add_argument("--stage4a4-bundle", required=True)
    parser.add_argument("--stage4a5-bundle", required=True)
    parser.add_argument("--runtime-scale", type=float, default=DEFAULT_RUNTIME_SCALE)
    parser.add_argument("--gaia-quantum-seconds", type=float, default=DEFAULT_GAIA_QUANTUM_SECONDS)
    parser.add_argument("--oracle-wait-seconds", type=float, default=DEFAULT_ORACLE_WAIT_SECONDS)
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


def main() -> int:
    args = parse_args()
    if args.runtime_scale <= 0 or args.gaia_quantum_seconds <= 0 or args.oracle_wait_seconds < 0:
        raise ValueError("runtime scale/GAIA quantum/oracle wait are invalid")
    months = [int(value) for value in args.months.split(",") if value.strip()]
    if not months:
        raise ValueError("months must be non-empty")

    a1 = Path(args.stage4a1_bundle)
    a2 = Path(args.stage4a2_bundle)
    a3 = Path(args.stage4a3_bundle)
    a4 = Path(args.stage4a4_bundle)
    a5 = Path(args.stage4a5_bundle)
    a1_summary = require_bundle(a1, "Stage 4A.1", require_passed=False)
    if a1_summary.get("hardware_preflight_passed") is not True:
        raise RuntimeError("Stage 4A.1 hardware preflight did not pass")
    if int(a1_summary.get("node_count") or 0) != len(load_cluster_config(args.cluster).nodes):
        raise RuntimeError("Stage 4A.1 node count does not match configured cluster")
    a2_summary = require_bundle(a2, "Stage 4A.2")
    a3_summary = require_bundle(a3, "Stage 4A.3")
    a4_summary = require_bundle(a4, "Stage 4A.4")
    a5_summary = require_bundle(a5, "Stage 4A.5")
    if a5_summary.get("ready_for_stage4b_runtime_model") is not True:
        raise RuntimeError("Stage 4A.5 did not approve the single-node slowdown runtime model for Stage 4B")
    if a5_summary.get("recommended_runtime_model") != "single_node_slowdown_factor":
        raise RuntimeError("Unexpected Stage 4A.5 runtime-model recommendation")

    cluster = load_cluster_config(args.cluster)
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
    edge_rows = load_stage4a1_edges(a1, a1_summary)
    expected_edges = len(cluster.nodes) * (len(cluster.nodes) - 1)
    if len(edge_rows) != expected_edges:
        raise RuntimeError(f"Stage 4A.1 edge count {len(edge_rows)} != expected {expected_edges}")
    gaia_queues = gaia_queue_parameters(calibrations, runtime_scale=args.runtime_scale)
    scenarios = annual_scenarios(class_ids=CORE_WORKLOADS, months=months)

    comparison_id = args.comparison_id or f"stage4b-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4B calibrated core comparison ==")
    print(f"comparison_id={comparison_id}")
    print(f"scenarios={len(scenarios)} policies={len(CORE_POLICIES)} expected_outcomes={len(scenarios) * len(CORE_POLICIES)}")
    print(f"runtime_scale={args.runtime_scale:g}x carbon_metric=lifecycle")

    scenario_rows = []
    all_outcome_rows = []
    trace_rows = []
    for index, scenario in enumerate(scenarios, start=1):
        calibration = calibrations[scenario.class_id]
        graph = FrozenCalibrationGraph(cluster=cluster, edge_rows=edge_rows, workload=calibration)
        outcomes = scenario_outcomes(
            scenario=scenario,
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            graph=graph,
            runtime_scale=args.runtime_scale,
            gaia_queues=gaia_queues,
            gaia_quantum_seconds=args.gaia_quantum_seconds,
            oracle_wait_seconds=args.oracle_wait_seconds,
        )
        rows = outcome_rows(scenario=scenario, calibration=calibration, outcomes=outcomes, policy=policy)
        all_outcome_rows.extend(rows)
        scenario_rows.append({
            "scenario_id": scenario.scenario_id,
            "class_id": scenario.class_id,
            "workload": calibration.workload,
            "variant": calibration.variant,
            "arrival_utc": scenario.arrival_utc.isoformat(),
            "canonical_runtime_seconds": calibration.canonical_runtime_seconds,
            "scaled_boston_work_seconds": calibration.scaled_work_seconds(args.runtime_scale),
            "power_kw": calibration.power_kw,
            "checkpoint_bytes": calibration.checkpoint_bytes,
        })
        for outcome in outcomes:
            trace_rows.append({
                "scenario_id": scenario.scenario_id,
                "class_id": scenario.class_id,
                "arrival_utc": scenario.arrival_utc.isoformat(),
                "outcome": outcome.model_dump(mode="json"),
            })
        print(f"[{index:02d}/{len(scenarios):02d}] {scenario.scenario_id} complete")

    policy_summary = summarize_policy_rows(all_outcome_rows)
    expected_outcomes = len(scenarios) * len(CORE_POLICIES)
    summary = {
        "comparison_id": comparison_id,
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "runtime_model": "single_node_slowdown_factor",
        "runtime_scale": args.runtime_scale,
        "carbon_metric": "lifecycle",
        "policy_names": list(CORE_POLICIES),
        "workload_classes": list(CORE_WORKLOADS),
        "months": months,
        "expected_scenario_count": len(months) * 2 * len(CORE_WORKLOADS),
        "observed_scenario_count": len(scenarios),
        "expected_outcome_count": expected_outcomes,
        "observed_outcome_count": len(all_outcome_rows),
        "passed": len(all_outcome_rows) == expected_outcomes and all(bool(row["completed"]) for row in all_outcome_rows),
    }
    calibration_model = {
        "node_slowdown_factors": node_slowdowns,
        "workloads": {key: value.as_dict() for key, value in calibrations.items()},
        "gaia_queue_parameters": gaia_queues,
        "stage4a1_edge_count": len(edge_rows),
        "stage4a1_calibration_id": a1_summary.get("calibration_id"),
        "stage4a2_calibration_id": a2_summary.get("calibration_id"),
        "stage4a3_calibration_id": a3_summary.get("calibration_id"),
        "stage4a4_calibration_id": a4_summary.get("calibration_id"),
        "stage4a5_calibration_id": a5_summary.get("calibration_id"),
    }
    gaia_reproduction = {
        "paper": "Hanafy et al., Going Green for Less Green, ASPLOS 2024",
        "artifact_repository": "https://github.com/umassos/GAIA",
        "artifact_policy_mapping": {"Carbon-Time": {"scheduling_policy": "carbon", "carbon_policy": "cst_average"}},
        "formula": "CST(t_start) = (C(t) - C(t_start)) / (t_start + J_avg - t)",
        "execution_model": "fixed Boston placement; select a submission-time start; uninterruptible execution to completion",
        "future_carbon_knowledge": "perfect within allowed waiting window, matching GAIA evaluation assumption",
        "short_queue_max_runtime_seconds": 7200,
        "short_queue_max_wait_seconds": 21600,
        "long_queue_max_wait_seconds": 86400,
        "queue_runtime_estimate": "mean scaled Boston runtime of Stage 4B classes assigned to the same GAIA queue",
        "candidate_quantum_seconds": args.gaia_quantum_seconds,
        "reproduction_note": "Equation-level Carbon-Time/cst_average reproduction inside the common Magellan replay harness; not a claim that upstream GAIA code was executed on Magellan traces.",
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4b_calibrated_core_policy_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "comparison": "All policies receive identical arrivals, 2024 lifecycle carbon traces, regional prices, Stage-4A.4 runtimes/slowdowns, Stage-4A.3 power, and frozen migration calibration.",
            "magellan": "Production evaluate_task is replayed causally with the configured linear-trend forecast and adaptive policy. Realized progress uses the validated per-node slowdown factor; destination slowdown is deliberately not injected into production migration scoring.",
            "migration": "Production candidate scoring and realized migrations use Stage-4A.1 affine WAN transfer calibration plus Stage-4A.2 workload checkpoint/restore/residual-overhead medians.",
            "best_static": "Free initial placement at arrival using actual calibrated execution-window metrics; no waiting or migration.",
            "oracle": "Clairvoyant static reference over node and hourly start time up to 24h; no migration. It is not claimed to be a universal optimal scheduler.",
            "runtime_scale": "Frozen physical completion time is multiplied by a constant to model long-running jobs while preserving measured power, working-set/checkpoint size, and regional slowdown factors.",
        },
    }

    write_csv(root / "scenarios.csv", scenario_rows, list(scenario_rows[0].keys()))
    write_csv(root / "outcomes.csv", all_outcome_rows, list(all_outcome_rows[0].keys()))
    write_csv(root / "policy_summary.csv", policy_summary, list(policy_summary[0].keys()))
    write_jsonl(root / "traces.jsonl", trace_rows)
    write_json(root / "policy_descriptive_metrics.json", descriptive_policy_metrics(all_outcome_rows))
    write_json(root / "calibration_model.json", calibration_model)
    write_json(root / "gaia_reproduction.json", gaia_reproduction)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4B_CORE_COMPARISON_PASS" if summary["passed"] else "STAGE_4B_CORE_COMPARISON_INCOMPLETE"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"scenarios: {len(scenarios)}/{summary['expected_scenario_count']}")
    print(f"outcomes: {len(all_outcome_rows)}/{expected_outcomes}")
    for row in policy_summary:
        print(f"{row['policy']}: carbon_ratio={float(row['carbon_ratio_mean']):.4f} time_ratio={float(row['time_ratio_mean']):.4f} cost_ratio={float(row['cost_ratio_mean']):.4f}")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
