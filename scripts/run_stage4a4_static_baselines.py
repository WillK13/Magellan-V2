#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import validate_checksums, write_checksums, write_csv, write_json
from magellan.experiments.stage4a4 import (
    REPRESENTATIVE_EQUIVALENCE_CLASS,
    StaticCase,
    build_static_cases,
    successful_static_bundle,
    summarize_canonical_runs,
    summarize_node_equivalence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4A.4 static completion and node-equivalence calibration.")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--stage4a3-bundle", required=True)
    parser.add_argument("--canonical-node", default="boston")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--target-seconds", type=float, default=100.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--minimum-samples-per-run", type=int, default=3)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dendro-definition", default="config/submissions/dendro-bssn-template.json")
    parser.add_argument("--dendro-solver", default="/home/WILL/dgr-build/BSSN_GR/bssnSolver")
    parser.add_argument("--dendro-parameter-template", default="/home/WILL/q1-magellan-magellan.toml")
    parser.add_argument("--llm-model", default="experiment-assets/models/distilgpt2")
    parser.add_argument("--llm-checkpoint-every", type=int, default=1)
    parser.add_argument("--llm-sleep-per-step", type=float, default=2.0)
    parser.add_argument("--llm-torch-threads", type=int, default=2)
    return parser.parse_args()


def run_child(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def child_command(
    *,
    args: argparse.Namespace,
    case: StaticCase,
    node_id: str,
    trial: int,
    scope: str,
    root: Path,
    case_id: str,
) -> list[str]:
    command = [
        "python", "scripts/measure_stage4a4_static.py",
        "--cluster", args.cluster,
        "--local-node-id", args.local_node_id,
        "--node", node_id,
        "--workload", case.workload,
        "--class-id", case.class_id,
        "--trial", str(trial),
        "--scope", scope,
        "--sample-interval-seconds", str(args.sample_interval_seconds),
        "--timeout-seconds", str(args.timeout_seconds),
        "--measurements-root", str(root / "measurements"),
        "--measurement-id", case_id,
    ]
    if case.workload == "benchmark":
        command.extend([
            "--benchmark", str(case.benchmark),
            "--size", str(case.size),
            "--benchmark-iterations", str(case.benchmark_iterations),
        ])
    elif case.workload == "dendro":
        command.extend([
            "--resolution", str(case.resolution),
            "--time-end", str(case.time_end),
            "--dendro-definition", args.dendro_definition,
            "--dendro-solver", args.dendro_solver,
            "--dendro-parameter-template", args.dendro_parameter_template,
        ])
    else:
        command.extend([
            "--model", args.llm_model,
            "--llm-max-steps", str(case.llm_max_steps),
            "--llm-checkpoint-every", str(args.llm_checkpoint_every),
            "--llm-sleep-per-step", str(args.llm_sleep_per_step),
            "--llm-torch-threads", str(args.llm_torch_threads),
        ])
    return command


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def row_from_summary(summary: dict[str, Any], *, physical_scope: str) -> dict[str, Any]:
    return {
        "measurement_id": summary["measurement_id"],
        "class_id": summary["class_id"],
        "workload": summary["workload"],
        "variant": summary["variant"],
        "node_id": summary["node_id"],
        "trial": summary["trial"],
        "scope": physical_scope,
        "run_id": summary["run_id"],
        "status": summary["status"],
        "generation": summary["generation"],
        "wall_seconds": summary["wall_seconds"],
        "telemetry_sample_count": summary["telemetry_sample_count"],
        "progress_completed_units": summary["progress_completed_units"],
        "progress_total_units": summary["progress_total_units"],
        "accumulated_runtime_seconds": summary["accumulated_runtime_seconds"],
        "accumulated_paused_seconds": summary["accumulated_paused_seconds"],
        "accumulated_migration_seconds": summary["accumulated_migration_seconds"],
        "accumulated_compute_cost_usd": summary["accumulated_compute_cost_usd"],
        "accumulated_transfer_cost_usd": summary["accumulated_transfer_cost_usd"],
        "accumulated_cost_usd": summary["accumulated_cost_usd"],
        "accumulated_compute_carbon_grams": summary["accumulated_compute_carbon_grams"],
        "accumulated_transfer_carbon_grams": summary["accumulated_transfer_carbon_grams"],
        "accumulated_carbon_grams": summary["accumulated_carbon_grams"],
    }


def main() -> int:
    args = parse_args()
    if args.trials < 1 or args.minimum_samples_per_run < 1:
        raise ValueError("trials/minimum samples must be positive")
    stage4a3 = Path(args.stage4a3_bundle)
    profile_csv = stage4a3 / "profile_classes.csv"
    if not profile_csv.is_file():
        raise FileNotFoundError(profile_csv)
    source_errors = validate_checksums(stage4a3)
    if source_errors:
        raise RuntimeError("Stage 4A.3 source bundle checksum validation failed: " + "; ".join(source_errors))
    source_summary_path = stage4a3 / "summary.json"
    if not source_summary_path.is_file() or json.loads(source_summary_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("Stage 4A.3 source bundle is not passed")
    cases = build_static_cases(profile_csv, target_seconds=args.target_seconds)
    case_by_id = {case.class_id: case for case in cases}
    equivalence_case = case_by_id[REPRESENTATIVE_EQUIVALENCE_CLASS]

    cluster = load_cluster_config(args.cluster)
    node_ids = [node.id for node in cluster.nodes]
    if args.canonical_node not in node_ids:
        raise ValueError(f"Unknown canonical node: {args.canonical_node}")

    calibration_id = args.calibration_id or f"stage4a4-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / calibration_id
    if root.exists() and not args.resume:
        raise FileExistsError(f"Calibration bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir(exist_ok=True)

    print("== Stage 4A.4 static execution baselines ==")
    print(f"calibration_id={calibration_id}")
    print(f"canonical_node={args.canonical_node} trials={args.trials} target_seconds={args.target_seconds:g}")

    canonical_summaries: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        for case in cases:
            case_id = f"{case.class_id}-{args.canonical_node}-trial{trial:02d}"
            bundle = root / "measurements" / case_id
            if args.resume and successful_static_bundle(bundle, minimum_samples=args.minimum_samples_per_run):
                print(f"[resume] already passed: {case_id}")
            else:
                if bundle.exists():
                    raise FileExistsError(f"Case bundle exists but is not resumable: {bundle}")
                run_child(child_command(args=args, case=case, node_id=args.canonical_node, trial=trial, scope="canonical", root=root, case_id=case_id))
            canonical_summaries.append(load_summary(bundle))

    equivalence_summaries: list[dict[str, Any]] = []
    for node_id in node_ids:
        if node_id == args.canonical_node:
            equivalence_summaries.extend(
                summary for summary in canonical_summaries if summary["class_id"] == REPRESENTATIVE_EQUIVALENCE_CLASS
            )
            continue
        for trial in range(1, args.trials + 1):
            case_id = f"equivalence-{REPRESENTATIVE_EQUIVALENCE_CLASS}-{node_id}-trial{trial:02d}"
            bundle = root / "measurements" / case_id
            if args.resume and successful_static_bundle(bundle, minimum_samples=args.minimum_samples_per_run):
                print(f"[resume] already passed: {case_id}")
            else:
                if bundle.exists():
                    raise FileExistsError(f"Case bundle exists but is not resumable: {bundle}")
                run_child(child_command(args=args, case=equivalence_case, node_id=node_id, trial=trial, scope="equivalence", root=root, case_id=case_id))
            equivalence_summaries.append(load_summary(bundle))

    canonical_rows = [row_from_summary(summary, physical_scope="canonical") for summary in canonical_summaries]
    additional_equivalence = [
        summary for summary in equivalence_summaries if summary["node_id"] != args.canonical_node
    ]
    physical_rows = canonical_rows + [row_from_summary(summary, physical_scope="equivalence") for summary in additional_equivalence]
    effective_equivalence_rows = [row_from_summary(summary, physical_scope="equivalence-evidence") for summary in equivalence_summaries]

    canonical_classes = summarize_canonical_runs(canonical_rows, trials=args.trials)
    node_equivalence = summarize_node_equivalence(
        effective_equivalence_rows,
        canonical_node_id=args.canonical_node,
        trials=args.trials,
    )

    write_csv(root / "static_runs.csv", physical_rows, list(physical_rows[0].keys()))
    write_csv(root / "static_classes.csv", canonical_classes, list(canonical_classes[0].keys()))
    write_csv(root / "node_equivalence.csv", node_equivalence, list(node_equivalence[0].keys()))
    case_summaries = canonical_summaries + additional_equivalence
    write_json(root / "case_summaries.json", case_summaries)

    expected_canonical_runs = len(cases) * args.trials
    expected_additional_equivalence_runs = (len(node_ids) - 1) * args.trials
    expected_physical_runs = expected_canonical_runs + expected_additional_equivalence_runs
    summary = {
        "calibration_id": calibration_id,
        "stage4a3_bundle": str(stage4a3),
        "canonical_node_id": args.canonical_node,
        "representative_equivalence_class": REPRESENTATIVE_EQUIVALENCE_CLASS,
        "trials_per_class": args.trials,
        "target_seconds": args.target_seconds,
        "authoritative_runtime_field": "wall_seconds",
        "expected_class_count": len(cases),
        "observed_class_count": len(canonical_classes),
        "expected_node_count": len(node_ids),
        "observed_node_count": len(node_equivalence),
        "expected_canonical_run_count": expected_canonical_runs,
        "expected_additional_equivalence_run_count": expected_additional_equivalence_runs,
        "expected_physical_run_count": expected_physical_runs,
        "observed_physical_run_count": len(physical_rows),
        "effective_equivalence_sample_count": len(effective_equivalence_rows),
        "passed": bool(
            len(canonical_classes) == len(cases)
            and len(node_equivalence) == len(node_ids)
            and len(physical_rows) == expected_physical_runs
            and all(successful_static_bundle(root / "measurements" / summary["measurement_id"], minimum_samples=args.minimum_samples_per_run) for summary in case_summaries)
        ),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4a4_static_execution_baselines",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": calibration_id,
        "cluster": args.cluster,
        "stage4a3_bundle": str(stage4a3),
        "cases": [case.__dict__ for case in cases],
        "node_ids": node_ids,
        "methodology": {
            "canonical_static_completion": "All 13 Stage 4A.3 workload classes run to natural completion on the canonical node for three trials with scheduler_mode=operator_only.",
            "finite_length_selection": "Benchmark iterations and LLM max_steps are derived from Stage 4A.3 median progress rates to target a stable finite completion window; Dendro physical resolution/time-end variants are unchanged.",
            "node_equivalence": f"{REPRESENTATIVE_EQUIVALENCE_CLASS} is measured on all seven identical final-hardware nodes; canonical-node trials are reused from the 13-class matrix.",
            "completion_reconciliation": "The child harness invokes the operator-only /runtime/reconcile endpoint while polling so naturally exited Dendro tasks are finalized promptly without waiting for a scheduler epoch.",
            "authoritative_runtime": "started_at_utc to validated completion-marker time (wall_seconds). Persisted accumulated accounting fields are diagnostic only in Stage 4A.4 and are not used for runtime or slowdown aggregation.",
            "accuracy_policy": "Node slowdown factors are recorded descriptively; no regional runtime samples are discarded by an accuracy threshold.",
        },
    }
    write_json(root / "summary.json", summary)
    write_json(root / "metadata.json", metadata)
    write_checksums(root)
    if summary["passed"] is not True:
        raise RuntimeError(f"Stage 4A.4 parent invariants failed: {summary}")

    print("\nSTAGE_4A4_STATIC_BASELINES_PASS")
    print(f"bundle: {root}")
    print(f"classes: {len(canonical_classes)}/{len(cases)}")
    print(f"nodes: {len(node_equivalence)}/{len(node_ids)}")
    print(f"physical_runs: {len(physical_rows)}/{expected_physical_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
