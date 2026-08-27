#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import validate_checksums, write_checksums, write_csv, write_json
from magellan.experiments.stage4a4 import StaticCase, build_static_cases, successful_static_bundle
from magellan.experiments.stage4a5 import (
    DEFAULT_MEDIAN_ERROR_GATE_PERCENT,
    DEFAULT_P95_ERROR_GATE_PERCENT,
    DEFAULT_VALIDATION_CLASSES,
    DEFAULT_VALIDATION_NODES,
    predict_runtime_seconds,
    read_csv,
    runtime_model_tables,
    runtime_validation_row,
    summarize_migration_evidence,
    summarize_runtime_validation,
)


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4A.5 calibration/model validation.")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--stage4a1-bundle", required=True)
    parser.add_argument("--stage4a2-bundle", required=True)
    parser.add_argument("--stage4a3-bundle", required=True)
    parser.add_argument("--stage4a4-bundle", required=True)
    parser.add_argument("--validation-nodes", default=",".join(DEFAULT_VALIDATION_NODES))
    parser.add_argument("--validation-classes", default=",".join(DEFAULT_VALIDATION_CLASSES))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--minimum-samples-per-run", type=int, default=3)
    parser.add_argument("--median-error-gate-percent", type=float, default=DEFAULT_MEDIAN_ERROR_GATE_PERCENT)
    parser.add_argument("--p95-error-gate-percent", type=float, default=DEFAULT_P95_ERROR_GATE_PERCENT)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--dendro-definition", default="config/submissions/dendro-bssn-template.json")
    parser.add_argument("--dendro-solver", default="/home/WILL/dgr-build/BSSN_GR/bssnSolver")
    parser.add_argument("--dendro-parameter-template", default="/home/WILL/q1-magellan-magellan.toml")
    parser.add_argument("--llm-model", default="experiment-assets/models/distilgpt2")
    parser.add_argument("--llm-checkpoint-every", type=int, default=1)
    parser.add_argument("--llm-sleep-per-step", type=float, default=2.0)
    parser.add_argument("--llm-torch-threads", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=2.5)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_source_bundle(path: Path, stage: str) -> dict[str, Any]:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{stage} checksum validation failed: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if stage == "Stage 4A.1":
        if summary.get("hardware_preflight_passed") is not True:
            raise RuntimeError("Stage 4A.1 source bundle did not pass hardware preflight")
    elif summary.get("passed") is not True:
        raise RuntimeError(f"{stage} source bundle is not passed")
    return summary


def child_command(
    *,
    args: argparse.Namespace,
    case: StaticCase,
    node_id: str,
    trial: int,
    root: Path,
    case_id: str,
) -> list[str]:
    command = [
        "python", "scripts/measure_stage4a4_static.py",
        "--cluster", args.cluster,
        "--local-node-id", args.local_node_id,
        "--ssh-user", args.ssh_user,
        "--remote-repo", args.remote_repo,
        "--node", node_id,
        "--workload", case.workload,
        "--class-id", case.class_id,
        "--trial", str(trial),
        "--scope", "validation",
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
            "--minimum-free-gib", str(args.minimum_free_gib),
        ])
    return command


def main() -> int:
    args = parse_args()
    if args.trials < 1 or args.minimum_samples_per_run < 1:
        raise ValueError("trials/minimum samples must be positive")

    a1 = Path(args.stage4a1_bundle)
    a2 = Path(args.stage4a2_bundle)
    a3 = Path(args.stage4a3_bundle)
    a4 = Path(args.stage4a4_bundle)
    a1_summary = require_source_bundle(a1, "Stage 4A.1")
    a2_summary = require_source_bundle(a2, "Stage 4A.2")
    a3_summary = require_source_bundle(a3, "Stage 4A.3")
    a4_summary = require_source_bundle(a4, "Stage 4A.4")

    validation_nodes = parse_csv_list(args.validation_nodes)
    validation_classes = parse_csv_list(args.validation_classes)
    if not validation_nodes or not validation_classes:
        raise ValueError("validation node/class sets must be non-empty")

    cluster = load_cluster_config(args.cluster)
    cluster_nodes = {node.id for node in cluster.nodes}
    unknown_nodes = sorted(set(validation_nodes) - cluster_nodes)
    if unknown_nodes:
        raise ValueError(f"Unknown validation nodes: {unknown_nodes}")

    target_seconds = float(a4_summary["target_seconds"])
    all_cases = build_static_cases(a3 / "profile_classes.csv", target_seconds=target_seconds)
    case_by_id = {case.class_id: case for case in all_cases}
    unknown_classes = sorted(set(validation_classes) - set(case_by_id))
    if unknown_classes:
        raise ValueError(f"Unknown validation classes: {unknown_classes}")
    selected_cases = [case_by_id[class_id] for class_id in validation_classes]
    class_runtime, node_slowdown = runtime_model_tables(
        a4 / "static_classes.csv",
        a4 / "node_equivalence.csv",
    )
    for node_id in validation_nodes:
        if node_id not in node_slowdown:
            raise ValueError(f"Stage 4A.4 has no slowdown factor for {node_id}")

    calibration_id = args.calibration_id or f"stage4a5-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / calibration_id
    if root.exists() and not args.resume:
        raise FileExistsError(f"Calibration bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements").mkdir(exist_ok=True)

    print("== Stage 4A.5 calibration/model validation ==")
    print(f"calibration_id={calibration_id}")
    print(f"nodes={','.join(validation_nodes)}")
    print(f"classes={','.join(validation_classes)} trials={args.trials}")

    rows: list[dict[str, Any]] = []
    for case in selected_cases:
        for node_id in validation_nodes:
            predicted = predict_runtime_seconds(
                class_id=case.class_id,
                node_id=node_id,
                class_runtime=class_runtime,
                node_slowdown=node_slowdown,
            )
            for trial in range(1, args.trials + 1):
                case_id = f"validation-{case.class_id}-{node_id}-trial{trial:02d}"
                bundle = root / "measurements" / case_id
                if args.resume and successful_static_bundle(bundle, minimum_samples=args.minimum_samples_per_run):
                    print(f"[resume] already passed: {case_id}")
                else:
                    if bundle.exists():
                        raise FileExistsError(f"Case bundle exists but is not resumable: {bundle}")
                    subprocess.run(
                        child_command(
                            args=args,
                            case=case,
                            node_id=node_id,
                            trial=trial,
                            root=root,
                            case_id=case_id,
                        ),
                        check=True,
                    )
                summary = read_json(bundle / "summary.json")
                rows.append(
                    runtime_validation_row(
                        class_id=case.class_id,
                        workload=case.workload,
                        node_id=node_id,
                        trial=trial,
                        run_id=str(summary["run_id"]),
                        measurement_id=case_id,
                        actual_seconds=float(summary["wall_seconds"]),
                        predicted_seconds=predicted,
                        telemetry_sample_count=int(summary["telemetry_sample_count"]),
                    )
                )

    runtime_validation = summarize_runtime_validation(
        rows,
        median_gate_percent=args.median_error_gate_percent,
        p95_gate_percent=args.p95_error_gate_percent,
    )
    write_csv(root / "runtime_validation_runs.csv", rows, list(rows[0].keys()))
    write_csv(root / "runtime_validation_by_class.csv", runtime_validation["by_class"], list(runtime_validation["by_class"][0].keys()))
    write_csv(root / "runtime_validation_by_node.csv", runtime_validation["by_node"], list(runtime_validation["by_node"][0].keys()))
    write_json(root / "runtime_validation.json", runtime_validation)

    migration_rows = read_csv(a2 / "migration_samples.csv")
    a1_network = a1_summary.get("network") or {}
    calibration_evidence = {
        "stage4a1": {
            "bundle": str(a1),
            "calibration_id": a1_summary.get("calibration_id"),
            "node_count": a1_summary.get("node_count"),
            "network": a1_network,
        },
        "stage4a2": {
            "bundle": str(a2),
            "calibration_id": a2_summary.get("calibration_id"),
            "migration_accuracy": a2_summary.get("migration_accuracy"),
            "migration_accuracy_by_checkpoint_scale": summarize_migration_evidence(migration_rows),
        },
        "stage4a3": {
            "bundle": str(a3),
            "calibration_id": a3_summary.get("calibration_id"),
            "class_count": a3_summary.get("observed_class_count"),
            "run_count": a3_summary.get("observed_run_count"),
        },
        "stage4a4": {
            "bundle": str(a4),
            "calibration_id": a4_summary.get("calibration_id"),
            "authoritative_runtime_field": a4_summary.get("authoritative_runtime_field"),
            "representative_equivalence_class": a4_summary.get("representative_equivalence_class"),
        },
    }
    write_json(root / "calibration_evidence.json", calibration_evidence)

    expected_runs = len(selected_cases) * len(validation_nodes) * args.trials
    summary = {
        "calibration_id": calibration_id,
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "validation_nodes": validation_nodes,
        "validation_classes": validation_classes,
        "trials_per_pair": args.trials,
        "expected_runtime_validation_run_count": expected_runs,
        "observed_runtime_validation_run_count": len(rows),
        "runtime_model_transfer_passed": runtime_validation["runtime_model_transfer_passed"],
        "recommended_runtime_model": runtime_validation["recommended_runtime_model"],
        "llm_regional_runtime_transfer_validated": "llm-distilgpt2" in validation_classes,
        "ready_for_stage4b_runtime_model": bool(
            runtime_validation["runtime_model_transfer_passed"]
            and "llm-distilgpt2" in validation_classes
        ),
        "passed": len(rows) == expected_runs,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4a5_calibration_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "held_out_runtime_validation": (
                "Predict runtime as Stage-4A.4 Boston class median multiplied by the Stage-4A.4 per-node "
                "matmul-medium slowdown factor, then compare with new physical JSON-medium, Dendro-r9, and DistilGPT2 runs."
            ),
            "validation_matrix": (
                "Default: Boston, South Australia, Ethiopia, Virginia x JSON-medium, Dendro-r9-t1p0, DistilGPT2 x 2 trials = 24 runs."
            ),
            "gate": (
                "The single-factor model passes only when overall, every class, and every node remain within the configured "
                "median and p95 absolute-percent-error gates. A failed gate completes Stage 4A.5 but blocks use of the single-factor model in Stage 4B."
            ),
            "llm_scope": (
                "DistilGPT2 is included in the held-out matrix. The identical local model snapshot must be provisioned on every validation node; the child harness verifies model presence and free disk before creating the task."
            ),
        },
    }
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    print("\nSTAGE_4A5_CALIBRATION_VALIDATION_COMPLETE")
    print(f"bundle: {root}")
    print(f"runtime_validation_runs: {len(rows)}/{expected_runs}")
    print("runtime_model_transfer: " + ("PASS" if runtime_validation["runtime_model_transfer_passed"] else "FAIL"))
    overall = runtime_validation["overall_absolute_error_percent"]
    print(f"runtime_error_median_pct: {float(overall['median']):.2f}")
    print(f"runtime_error_p95_pct: {float(overall['p95']):.2f}")
    print(f"recommended_runtime_model: {runtime_validation['recommended_runtime_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
