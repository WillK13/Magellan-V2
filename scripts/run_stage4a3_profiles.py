#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.stage4a2 import summarize_profile_samples


BENCHMARKS = ("nbody", "json", "matmul")
SIZES = ("small", "medium", "large")
DENDRO_VARIANTS = ((8, 3.0), (9, 1.0), (10, 2.0))
MINIMUM_SAMPLES_PER_RUN = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4A.3 scheduler-isolated workload resource profiles on one "
            "canonical final-hardware node. No case performs a migration."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--node", default="boston")
    parser.add_argument("--partner-node", default="virginia")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile-seconds", type=float, default=20.0)
    parser.add_argument("--dendro-profile-seconds", type=float, default=20.0)
    parser.add_argument("--llm-profile-seconds", type=float, default=20.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--llm-model", default="experiment-assets/models/distilgpt2")
    parser.add_argument("--llm-sleep-per-step", type=float, default=2.0)
    parser.add_argument(
        "--phases",
        default="benchmarks,dendro,llm",
        help="Comma-separated subset of benchmarks,dendro,llm",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def profile_summary_passes(summary: dict[str, Any]) -> bool:
    profile = summary.get("profile") or {}
    sample_count = int(profile.get("sample_count") or 0)
    return (
        bool(summary.get("passed"))
        and summary.get("profile_only") is True
        and sample_count >= MINIMUM_SAMPLES_PER_RUN
    )


def successful_profile_bundle(path: Path) -> bool:
    try:
        summary = read_json(path / "summary.json")
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return profile_summary_passes(summary)


def run_case(command: list[str], bundle: Path, resume: bool) -> None:
    if bundle.exists():
        if resume and successful_profile_bundle(bundle):
            print(f"[resume] already passed: {bundle.name}")
            return
        raise FileExistsError(
            f"Stage 4A.3 case already exists but is not resumable: {bundle}. "
            "Inspect/remove only that failed case or use a new --calibration-id."
        )
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def safe_metric(profile: dict[str, Any], field: str, statistic: str) -> Any:
    value = profile.get(field) or {}
    return value.get(statistic) if isinstance(value, dict) else None


def main() -> int:
    args = parse_args()
    if args.node == args.partner_node:
        raise ValueError("--node and --partner-node must differ")
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if min(
        args.profile_seconds,
        args.dendro_profile_seconds,
        args.llm_profile_seconds,
        args.sample_interval_seconds,
    ) <= 0:
        raise ValueError("profile/sample intervals must be positive")

    phases = {item.strip() for item in args.phases.split(",") if item.strip()}
    unknown = phases - {"benchmarks", "dendro", "llm"}
    if unknown or not phases:
        raise ValueError(f"Invalid --phases: {sorted(unknown)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibration_id = args.calibration_id or f"stage4a3-{timestamp}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / calibration_id
    case_root = root / "measurements"
    if root.exists() and not args.resume:
        raise FileExistsError(f"Stage 4A.3 bundle already exists: {root}")
    case_root.mkdir(parents=True, exist_ok=True)

    print("== Stage 4A.3 workload resource profiling ==")
    print(f"calibration_id={calibration_id}")
    print(f"node={args.node} trials={args.trials}")

    for trial in range(1, args.trials + 1):
        trial_slug = f"trial{trial:02d}"
        if "benchmarks" in phases:
            for benchmark in BENCHMARKS:
                for size in SIZES:
                    case_id = f"benchmark-{benchmark}-{size}-{trial_slug}"
                    run_case(
                        [
                            "python", "scripts/measure_stage4a2_workload.py",
                            "--cluster", args.cluster,
                            "--local-node-id", args.local_node_id,
                            "--ssh-user", args.ssh_user,
                            "--workload", "benchmark",
                            "--benchmark", benchmark,
                            "--size", size,
                            "--source", args.node,
                            "--destination", args.partner_node,
                            "--profile-only",
                            "--profile-seconds", str(args.profile_seconds),
                            "--sample-interval-seconds", str(args.sample_interval_seconds),
                            "--timeout-seconds", str(args.timeout_seconds),
                            "--measurements-root", str(case_root),
                            "--measurement-id", case_id,
                        ],
                        case_root / case_id,
                        args.resume,
                    )

        if "dendro" in phases:
            for resolution, time_end in DENDRO_VARIANTS:
                time_slug = str(time_end).replace(".", "p")
                case_id = f"dendro-r{resolution}-t{time_slug}-{trial_slug}"
                run_case(
                    [
                        "python", "scripts/measure_stage4a2_workload.py",
                        "--cluster", args.cluster,
                        "--local-node-id", args.local_node_id,
                        "--ssh-user", args.ssh_user,
                        "--workload", "dendro",
                        "--resolution", str(resolution),
                        "--time-end", str(time_end),
                        "--source", args.node,
                        "--destination", args.partner_node,
                        "--profile-only",
                        "--profile-seconds", str(args.dendro_profile_seconds),
                        "--sample-interval-seconds", str(args.sample_interval_seconds),
                        "--timeout-seconds", str(args.timeout_seconds),
                        "--measurements-root", str(case_root),
                        "--measurement-id", case_id,
                    ],
                    case_root / case_id,
                    args.resume,
                )

        if "llm" in phases:
            case_id = f"llm-distilgpt2-{trial_slug}"
            run_case(
                [
                    "python", "scripts/measure_llm_migration.py",
                    "--cluster", args.cluster,
                    "--local-node-id", args.local_node_id,
                    "--ssh-user", args.ssh_user,
                    "--path", f"{args.node},{args.partner_node}",
                    "--model", args.llm_model,
                    "--migrate-after-step", "2",
                    "--checkpoint-every", "1",
                    "--sleep-per-step", str(args.llm_sleep_per_step),
                    "--torch-threads", "2",
                    "--timeout-seconds", str(args.timeout_seconds),
                    "--profile-only",
                    "--profile-seconds", str(args.llm_profile_seconds),
                    "--profile-sample-interval-seconds", str(args.sample_interval_seconds),
                    "--measurements-root", str(case_root),
                    "--measurement-id", case_id,
                ],
                case_root / case_id,
                args.resume,
            )

    case_summaries: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    samples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_meta: dict[str, tuple[str, str]] = {}

    for case in sorted(item for item in case_root.iterdir() if item.is_dir()):
        summary_path = case / "summary.json"
        sample_path = case / "profile_samples.csv"
        if not summary_path.is_file() or not sample_path.is_file():
            continue
        summary = read_json(summary_path)
        case_summaries.append({"case_id": case.name, **summary})
        workload = str(summary.get("workload") or ("llm" if summary.get("model") else ""))
        variant = str(summary.get("variant") or summary.get("model") or "")
        class_id = case.name.rsplit("-trial", 1)[0]
        samples = read_csv(sample_path)
        samples_by_class[class_id].extend(samples)
        class_meta[class_id] = (workload, variant)
        profile = summary.get("profile") or summarize_profile_samples(samples)
        run_rows.append(
            {
                "case_id": case.name,
                "class_id": class_id,
                "workload": workload,
                "variant": variant,
                "source_node_id": summary.get("source_node_id"),
                "sample_count": profile.get("sample_count"),
                "process_count_median": safe_metric(profile, "process_count", "median"),
                "cpu_median_percent": safe_metric(profile, "cpu_utilization_percent", "median"),
                "cpu_p95_percent": safe_metric(profile, "cpu_utilization_percent", "p95"),
                "memory_median_mb": safe_metric(profile, "memory_rss_mb", "median"),
                "memory_p95_mb": safe_metric(profile, "memory_rss_mb", "p95"),
                "checkpoint_p95_bytes": safe_metric(profile, "checkpoint_bytes", "p95"),
                "progress_rate_median_units_per_second": safe_metric(profile, "progress_rate_units_per_second", "median"),
                "power_median_kw": safe_metric(profile, "measured_power_kw", "median"),
                "power_p95_kw": safe_metric(profile, "measured_power_kw", "p95"),
            }
        )

    class_rows: list[dict[str, Any]] = []
    for class_id in sorted(samples_by_class):
        profile = summarize_profile_samples(samples_by_class[class_id])
        workload, variant = class_meta[class_id]
        class_rows.append(
            {
                "class_id": class_id,
                "workload": workload,
                "variant": variant,
                "trial_count": sum(1 for row in run_rows if row["class_id"] == class_id),
                "sample_count": profile.get("sample_count"),
                "process_count_median": safe_metric(profile, "process_count", "median"),
                "cpu_median_percent": safe_metric(profile, "cpu_utilization_percent", "median"),
                "cpu_p95_percent": safe_metric(profile, "cpu_utilization_percent", "p95"),
                "memory_median_mb": safe_metric(profile, "memory_rss_mb", "median"),
                "memory_p95_mb": safe_metric(profile, "memory_rss_mb", "p95"),
                "checkpoint_p95_bytes": safe_metric(profile, "checkpoint_bytes", "p95"),
                "progress_rate_median_units_per_second": safe_metric(profile, "progress_rate_units_per_second", "median"),
                "power_median_kw": safe_metric(profile, "measured_power_kw", "median"),
                "power_p95_kw": safe_metric(profile, "measured_power_kw", "p95"),
            }
        )

    expected_classes = (9 if "benchmarks" in phases else 0) + (3 if "dendro" in phases else 0) + (1 if "llm" in phases else 0)
    expected_runs = expected_classes * args.trials
    passed = [item for item in case_summaries if profile_summary_passes(item)]
    summary = {
        "calibration_id": calibration_id,
        "node_id": args.node,
        "partner_node_id": args.partner_node,
        "phases": sorted(phases),
        "trials_per_class": args.trials,
        "minimum_samples_per_run": MINIMUM_SAMPLES_PER_RUN,
        "expected_class_count": expected_classes,
        "observed_class_count": len(class_rows),
        "expected_run_count": expected_runs,
        "observed_run_count": len(case_summaries),
        "passed_run_count": len(passed),
        "passed": (
            len(class_rows) == expected_classes
            and len(case_summaries) == expected_runs
            and len(passed) == expected_runs
        ),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4a3_workload_resource_profiles",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": calibration_id,
        "node_id": args.node,
        "partner_node_id": args.partner_node,
        "methodology": {
            "scheduler": "Every task is labeled scheduler_mode=operator_only and no operator migration is requested.",
            "hardware_scope": "One canonical final-hardware e2-highmem-2 node; Stage 4A.4 measures node-to-node static execution behavior.",
            "replication": (
                f"{args.trials} independent task runs per workload class; "
                f"each passing run requires at least {MINIMUM_SAMPLES_PER_RUN} telemetry samples."
            ),
            "benchmark_matrix": "nbody/json/matmul x small/medium/large",
            "dendro_matrix": "(resolution,time_end) = (8,3.0), (9,1.0), (10,2.0)",
            "llm_matrix": f"one real CPU causal-LM profile using {args.llm_model}",
            "telemetry": "Aggregate Linux workload-session CPU/RSS plus checkpoint footprint, progress rate, and utilization-based power model.",
        },
    }

    if run_rows:
        write_csv(root / "profile_runs.csv", run_rows, list(run_rows[0].keys()))
    if class_rows:
        write_csv(root / "profile_classes.csv", class_rows, list(class_rows[0].keys()))
    write_json(root / "case_summaries.json", case_summaries)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    if not summary["passed"]:
        print("STAGE_4A3_PROFILES_INCOMPLETE")
        print(f"bundle: {root}")
        print(f"passed_runs: {len(passed)}/{expected_runs}")
        return 2
    print("\nSTAGE_4A3_PROFILES_PASS")
    print(f"bundle: {root}")
    print(f"classes: {len(class_rows)}/{expected_classes}")
    print(f"runs: {len(passed)}/{expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
