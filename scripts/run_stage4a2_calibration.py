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

from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.stage4a2 import (
    select_representative_edges,
    summarize_migration_accuracy,
)


BENCHMARKS = ("nbody", "json", "matmul")
SIZES = ("small", "medium", "large")
DENDRO_VARIANTS = (
    ("short", 8, 0.5),
    ("medium", 9, 1.0),
    ("long", 10, 2.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4A.2 final-hardware workload and migration calibration. "
            "Representative short/medium/long directed edges are selected from "
            "the immutable Stage 4A.1 WAN bundle."
        )
    )
    parser.add_argument("--stage4a1-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--calibration-id", default=None)
    parser.add_argument(
        "--phases",
        default="benchmarks,dendro,llm",
        help="Comma-separated subset of benchmarks,dendro,llm",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-seconds", type=float, default=20.0)
    parser.add_argument("--dendro-profile-seconds", type=float, default=30.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--llm-model",
        default="experiment-assets/models/distilgpt2",
        help="Repository-relative pre-staged local model snapshot.",
    )
    parser.add_argument("--llm-sleep-per-step", type=float, default=2.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def successful_bundle(path: Path) -> bool:
    try:
        return bool(read_json(path / "summary.json").get("passed"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def run_case(command: list[str], bundle: Path, resume: bool) -> None:
    if bundle.exists():
        if resume and successful_bundle(bundle):
            print(f"[resume] already passed: {bundle.name}")
            return
        raise FileExistsError(
            f"Case bundle already exists but is not resumable: {bundle}. "
            "Use a new --calibration-id or inspect/remove only the failed case."
        )
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    phases = {item.strip() for item in args.phases.split(",") if item.strip()}
    unknown = phases - {"benchmarks", "dendro", "llm"}
    if unknown or not phases:
        raise ValueError(f"Invalid --phases: {sorted(unknown)}")

    stage4a1 = Path(args.stage4a1_bundle)
    edges_path = stage4a1 / "network" / "directed-mesh" / "edges.csv"
    if not edges_path.is_file():
        raise FileNotFoundError(f"Stage 4A.1 edge table not found: {edges_path}")
    subprocess.run(
        ["python", "scripts/validate_stage4a1_calibration.py", str(stage4a1)],
        check=True,
    )
    edges = select_representative_edges(edges_path)
    edge_by_role = {edge.role: edge for edge in edges}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibration_id = args.calibration_id or f"stage4a2-{timestamp}-{uuid4().hex[:8]}"
    root = Path(args.measurements_root) / calibration_id
    case_root = root / "measurements"
    if root.exists() and not args.resume:
        raise FileExistsError(f"Stage 4A.2 bundle already exists: {root}")
    case_root.mkdir(parents=True, exist_ok=True)

    print("== Stage 4A.2 workload + migration calibration ==")
    print(f"calibration_id={calibration_id}")
    for edge in edges:
        print(
            f"[{edge.role}] {edge.edge} "
            f"bandwidth={edge.bandwidth_mbps:.2f}Mbps rtt={edge.rtt_ms:.2f}ms"
        )
    write_json(root / "representative_edges.json", [edge.as_dict() for edge in edges])

    if "benchmarks" in phases:
        for benchmark in BENCHMARKS:
            for size in SIZES:
                role = {"small": "short", "medium": "medium", "large": "long"}[size]
                edge = edge_by_role[role]
                case_id = f"benchmark-{benchmark}-{size}-{role}"
                run_case(
                    [
                        "python", "scripts/measure_stage4a2_workload.py",
                        "--cluster", args.cluster,
                        "--local-node-id", args.local_node_id,
                        "--ssh-user", args.ssh_user,
                        "--workload", "benchmark",
                        "--benchmark", benchmark,
                        "--size", size,
                        "--source", edge.source_node_id,
                        "--destination", edge.destination_node_id,
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
        for role, resolution, time_end in DENDRO_VARIANTS:
            edge = edge_by_role[role]
            time_slug = str(time_end).replace(".", "p")
            case_id = f"dendro-r{resolution}-t{time_slug}-{role}"
            run_case(
                [
                    "python", "scripts/measure_stage4a2_workload.py",
                    "--cluster", args.cluster,
                    "--local-node-id", args.local_node_id,
                    "--ssh-user", args.ssh_user,
                    "--workload", "dendro",
                    "--resolution", str(resolution),
                    "--time-end", str(time_end),
                    "--source", edge.source_node_id,
                    "--destination", edge.destination_node_id,
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
        for role in ("short", "medium", "long"):
            edge = edge_by_role[role]
            case_id = f"llm-{role}"
            run_case(
                [
                    "python", "scripts/measure_llm_migration.py",
                    "--cluster", args.cluster,
                    "--local-node-id", args.local_node_id,
                    "--ssh-user", args.ssh_user,
                    "--path", f"{edge.source_node_id},{edge.destination_node_id}",
                    "--model", args.llm_model,
                    "--migrate-after-step", "2",
                    "--post-path-steps", "1",
                    "--checkpoint-every", "1",
                    "--sleep-per-step", str(args.llm_sleep_per_step),
                    "--torch-threads", "2",
                    "--timeout-seconds", str(args.timeout_seconds),
                    "--measurements-root", str(case_root),
                    "--measurement-id", case_id,
                    "--profile-seconds", str(args.profile_seconds),
                    "--profile-sample-interval-seconds", str(args.sample_interval_seconds),
                ],
                case_root / case_id,
                args.resume,
            )

    profile_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in sorted(item for item in case_root.iterdir() if item.is_dir()):
        summary_path = case / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        case_summaries.append({"case_id": case.name, **summary})
        if (case / "migration.csv").is_file():
            migration_rows.extend(read_csv(case / "migration.csv"))
            profile = summary.get("profile") or {}
            profile_rows.append(
                {
                    "case_id": case.name,
                    "workload": summary.get("workload"),
                    "variant": summary.get("variant"),
                    "source_node_id": summary.get("source_node_id"),
                    "destination_node_id": summary.get("destination_node_id"),
                    "sample_count": profile.get("sample_count"),
                    "cpu_median_percent": (profile.get("cpu_utilization_percent") or {}).get("median"),
                    "memory_median_mb": (profile.get("memory_rss_mb") or {}).get("median"),
                    "memory_p95_mb": (profile.get("memory_rss_mb") or {}).get("p95"),
                    "checkpoint_median_bytes": (profile.get("checkpoint_bytes") or {}).get("median"),
                    "progress_rate_median_units_per_second": (profile.get("progress_rate_units_per_second") or {}).get("median"),
                    "power_median_kw": (profile.get("measured_power_kw") or {}).get("median"),
                }
            )
        elif (case / "llm_migrations.csv").is_file():
            rows = read_csv(case / "llm_migrations.csv")
            for row in rows:
                migration_rows.append({"workload": "llm", "variant": args.llm_model, **row})
            llm_profile = summary.get("profile") or {}
            profile_rows.append(
                {
                    "case_id": case.name,
                    "workload": "llm",
                    "variant": args.llm_model,
                    "source_node_id": (summary.get("path") or [None])[0],
                    "destination_node_id": (summary.get("path") or [None, None])[-1],
                    "sample_count": llm_profile.get("sample_count"),
                    "cpu_median_percent": (llm_profile.get("cpu_utilization_percent") or {}).get("median"),
                    "memory_median_mb": (llm_profile.get("memory_rss_mb") or {}).get("median"),
                    "memory_p95_mb": (llm_profile.get("memory_rss_mb") or {}).get("p95"),
                    "checkpoint_median_bytes": (llm_profile.get("checkpoint_bytes") or {}).get("median"),
                    "progress_rate_median_units_per_second": (llm_profile.get("progress_rate_units_per_second") or {}).get("median"),
                    "power_median_kw": (llm_profile.get("measured_power_kw") or {}).get("median"),
                }
            )

    if profile_rows:
        write_csv(root / "workload_profiles.csv", profile_rows, list(profile_rows[0].keys()))
    if migration_rows:
        all_fields: list[str] = []
        for row in migration_rows:
            for key in row:
                if key not in all_fields:
                    all_fields.append(key)
        write_csv(root / "migration_samples.csv", migration_rows, all_fields)

    expected = 0
    if "benchmarks" in phases:
        expected += 9
    if "dendro" in phases:
        expected += 3
    if "llm" in phases:
        expected += 3
    passed_cases = [case for case in case_summaries if case.get("passed") is True]
    summary = {
        "calibration_id": calibration_id,
        "stage4a1_bundle": str(stage4a1),
        "phases": sorted(phases),
        "representative_edges": [edge.as_dict() for edge in edges],
        "expected_case_count_for_selected_phases": expected,
        "observed_case_count": len(case_summaries),
        "passed_case_count": len(passed_cases),
        "profile_case_count": len(profile_rows),
        "migration_sample_count": len(migration_rows),
        "migration_accuracy": summarize_migration_accuracy(migration_rows),
        "passed": len(case_summaries) == expected and len(passed_cases) == expected,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4a2_workload_migration_calibration",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_id": calibration_id,
        "stage4a1_bundle": str(stage4a1),
        "methodology": {
            "edge_selection": (
                "short=highest measured Stage-4A.1 bandwidth; long=lowest measured bandwidth; "
                "medium=directed edge nearest the Stage-4A.1 median bandwidth"
            ),
            "benchmark_matrix": "nbody/json/matmul x small/medium/large, with size mapped to short/medium/long WAN regime",
            "dendro_matrix": "resolution/time pairs (8,0.5), (9,1.0), (10,2.0) mapped to short/medium/long WAN regimes",
            "llm_matrix": "same pre-staged distilgpt2 snapshot migrated once on each WAN regime",
            "accuracy_policy": "All observed prediction errors are retained; no accuracy threshold drops a sample.",
        },
    }
    write_json(root / "case_summaries.json", case_summaries)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    if not summary["passed"]:
        print("STAGE_4A2_CALIBRATION_INCOMPLETE")
        print(f"bundle: {root}")
        print(f"passed_cases: {len(passed_cases)}/{expected}")
        return 2
    print("\nSTAGE_4A2_CALIBRATION_PASS")
    print(f"bundle: {root}")
    print(f"cases: {len(passed_cases)}/{expected}")
    print(f"migration_samples: {len(migration_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
