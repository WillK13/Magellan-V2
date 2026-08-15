#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.migration_matrix import summarize_migration_rows


DEFAULT_EDGES = [
    "boston:virginia",
    "boston:france",
    "california:south-australia",
    "boston:nepal",
]
DEFAULT_SIZES = [
    10 * 1024 * 1024,
    100 * 1024 * 1024,
    500 * 1024 * 1024,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compact Stage-3A real-system migration validation matrix and "
            "summarize calibrated prediction accuracy by edge and checkpoint size."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--state-root", default="runtime-state-gcp-measurement")
    parser.add_argument(
        "--edge",
        action="append",
        default=None,
        help="Directed source:destination edge. Repeat to override the default matrix.",
    )
    parser.add_argument(
        "--checkpoint-bytes",
        action="append",
        type=int,
        default=None,
        help="Checkpoint payload size in bytes. Repeat to override the default matrix.",
    )
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=3600.0,
        help="Per-migration timeout. The default allows 500 MiB on slow WAN edges.",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Print the matrix and nominal checkpoint-transfer volume without "
            "running it."
        ),
    )
    return parser.parse_args()


def _gib(value: int) -> float:
    return value / (1024**3)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _print_plan(edges: list[str], sizes: list[int], samples_per_case: int) -> None:
    cases = len(edges) * len(sizes)
    runs = cases * samples_per_case
    transfer_bytes = sum(sizes) * len(edges) * samples_per_case
    print("== Stage 3A compact migration validation matrix ==")
    print(f"edges={len(edges)} checkpoint_sizes={len(sizes)} cases={cases} runs={runs}")
    for edge in edges:
        print(f"  edge: {edge}")
    print("  checkpoint_bytes:", ", ".join(str(size) for size in sizes))
    print(f"nominal_checkpoint_transfer_volume={_gib(transfer_bytes):.2f} GiB")
    print("NOTE: live edge calibration probes add a small amount of additional traffic")
    print("NOTE: this is model validation, not a scheduler-policy comparison")


def main() -> int:
    args = parse_args()
    edges = args.edge or list(DEFAULT_EDGES)
    sizes = args.checkpoint_bytes or list(DEFAULT_SIZES)
    if args.samples_per_case < 1:
        raise ValueError("--samples-per-case must be positive")
    if any(size <= 0 for size in sizes):
        raise ValueError("Checkpoint sizes must be positive")

    _print_plan(edges, sizes, args.samples_per_case)
    if args.plan_only:
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurement_id = args.measurement_id or (
        f"migration-matrix-{timestamp}-{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / measurement_id

    command = [
        sys.executable,
        "scripts/measure_migration_model.py",
        "--cluster",
        args.cluster,
        "--local-node-id",
        args.local_node_id,
        "--ssh-user",
        args.ssh_user,
        "--remote-repo",
        args.remote_repo,
        "--state-root",
        args.state_root,
        "--samples-per-case",
        str(args.samples_per_case),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--poll-seconds",
        str(args.poll_seconds),
        "--measurements-root",
        args.measurements_root,
        "--measurement-id",
        measurement_id,
        "--expected-carbon-metric",
        args.expected_carbon_metric,
        "--expected-state-token",
        args.expected_state_token,
    ]
    for edge in edges:
        command.extend(["--edge", edge])
    for size in sizes:
        command.extend(["--checkpoint-bytes", str(size)])

    print("\n== Collect real migration samples ==")
    subprocess.run(command, check=True)

    print("\n== Validate base migration bundle ==")
    subprocess.run(
        [sys.executable, "scripts/validate_migration_measurement.py", str(root)],
        check=True,
    )

    rows = _load_rows(root / "migration_samples.csv")
    summary = summarize_migration_rows(rows)
    summary["matrix"] = {
        "edges": edges,
        "checkpoint_bytes": sizes,
        "samples_per_case": args.samples_per_case,
        "nominal_checkpoint_transfer_bytes": sum(sizes)
        * len(edges)
        * args.samples_per_case,
        "calibrated_definition": (
            "Samples whose candidate used measured_migration_ema workload timing "
            "and a live measured transfer model. Cold/fallback samples remain in "
            "migration_samples.csv but are excluded from headline calibrated accuracy."
        ),
    }
    write_json(root / "matrix_summary.json", summary)

    case_rows = list(summary.get("cases", []))
    if case_rows:
        write_csv(root / "matrix_cases.csv", case_rows, list(case_rows[0].keys()))

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["measurement_type"] = "migration_validation_matrix"
    metadata["matrix_summary_file"] = "matrix_summary.json"
    metadata["matrix_cases_file"] = "matrix_cases.csv"
    metadata["matrix_design"] = {
        "purpose": "Stage 3A final real-system model validation; not policy comparison",
        "default_edges": DEFAULT_EDGES,
        "default_checkpoint_bytes": DEFAULT_SIZES,
        "headline_population": "calibrated samples only",
        "cold_samples_retained": True,
    }
    write_json(metadata_path, metadata)
    write_checksums(root)

    overall = summary.get("overall_calibrated", {})
    transfer = overall.get("transfer_absolute_error_percent") or {}
    downtime = overall.get("downtime_absolute_error_percent") or {}

    print("\nMIGRATION VALIDATION MATRIX PASSED")
    print(f"bundle: {root}")
    print(f"total_samples: {summary['total_sample_count']}")
    print(f"calibrated_samples: {summary['calibrated_sample_count']}")
    print(
        "cold_or_uncalibrated_samples: "
        f"{summary['cold_or_uncalibrated_sample_count']}"
    )
    if transfer:
        print(f"calibrated_transfer_ape_median_pct: {float(transfer['median']):.2f}")
        print(f"calibrated_transfer_ape_p95_pct: {float(transfer['p95']):.2f}")
    if downtime:
        print(f"calibrated_downtime_ape_median_pct: {float(downtime['median']):.2f}")
        print(f"calibrated_downtime_ape_p95_pct: {float(downtime['p95']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
