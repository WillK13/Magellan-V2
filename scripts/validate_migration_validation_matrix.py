#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage-3A migration matrix bundle"
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = [
        "metadata.json",
        "migration_samples.csv",
        "matrix_summary.json",
        "matrix_cases.csv",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")

    metadata = {}
    summary = {}
    rows: list[dict[str, str]] = []
    cases: list[dict[str, str]] = []
    if (root / "metadata.json").is_file():
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (root / "matrix_summary.json").is_file():
        summary = json.loads((root / "matrix_summary.json").read_text(encoding="utf-8"))
    if (root / "migration_samples.csv").is_file():
        with (root / "migration_samples.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
    if (root / "matrix_cases.csv").is_file():
        with (root / "matrix_cases.csv").open(encoding="utf-8", newline="") as handle:
            cases = list(csv.DictReader(handle))

    if metadata.get("measurement_type") != "migration_validation_matrix":
        errors.append("metadata measurement_type is not migration_validation_matrix")

    parameters = metadata.get("parameters", {})
    edge_count = len(parameters.get("edges", []))
    size_count = len(parameters.get("checkpoint_bytes", []))
    samples_per_case = int(parameters.get("samples_per_case", 0) or 0)
    expected_rows = edge_count * size_count * samples_per_case
    if len(rows) != expected_rows:
        errors.append(f"Expected {expected_rows} migration rows, found {len(rows)}")

    if int(summary.get("total_sample_count", -1)) != len(rows):
        errors.append("matrix_summary total_sample_count does not match CSV")

    calibrated_count = int(summary.get("calibrated_sample_count", 0) or 0)
    cold_count = int(summary.get("cold_or_uncalibrated_sample_count", 0) or 0)
    if calibrated_count + cold_count != len(rows):
        errors.append("calibrated + cold sample counts do not cover all rows")
    if calibrated_count == 0:
        errors.append("No calibrated samples were captured")

    expected_cases = edge_count * size_count
    if len(cases) > expected_cases:
        errors.append(f"Found {len(cases)} case rows for only {expected_cases} cases")

    raw_dir = root / "raw"
    for row in rows:
        run_id = row.get("run_id")
        if not run_id or not (raw_dir / f"{run_id}.json").is_file():
            errors.append(f"Missing raw evidence for run {run_id}")
        if row.get("final_status") != "completed":
            errors.append(f"Run did not complete: {run_id}")

    if errors:
        print("MIGRATION VALIDATION MATRIX FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    overall = summary.get("overall_calibrated", {})
    transfer = overall.get("transfer_absolute_error_percent") or {}
    downtime = overall.get("downtime_absolute_error_percent") or {}

    print("MIGRATION VALIDATION MATRIX BUNDLE PASSED")
    print(f"measurement_id: {metadata.get('measurement_id')}")
    print(f"edges: {edge_count}")
    print(f"checkpoint_sizes: {size_count}")
    print(f"samples: {len(rows)}")
    print(f"calibrated_samples: {calibrated_count}")
    if transfer:
        print(f"transfer_ape_median_pct: {float(transfer['median']):.2f}")
        print(f"transfer_ape_p95_pct: {float(transfer['p95']):.2f}")
    if downtime:
        print(f"downtime_ape_median_pct: {float(downtime['median']):.2f}")
        print(f"downtime_ape_p95_pct: {float(downtime['p95']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
