#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a controlled migration measurement bundle"
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    metadata_path = root / "metadata.json"
    samples_path = root / "migration_samples.csv"
    if not metadata_path.is_file():
        errors.append("Missing metadata.json")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not samples_path.is_file():
        errors.append("Missing migration_samples.csv")
        rows: list[dict[str, str]] = []
    else:
        with samples_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    parameters = metadata.get("parameters", {})
    expected = (
        len(parameters.get("edges", []))
        * len(parameters.get("checkpoint_bytes", []))
        * int(parameters.get("samples_per_case", 0))
    )
    if len(rows) != expected:
        errors.append(f"Expected {expected} migration samples, found {len(rows)}")

    run_ids = {row.get("run_id") for row in rows}
    if len(run_ids) != len(rows):
        errors.append("Migration sample run IDs are not unique")

    for row in rows:
        prefix = (
            f"{row.get('source_node_id')}->{row.get('destination_node_id')} "
            f"run={row.get('run_id')}"
        )
        if row.get("source_node_id") == row.get("destination_node_id"):
            errors.append(f"Self migration found: {prefix}")
        if row.get("final_status") != "completed":
            errors.append(f"Migration workload did not complete: {prefix}")
        try:
            if int(row.get("final_generation", "0")) < 1:
                errors.append(f"Generation did not advance: {prefix}")
        except ValueError:
            errors.append(f"Invalid final generation: {prefix}")
        for field in (
            "actual_checkpoint_seconds",
            "actual_transfer_seconds",
            "actual_restore_seconds",
            "actual_downtime_seconds",
            "predicted_checkpoint_seconds",
            "predicted_transfer_seconds",
            "predicted_restore_seconds",
            "predicted_downtime_seconds",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"Invalid {field}: {prefix}")
                continue
            if value < 0:
                errors.append(f"Negative {field}: {prefix}")

        raw_path = root / "raw" / f"{row.get('run_id')}.json"
        if not raw_path.is_file():
            errors.append(f"Missing raw sample evidence: {raw_path.name}")

    if errors:
        print("MIGRATION MEASUREMENT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("MIGRATION MEASUREMENT BUNDLE PASSED")
    print(f"measurement_id: {metadata.get('measurement_id')}")
    print(f"samples: {len(rows)}")
    print(f"edges: {len(parameters.get('edges', []))}")
    print(f"checkpoint_sizes: {len(parameters.get('checkpoint_bytes', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
