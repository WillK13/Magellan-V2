#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 3B.1 real LLM migration measurement bundle."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors: list[str] = []

    for name in ("metadata.json", "summary.json", "llm_migrations.csv", "checksums.sha256"):
        if not (root / name).is_file():
            errors.append(f"Missing {name}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with (root / "llm_migrations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if metadata.get("measurement_type") != "real_llm_migration_validation":
        errors.append("Unexpected measurement_type")
    if int(summary.get("migration_count", -1)) != len(rows):
        errors.append("summary migration_count does not match CSV")
    if not rows:
        errors.append("No LLM migration rows")

    for index, row in enumerate(rows, start=1):
        for field in (
            "source_shutdown_checkpoint_id",
            "destination_resumed_checkpoint_id",
            "actual_checkpoint_bytes",
            "actual_checkpoint_seconds",
            "actual_transfer_seconds",
            "actual_downtime_seconds",
        ):
            if row.get(field) in (None, ""):
                errors.append(f"row {index} missing {field}")
        if row.get("checkpoint_id_matches") != "True":
            errors.append(f"row {index} checkpoint ID did not match")
        if row.get("destination_optimizer_state_loaded") != "True":
            errors.append(f"row {index} optimizer state was not loaded")
        if row.get("resumed_at_same_step") != "True":
            errors.append(f"row {index} did not resume at the checkpoint step")
        if row.get("progress_continued") != "True":
            errors.append(f"row {index} did not advance after resume")
        if row.get("resume_validation_passed") != "True":
            errors.append(f"row {index} resume validation failed")

    if int(summary.get("resume_validations_passed", -1)) != len(rows):
        errors.append("Not all resume validations passed")
    if not bool(summary.get("passed")):
        errors.append("summary passed=false")

    errors.extend(validate_checksums(root))

    if errors:
        print("LLM MIGRATION MEASUREMENT BUNDLE FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("LLM MIGRATION MEASUREMENT BUNDLE PASSED")
    print(f"measurement_id: {summary['measurement_id']}")
    print(f"model: {summary['model']}")
    print(f"migrations: {len(rows)}")
    print(f"resume_validations: {summary['resume_validations_passed']}")
    print(f"checkpoint_bytes_median: {summary['checkpoint_bytes_median']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
