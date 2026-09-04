#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5e1 import STAGE5E1_CASES, stage5e1_passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 5E.1 real workload smoke bundle")
    parser.add_argument("bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = ("summary.json", "metadata.json", "cases.csv", "checksums.sha256")
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with (root / "cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "child_passed": row.get("child_passed", "").lower() == "true",
                "resume_validation_passed": row.get("resume_validation_passed", "").lower() == "true",
                "migration_count": int(row.get("migration_count") or 0),
            }
        )

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if not stage5e1_passes(normalized):
        errors.append("case coverage or migration/resume invariants failed")

    stage5a = Path(str(summary.get("source_stage5a_bundle") or ""))
    stage4a3 = Path(str(summary.get("source_stage4a3_bundle") or ""))
    for path, label in ((stage5a, "Stage 5A"), (stage4a3, "Stage 4A.3")):
        if not path.is_dir():
            errors.append(f"{label} source bundle missing: {path}")
        elif validate_checksums(path):
            errors.append(f"{label} source bundle checksum validation failed")

    for case in STAGE5E1_CASES:
        child = root / "cases" / case.case_id
        if not child.is_dir():
            errors.append(f"Missing child bundle for {case.case_id}")
        else:
            child_errors = validate_checksums(child)
            if child_errors:
                errors.append(f"{case.case_id} child checksum validation failed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5E1_REAL_WORKLOAD_SMOKES_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print(f"cases: {len(normalized)}/{len(STAGE5E1_CASES)}")
    for row in normalized:
        print(
            f"{row['case_id']}: {row['source_node_id']}->{row['destination_node_id']} "
            f"migration={row['migration_count']} resume={row['resume_validation_passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
