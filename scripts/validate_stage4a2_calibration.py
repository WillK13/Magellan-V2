#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4A.2 calibration bundle")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    for required in (
        "metadata.json",
        "summary.json",
        "representative_edges.json",
        "case_summaries.json",
        "workload_profiles.csv",
        "migration_samples.csv",
    ):
        if not (root / required).is_file():
            errors.append(f"Missing {required}")
    if errors:
        raise SystemExit("STAGE_4A2_CALIBRATION_BUNDLE_FAIL\n- " + "\n- ".join(errors))

    metadata = read_json(root / "metadata.json")
    summary = read_json(root / "summary.json")
    edges = read_json(root / "representative_edges.json")
    cases = read_json(root / "case_summaries.json")

    if metadata.get("measurement_type") != "stage4a2_workload_migration_calibration":
        errors.append("Unexpected measurement_type")
    if [edge.get("role") for edge in edges] != ["short", "medium", "long"]:
        errors.append("Representative edge roles must be short,medium,long")
    if len({edge.get("edge") for edge in edges}) != 3:
        errors.append("Representative edges must be distinct")
    if summary.get("passed") is not True:
        errors.append("Stage 4A.2 summary passed=false")
    expected = int(summary.get("expected_case_count_for_selected_phases") or 0)
    observed = int(summary.get("observed_case_count") or 0)
    passed = int(summary.get("passed_case_count") or 0)
    if expected <= 0 or observed != expected or passed != expected:
        errors.append(f"Case completeness mismatch expected={expected} observed={observed} passed={passed}")
    if len(cases) != expected:
        errors.append(f"case_summaries count={len(cases)} expected={expected}")
    if any(case.get("passed") is not True for case in cases):
        errors.append("One or more case summaries did not pass")
    profile_count = count_csv(root / "workload_profiles.csv")
    migration_count = count_csv(root / "migration_samples.csv")
    if profile_count != int(summary.get("profile_case_count") or -1):
        errors.append("workload_profiles.csv count mismatch")
    if migration_count != int(summary.get("migration_sample_count") or -1):
        errors.append("migration_samples.csv count mismatch")
    if migration_count < expected:
        errors.append(f"migration sample count={migration_count}; expected at least {expected}")

    for case in cases:
        case_id = case.get("case_id")
        child = root / "measurements" / str(case_id)
        if not child.is_dir():
            errors.append(f"Missing child bundle {case_id}")
            continue
        child_errors = validate_checksums(child)
        errors.extend(f"{case_id}: {error}" for error in child_errors)

    if errors:
        raise SystemExit("STAGE_4A2_CALIBRATION_BUNDLE_FAIL\n- " + "\n- ".join(errors))

    print("STAGE_4A2_CALIBRATION_BUNDLE_PASS")
    print(f"calibration_id: {summary['calibration_id']}")
    print(f"cases: {passed}/{expected}")
    print(f"migration_samples: {migration_count}")
    print(f"stage4a1_bundle: {summary['stage4a1_bundle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
