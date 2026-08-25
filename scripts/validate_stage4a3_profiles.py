#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4A.3 workload-profile bundle.")
    parser.add_argument("bundle")
    parser.add_argument("--minimum-samples-per-run", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    for name in ("summary.json", "metadata.json", "case_summaries.json", "profile_runs.csv", "profile_classes.csv"):
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = read_json(root / "summary.json")
    cases = read_json(root / "case_summaries.json")
    runs = read_csv(root / "profile_runs.csv")
    classes = read_csv(root / "profile_classes.csv")
    expected_runs = int(summary.get("expected_run_count", -1))
    expected_classes = int(summary.get("expected_class_count", -1))
    if summary.get("passed") is not True:
        errors.append("Parent summary is not passed")
    if len(cases) != expected_runs or len(runs) != expected_runs:
        errors.append(f"Run count mismatch: cases={len(cases)} rows={len(runs)} expected={expected_runs}")
    if len(classes) != expected_classes:
        errors.append(f"Class count mismatch: {len(classes)} expected={expected_classes}")
    for case in cases:
        if case.get("passed") is not True or case.get("profile_only") is not True:
            errors.append(f"Case not passed/profile-only: {case.get('case_id')}")
        sample_count = int((case.get("profile") or {}).get("sample_count") or 0)
        if sample_count < args.minimum_samples_per_run:
            errors.append(
                f"Insufficient samples for {case.get('case_id')}: {sample_count} < {args.minimum_samples_per_run}"
            )
    for row in classes:
        if int(float(row.get("trial_count") or 0)) != int(summary.get("trials_per_class", -1)):
            errors.append(f"Trial count mismatch for {row.get('class_id')}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4A3_PROFILE_BUNDLE_PASS")
    print(f"calibration_id: {summary.get('calibration_id')}")
    print(f"node: {summary.get('node_id')}")
    print(f"classes: {len(classes)}/{expected_classes}")
    print(f"runs: {len(runs)}/{expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
