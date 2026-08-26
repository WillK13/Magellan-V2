#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4a4 import REPRESENTATIVE_EQUIVALENCE_CLASS, successful_static_bundle


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4A.4 static-baseline bundle.")
    parser.add_argument("bundle")
    parser.add_argument("--minimum-samples-per-run", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = ("summary.json", "metadata.json", "case_summaries.json", "static_runs.csv", "static_classes.csv", "node_equivalence.csv")
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = read_json(root / "summary.json")
    cases = read_json(root / "case_summaries.json")
    runs = read_csv(root / "static_runs.csv")
    classes = read_csv(root / "static_classes.csv")
    equivalence = read_csv(root / "node_equivalence.csv")
    if summary.get("passed") is not True:
        errors.append("Parent summary is not passed")
    expected_runs = int(summary.get("expected_physical_run_count", -1))
    expected_classes = int(summary.get("expected_class_count", -1))
    expected_nodes = int(summary.get("expected_node_count", -1))
    trials = int(summary.get("trials_per_class", -1))
    canonical_node = str(summary.get("canonical_node_id"))
    if len(cases) != expected_runs or len(runs) != expected_runs:
        errors.append(f"Physical run count mismatch: cases={len(cases)} rows={len(runs)} expected={expected_runs}")
    if len(classes) != expected_classes:
        errors.append(f"Class count mismatch: {len(classes)} expected={expected_classes}")
    if len(equivalence) != expected_nodes:
        errors.append(f"Node count mismatch: {len(equivalence)} expected={expected_nodes}")

    for case in cases:
        bundle = root / "measurements" / str(case.get("measurement_id"))
        if not successful_static_bundle(bundle, minimum_samples=args.minimum_samples_per_run):
            errors.append(f"Case not a valid static completion: {case.get('measurement_id')}")
        if case.get("status") != "completed":
            errors.append(f"Case did not complete naturally: {case.get('measurement_id')}")
        if int(case.get("generation") or 0) != 0:
            errors.append(f"Generation changed in static case: {case.get('measurement_id')}")

    canonical_counts = Counter(
        row["class_id"] for row in runs if row.get("scope") == "canonical" and row.get("node_id") == canonical_node
    )
    for row in classes:
        class_id = row["class_id"]
        if canonical_counts[class_id] != trials:
            errors.append(f"Canonical trial count mismatch for {class_id}: {canonical_counts[class_id]} != {trials}")
        if int(float(row.get("trial_count") or 0)) != trials:
            errors.append(f"Class summary trial count mismatch for {class_id}")

    if summary.get("representative_equivalence_class") != REPRESENTATIVE_EQUIVALENCE_CLASS:
        errors.append("Unexpected representative equivalence class")
    eq_nodes = {row["node_id"] for row in equivalence}
    if canonical_node not in eq_nodes:
        errors.append("Canonical node missing from node equivalence table")
    for row in equivalence:
        if int(float(row.get("trial_count") or 0)) != trials:
            errors.append(f"Equivalence trial count mismatch for {row.get('node_id')}")
        if float(row.get("runtime_seconds_median") or 0) <= 0:
            errors.append(f"Non-positive equivalence runtime for {row.get('node_id')}")
        if float(row.get("slowdown_vs_canonical") or 0) <= 0:
            errors.append(f"Non-positive slowdown for {row.get('node_id')}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4A4_STATIC_BASELINE_BUNDLE_PASS")
    print(f"calibration_id: {summary.get('calibration_id')}")
    print(f"canonical_node: {canonical_node}")
    print(f"classes: {len(classes)}/{expected_classes}")
    print(f"nodes: {len(equivalence)}/{expected_nodes}")
    print(f"physical_runs: {len(runs)}/{expected_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
