#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4A.5 calibration-validation bundle.")
    parser.add_argument("bundle")
    args = parser.parse_args()
    root = Path(args.bundle)

    errors = validate_checksums(root)
    required = [
        "summary.json",
        "metadata.json",
        "calibration_evidence.json",
        "runtime_validation.json",
        "runtime_validation_runs.csv",
        "runtime_validation_by_class.csv",
        "runtime_validation_by_node.csv",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        raise SystemExit("STAGE_4A5_CALIBRATION_VALIDATION_BUNDLE_FAIL\n- " + "\n- ".join(errors))

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / "runtime_validation.json").read_text(encoding="utf-8"))
    rows = read_csv(root / "runtime_validation_runs.csv")
    by_class = read_csv(root / "runtime_validation_by_class.csv")
    by_node = read_csv(root / "runtime_validation_by_node.csv")

    expected = int(summary.get("expected_runtime_validation_run_count") or 0)
    if summary.get("passed") is not True:
        errors.append("Parent summary passed=false")
    if expected <= 0 or len(rows) != expected:
        errors.append(f"Runtime validation row count {len(rows)} != expected {expected}")
    if int(summary.get("observed_runtime_validation_run_count") or 0) != len(rows):
        errors.append("Observed runtime validation count mismatch")
    if int(runtime.get("sample_count") or 0) != len(rows):
        errors.append("Runtime validation sample_count mismatch")

    expected_classes = set(summary.get("validation_classes") or [])
    expected_nodes = set(summary.get("validation_nodes") or [])
    if {row.get("class_id") for row in by_class} != expected_classes:
        errors.append("Runtime validation class set mismatch")
    if {row.get("node_id") for row in by_node} != expected_nodes:
        errors.append("Runtime validation node set mismatch")
    if runtime.get("runtime_model_transfer_passed") != summary.get("runtime_model_transfer_passed"):
        errors.append("Runtime-model gate mismatch")
    if runtime.get("recommended_runtime_model") != summary.get("recommended_runtime_model"):
        errors.append("Runtime-model recommendation mismatch")
    expected_ready = bool(
        summary.get("runtime_model_transfer_passed")
        and summary.get("llm_regional_runtime_transfer_validated")
    )
    if bool(summary.get("ready_for_stage4b_runtime_model")) != expected_ready:
        errors.append("Stage 4B runtime-model readiness mismatch")

    for row in rows:
        if float(row.get("actual_runtime_seconds") or 0) <= 0:
            errors.append(f"Non-positive actual runtime: {row.get('measurement_id')}")
        if float(row.get("predicted_runtime_seconds") or 0) <= 0:
            errors.append(f"Non-positive predicted runtime: {row.get('measurement_id')}")
        if int(float(row.get("telemetry_sample_count") or 0)) < 3:
            errors.append(f"Too few telemetry samples: {row.get('measurement_id')}")

    if errors:
        raise SystemExit("STAGE_4A5_CALIBRATION_VALIDATION_BUNDLE_FAIL\n- " + "\n- ".join(errors))

    print("STAGE_4A5_CALIBRATION_VALIDATION_BUNDLE_PASS")
    print(f"calibration_id: {summary.get('calibration_id')}")
    print(f"runtime_validation_runs: {len(rows)}/{expected}")
    print("runtime_model_transfer: " + ("PASS" if summary.get("runtime_model_transfer_passed") else "FAIL"))
    print(f"recommended_runtime_model: {summary.get('recommended_runtime_model')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
