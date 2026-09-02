#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4e1 import SCALE_SIZES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4E.2 control-plane bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = [
        "summary.json",
        "metadata.json",
        "control_plane_summary.csv",
        "latency_samples.csv",
        "checksums.sha256",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(root / "control_plane_summary.csv")
    samples = read_csv(root / "latency_samples.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("scale_sizes") or []) != list(SCALE_SIZES):
        errors.append("summary scale_sizes mismatch")

    e1 = Path(str(summary.get("source_stage4e1_bundle") or ""))
    if not e1.is_dir():
        errors.append(f"source Stage 4E.1 bundle not found: {e1}")
    elif validate_checksums(e1):
        errors.append("source Stage 4E.1 checksum validation failed")

    if len(rows) != len(SCALE_SIZES):
        errors.append(f"summary coverage {len(rows)} != {len(SCALE_SIZES)}")
    observed_sizes = [int(row["task_count"]) for row in rows]
    if observed_sizes != list(SCALE_SIZES):
        errors.append(f"task_count coverage mismatch: {observed_sizes}")

    repetitions = int(summary.get("repetitions") or 0)
    expected_samples = len(SCALE_SIZES) * repetitions
    if len(samples) != expected_samples:
        errors.append(f"latency sample coverage {len(samples)} != {expected_samples}")

    for row in rows:
        n = int(row["task_count"])
        if int(row["bid_count"]) != n:
            errors.append(f"n={n} bid_count does not equal task_count")
        for field in (
            "cold_epoch_wall_ms",
            "decision_wall_ms_median",
            "decision_wall_ms_p95",
            "auction_wall_ms_median",
            "auction_wall_ms_p95",
            "epoch_wall_ms_median",
            "epoch_wall_ms_p95",
            "epoch_cpu_ms_median",
            "peak_incremental_tracemalloc_kb",
        ):
            if float(row[field]) <= 0:
                errors.append(f"n={n} non-positive {field}")
        if float(row["epoch_wall_ms_median"]) < float(row["decision_wall_ms_median"]):
            errors.append(f"n={n} epoch median is smaller than decision median")
        if float(row["epoch_wall_ms_median"]) < float(row["auction_wall_ms_median"]):
            errors.append(f"n={n} epoch median is smaller than auction median")

    samples_by_n = {}
    for row in samples:
        samples_by_n.setdefault(int(row["task_count"]), []).append(row)
    for n in SCALE_SIZES:
        subset = samples_by_n.get(n, [])
        if len(subset) != repetitions:
            errors.append(f"n={n} sample count {len(subset)} != {repetitions}")
        reps = sorted(int(row["repetition"]) for row in subset)
        if reps != list(range(1, repetitions + 1)):
            errors.append(f"n={n} repetition ids mismatch: {reps}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4E2_CONTROL_PLANE_SCALING_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"sizes: {','.join(str(value) for value in SCALE_SIZES)}")
    print(f"repetitions: {repetitions}")
    print(f"summary_rows: {len(rows)}/{len(SCALE_SIZES)}")
    print(f"latency_samples: {len(samples)}/{expected_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
