#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4e1 import SCALE_SIZES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4E.3 profile bundle.")
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
        "profile_summary.csv",
        "category_profile.csv",
        "function_profile.csv",
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
    rows = read_csv(root / "profile_summary.csv")
    categories = read_csv(root / "category_profile.csv")
    functions = read_csv(root / "function_profile.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("scale_sizes") or []) != list(SCALE_SIZES):
        errors.append("summary scale_sizes mismatch")

    e2 = Path(str(summary.get("source_stage4e2_bundle") or ""))
    if not e2.is_dir():
        errors.append(f"source Stage 4E.2 bundle not found: {e2}")
    elif validate_checksums(e2):
        errors.append("source Stage 4E.2 checksum validation failed")

    if len(rows) != len(SCALE_SIZES):
        errors.append(f"profile summary coverage {len(rows)} != {len(SCALE_SIZES)}")
    observed = [int(row["task_count"]) for row in rows]
    if observed != list(SCALE_SIZES):
        errors.append(f"profile task_count coverage mismatch: {observed}")

    for row in rows:
        n = int(row["task_count"])
        if int(row["evaluate_task_calls"]) != n:
            errors.append(f"n={n} evaluate_task calls do not equal task_count")
        if int(row["adaptive_store_put_calls"]) <= 0:
            errors.append(f"n={n} has no adaptive store put calls")
        if int(row["adaptive_store_persist_calls"]) <= 0:
            errors.append(f"n={n} has no adaptive store persist calls")
        if float(row["profiled_epoch_wall_ms"]) <= 0:
            errors.append(f"n={n} has non-positive profiled wall time")
        share = float(row["adaptive_store_persist_fraction_of_profiled_wall"])
        if share < 0:
            errors.append(f"n={n} has negative persist fraction")

    for n in SCALE_SIZES:
        if not any(int(row["task_count"]) == n for row in categories):
            errors.append(f"n={n} missing category profile")
        if not any(int(row["task_count"]) == n for row in functions):
            errors.append(f"n={n} missing function profile")

    category_self_by_n = {}
    for n in SCALE_SIZES:
        category_self_by_n[n] = sum(
            float(row["self_ms"])
            for row in categories
            if int(row["task_count"]) == n
        )
        if category_self_by_n[n] <= 0:
            errors.append(f"n={n} category self time is non-positive")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    n100 = next(row for row in rows if int(row["task_count"]) == 100)
    print("STAGE_4E3_DECISION_PROFILE_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"sizes: {','.join(str(value) for value in SCALE_SIZES)}")
    print(f"profile_rows: {len(rows)}/{len(SCALE_SIZES)}")
    print(f"category_rows: {len(categories)}")
    print(f"function_rows: {len(functions)}")
    print(f"n100_store_put_calls: {n100['adaptive_store_put_calls']}")
    print(f"n100_store_persist_calls: {n100['adaptive_store_persist_calls']}")
    print(
        "n100_store_persist_share: "
        f"{100*float(n100['adaptive_store_persist_fraction_of_profiled_wall']):.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
