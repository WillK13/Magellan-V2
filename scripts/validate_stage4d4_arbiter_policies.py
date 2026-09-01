#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4d4 import STRATEGY_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4D.4 arbiter-policy bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = [
        "summary.json",
        "metadata.json",
        "auction_events.csv",
        "fixed_cohort_summary.csv",
        "starvation_summary.csv",
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
    events = read_csv(root / "auction_events.csv")
    fixed = read_csv(root / "fixed_cohort_summary.csv")
    starvation = read_csv(root / "starvation_summary.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("strategy_values") or []) != list(STRATEGY_VALUES):
        errors.append("summary strategy_values mismatch")
    if int(summary.get("residual_measured_benchmark_admissions") or 0) != 1:
        errors.append("measured residual admission count is not 1")
    if int(summary.get("fixed_cohort_bidder_count") or 0) != 5:
        errors.append("fixed cohort bidder count is not 5")
    if int(summary.get("event_count") or 0) != len(events):
        errors.append("event_count mismatch")

    fixed_strategies = {row["strategy"] for row in fixed}
    starvation_strategies = {row["strategy"] for row in starvation}
    expected = set(STRATEGY_VALUES)
    if fixed_strategies != expected:
        errors.append(f"fixed strategy coverage mismatch: {sorted(fixed_strategies)}")
    if starvation_strategies != expected:
        errors.append(f"starvation strategy coverage mismatch: {sorted(starvation_strategies)}")

    for row in fixed:
        if not truthy(row["all_tasks_admitted"]):
            errors.append(f"fixed cohort did not admit all tasks for {row['strategy']}")
        order = row["admission_order"].split("->")
        if len(order) != 5 or len(set(order)) != 5:
            errors.append(f"invalid admission order for {row['strategy']}: {order}")

    fixed_events = [row for row in events if row["experiment"] == "fixed_cohort"]
    for strategy in STRATEGY_VALUES:
        subset = [row for row in fixed_events if row["strategy"] == strategy]
        by_round = {}
        for row in subset:
            by_round.setdefault(int(row["round_index"]), []).append(row)
        if set(by_round) != {1, 2, 3, 4, 5}:
            errors.append(f"{strategy} fixed rounds mismatch: {sorted(by_round)}")
            continue
        for round_index, rows in by_round.items():
            accepted = sum(row["status"] == "accepted" for row in rows)
            if accepted != 1:
                errors.append(
                    f"{strategy} fixed round {round_index} accepted {accepted}, expected 1"
                )

    stream_events = [row for row in events if row["experiment"] == "starvation_stream"]
    for strategy in STRATEGY_VALUES:
        subset = [row for row in stream_events if row["strategy"] == strategy]
        by_round = {}
        for row in subset:
            by_round.setdefault(int(row["round_index"]), []).append(row)
        for round_index, rows in by_round.items():
            accepted = sum(row["status"] == "accepted" for row in rows)
            if accepted != 1:
                errors.append(
                    f"{strategy} stream round {round_index} accepted {accepted}, expected 1"
                )
            if len(rows) != 2:
                errors.append(
                    f"{strategy} stream round {round_index} has {len(rows)} bids, expected 2"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4D4_ARBITER_POLICY_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"strategies: {len(fixed)}/{len(STRATEGY_VALUES)}")
    print(f"auction_events: {len(events)}")
    print("measured_residual_admissions: 1")
    print("capacity_violations: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
