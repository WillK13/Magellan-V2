#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5e4 import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_LOAD_ID,
    EXPECTED_SEASON,
    EXPECTED_SOURCE_SCENARIO_ID,
    EXPECTED_TASK_COUNT,
    MAGELLAN_POLICY,
    POLICIES,
    STATIC_POLICY,
    layout_fingerprint,
    stage5e4_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 5E.4 live heterogeneous comparison bundle."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = [
        "summary.json",
        "metadata.json",
        "frozen_initial_layout.csv",
        "disk_preflight.csv",
        "policy_summary.csv",
        "checksums.sha256",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")

    summary = {}
    if (root / "summary.json").is_file():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    for policy in POLICIES:
        trial_root = root / policy
        trial_required = [
            "summary.json",
            "initial_layout.csv",
            "pre_live_witness.csv",
            "resource_samples.csv",
            "evaluation_triggers.csv",
            "decisions.csv",
            "bids.csv",
            "migrations.csv",
            "ownership.csv",
            "final_tasks.csv",
            "cleanup.csv",
            "checksums.sha256",
        ]
        for name in trial_required:
            if not (trial_root / name).is_file():
                errors.append(f"Missing {policy}/{name}")
        if trial_root.is_dir():
            errors.extend(f"{policy}: {item}" for item in validate_checksums(trial_root))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    layout = read_csv(root / "frozen_initial_layout.csv")
    trial_summaries = [
        json.loads((root / policy / "summary.json").read_text(encoding="utf-8"))
        for policy in POLICIES
    ]

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if summary.get("season") != EXPECTED_SEASON:
        errors.append("season mismatch")
    if summary.get("load_id") != EXPECTED_LOAD_ID:
        errors.append("load-id mismatch")
    if summary.get("source_stage4d2_scenario_id") != EXPECTED_SOURCE_SCENARIO_ID:
        errors.append("source scenario mismatch")
    if int(summary.get("task_count") or 0) != EXPECTED_TASK_COUNT:
        errors.append("task-count mismatch")
    if dict(summary.get("class_counts") or {}) != EXPECTED_CLASS_COUNTS:
        errors.append("class-count mismatch")

    for key, label in (
        ("source_stage5a_bundle", "Stage 5A"),
        ("source_stage5e3_bundle", "Stage 5E.3"),
        ("source_stage5e2_bundle", "Stage 5E.2"),
        ("source_stage5e1_bundle", "Stage 5E.1"),
        ("source_stage4d3_bundle", "Stage 4D.3"),
        ("source_stage4d1_bundle", "Stage 4D.1"),
    ):
        path = Path(str(summary.get(key) or ""))
        if not path.is_dir():
            errors.append(f"{label} source bundle missing: {path}")
        elif validate_checksums(path):
            errors.append(f"{label} source checksum validation failed")

    if not stage5e4_passes(
        trial_summaries=trial_summaries,
        expected_layout_fingerprint=layout_fingerprint(layout),
    ):
        errors.append("Stage 5E.4 pass invariants failed")

    by_policy = {str(row["policy"]): row for row in trial_summaries}
    static = by_policy.get(STATIC_POLICY, {})
    magellan = by_policy.get(MAGELLAN_POLICY, {})
    if int(static.get("scheduler_decision_count") or 0) != 0:
        errors.append("static trial contains scheduler decisions")
    if int(static.get("bid_count") or 0) != 0:
        errors.append("static trial contains bids")
    if int(static.get("successful_migration_count") or 0) != 0:
        errors.append("static trial contains migrations")
    if int(magellan.get("scheduler_decision_count") or 0) != 6:
        errors.append("Magellan trial does not contain six decision-cohort evaluations")
    if int(magellan.get("successful_migration_count") or 0) < 1:
        errors.append("Magellan trial did not exercise a successful migration")
    if int(magellan.get("failed_migration_count") or 0) != 0:
        errors.append("Magellan trial contains failed migrations")
    if float(magellan.get("evaluation_trigger_delay_seconds_after_witness") or 0.0) > 2.0:
        errors.append("Magellan evaluation triggers were not issued promptly after the 9/9 physical witness")

    for policy in POLICIES:
        trial_root = root / policy
        witness = read_csv(trial_root / "pre_live_witness.csv")
        samples = read_csv(trial_root / "resource_samples.csv")
        cleanup = read_csv(trial_root / "cleanup.csv")
        final_tasks = read_csv(trial_root / "final_tasks.csv")
        if len(witness) != EXPECTED_TASK_COUNT or sum(truthy(row.get("live")) for row in witness) != EXPECTED_TASK_COUNT:
            errors.append(f"{policy} pre-live witness is not 9/9")
        if not samples:
            errors.append(f"{policy} has no resource samples")
        if any(not truthy(row.get("capacity_respected")) for row in samples):
            errors.append(f"{policy} contains a capacity violation sample")
        if len(cleanup) != EXPECTED_TASK_COUNT or sum(truthy(row.get("cleanup_ok")) for row in cleanup) != EXPECTED_TASK_COUNT:
            errors.append(f"{policy} cleanup is not 9/9")
        if len(final_tasks) != EXPECTED_TASK_COUNT:
            errors.append(f"{policy} final-task coverage is not 9/9")
        if any(row.get("status") == "failed" for row in final_tasks):
            errors.append(f"{policy} contains a failed workload")

    static_carbon = float(summary.get("static_total_carbon_grams") or 0.0)
    magellan_carbon = float(summary.get("magellan_total_carbon_grams") or 0.0)
    ratio = summary.get("carbon_ratio_vs_static")
    if static_carbon <= 0 or magellan_carbon <= 0:
        errors.append("non-positive policy carbon total")
    elif ratio is None or abs(float(ratio) - magellan_carbon / static_carbon) > 1e-9:
        errors.append("carbon ratio mismatch")

    if int(summary.get("capacity_violation_sample_count") or 0) != 0:
        errors.append("summary reports capacity violations")
    if float(summary.get("trace_anchor_delta_seconds") or 0.0) > 300.0:
        errors.append("policy trace anchors are too far apart")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5E4_LIVE_HETEROGENEOUS_COMPARISON_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print("source_case: winter/u75 (3 benchmark + 3 llm + 3 dendro)")
    print(f"planned_cpu_fraction: {float(summary.get('planned_cpu_fraction') or 0.0) * 100.0:.2f}%")
    print(
        "static: "
        f"carbon={static_carbon:.6f}g cost=${float(summary.get('static_total_cost_usd') or 0.0):.6f} "
        "decisions=0 bids=0 migrations=0"
    )
    print(
        "magellan: "
        f"carbon={magellan_carbon:.6f}g cost=${float(summary.get('magellan_total_cost_usd') or 0.0):.6f} "
        f"decisions={magellan.get('scheduler_decision_count')} bids={magellan.get('bid_count')} "
        f"migrations={magellan.get('successful_migration_count')}"
    )
    print(f"carbon_ratio_vs_static: {float(summary.get('carbon_ratio_vs_static') or 0.0):.4f}")
    print(f"carbon_savings_percent_vs_static: {float(summary.get('carbon_savings_percent_vs_static') or 0.0):.2f}%")
    print("capacity_violations: 0")
    print("cleanup: 9/9 per policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
