#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4b import CORE_POLICIES, CORE_WORKLOADS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 4B calibrated core-comparison bundle.")
    parser.add_argument("bundle")
    args = parser.parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = [
        "summary.json",
        "metadata.json",
        "calibration_model.json",
        "gaia_reproduction.json",
        "scenarios.csv",
        "outcomes.csv",
        "policy_summary.csv",
        "policy_descriptive_metrics.json",
        "traces.jsonl",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        raise SystemExit("STAGE_4B_CORE_COMPARISON_BUNDLE_FAIL\n- " + "\n- ".join(errors))

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration_model.json").read_text(encoding="utf-8"))
    gaia = json.loads((root / "gaia_reproduction.json").read_text(encoding="utf-8"))
    scenarios = read_csv(root / "scenarios.csv")
    outcomes = read_csv(root / "outcomes.csv")
    policy_summary = read_csv(root / "policy_summary.csv")

    if summary.get("passed") is not True:
        errors.append("Parent summary passed=false")
    if summary.get("runtime_model") != "single_node_slowdown_factor":
        errors.append("Unexpected runtime model")
    if summary.get("carbon_metric") != "lifecycle":
        errors.append("Stage 4B core comparison must use lifecycle carbon")
    if set(summary.get("policy_names") or []) != set(CORE_POLICIES):
        errors.append("Core policy set mismatch")
    if set(summary.get("workload_classes") or []) != set(CORE_WORKLOADS):
        errors.append("Core workload set mismatch")

    expected_scenarios = int(summary.get("expected_scenario_count") or 0)
    expected_outcomes = int(summary.get("expected_outcome_count") or 0)
    if len(scenarios) != expected_scenarios or int(summary.get("observed_scenario_count") or 0) != len(scenarios):
        errors.append(f"Scenario count {len(scenarios)} != expected {expected_scenarios}")
    if len(outcomes) != expected_outcomes or int(summary.get("observed_outcome_count") or 0) != len(outcomes):
        errors.append(f"Outcome count {len(outcomes)} != expected {expected_outcomes}")
    if len(policy_summary) != len(CORE_POLICIES):
        errors.append("Policy summary row count mismatch")

    scenario_ids = {row["scenario_id"] for row in scenarios}
    pairs = [(row["scenario_id"], row["policy"]) for row in outcomes]
    if len(set(pairs)) != len(pairs):
        errors.append("Duplicate scenario-policy outcome")
    counts = Counter(row["scenario_id"] for row in outcomes)
    if set(counts) != scenario_ids or any(value != len(CORE_POLICIES) for value in counts.values()):
        errors.append("Every scenario must contain exactly one outcome per core policy")

    for row in outcomes:
        label = f"{row.get('scenario_id')}:{row.get('policy')}"
        if str(row.get("completed")).lower() not in {"true", "1"}:
            errors.append(f"Incomplete outcome {label}")
        for field in ("makespan_seconds", "compute_seconds", "carbon_grams", "cost_usd"):
            if float(row.get(field) or 0) < 0:
                errors.append(f"Negative {field}: {label}")
        if float(row.get("makespan_seconds") or 0) <= 0 or float(row.get("compute_seconds") or 0) <= 0:
            errors.append(f"Non-positive execution time: {label}")
        policy_name = row.get("policy")
        if policy_name == "boston_static":
            if row.get("start_node_id") != "boston" or row.get("final_node_id") != "boston":
                errors.append(f"Boston-static moved: {label}")
            if int(float(row.get("migrations") or 0)) or int(float(row.get("pauses") or 0)):
                errors.append(f"Boston-static has pause/migration: {label}")
        elif policy_name == "best_static":
            if row.get("start_node_id") != row.get("final_node_id") or int(float(row.get("migrations") or 0)):
                errors.append(f"Best-static moved after placement: {label}")
            if float(row.get("submission_wait_seconds") or 0) > 1e-9:
                errors.append(f"Best-static waited: {label}")
        elif policy_name == "gaia_carbon_time":
            if row.get("start_node_id") != "boston" or row.get("final_node_id") != "boston":
                errors.append(f"GAIA reproduction left Boston: {label}")
            if int(float(row.get("migrations") or 0)) or int(float(row.get("pauses") or 0)):
                errors.append(f"GAIA reproduction used pause/migration: {label}")
            if float(row.get("submission_wait_seconds") or 0) < 0 or float(row.get("submission_wait_seconds") or 0) >= 86400 + 1e-9:
                errors.append(f"GAIA wait outside published maximum window: {label}")
        elif policy_name == "clairvoyant_spatiotemporal_static_oracle":
            if row.get("start_node_id") != row.get("final_node_id") or int(float(row.get("migrations") or 0)):
                errors.append(f"Static oracle migrated: {label}")
        elif policy_name == "magellan_causal":
            if row.get("start_node_id") != "boston":
                errors.append(f"Magellan did not start in Boston: {label}")

    if gaia.get("artifact_policy_mapping", {}).get("Carbon-Time") != {"scheduling_policy": "carbon", "carbon_policy": "cst_average"}:
        errors.append("GAIA Carbon-Time artifact mapping mismatch")
    if float(gaia.get("short_queue_max_runtime_seconds") or 0) != 7200:
        errors.append("GAIA short-queue boundary mismatch")
    if float(gaia.get("short_queue_max_wait_seconds") or 0) != 21600 or float(gaia.get("long_queue_max_wait_seconds") or 0) != 86400:
        errors.append("GAIA wait-window configuration mismatch")

    slowdowns = calibration.get("node_slowdown_factors") or {}
    if len(slowdowns) != 7 or abs(float(slowdowns.get("boston") or 0) - 1.0) > 1e-6:
        errors.append("Frozen Stage 4A.4 slowdown table is incomplete")
    if set((calibration.get("workloads") or {}).keys()) != set(CORE_WORKLOADS):
        errors.append("Frozen workload calibration set mismatch")
    if int(calibration.get("stage4a1_edge_count") or 0) != 42:
        errors.append("Frozen Stage 4A.1 directed edge count must be 42")

    for source_key in ("stage4a1_bundle", "stage4a2_bundle", "stage4a3_bundle", "stage4a4_bundle", "stage4a5_bundle"):
        source = Path(str(summary.get(source_key) or ""))
        if not source.is_dir():
            errors.append(f"Missing source bundle referenced by {source_key}")
        else:
            source_errors = validate_checksums(source)
            if source_errors:
                errors.append(f"Source bundle checksum failure {source_key}: {'; '.join(source_errors)}")
    a5 = Path(str(summary.get("stage4a5_bundle") or ""))
    if (a5 / "summary.json").is_file():
        a5_summary = json.loads((a5 / "summary.json").read_text(encoding="utf-8"))
        if a5_summary.get("ready_for_stage4b_runtime_model") is not True or a5_summary.get("recommended_runtime_model") != "single_node_slowdown_factor":
            errors.append("Stage 4A.5 source does not authorize the runtime model")

    if errors:
        raise SystemExit("STAGE_4B_CORE_COMPARISON_BUNDLE_FAIL\n- " + "\n- ".join(errors))
    print("STAGE_4B_CORE_COMPARISON_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"scenarios: {len(scenarios)}/{expected_scenarios}")
    print(f"outcomes: {len(outcomes)}/{expected_outcomes}")
    print(f"policies: {len(CORE_POLICIES)}/{len(CORE_POLICIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
