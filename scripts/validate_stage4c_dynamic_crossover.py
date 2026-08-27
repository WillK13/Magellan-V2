#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage4b import CORE_WORKLOADS
from magellan.experiments.stage4c import DYNAMIC_POLICIES, TARGET_BOSTON_RUNTIME_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4C 72-hour dynamic crossover bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return abs(left - right) <= max(atol, rtol * max(abs(left), abs(right)))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors: list[str] = []
    if not root.is_dir():
        print(f"ERROR: Missing bundle directory: {root}")
        return 2

    errors.extend(validate_checksums(root))
    required = [
        "summary.json",
        "metadata.json",
        "calibration_model.json",
        "dynamic_summary.json",
        "candidate_windows.csv",
        "selected_windows.csv",
        "scenarios.csv",
        "outcomes.csv",
        "policy_summary.csv",
        "magellan_dynamic_summary.csv",
        "leadership_timeline.csv",
        "leadership_windows.csv",
        "magellan_migrations.csv",
        "magellan_residence.csv",
        "traces.jsonl",
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
    dynamic = json.loads((root / "dynamic_summary.json").read_text(encoding="utf-8"))
    candidates = read_csv(root / "candidate_windows.csv")
    selected = read_csv(root / "selected_windows.csv")
    scenarios = read_csv(root / "scenarios.csv")
    outcomes = read_csv(root / "outcomes.csv")
    dynamic_rows = read_csv(root / "magellan_dynamic_summary.csv")
    leadership = read_csv(root / "leadership_timeline.csv")
    windows = read_csv(root / "leadership_windows.csv")
    migrations = read_csv(root / "magellan_migrations.csv")
    residence = read_csv(root / "magellan_residence.csv")

    target_seconds = float(summary.get("target_boston_runtime_seconds") or 0.0)
    expected_scenarios = int(summary.get("expected_scenario_count") or 0)
    expected_outcomes = int(summary.get("expected_outcome_count") or 0)
    samples_per_scenario = int(summary.get("expected_leadership_samples_per_scenario") or 0)
    windows_per_season = int(summary.get("windows_per_season") or 0)
    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if not close(target_seconds, TARGET_BOSTON_RUNTIME_SECONDS):
        errors.append(f"target runtime {target_seconds} is not the canonical 72h value {TARGET_BOSTON_RUNTIME_SECONDS}")
    if set(summary.get("policy_names") or []) != set(DYNAMIC_POLICIES):
        errors.append("summary policy set does not match Stage 4C dynamic policies")
    if set(summary.get("workload_classes") or []) != set(CORE_WORKLOADS):
        errors.append("summary workload classes do not match the frozen core workloads")
    if windows_per_season != 1:
        errors.append(f"canonical Stage 4C requires windows_per_season=1, got {windows_per_season}")
    if len(candidates) != 24:
        errors.append(f"candidate annual window coverage {len(candidates)} != 24")
    if len(selected) != 4:
        errors.append(f"selected seasonal window coverage {len(selected)} != 4")
    if {row["season"] for row in selected} != {"winter", "spring", "summer", "fall"}:
        errors.append("selected windows do not contain exactly one window per season")
    if len(scenarios) != expected_scenarios or expected_scenarios != 12:
        errors.append(f"scenario coverage {len(scenarios)}/{expected_scenarios}, expected canonical 12")
    if len(outcomes) != expected_outcomes or expected_outcomes != 24:
        errors.append(f"outcome coverage {len(outcomes)}/{expected_outcomes}, expected canonical 24")
    if len(dynamic_rows) != expected_scenarios:
        errors.append(f"dynamic scenario rows {len(dynamic_rows)} != {expected_scenarios}")
    if len(leadership) != expected_scenarios * samples_per_scenario:
        errors.append(
            f"leadership sample coverage {len(leadership)} != {expected_scenarios}*{samples_per_scenario}"
        )

    # Selection must be trace-only and deterministic: one top-ranked candidate per season.
    selected_by_season = {row["season"]: row for row in selected}
    for season in ("winter", "spring", "summer", "fall"):
        rows = [row for row in candidates if row["season"] == season]
        rows.sort(
            key=lambda row: (
                -int(row["sustained_scheduler_leader_transitions"]),
                -int(row["sustained_scheduler_unique_leaders"]),
                -int(row["scheduler_leader_changes"]),
                row["arrival_utc"],
            )
        )
        if not rows:
            errors.append(f"no candidate windows for {season}")
            continue
        chosen = selected_by_season.get(season)
        if chosen is None or chosen["arrival_utc"] != rows[0]["arrival_utc"]:
            errors.append(f"{season} selected window is not deterministic rank 1")
        if chosen is not None and int(chosen["selection_rank_within_season"] or 0) != 1:
            errors.append(f"{season} selected window rank is not 1")

    scenario_ids = {row["scenario_id"] for row in scenarios}
    if len(scenario_ids) != expected_scenarios:
        errors.append("scenario IDs are not unique")
    selected_arrivals = {row["arrival_utc"] for row in selected}
    arrival_class_counts: Counter[str] = Counter()
    for row in scenarios:
        arrival_class_counts[row["arrival_utc"]] += 1
        if row["arrival_utc"] not in selected_arrivals:
            errors.append(f"{row['scenario_id']} uses a non-selected arrival")
        if not close(float(row["target_boston_runtime_seconds"]), target_seconds):
            errors.append(f"{row['scenario_id']} has inconsistent target runtime")
        if not close(float(row["scaled_boston_work_seconds"]), target_seconds, atol=1e-3):
            errors.append(f"{row['scenario_id']} does not scale to 72h Boston-static work")
    for arrival in selected_arrivals:
        if arrival_class_counts[arrival] != len(CORE_WORKLOADS):
            errors.append(f"selected arrival {arrival} does not have all workload classes")

    by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in outcomes:
        by_scenario[row["scenario_id"]].append(row)
    for scenario_id in scenario_ids:
        rows = by_scenario.get(scenario_id, [])
        policies = {row["policy"] for row in rows}
        if policies != set(DYNAMIC_POLICIES):
            errors.append(f"{scenario_id} policy set mismatch: {sorted(policies)}")
            continue
        for row in rows:
            if str(row["completed"]).lower() not in {"true", "1"}:
                errors.append(f"{scenario_id}/{row['policy']} did not complete")
        boston = next(row for row in rows if row["policy"] == "boston_static")
        if not close(float(boston["makespan_seconds"]), target_seconds, atol=1e-3):
            errors.append(f"{scenario_id} Boston-static makespan is not 72h")

    leadership_counts = Counter(row["scenario_id"] for row in leadership)
    for scenario_id in scenario_ids:
        if leadership_counts[scenario_id] != samples_per_scenario:
            errors.append(
                f"{scenario_id} leadership samples {leadership_counts[scenario_id]} != {samples_per_scenario}"
            )

    dynamic_by_id = {row["scenario_id"]: row for row in dynamic_rows}
    magellan_by_id = {
        row["scenario_id"]: row
        for row in outcomes
        if row["policy"] == "magellan_causal"
    }
    migration_counts = Counter(row["scenario_id"] for row in migrations)
    residence_compute: dict[str, float] = defaultdict(float)
    for row in residence:
        residence_compute[row["scenario_id"]] += float(row["compute_seconds"])

    for scenario_id in scenario_ids:
        if scenario_id not in dynamic_by_id or scenario_id not in magellan_by_id:
            errors.append(f"{scenario_id} missing dynamic/Magellan row")
            continue
        dyn = dynamic_by_id[scenario_id]
        mag = magellan_by_id[scenario_id]
        expected_migrations = int(mag["migrations"])
        if int(dyn["migrations"]) != expected_migrations:
            errors.append(f"{scenario_id} dynamic migration count mismatch")
        if migration_counts[scenario_id] != expected_migrations:
            errors.append(f"{scenario_id} migration event count mismatch")
        if dyn["owner_path"] != mag["owner_path"]:
            errors.append(f"{scenario_id} owner path mismatch")
        if not close(residence_compute[scenario_id], float(mag["compute_seconds"]), atol=1e-3):
            errors.append(f"{scenario_id} residence compute total mismatch")

    if int(summary.get("observed_magellan_migration_event_count") or 0) != len(migrations):
        errors.append("summary migration event count does not match magellan_migrations.csv")
    if int(dynamic.get("magellan_migrations_total") or 0) != len(migrations):
        errors.append("dynamic summary migration total does not match magellan_migrations.csv")
    multi_count = sum(int(row["migrations"]) >= 2 for row in dynamic_rows)
    if int(dynamic.get("scenarios_multi_migration") or 0) != multi_count:
        errors.append("dynamic summary multi-migration scenario count mismatch")
    observed = bool(dynamic.get("dynamic_traversal_observed"))
    if observed != (multi_count > 0):
        errors.append("dynamic_traversal_observed is inconsistent with scenario migration counts")
    if bool(summary.get("dynamic_traversal_observed")) != observed:
        errors.append("summary dynamic_traversal_observed disagrees with dynamic_summary.json")

    bad_window_ids = {row["scenario_id"] for row in windows if row["scenario_id"] not in scenario_ids}
    if bad_window_ids:
        errors.append(f"leadership windows reference unknown scenarios: {sorted(bad_window_ids)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_4C_DYNAMIC_CROSSOVER_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"candidate_arrivals: {len(candidates)}/24")
    print(f"selected_arrivals: {len(selected)}/4")
    print(f"scenarios: {len(scenarios)}/12")
    print(f"outcomes: {len(outcomes)}/24")
    print(f"target_hours: {target_seconds / 3600.0:.1f}")
    print(f"magellan_migrations: {len(migrations)}")
    print(f"multi_migration_scenarios: {multi_count}/12")
    print(f"dynamic_traversal_observed: {observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
