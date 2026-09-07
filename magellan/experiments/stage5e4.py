from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from magellan.experiments.stage5e2 import (
    BENCHMARK_CLASS_ID,
    DENDRO_CLASS_ID,
    LLM_CLASS_ID,
)

STATIC_POLICY = "static_initial_layout"
MAGELLAN_POLICY = "magellan_lowest_score"
POLICIES = (STATIC_POLICY, MAGELLAN_POLICY)
EXPECTED_SEASON = "winter"
EXPECTED_LOAD_ID = "u75"
EXPECTED_SOURCE_SCENARIO_ID = "winter-20240105T0000Z-layout0"
EXPECTED_CLASS_COUNTS = {
    BENCHMARK_CLASS_ID: 3,
    DENDRO_CLASS_ID: 3,
    LLM_CLASS_ID: 3,
}
EXPECTED_TASK_COUNT = 9
DECISION_CLASSES = {BENCHMARK_CLASS_ID, LLM_CLASS_ID}
EXPECTED_DECISION_TASK_COUNT = 6


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_stage4d3_u75_layout(stage4d3_bundle: str | Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    root = Path(stage4d3_bundle)
    with (root / "load_cases.csv").open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))
    matches = [
        row for row in cases
        if row.get("season") == EXPECTED_SEASON and row.get("load_id") == EXPECTED_LOAD_ID
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {EXPECTED_SEASON}/{EXPECTED_LOAD_ID} load case, found {len(matches)}"
        )
    case = matches[0]
    if case.get("source_stage4d2_scenario_id") != EXPECTED_SOURCE_SCENARIO_ID:
        raise ValueError(
            "Frozen Stage 5E.4 source scenario drifted: "
            f"{case.get('source_stage4d2_scenario_id')} != {EXPECTED_SOURCE_SCENARIO_ID}"
        )
    scenario_id = str(case["scenario_id"])
    with (root / "initial_layout.csv").open(encoding="utf-8", newline="") as handle:
        layout = [row for row in csv.DictReader(handle) if row.get("scenario_id") == scenario_id]
    if len(layout) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Frozen Stage 4D.3 u75 layout has {len(layout)} tasks, expected {EXPECTED_TASK_COUNT}")
    counts = Counter(row.get("class_id") for row in layout)
    if dict(counts) != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Frozen Stage 4D.3 u75 class mix drifted: {dict(counts)}")
    if int(case.get("task_count") or 0) != EXPECTED_TASK_COUNT:
        raise ValueError("Frozen Stage 4D.3 u75 load-case task count drifted")
    return case, sorted(layout, key=lambda row: str(row.get("task_id")))


def layout_fingerprint(rows: Iterable[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(row.get("initial_node_id")), str(row.get("class_id"))) for row in rows))


def trial_passes(
    trial: dict[str, Any],
    *,
    expected_layout_fingerprint: tuple[tuple[str, str], ...],
) -> bool:
    if str(trial.get("policy")) not in POLICIES:
        return False
    if int(trial.get("task_count") or 0) != EXPECTED_TASK_COUNT:
        return False
    if dict(trial.get("class_counts") or {}) != EXPECTED_CLASS_COUNTS:
        return False
    observed_fp = tuple(tuple(item) for item in trial.get("layout_fingerprint") or [])
    if observed_fp != expected_layout_fingerprint:
        return False
    if int(trial.get("pre_live_witness_count") or 0) != EXPECTED_TASK_COUNT:
        return False
    if int(trial.get("cleanup_ok_count") or 0) != EXPECTED_TASK_COUNT:
        return False
    if int(trial.get("capacity_violation_sample_count") or 0) != 0:
        return False
    if int(trial.get("failed_migration_count") or 0) != 0:
        return False
    if not _truthy(trial.get("ownership_converged")):
        return False
    if float(trial.get("actual_trial_seconds") or 0.0) + 1e-6 < float(trial.get("configured_trial_seconds") or 0.0):
        return False
    if float(trial.get("total_carbon_grams") or 0.0) <= 0.0:
        return False
    if float(trial.get("total_cost_usd") or 0.0) <= 0.0:
        return False

    policy = str(trial["policy"])
    decisions = int(trial.get("scheduler_decision_count") or 0)
    bids = int(trial.get("bid_count") or 0)
    migrations = int(trial.get("successful_migration_count") or 0)
    if policy == STATIC_POLICY:
        if decisions != 0 or bids != 0 or migrations != 0:
            return False
    else:
        if decisions != EXPECTED_DECISION_TASK_COUNT:
            return False
        if bids < 1:
            return False
        if migrations < 1:
            return False
        # The exact Dendro r9/t1p0 jobs are intentionally short.  They must be
        # physically present at the scheduling epoch, so the six evaluation
        # triggers are submitted immediately after the 9/9 direct witness.
        if float(trial.get("evaluation_trigger_delay_seconds_after_witness") or 0.0) > 2.0:
            return False
    return True


def stage5e4_passes(
    *,
    trial_summaries: list[dict[str, Any]],
    expected_layout_fingerprint: tuple[tuple[str, str], ...],
) -> bool:
    if {str(row.get("policy")) for row in trial_summaries} != set(POLICIES):
        return False
    if len(trial_summaries) != len(POLICIES):
        return False
    by_policy = {str(row["policy"]): row for row in trial_summaries}
    if not all(
        trial_passes(row, expected_layout_fingerprint=expected_layout_fingerprint)
        for row in trial_summaries
    ):
        return False
    static = by_policy[STATIC_POLICY]
    magellan = by_policy[MAGELLAN_POLICY]
    # Same fixed measurement window; a small wall-clock overrun from cleanup/accounting
    # is not part of the trial summary itself.
    if abs(
        float(static.get("configured_trial_seconds") or 0.0)
        - float(magellan.get("configured_trial_seconds") or 0.0)
    ) > 1e-9:
        return False
    if str(static.get("trace_date_utc")) != "2024-01-05":
        return False
    if str(magellan.get("trace_date_utc")) != "2024-01-05":
        return False
    return True
