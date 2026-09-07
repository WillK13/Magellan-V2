from __future__ import annotations

import csv
import json
from pathlib import Path

from magellan.experiments.stage5e4 import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_SOURCE_SCENARIO_ID,
    MAGELLAN_POLICY,
    STATIC_POLICY,
    layout_fingerprint,
    read_stage4d3_u75_layout,
    stage5e4_passes,
    trial_passes,
)


def _layout() -> list[dict[str, str]]:
    # Exact winter/layout0 u75 nested subset frozen by Stage 4D.3.
    return [
        {"scenario_id": "winter-u75", "task_id": "b1", "class_id": "benchmark-json-medium", "initial_node_id": "boston"},
        {"scenario_id": "winter-u75", "task_id": "b2", "class_id": "benchmark-json-medium", "initial_node_id": "boston"},
        {"scenario_id": "winter-u75", "task_id": "b3", "class_id": "benchmark-json-medium", "initial_node_id": "california"},
        {"scenario_id": "winter-u75", "task_id": "l1", "class_id": "llm-distilgpt2", "initial_node_id": "california"},
        {"scenario_id": "winter-u75", "task_id": "d1", "class_id": "dendro-r9-t1p0", "initial_node_id": "south-australia"},
        {"scenario_id": "winter-u75", "task_id": "l2", "class_id": "llm-distilgpt2", "initial_node_id": "nepal"},
        {"scenario_id": "winter-u75", "task_id": "d2", "class_id": "dendro-r9-t1p0", "initial_node_id": "ethiopia"},
        {"scenario_id": "winter-u75", "task_id": "l3", "class_id": "llm-distilgpt2", "initial_node_id": "france"},
        {"scenario_id": "winter-u75", "task_id": "d3", "class_id": "dendro-r9-t1p0", "initial_node_id": "virginia"},
    ]


def _trial(policy: str) -> dict:
    return {
        "policy": policy,
        "task_count": 9,
        "class_counts": dict(EXPECTED_CLASS_COUNTS),
        "layout_fingerprint": [list(item) for item in layout_fingerprint(_layout())],
        "configured_trial_seconds": 240.0,
        "actual_trial_seconds": 240.2,
        "trace_anchor_utc": "2024-01-05T00:02:00+00:00",
        "trace_date_utc": "2024-01-05",
        "pre_live_witness_count": 9,
        "evaluation_trigger_delay_seconds_after_witness": 0.0 if policy == STATIC_POLICY else 0.2,
        "scheduler_decision_count": 0 if policy == STATIC_POLICY else 6,
        "bid_count": 0 if policy == STATIC_POLICY else 4,
        "accepted_or_consumed_bid_count": 0 if policy == STATIC_POLICY else 1,
        "rejected_bid_count": 0 if policy == STATIC_POLICY else 3,
        "successful_migration_count": 0 if policy == STATIC_POLICY else 1,
        "failed_migration_count": 0,
        "migration_downtime_seconds_total": 0.0 if policy == STATIC_POLICY else 12.0,
        "ownership_converged": True,
        "completed_task_count": 3,
        "total_runtime_seconds": 1200.0,
        "total_carbon_grams": 12.0 if policy == STATIC_POLICY else 10.0,
        "total_cost_usd": 0.02,
        "average_cluster_resource_busy_fraction": 0.5,
        "mean_observed_cluster_task_cpu_percent": 700.0,
        "capacity_violation_sample_count": 0,
        "sample_round_count": 17,
        "cleanup_ok_count": 9,
    }


def test_read_stage4d3_u75_layout(tmp_path: Path) -> None:
    with (tmp_path / "load_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id", "season", "source_stage4d2_scenario_id", "load_id",
                "task_count", "achieved_initial_cpu_fraction",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": "winter-u75",
                "season": "winter",
                "source_stage4d2_scenario_id": EXPECTED_SOURCE_SCENARIO_ID,
                "load_id": "u75",
                "task_count": 9,
                "achieved_initial_cpu_fraction": 0.7578,
            }
        )
    layout = _layout()
    with (tmp_path / "initial_layout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layout[0].keys()))
        writer.writeheader()
        writer.writerows(layout)

    case, rows = read_stage4d3_u75_layout(tmp_path)
    assert case["source_stage4d2_scenario_id"] == EXPECTED_SOURCE_SCENARIO_ID
    assert len(rows) == 9
    assert {row["initial_node_id"] for row in rows} == {
        "boston", "california", "south-australia", "nepal", "ethiopia", "france", "virginia"
    }


def test_static_trial_rejects_scheduler_activity() -> None:
    trial = _trial(STATIC_POLICY)
    trial["bid_count"] = 1
    assert not trial_passes(trial, expected_layout_fingerprint=layout_fingerprint(_layout()))


def test_magellan_trial_requires_real_movement() -> None:
    trial = _trial(MAGELLAN_POLICY)
    trial["successful_migration_count"] = 0
    assert not trial_passes(trial, expected_layout_fingerprint=layout_fingerprint(_layout()))



def test_magellan_trial_rejects_late_evaluation_after_physical_witness() -> None:
    trial = _trial(MAGELLAN_POLICY)
    trial["evaluation_trigger_delay_seconds_after_witness"] = 2.5
    assert not trial_passes(trial, expected_layout_fingerprint=layout_fingerprint(_layout()))

def test_stage5e4_comparison_passes_without_requiring_carbon_win() -> None:
    static = _trial(STATIC_POLICY)
    magellan = _trial(MAGELLAN_POLICY)
    # Results are evidence, not a pass-condition. A live trial remains valid even
    # if migration overhead makes this short window carbon-positive.
    magellan["total_carbon_grams"] = 13.0
    assert stage5e4_passes(
        trial_summaries=[static, magellan],
        expected_layout_fingerprint=layout_fingerprint(_layout()),
    )


def test_stage5e4_rejects_capacity_violation() -> None:
    static = _trial(STATIC_POLICY)
    magellan = _trial(MAGELLAN_POLICY)
    magellan["capacity_violation_sample_count"] = 1
    assert not stage5e4_passes(
        trial_summaries=[static, magellan],
        expected_layout_fingerprint=layout_fingerprint(_layout()),
    )


def test_stage5e4_policy_trace_is_frozen_to_winter_window() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "config" / "policy.stage5e4.json").read_text(encoding="utf-8"))
    assert policy["clock"]["mode"] == "trace"
    assert policy["clock"]["trace_start_utc"] == "2024-01-05T00:00:00Z"
    assert policy["clock"]["trace_seconds_per_real_second"] == 1
    assert policy["auction"]["strategy"] == "lowest_score"
    assert policy["weights"] == {"time": 0.25, "carbon": 0.5, "cost": 0.25}


def test_production_policy_trace_anchor_remains_generic() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "config" / "policy.prod.json").read_text(encoding="utf-8"))
    assert policy["clock"]["trace_start_utc"] == "2024-01-01T00:00:00Z"


def test_stage5e4_policy_differs_from_production_only_by_trace_anchor() -> None:
    root = Path(__file__).resolve().parents[1]
    production = json.loads((root / "config" / "policy.prod.json").read_text(encoding="utf-8"))
    experiment = json.loads((root / "config" / "policy.stage5e4.json").read_text(encoding="utf-8"))
    assert production["clock"]["trace_start_utc"] == "2024-01-01T00:00:00Z"
    assert experiment["clock"]["trace_start_utc"] == "2024-01-05T00:00:00Z"
    production["clock"]["trace_start_utc"] = experiment["clock"]["trace_start_utc"]
    assert production == experiment
