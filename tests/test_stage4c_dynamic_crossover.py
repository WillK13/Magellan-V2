from __future__ import annotations

import pandas as pd
import pytest

from magellan.config.loader import load_cluster_config
from magellan.experiments.comparison import PolicyOutcome, ReplayStep
from magellan.experiments.stage4b import Scenario, WorkloadCalibration
from magellan.experiments.stage4c import (
    TARGET_BOSTON_RUNTIME_SECONDS,
    aggregate_dynamic_summary,
    dynamic_scenario_summary,
    leader_change_count,
    leadership_timeline,
    leadership_windows,
    migration_diagnostics,
    residence_rows,
    runtime_scales_for_target,
)


def calibration(runtime: float = 120.0) -> WorkloadCalibration:
    return WorkloadCalibration(
        class_id="benchmark-json-medium",
        workload="json",
        variant="medium",
        canonical_runtime_seconds=runtime,
        power_kw=0.05,
        checkpoint_bytes=100_000_000,
        checkpoint_seconds=2.0,
        restore_seconds=3.0,
        migration_overhead_seconds=1.0,
    )


class CrossoverCarbon:
    def value_at(self, node_id, at_utc):
        hour = pd.Timestamp(at_utc).hour
        if node_id == "ethiopia":
            return 20.0 if hour < 2 else 80.0
        if node_id == "france":
            return 50.0 if hour < 2 else 25.0
        if node_id == "boston":
            return 500.0
        return 300.0

    def average(self, node_id, start_utc, seconds):
        return self.value_at(node_id, start_utc)


def scenario() -> Scenario:
    return Scenario(
        scenario_id="20240105T0000Z-benchmark-json-medium",
        class_id="benchmark-json-medium",
        arrival_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
    )


def test_runtime_scaling_targets_exact_72h_boston():
    scales = runtime_scales_for_target(
        {"benchmark-json-medium": calibration(runtime=120.0)},
        node_slowdowns={"boston": 1.0},
    )
    assert scales["benchmark-json-medium"] == pytest.approx(2160.0)
    assert 120.0 * scales["benchmark-json-medium"] == pytest.approx(TARGET_BOSTON_RUNTIME_SECONDS)


def test_leadership_timeline_detects_real_trace_crossover_shape():
    cluster = load_cluster_config("config/cluster.gcp.json")
    slowdowns = {node.id: 1.0 for node in cluster.nodes}
    rows = leadership_timeline(
        scenario=scenario(),
        cluster=cluster,
        node_slowdowns=slowdowns,
        carbon_store=CrossoverCarbon(),  # type: ignore[arg-type]
        horizon_seconds=4 * 3600,
        quantum_seconds=3600,
    )
    assert len(rows) == 4
    assert [row["scheduler_carbon_leader_id"] for row in rows] == [
        "ethiopia",
        "ethiopia",
        "france",
        "france",
    ]
    assert leader_change_count(rows, "scheduler_carbon_leader_id") == 1


def _migration_step(*, start_hour: float = 0.5, destination: str = "ethiopia") -> ReplayStep:
    start = pd.Timestamp("2024-01-05T00:00:00Z") + pd.Timedelta(hours=start_hour)
    finish = start + pd.Timedelta(minutes=10)
    return ReplayStep(
        index=1,
        action="migrate",
        source_node_id="boston",
        destination_node_id=destination,
        started_at_utc=start.to_pydatetime(),
        finished_at_utc=finish.to_pydatetime(),
        elapsed_seconds=600.0,
        migration_seconds=600.0,
        carbon_grams=1.0,
        cost_usd=0.1,
        remaining_seconds_after=3600.0,
        details={
            "decision": {
                "selected": {"action": "migrate", "carbon_grams": 5.0, "score": 0.2},
                "ranked_actions": [
                    {"action": "migrate", "carbon_grams": 5.0, "score": 0.2},
                    {"action": "continue", "carbon_grams": 20.0, "score": 0.6},
                ],
            }
        },
    )


def _outcome(steps: list[ReplayStep], *, owner_path: list[str], makespan_seconds: float = 4 * 3600) -> PolicyOutcome:
    return PolicyOutcome(
        policy="magellan_causal",
        start_node_id="boston",
        final_node_id=owner_path[-1],
        selected_initial_node_id="boston",
        completed=True,
        makespan_seconds=makespan_seconds,
        compute_seconds=3600.0,
        paused_idle_seconds=0.0,
        pause_overhead_seconds=0.0,
        migration_seconds=sum(step.migration_seconds for step in steps),
        carbon_grams=10.0,
        cost_usd=1.0,
        migrations=sum(step.action == "migrate" for step in steps),
        pauses=sum(step.action == "pause" for step in steps),
        decision_count=len(steps),
        owner_path=owner_path,
        steps=steps,
    )


def test_leadership_windows_distinguish_exploited_and_ignored_opportunities():
    cluster = load_cluster_config("config/cluster.gcp.json")
    slowdowns = {node.id: 1.0 for node in cluster.nodes}
    timeline = leadership_timeline(
        scenario=scenario(),
        cluster=cluster,
        node_slowdowns=slowdowns,
        carbon_store=CrossoverCarbon(),  # type: ignore[arg-type]
        horizon_seconds=4 * 3600,
        quantum_seconds=3600,
    )
    magellan = _outcome([_migration_step()], owner_path=["boston", "ethiopia"])
    windows = leadership_windows(
        scenario=scenario(),
        timeline_rows=timeline,
        magellan=magellan,
        minimum_opportunity_seconds=3600,
        horizon_seconds=4 * 3600,
        quantum_seconds=3600,
    )
    scheduler = [row for row in windows if row["leader_kind"] == "scheduler_carbon"]
    assert scheduler[0]["leader_node_id"] == "ethiopia"
    assert scheduler[0]["is_cross_region_opportunity"] is True
    assert scheduler[0]["exploited_by_migration"] is True
    assert scheduler[1]["leader_node_id"] == "france"
    assert scheduler[1]["ignored_opportunity"] is True


def test_migration_diagnostic_records_prediction_and_clairvoyant_net_savings():
    cluster = load_cluster_config("config/cluster.gcp.json")
    slowdowns = {node.id: 1.0 for node in cluster.nodes}
    step = _migration_step()
    magellan = _outcome([step], owner_path=["boston", "ethiopia"])
    rows = migration_diagnostics(
        scenario=scenario(),
        magellan=magellan,
        calibration=calibration(),
        cluster=cluster,
        node_slowdowns=slowdowns,
        carbon_store=CrossoverCarbon(),  # type: ignore[arg-type]
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["projected_carbon_savings_vs_continue_grams"] == pytest.approx(15.0)
    assert row["score_improvement_vs_continue"] == pytest.approx(0.4)
    assert row["clairvoyant_migration_beneficial"] is True
    assert row["clairvoyant_net_carbon_saved_grams"] > 0


def test_residence_and_dynamic_summary_preserve_owner_path_and_multi_migration():
    start = pd.Timestamp("2024-01-05T00:00:00Z")
    steps = [
        ReplayStep(
            index=1,
            action="continue",
            source_node_id="boston",
            started_at_utc=start.to_pydatetime(),
            finished_at_utc=(start + pd.Timedelta(hours=1)).to_pydatetime(),
            elapsed_seconds=3600.0,
            compute_seconds=3600.0,
            carbon_grams=10.0,
            cost_usd=0.1,
            remaining_seconds_after=7200.0,
        ),
        _migration_step(start_hour=1.0, destination="ethiopia").model_copy(update={"index": 2}),
        ReplayStep(
            index=3,
            action="migrate",
            source_node_id="ethiopia",
            destination_node_id="france",
            started_at_utc=(start + pd.Timedelta(hours=2)).to_pydatetime(),
            finished_at_utc=(start + pd.Timedelta(hours=2, minutes=10)).to_pydatetime(),
            elapsed_seconds=600.0,
            migration_seconds=600.0,
            carbon_grams=1.0,
            cost_usd=0.1,
            remaining_seconds_after=3600.0,
        ),
    ]
    magellan = _outcome(steps, owner_path=["boston", "ethiopia", "france"])
    residence = residence_rows(scenario=scenario(), magellan=magellan)
    assert sum(float(row["compute_seconds"]) for row in residence) == pytest.approx(3600.0)
    timeline = [
        {"scheduler_carbon_leader_id": "ethiopia", "realized_work_carbon_leader_id": "ethiopia"},
        {"scheduler_carbon_leader_id": "france", "realized_work_carbon_leader_id": "france"},
    ]
    windows = [
        {
            "leader_kind": "scheduler_carbon",
            "is_cross_region_opportunity": True,
            "exploited_by_migration": True,
            "ignored_opportunity": False,
        },
        {
            "leader_kind": "realized_work_carbon",
            "is_cross_region_opportunity": True,
            "exploited_by_migration": False,
            "ignored_opportunity": True,
        },
    ]
    row = dynamic_scenario_summary(
        scenario=scenario(),
        magellan=magellan,
        timeline_rows=timeline,
        window_rows=windows,
        migration_rows=[{"clairvoyant_migration_beneficial": True}, {"clairvoyant_migration_beneficial": True}],
        residence=residence,
        outcome_row={
            "carbon_ratio_vs_boston_static": 0.2,
            "time_ratio_vs_boston_static": 0.9,
            "cost_ratio_vs_boston_static": 1.1,
        },
    )
    assert row["multi_migration"] is True
    assert row["distinct_nodes_visited"] == 3
    assert row["owner_path"] == "boston->ethiopia->france"
    aggregate = aggregate_dynamic_summary([row])
    assert aggregate["dynamic_traversal_observed"] is True
    assert aggregate["scenarios_multi_migration"] == 1
    assert aggregate["magellan_migrations_total"] == 2


def test_dynamic_pass_condition_is_not_tied_to_multi_migration():
    base = {
        "scenario_id": "s1",
        "class_id": "benchmark-json-medium",
        "owner_path": "boston->ethiopia",
        "migrations": 1,
        "pauses": 0,
        "scheduler_carbon_leader_changes_72h": 4,
        "realized_work_carbon_leader_changes_72h": 3,
        "scheduler_opportunity_windows": 2,
        "scheduler_opportunities_exploited": 1,
        "scheduler_opportunities_ignored": 1,
        "realized_work_opportunity_windows": 2,
        "realized_work_opportunities_exploited": 0,
        "realized_work_opportunities_ignored": 2,
        "beneficial_migrations_clairvoyant_diagnostic": 1,
    }
    aggregate = aggregate_dynamic_summary([base])
    assert aggregate["dynamic_traversal_observed"] is False
    assert aggregate["scenarios_one_migration"] == 1


def test_trace_only_selection_picks_highest_crossover_window_per_season():
    from magellan.experiments.stage4c import select_crossover_arrivals

    rows = [
        {"arrival_utc": "2024-01-05T00:00:00+00:00", "season": "winter", "sustained_scheduler_leader_transitions": 0, "sustained_scheduler_unique_leaders": 1, "scheduler_leader_changes": 0},
        {"arrival_utc": "2024-02-05T00:00:00+00:00", "season": "winter", "sustained_scheduler_leader_transitions": 1, "sustained_scheduler_unique_leaders": 2, "scheduler_leader_changes": 1},
        {"arrival_utc": "2024-04-05T00:00:00+00:00", "season": "spring", "sustained_scheduler_leader_transitions": 2, "sustained_scheduler_unique_leaders": 2, "scheduler_leader_changes": 3},
        {"arrival_utc": "2024-05-05T00:00:00+00:00", "season": "spring", "sustained_scheduler_leader_transitions": 2, "sustained_scheduler_unique_leaders": 2, "scheduler_leader_changes": 4},
    ]
    selected = select_crossover_arrivals(rows, windows_per_season=1)
    by_season = {row["season"]: row for row in selected}
    assert by_season["winter"]["arrival_utc"].startswith("2024-02-05")
    assert by_season["spring"]["arrival_utc"].startswith("2024-05-05")
    assert all(row["selection_rank_within_season"] == 1 for row in selected)


def test_dynamic_outcome_rows_compare_only_boston_and_magellan():
    from magellan.config.loader import load_policy_config
    from magellan.experiments.stage4c import dynamic_outcome_rows

    policy = load_policy_config("config/policy.prod.json")
    boston = _outcome([], owner_path=["boston"], makespan_seconds=100.0).model_copy(
        update={"policy": "boston_static", "carbon_grams": 100.0, "cost_usd": 10.0, "compute_seconds": 100.0}
    )
    magellan = _outcome([_migration_step()], owner_path=["boston", "ethiopia"], makespan_seconds=90.0).model_copy(
        update={"carbon_grams": 40.0, "cost_usd": 11.0}
    )
    rows = dynamic_outcome_rows(
        scenario=scenario(),
        calibration=calibration(),
        boston_static=boston,
        magellan=magellan,
        policy=policy,
    )
    assert [row["policy"] for row in rows] == ["boston_static", "magellan_causal"]
    assert rows[1]["time_ratio_vs_boston_static"] == pytest.approx(0.9)
    assert rows[1]["carbon_ratio_vs_boston_static"] == pytest.approx(0.4)
    assert rows[1]["cost_ratio_vs_boston_static"] == pytest.approx(1.1)
