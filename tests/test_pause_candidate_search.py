from __future__ import annotations

import pandas as pd

from magellan.carbon.forecast import CarbonForecastEstimate
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import (
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
)
from magellan.graph.topology import ClusterGraph
from magellan.models.types import ActionType, TaskProfile
from magellan.scheduler.scoring import build_raw_actions


class ScheduledCarbonStore:
    def forecast(
        self,
        *,
        node_id,
        observed_at_utc,
        forecast_start_utc,
        duration_seconds,
        policy,
    ):
        del policy
        observed = pd.Timestamp(observed_at_utc)
        start = pd.Timestamp(forecast_start_utc)
        value = 500.0 if (start - observed).total_seconds() < 900 else 10.0
        return CarbonForecastEstimate(
            node_id=node_id,
            average_g_per_kwh=value,
            current_g_per_kwh=500,
            source="test",
            confidence=1,
            freshness="fresh",
            history_points=8,
            forecast_start_utc=start.to_pydatetime(),
            forecast_horizon_seconds=duration_seconds,
            generated_at_utc=observed.to_pydatetime(),
        )


def node(node_id: str, longitude: float) -> NodeConfig:
    return NodeConfig(
        id=node_id,
        name=node_id.title(),
        vm_name=node_id,
        zone="test-zone",
        internal_ip="10.0.0.1" if node_id == "boston" else "10.0.0.2",
        carbon_region=node_id,
        dataset_file=f"{node_id}.csv",
        latitude=0,
        longitude=longitude,
        pue=1,
        compute_price_usd_per_hour=0,
    )


def test_build_raw_actions_scores_each_pause_duration() -> None:
    cluster = ClusterConfig(
        nodes=[node("boston", 0), node("virginia", 1)],
    )
    policy = ScoringPolicy(
        horizon_seconds=600,
        weights=ObjectiveWeights(time=0.2, carbon=0.8, cost=0),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=300,
            candidate_idle_seconds=[0, 300, 900, 1800],
            resume_seconds=0,
            max_pause_window_seconds=3600,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        clock=ClockPolicy(mode="wall"),
    )
    task = TaskProfile(
        task_id="pause-search",
        workload_type="test",
        current_node_id="boston",
        power_kw=1,
        checkpoint_bytes=0,
        estimated_remaining_seconds=600,
    )

    actions = build_raw_actions(
        task=task,
        cluster=cluster,
        policy=policy,
        graph=ClusterGraph(cluster),
        carbon_store=ScheduledCarbonStore(),
        at_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        compatible_destination_ids=set(),
    )
    pauses = [item for item in actions if item.action == ActionType.PAUSE]

    assert [item.details["idle_seconds"] for item in pauses] == [0, 300, 900, 1800]
    assert pauses[2].carbon_grams < pauses[0].carbon_grams
    assert pauses[2].details["pause_duration_seconds"] == 900
