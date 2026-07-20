import pandas as pd

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


class ConstantCarbonStore:
    def average(self, _node_id, _start, _duration):
        return 100.0


def node(node_id, ip, price):
    return NodeConfig(
        id=node_id,
        name=node_id,
        vm_name=node_id,
        zone="test-zone",
        internal_ip=ip,
        carbon_region=node_id,
        dataset_file="unused.csv",
        latitude=40 if node_id == "boston" else 38,
        longitude=-71 if node_id == "boston" else -78,
        compute_price_usd_per_hour=price,
        egress_price_usd_per_gb=0.1,
    )


def test_accumulated_cost_prunes_migration() -> None:
    cluster = ClusterConfig(
        nodes=[
            node("boston", "10.0.0.1", 0.1),
            node("virginia", "10.0.0.2", 1.0),
        ]
    )
    policy = ScoringPolicy(
        horizon_seconds=3600,
        weights=ObjectiveWeights(time=1, carbon=1, cost=1),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=60,
            resume_seconds=0,
            max_pause_window_seconds=7200,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        clock=ClockPolicy(mode="wall"),
    )
    task = TaskProfile(
        task_id="budget-task",
        workload_type="test",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=1_000_000_000,
        accumulated_cost_usd=9.9,
        cost_cap_usd=10.0,
        estimated_remaining_seconds=3600,
    )

    actions = build_raw_actions(
        task=task,
        cluster=cluster,
        policy=policy,
        graph=ClusterGraph(cluster),
        carbon_store=ConstantCarbonStore(),
        at_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
    )

    assert all(action.action != ActionType.MIGRATE for action in actions)
    assert {action.action for action in actions} == {
        ActionType.CONTINUE,
        ActionType.PAUSE,
    }
