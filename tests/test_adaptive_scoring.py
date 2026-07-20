from __future__ import annotations

from datetime import datetime, timezone

from magellan.config.models import ClusterConfig, NetworkEdgeConfig, NodeConfig
from magellan.config.policy_models import (
    AdaptivePolicy,
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
)
from magellan.graph.topology import ClusterGraph
from magellan.models.types import ActionType, TaskProfile
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task


class Carbon:
    def average(self, node_id, *_args):
        return 200.0 if node_id == "boston" else 20.0


def cluster() -> ClusterConfig:
    return ClusterConfig(
        nodes=[
            NodeConfig(
                id="boston",
                name="Boston",
                vm_name="boston",
                zone="a",
                internal_ip="10.0.0.1",
                carbon_region="Boston",
                dataset_file="unused",
                latitude=42,
                longitude=-71,
                egress_price_usd_per_gb=0.1,
            ),
            NodeConfig(
                id="virginia",
                name="Virginia",
                vm_name="virginia",
                zone="b",
                internal_ip="10.0.0.2",
                carbon_region="Virginia",
                dataset_file="unused",
                latitude=37,
                longitude=-78,
                compute_price_usd_per_hour=0.05,
            ),
        ],
        edges=[
            NetworkEdgeConfig(
                source_node_id="boston",
                destination_node_id="virginia",
                bandwidth_mbps=100,
                latency_ms=10,
            )
        ],
    )


def policy() -> ScoringPolicy:
    return ScoringPolicy(
        horizon_seconds=3600,
        weights=ObjectiveWeights(time=0.25, carbon=0.5, cost=0.25),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=300,
            resume_seconds=0,
            max_pause_window_seconds=3600,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        adaptive=AdaptivePolicy(),
        clock=ClockPolicy(mode="wall"),
    )


def test_evaluate_task_records_explainable_adaptive_metadata(tmp_path) -> None:
    scoring_policy = policy()
    adaptive = AdaptivePolicyService(
        scoring_policy.adaptive,
        scoring_policy.weights,
        AdaptivePolicyStore(tmp_path),
    )
    task = TaskProfile(
        task_id="task",
        workload_type="counter",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=100,
        estimated_remaining_seconds=3600,
        accumulated_cost_usd=9,
        cost_cap_usd=10,
    )

    result = evaluate_task(
        task,
        cluster(),
        scoring_policy,
        ClusterGraph(cluster()),
        Carbon(),
        datetime.now(timezone.utc),
        adaptive_service=adaptive,
        telemetry_confidence=1.0,
    )

    assert result.policy_metadata["decision_count"] == 1
    assert result.policy_metadata["effective_weights"]["cost"] > 0.25
    assert result.policy_metadata["effective_weights"]["carbon"] > 0.5
    assert result.policy_metadata["normalization_bounds"]["time_max"] > 0


def test_hard_cost_cap_prunes_migration_before_adaptive_scoring(tmp_path) -> None:
    scoring_policy = policy()
    adaptive = AdaptivePolicyService(
        scoring_policy.adaptive,
        scoring_policy.weights,
        AdaptivePolicyStore(tmp_path),
    )
    task = TaskProfile(
        task_id="capped",
        workload_type="counter",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=1_000_000_000,
        estimated_remaining_seconds=3600,
        accumulated_cost_usd=0.099,
        cost_cap_usd=0.1,
    )

    result = evaluate_task(
        task,
        cluster(),
        scoring_policy,
        ClusterGraph(cluster()),
        Carbon(),
        datetime.now(timezone.utc),
        adaptive_service=adaptive,
        telemetry_confidence=1.0,
    )

    assert all(
        action.action != ActionType.MIGRATE
        for action in result.ranked_actions
    )
    assert (
        result.policy_metadata["hard_constraints"][
            "cost_cap_pruned_migrations"
        ]
        == 1.0
    )
