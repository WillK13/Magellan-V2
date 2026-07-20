import os

import pandas as pd
import pytest

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import (
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
)
from magellan.daemon.scheduler_service import SchedulerService
from magellan.graph.topology import ClusterGraph
from magellan.models.types import (
    ActionType,
    DecisionResult,
    ScoredAction,
    TaskProfile,
)
from magellan.runtime.checkpoint import CheckpointValidationError
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class NoopAccounting:
    def settle_task(self, *_args, **_kwargs):
        return None


class MissingCheckpoint:
    def validate(self, _task_id):
        raise CheckpointValidationError("first checkpoint not written")


class PrefetchMustNotRun:
    async def missing_bytes(self, **_kwargs):
        raise AssertionError("prefetch must not run without a checkpoint")


@pytest.mark.asyncio
async def test_scheduler_still_evaluates_local_actions_without_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    node = NodeConfig(
        id="boston",
        name="Boston",
        vm_name="boston",
        zone="us-east1-c",
        internal_ip="10.0.0.1",
        carbon_region="Boston",
        dataset_file="unused.csv",
        latitude=42,
        longitude=-71,
    )
    peer = NodeConfig(
        id="virginia",
        name="Virginia",
        vm_name="virginia",
        zone="us-east4-c",
        internal_ip="10.0.0.2",
        carbon_region="Virginia",
        dataset_file="unused.csv",
        latitude=37,
        longitude=-78,
    )
    cluster = ClusterConfig(nodes=[node, peer])
    policy = ScoringPolicy(
        horizon_seconds=60,
        weights=ObjectiveWeights(time=1, carbon=1, cost=1),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=30,
            resume_seconds=0,
            max_pause_window_seconds=120,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        clock=ClockPolicy(mode="wall"),
    )
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="no-checkpoint-yet",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=4096,
        ),
        runtime=LocalProcessSpec(module="example.module"),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )
    registry.mark_running("no-checkpoint-yet", os.getpid())

    continue_action = ScoredAction(
        action=ActionType.CONTINUE,
        source_node_id="boston",
        time_seconds=60,
        carbon_grams=1,
        cost_usd=1,
        normalized_time=0,
        normalized_carbon=0,
        normalized_cost=0,
        score=0,
    )
    decision = DecisionResult(
        selected=continue_action,
        ranked_actions=[continue_action],
        reason="Continue has the lowest score",
    )
    captured = {}

    def fake_evaluate_task(**kwargs):
        captured.update(kwargs)
        return decision

    monkeypatch.setattr(
        "magellan.daemon.scheduler_service.evaluate_task",
        fake_evaluate_task,
    )

    service = SchedulerService(
        local_node=node,
        cluster=cluster,
        policy=policy,
        graph=ClusterGraph(cluster),
        carbon_store=object(),
        clock=object(),
        registry=registry,
        runtime=object(),
        bid_client=object(),
        migration_service=object(),
        checkpoint_manager=MissingCheckpoint(),
        prefetch_service=PrefetchMustNotRun(),
        broadcaster=object(),
        pause_service=object(),
        accounting_service=NoopAccounting(),
    )

    at = pd.Timestamp("2024-01-01T00:00:00Z")
    await service._evaluate_task("no-checkpoint-yet", at)

    assert captured["compatible_destination_ids"] == set()
    assert captured["static_data_bytes_by_destination"] == {}
    assert captured["task"].checkpoint_bytes == 4096
