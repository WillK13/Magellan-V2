import os
from types import SimpleNamespace

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
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class NoopAccounting:
    def settle_task(self, *_args, **_kwargs):
        return None


class RecordingPauseService:
    def __init__(self):
        self.calls = []

    async def pause(self, **kwargs):
        self.calls.append(kwargs)
        return None


class Checkpoint:
    def validate(self, _task_id):
        return SimpleNamespace(size_bytes=10, file_count=1)


class NoopPrefetch:
    async def missing_bytes(self, **_kwargs):
        return 0


@pytest.mark.asyncio
async def test_scheduler_executes_selected_pause(monkeypatch, tmp_path) -> None:
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
    cluster = ClusterConfig(nodes=[node])
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
            task_id="scheduler-pause",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=10,
        ),
        runtime=LocalProcessSpec(module="example.module"),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )
    registry.mark_running("scheduler-pause", os.getpid())

    pause_action = ScoredAction(
        action=ActionType.PAUSE,
        source_node_id="boston",
        time_seconds=30,
        carbon_grams=0,
        cost_usd=0,
        details={"idle_seconds": 30},
        normalized_time=0,
        normalized_carbon=0,
        normalized_cost=0,
        score=0,
    )
    continue_action = ScoredAction(
        action=ActionType.CONTINUE,
        source_node_id="boston",
        time_seconds=60,
        carbon_grams=1,
        cost_usd=1,
        normalized_time=1,
        normalized_carbon=1,
        normalized_cost=1,
        score=1,
    )
    decision = DecisionResult(
        selected=pause_action,
        ranked_actions=[pause_action, continue_action],
        reason="Pause has the lowest score",
    )
    monkeypatch.setattr(
        "magellan.daemon.scheduler_service.evaluate_task",
        lambda **_kwargs: decision,
    )

    pause_service = RecordingPauseService()
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
        checkpoint_manager=Checkpoint(),
        prefetch_service=NoopPrefetch(),
        broadcaster=object(),
        pause_service=pause_service,
        accounting_service=NoopAccounting(),
    )

    at = pd.Timestamp("2024-01-01T00:00:00Z")
    await service._evaluate_task("scheduler-pause", at)

    assert len(pause_service.calls) == 1
    assert pause_service.calls[0]["task_id"] == "scheduler-pause"
    assert pause_service.calls[0]["idle_seconds"] == 30
