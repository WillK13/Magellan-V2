import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from magellan.bidding.models import BidRecord, BidStatus
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


class Checkpoint:
    def validate(self, _task_id):
        return SimpleNamespace(size_bytes=10, file_count=1)


class NoopPrefetch:
    async def missing_bytes(self, **_kwargs):
        return 0


class RecordingBidClient:
    def __init__(self):
        self.calls = 0
        self.submitted = asyncio.Event()

    async def submit_and_wait(self, bid):
        self.calls += 1
        self.submitted.set()
        await asyncio.sleep(0.05)
        now = datetime.now(timezone.utc)
        return BidRecord(
            **bid.model_dump(),
            status=BidStatus.ACCEPTED,
            received_at_utc=now,
            decided_at_utc=now,
            decision_reason="accepted",
            reservation_expires_at_utc=now,
        )


class WinningMigrationService:
    def __init__(self, registry):
        self.registry = registry
        self.calls = 0

    async def migrate(
        self,
        task_id,
        destination_node_id,
        migration_at_utc,
        bid_id,
    ):
        self.calls += 1
        self.registry.mark_remote(
            task_id=task_id,
            owner_node_id=destination_node_id,
            generation=1,
            migration_id=bid_id,
            migration_at_utc=migration_at_utc,
        )
        return True


@pytest.mark.asyncio
async def test_operator_migration_is_idempotent_when_scheduler_wins_race(
    monkeypatch,
    tmp_path,
) -> None:
    boston = NodeConfig(
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
    virginia = NodeConfig(
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
    cluster = ClusterConfig(nodes=[boston, virginia], epoch_seconds=30)
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
            task_id="race-task",
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
    registry.mark_running("race-task", os.getpid())

    migrate_action = ScoredAction(
        action=ActionType.MIGRATE,
        source_node_id="boston",
        destination_node_id="virginia",
        time_seconds=30,
        carbon_grams=1,
        cost_usd=1,
        normalized_time=0,
        normalized_carbon=0,
        normalized_cost=0,
        score=0,
    )
    continue_action = ScoredAction(
        action=ActionType.CONTINUE,
        source_node_id="boston",
        time_seconds=60,
        carbon_grams=2,
        cost_usd=2,
        normalized_time=1,
        normalized_carbon=1,
        normalized_cost=1,
        score=1,
    )
    decision = DecisionResult(
        selected=migrate_action,
        ranked_actions=[migrate_action, continue_action],
        reason="Migrate has the lowest score",
    )
    monkeypatch.setattr(
        "magellan.daemon.scheduler_service.evaluate_task",
        lambda **_kwargs: decision,
    )

    bid_client = RecordingBidClient()
    migration_service = WinningMigrationService(registry)
    service = SchedulerService(
        local_node=boston,
        cluster=cluster,
        policy=policy,
        graph=ClusterGraph(cluster),
        carbon_store=object(),
        clock=object(),
        registry=registry,
        runtime=object(),
        bid_client=bid_client,
        migration_service=migration_service,
        checkpoint_manager=Checkpoint(),
        prefetch_service=NoopPrefetch(),
        broadcaster=object(),
        pause_service=object(),
        accounting_service=NoopAccounting(),
    )

    automatic = asyncio.create_task(
        service._evaluate_task(
            "race-task",
            pd.Timestamp("2024-01-01T00:00:00Z"),
        )
    )
    await bid_client.submitted.wait()

    operator = asyncio.create_task(
        service.request_migration("race-task", "virginia")
    )

    await automatic
    result = await operator

    assert bid_client.calls == 1
    assert migration_service.calls == 1
    assert result["migrated"] is True
    assert result["already_migrated"] is True
    assert result["state"]["owner_node_id"] == "virginia"
