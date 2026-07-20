from datetime import datetime, timezone

import pytest

from magellan.artifacts.manager import ArtifactManager
from magellan.bidding.models import BidRequest, BidStatus
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.migration.models import MigrationActivationRequest
from magellan.migration.service import MigrationService
from magellan.models.types import ActionType, ScoredAction, TaskProfile
from magellan.runtime.checkpoint import CheckpointManager
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
    TaskStatus,
)


class FailingRuntime:
    def start(self, _task_id):
        raise RuntimeError("restore failed")

    def stop(self, _task_id):
        raise AssertionError("runtime was never started")


class Unused:
    pass


@pytest.mark.asyncio
async def test_destination_activation_rolls_back_on_start_failure(
    tmp_path,
) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="migrate-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(
            module="example.module",
            checkpoint_relative_path="checkpoint/state.json",
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="virginia",
    )

    cluster = ClusterConfig(
        api_port=8040,
        reservation_ttl_seconds=30,
        reservation_renew_interval_seconds=5,
        nodes=[
            NodeConfig(
                id="boston",
                name="Boston",
                vm_name="boston-vm",
                zone="us-east1-c",
                internal_ip="10.0.0.1",
                carbon_region="Boston_24H",
                dataset_file="Boston_24H.csv",
                latitude=42,
                longitude=-71,
            ),
            NodeConfig(
                id="virginia",
                name="Virginia",
                vm_name="virginia-vm",
                zone="northamerica-northeast1-c",
                internal_ip="10.0.0.2",
                carbon_region="France_24H",
                dataset_file="France_24H.csv",
                latitude=37,
                longitude=-78,
            ),
        ],
    )

    bid_store = BidStore(reservation_ttl_seconds=30)
    candidate = ScoredAction(
        action=ActionType.MIGRATE,
        source_node_id="boston",
        destination_node_id="virginia",
        time_seconds=10,
        carbon_grams=1,
        cost_usd=0.1,
        normalized_time=0.1,
        normalized_carbon=0.1,
        normalized_cost=0.1,
        score=0.1,
    )
    bid = BidRequest(
        bid_id="bid-rollback",
        epoch_id="epoch",
        task_id="migrate-task",
        source_node_id="boston",
        destination_node_id="virginia",
        candidate=candidate,
        submitted_at_utc=datetime.now(timezone.utc),
    )
    await bid_store.submit(bid)
    await bid_store.decide(
        bid.bid_id,
        BidStatus.ACCEPTED,
        "accepted",
    )

    migration_id = "migration-rollback"
    incoming = (
        registry.state_root
        / "incoming"
        / migration_id
        / "migrate-task"
        / "checkpoint"
    )
    incoming.mkdir(parents=True)
    (incoming / "state.json").write_text(
        "{}",
        encoding="utf-8",
    )

    artifact_manager = ArtifactManager(registry)
    service = MigrationService(
        local_node=cluster.get_node("virginia"),
        cluster=cluster,
        registry=registry,
        runtime=FailingRuntime(),
        transfer=Unused(),
        client=Unused(),
        broadcaster=Unused(),
        checkpoint_manager=CheckpointManager(registry),
        artifact_manager=artifact_manager,
        prefetch_service=Unused(),
        bid_client=Unused(),
        bid_store=bid_store,
    )

    response = await service.activate_incoming(
        MigrationActivationRequest(
            migration_id=migration_id,
            bid_id=bid.bid_id,
            task_id="migrate-task",
            source_node_id="boston",
            destination_node_id="virginia",
            generation=1,
            migration_at_utc=datetime.now(timezone.utc),
            artifacts=[],
        )
    )

    state = registry.get_state("migrate-task")
    record = await bid_store.get(bid.bid_id)

    assert response.activated is False
    assert state.owner_node_id == "boston"
    assert state.status == TaskStatus.REMOTE
    assert not registry.checkpoint_directory(
        "migrate-task"
    ).exists()
    assert record is not None
    assert record.status == BidStatus.CANCELLED
