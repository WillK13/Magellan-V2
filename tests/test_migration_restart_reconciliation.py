import os
from datetime import datetime, timezone

import pytest

from magellan.artifacts.manager import ArtifactManager
from magellan.bidding.models import BidRequest, BidStatus
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.migration.journal import MigrationJournal
from magellan.migration.models import (
    MigrationRecord,
    MigrationRole,
    MigrationStatus,
)
from magellan.migration.service import MigrationService
from magellan.models.types import ActionType, ScoredAction, TaskProfile
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition, TaskStatus


class Unused:
    pass


class Runtime:
    def stop(self, _task_id):
        raise AssertionError("no process should need stopping")


def cluster():
    nodes = [
        NodeConfig(
            id="boston", name="Boston", vm_name="b", zone="z1",
            internal_ip="10.0.0.1", carbon_region="Boston",
            dataset_file="b.csv", latitude=42, longitude=-71,
        ),
        NodeConfig(
            id="virginia", name="Virginia", vm_name="v", zone="z2",
            internal_ip="10.0.0.2", carbon_region="Virginia",
            dataset_file="v.csv", latitude=37, longitude=-78,
        ),
    ]
    return ClusterConfig(
        nodes=nodes,
        reservation_ttl_seconds=30,
        reservation_renew_interval_seconds=5,
    )


def registry(tmp_path):
    return PersistentTaskRegistry(
        definitions=[
            TaskDefinition(
                profile=TaskProfile(
                    task_id="task", workload_type="test",
                    current_node_id="boston", power_kw=0.1,
                    checkpoint_bytes=1,
                ),
                runtime=LocalProcessSpec(
                    module="example.module",
                    checkpoint_relative_path="checkpoint/state.json",
                ),
            )
        ],
        state_root=tmp_path,
        local_node_id="virginia",
    )


def bid_request():
    return BidRequest(
        bid_id="bid",
        epoch_id="epoch",
        task_id="task",
        source_node_id="boston",
        destination_node_id="virginia",
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=1,
            carbon_grams=1,
            cost_usd=1,
            normalized_time=0,
            normalized_carbon=0,
            normalized_cost=0,
            score=0,
        ),
    )


def service(registry, bid_store, journal):
    cfg = cluster()
    return MigrationService(
        local_node=cfg.get_node("virginia"),
        cluster=cfg,
        registry=registry,
        runtime=Runtime(),
        transfer=Unused(),
        client=Unused(),
        broadcaster=Unused(),
        checkpoint_manager=Unused(),
        artifact_manager=ArtifactManager(registry),
        prefetch_service=Unused(),
        bid_client=Unused(),
        bid_store=bid_store,
        journal=journal,
    )


@pytest.mark.asyncio
async def test_restart_finalizes_committed_destination_activation(tmp_path):
    reg = registry(tmp_path)
    store = BidStore(30, state_file=tmp_path / "control" / "bids.json")
    request = bid_request()
    await store.submit(request)
    await store.decide("bid", BidStatus.ACCEPTED, "accepted")
    await store.begin_activation("bid", "task", "boston", "virginia")

    original = reg.get_state("task")
    reg.claim_local(
        task_id="task",
        generation=1,
        migration_id="migration",
        migration_at_utc=datetime.now(timezone.utc),
        artifact_digests={},
    )
    reg.mark_running("task", os.getpid())
    journal = MigrationJournal(tmp_path)
    journal.put(
        MigrationRecord(
            migration_id="migration", bid_id="bid", task_id="task",
            source_node_id="boston", destination_node_id="virginia",
            generation=1, migration_at_utc=datetime.now(timezone.utc),
            role=MigrationRole.DESTINATION,
            status=MigrationStatus.ACTIVATING,
            original_state=original,
        )
    )

    repaired = await service(reg, store, journal).reconcile_durable_state()

    assert repaired == 1
    assert journal.get("migration").status == MigrationStatus.ACTIVATED
    assert (await store.get("bid")).status == BidStatus.CONSUMED
    assert reg.get_state("task").status == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_restart_rolls_back_uncommitted_destination_activation(tmp_path):
    reg = registry(tmp_path)
    store = BidStore(30, state_file=tmp_path / "control" / "bids.json")
    request = bid_request()
    await store.submit(request)
    await store.decide("bid", BidStatus.ACCEPTED, "accepted")
    await store.begin_activation("bid", "task", "boston", "virginia")
    original = reg.get_state("task")

    checkpoint = reg.checkpoint_directory("task")
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text("incoming", encoding="utf-8")
    backup = checkpoint.with_name("checkpoint.rollback-migration")
    backup.mkdir(parents=True)
    (backup / "state.json").write_text("original", encoding="utf-8")

    journal = MigrationJournal(tmp_path)
    journal.put(
        MigrationRecord(
            migration_id="migration", bid_id="bid", task_id="task",
            source_node_id="boston", destination_node_id="virginia",
            generation=1, migration_at_utc=datetime.now(timezone.utc),
            role=MigrationRole.DESTINATION,
            status=MigrationStatus.ACTIVATING,
            original_state=original,
        )
    )

    repaired = await service(reg, store, journal).reconcile_durable_state()

    assert repaired == 1
    assert journal.get("migration").status == MigrationStatus.ROLLED_BACK
    assert (await store.get("bid")).status == BidStatus.CANCELLED
    assert reg.get_state("task").owner_node_id == "boston"
    assert (checkpoint / "state.json").read_text() == "original"
