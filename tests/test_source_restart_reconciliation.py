import os
from datetime import datetime, timezone

import pytest

from magellan.artifacts.manager import ArtifactManager
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.journal import MigrationJournal
from magellan.migration.models import (
    MigrationRecord,
    MigrationRole,
    MigrationStatus,
)
from magellan.migration.service import MigrationService
from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition, TaskStatus


class Runtime:
    def __init__(self, registry):
        self.registry = registry
        self.started = 0

    def start(self, task_id):
        self.started += 1
        return self.registry.mark_running(task_id, 54321)


class DestinationHasNoRecord:
    async def status(self, _destination_node_id, _migration_id):
        return None


class Unused:
    pass


@pytest.mark.asyncio
async def test_source_restart_rolls_back_when_destination_confirms_not_seen(
    tmp_path,
) -> None:
    boston = NodeConfig(
        id="boston", name="Boston", vm_name="b", zone="z1",
        internal_ip="10.0.0.1", carbon_region="Boston",
        dataset_file="b.csv", latitude=42, longitude=-71,
    )
    virginia = NodeConfig(
        id="virginia", name="Virginia", vm_name="v", zone="z2",
        internal_ip="10.0.0.2", carbon_region="Virginia",
        dataset_file="v.csv", latitude=37, longitude=-78,
    )
    cluster = ClusterConfig(
        nodes=[boston, virginia],
        reservation_ttl_seconds=30,
        reservation_renew_interval_seconds=5,
    )
    registry = PersistentTaskRegistry(
        definitions=[
            TaskDefinition(
                profile=TaskProfile(
                    task_id="task", workload_type="test",
                    current_node_id="boston", power_kw=0.1,
                    checkpoint_bytes=1,
                ),
                runtime=LocalProcessSpec(module="example.module"),
            )
        ],
        state_root=tmp_path,
        local_node_id="boston",
    )
    registry.mark_running("task", os.getpid())
    original = registry.get_state("task")
    registry.mark_migrating("task", "migration")
    registry.mark_stopped("task")

    journal = MigrationJournal(tmp_path)
    journal.put(
        MigrationRecord(
            migration_id="migration", bid_id="bid", task_id="task",
            source_node_id="boston", destination_node_id="virginia",
            generation=1, migration_at_utc=datetime.now(timezone.utc),
            role=MigrationRole.SOURCE,
            status=MigrationStatus.TRANSFERRING,
            original_state=original,
        )
    )
    runtime = Runtime(registry)
    service = MigrationService(
        local_node=boston, cluster=cluster, registry=registry,
        runtime=runtime, transfer=Unused(), client=DestinationHasNoRecord(),
        broadcaster=Unused(), checkpoint_manager=Unused(),
        artifact_manager=ArtifactManager(registry),
        prefetch_service=Unused(), bid_client=Unused(),
        bid_store=BidStore(30), journal=journal,
        reconciliation_policy=ReconciliationPolicy(
            activation_resolution_timeout_seconds=0.1,
            activation_resolution_poll_seconds=0.01,
        ),
    )

    repaired = await service.reconcile_durable_state()
    state = registry.get_state("task")

    assert repaired == 1
    assert runtime.started == 1
    assert state.owner_node_id == "boston"
    assert state.status == TaskStatus.RUNNING
    assert state.pid == 54321
    assert journal.get("migration").status == MigrationStatus.ROLLED_BACK
