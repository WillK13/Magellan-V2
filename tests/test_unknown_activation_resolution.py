from datetime import datetime, timezone

import pytest

from magellan.artifacts.manager import ArtifactManager
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.client import ActivationOutcomeUnknownError
from magellan.migration.journal import MigrationJournal
from magellan.migration.models import (
    MigrationRecord,
    MigrationRole,
    MigrationStatus,
)
from magellan.migration.service import MigrationService
from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class Runtime:
    def __init__(self, registry):
        self.registry = registry

    def stop(self, task_id):
        return self.registry.mark_stopped(task_id)

    def start(self, task_id):
        return self.registry.mark_running(task_id, 12345)


class Checkpoint:
    def validate(self, _task_id):
        return type("Summary", (), {"size_bytes": 10, "file_count": 1})()


class Prefetch:
    async def missing_bytes(self, **_kwargs):
        return 0

    async def prefetch(self, **_kwargs):
        return []


class Transfer:
    def send(self, *_args):
        return None


class LostResponseButActivated:
    async def activate(self, _request):
        raise ActivationOutcomeUnknownError("response lost")

    async def status(self, destination_node_id, migration_id):
        return MigrationRecord(
            migration_id=migration_id,
            bid_id="bid",
            task_id="task",
            source_node_id="boston",
            destination_node_id=destination_node_id,
            generation=1,
            migration_at_utc=datetime.now(timezone.utc),
            role=MigrationRole.DESTINATION,
            status=MigrationStatus.ACTIVATED,
            pid=999,
        )


class BidClient:
    async def renew(self, **_kwargs):
        return type("Renewed", (), {"reservation_expires_at_utc": None})()

    async def cancel(self, **_kwargs):
        raise AssertionError("activated migration must not be cancelled")


class Broadcaster:
    def __init__(self):
        self.updates = []

    async def broadcast(self, update):
        self.updates.append(update)


class Unused:
    pass


@pytest.mark.asyncio
async def test_lost_activation_response_is_resolved_from_destination_journal(
    tmp_path,
) -> None:
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
    cluster = ClusterConfig(
        nodes=nodes,
        reservation_ttl_seconds=30,
        reservation_renew_interval_seconds=5,
    )
    registry = PersistentTaskRegistry(
        definitions=[
            TaskDefinition(
                profile=TaskProfile(
                    task_id="task", workload_type="test",
                    current_node_id="boston", power_kw=0.1,
                    checkpoint_bytes=10,
                ),
                runtime=LocalProcessSpec(module="example.module"),
            )
        ],
        state_root=tmp_path,
        local_node_id="boston",
    )
    registry.mark_running("task", 12345)
    broadcaster = Broadcaster()
    service = MigrationService(
        local_node=nodes[0], cluster=cluster, registry=registry,
        runtime=Runtime(registry), transfer=Transfer(),
        client=LostResponseButActivated(), broadcaster=broadcaster,
        checkpoint_manager=Checkpoint(),
        artifact_manager=ArtifactManager(registry),
        prefetch_service=Prefetch(), bid_client=BidClient(),
        bid_store=BidStore(30), journal=MigrationJournal(tmp_path),
        reconciliation_policy=ReconciliationPolicy(
            activation_resolution_timeout_seconds=1,
            activation_resolution_poll_seconds=0.01,
        ),
    )

    migrated = await service.migrate(
        task_id="task",
        destination_node_id="virginia",
        migration_at_utc=datetime.now(timezone.utc),
        bid_id="bid",
    )

    state = registry.get_state("task")
    records = service.list_records()
    assert migrated is True
    assert state.owner_node_id == "virginia"
    assert state.generation == 1
    assert records[0].status == MigrationStatus.ACTIVATED
    assert len(broadcaster.updates) == 1
