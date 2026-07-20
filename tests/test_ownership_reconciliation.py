from datetime import datetime, timezone

import pytest

from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.models import OwnershipUpdate
from magellan.models.types import TaskProfile
from magellan.reconciliation.models import OwnershipSnapshot
from magellan.reconciliation.service import DistributedReconciliationService
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class Snapshots:
    async def fetch_all(self):
        return [
            OwnershipSnapshot(
                reporting_node_id="virginia",
                updates=[
                    OwnershipUpdate(
                        task_id="task",
                        owner_node_id="virginia",
                        generation=1,
                        last_migration_id="migration-1",
                        migration_at_utc=datetime.now(timezone.utc),
                    )
                ],
            )
        ]


class NoopMigrationReconcile:
    async def reconcile_durable_state(self):
        return 0


@pytest.mark.asyncio
async def test_anti_entropy_repairs_missed_ownership_broadcast(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[
            TaskDefinition(
                profile=TaskProfile(
                    task_id="task",
                    workload_type="test",
                    current_node_id="boston",
                    power_kw=0.1,
                    checkpoint_bytes=1,
                ),
                runtime=LocalProcessSpec(module="example.module"),
            )
        ],
        state_root=tmp_path,
        local_node_id="boston",
    )
    service = DistributedReconciliationService(
        local_node_id="boston",
        policy=ReconciliationPolicy(scan_interval_seconds=1),
        registry=registry,
        client=Snapshots(),
        migration_service=NoopMigrationReconcile(),
    )

    applied = await service.run_once()
    state = registry.get_state("task")

    assert applied == 1
    assert state.owner_node_id == "virginia"
    assert state.generation == 1
    assert state.last_migration_id == "migration-1"


def test_equal_generation_conflict_does_not_replace_owner(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[
            TaskDefinition(
                profile=TaskProfile(
                    task_id="task",
                    workload_type="test",
                    current_node_id="boston",
                    power_kw=0.1,
                    checkpoint_bytes=1,
                ),
                runtime=LocalProcessSpec(module="example.module"),
            )
        ],
        state_root=tmp_path,
        local_node_id="boston",
    )

    applied = registry.apply_ownership(
        task_id="task",
        owner_node_id="virginia",
        generation=0,
    )

    assert applied is False
    assert registry.get_state("task").owner_node_id == "boston"
