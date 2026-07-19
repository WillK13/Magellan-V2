from __future__ import annotations

import pytest

from magellan.config.policy_models import ReconciliationPolicy
from magellan.reconciliation.service import DistributedReconciliationService
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.submission.catalog import TaskCatalogStore
from magellan.submission.models import (
    TaskDefinitionSubmission,
    TaskRunSubmission,
    TaskTemplateProfile,
)
from magellan.state.task_models import LocalProcessSpec


class NoOwnershipSnapshots:
    async def fetch_all(self):
        return []


class NoopMigrationReconcile:
    async def reconcile_durable_state(self):
        return 0


class CatalogSnapshots:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    async def fetch_all(self):
        return [self.snapshot]


@pytest.mark.asyncio
async def test_reconciliation_installs_remote_definition_and_run(tmp_path) -> None:
    source = TaskCatalogStore(tmp_path / "source", "boston")
    definition, _ = source.register_definition(
        TaskDefinitionSubmission(
            definition_id="dynamic-counter",
            profile=TaskTemplateProfile(
                workload_type="counter",
                power_kw=0.1,
                checkpoint_bytes=1,
            ),
            runtime=LocalProcessSpec(module="magellan.workloads.counter"),
        )
    )
    run, _ = source.create_run(
        TaskRunSubmission(
            definition_id=definition.definition_id,
            idempotency_key="replicate-me",
            initial_owner_node_id="boston",
        ),
        owner_node_id="boston",
    )

    destination = TaskCatalogStore(tmp_path / "destination", "virginia")
    registry = PersistentTaskRegistry(
        definitions=[],
        state_root=tmp_path / "destination",
        local_node_id="virginia",
    )
    service = DistributedReconciliationService(
        local_node_id="virginia",
        policy=ReconciliationPolicy(scan_interval_seconds=1),
        registry=registry,
        client=NoOwnershipSnapshots(),
        migration_service=NoopMigrationReconcile(),
        catalog=destination,
        catalog_client=CatalogSnapshots(source.snapshot()),
    )

    applied = await service.run_once()

    assert applied == 2
    assert destination.get_definition("dynamic-counter").digest == definition.digest
    assert destination.get_run(run.run_id) == run
    assert registry.get_state(run.run_id).owner_node_id == "boston"
    assert registry.get_state(run.run_id).status.value == "remote"
