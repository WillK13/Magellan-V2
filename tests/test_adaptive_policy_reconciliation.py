from datetime import datetime, timezone

import pytest

from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.models import OwnershipUpdate
from magellan.models.types import TaskProfile
from magellan.policy.models import AdaptiveTaskPolicyState, WeightVector
from magellan.policy.store import AdaptivePolicyStore
from magellan.reconciliation.models import OwnershipSnapshot
from magellan.reconciliation.service import DistributedReconciliationService
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class Snapshots:
    def __init__(self, policy_state):
        self.policy_state = policy_state

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
                        adaptive_policy=self.policy_state,
                    )
                ],
            )
        ]


class NoopMigrationReconcile:
    async def reconcile_durable_state(self):
        return 0


@pytest.mark.asyncio
async def test_anti_entropy_carries_adaptive_policy_state(tmp_path) -> None:
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
        state_root=tmp_path / "registry",
        local_node_id="boston",
    )
    remote_policy = AdaptiveTaskPolicyState(
        task_id="task",
        baseline_weights=WeightVector(time=0.25, carbon=0.5, cost=0.25),
        effective_weights=WeightVector(time=0.2, carbon=0.55, cost=0.25),
        decision_count=4,
    )
    policy_store = AdaptivePolicyStore(tmp_path / "policy")
    service = DistributedReconciliationService(
        local_node_id="boston",
        policy=ReconciliationPolicy(scan_interval_seconds=1),
        registry=registry,
        client=Snapshots(remote_policy),
        migration_service=NoopMigrationReconcile(),
        adaptive_policy_store=policy_store,
    )

    await service.run_once()

    restored = policy_store.get("task")
    assert restored is not None
    assert restored.decision_count == 4
    assert restored.effective_weights.carbon == 0.55
