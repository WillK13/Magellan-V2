from datetime import datetime, timezone

from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


def test_accounting_snapshot_follows_migration_ownership(tmp_path) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="handoff-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(module="example.module"),
    )
    source = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path / "source",
        local_node_id="boston",
    )
    destination = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path / "destination",
        local_node_id="virginia",
    )

    source.record_accounting(
        "handoff-task",
        runtime_seconds=100,
        migration_seconds=5,
        compute_cost_usd=2,
        transfer_cost_usd=0.5,
        compute_carbon_grams=30,
        transfer_carbon_grams=4,
        estimated_remaining_seconds=900,
        progress_completed_units=10,
        progress_total_units=100,
        progress_fraction=0.1,
    )
    snapshot = source.accounting_snapshot("handoff-task")

    destination.claim_local(
        task_id="handoff-task",
        generation=1,
        migration_id="migration-1",
        artifact_digests={},
        migration_at_utc=datetime.now(timezone.utc),
        accounting=snapshot,
    )

    state = destination.get_state("handoff-task")
    assert state.accumulated_runtime_seconds == 100
    assert state.accumulated_migration_seconds == 5
    assert state.accumulated_cost_usd == 2.5
    assert state.accumulated_carbon_grams == 34
    assert state.estimated_remaining_seconds == 900
    assert state.progress_fraction == 0.1
