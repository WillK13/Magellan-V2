from datetime import datetime, timezone

import pytest

from magellan.config.policy_models import RecoveryPolicy
from magellan.models.types import TaskProfile
from magellan.runtime.checkpoint import CheckpointManager
from magellan.runtime.recovery import FailureRecoveryService
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
    TaskStatus,
)


class FakeRuntime:
    def __init__(self, registry):
        self.registry = registry
        self.starts = 0

    def start(self, task_id):
        self.starts += 1
        return self.registry.mark_running(task_id, pid=4321)


@pytest.mark.asyncio
async def test_failed_task_restarts_from_valid_checkpoint(
    tmp_path,
) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="recover-task",
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
        local_node_id="boston",
    )
    checkpoint = registry.checkpoint_file("recover-task")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")
    registry.mark_failed("recover-task", "crashed", exit_code=2)

    runtime = FakeRuntime(registry)
    service = FailureRecoveryService(
        local_node_id="boston",
        policy=RecoveryPolicy(
            enabled=True,
            max_restart_attempts=2,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
            scan_interval_seconds=1,
        ),
        registry=registry,
        runtime=runtime,
        checkpoint_manager=CheckpointManager(registry),
    )

    now = datetime.now(timezone.utc)
    await service.run_once(now)
    await service.run_once(now)

    state = registry.get_state("recover-task")
    assert runtime.starts == 1
    assert state.status == TaskStatus.RUNNING
    assert state.recovery_attempts == 1
