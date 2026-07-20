import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from magellan.artifacts.manager import ArtifactManager
from magellan.config.policy_models import ClockPolicy, PausePolicy
from magellan.models.types import TaskProfile
from magellan.runtime.clock import MagellanClock
from magellan.runtime.completion import CompletionManager
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.runtime.pause import PauseService
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
    TaskStatus,
)


def checkpoint_value(path) -> int:
    return int(json.loads(path.read_text())["value"])


def wait_for_value(path, minimum: int, timeout: float = 3.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            value = checkpoint_value(path)
            if value >= minimum:
                return value
        time.sleep(0.05)
    raise AssertionError(f"Counter did not reach {minimum}")


@pytest.mark.asyncio
async def test_pause_persists_and_resumes_after_runtime_restart(
    tmp_path,
) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="pause-task",
            workload_type="counter",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(
            module="magellan.workloads.counter",
            arguments=[
                "--checkpoint-file",
                "{checkpoint_file}",
                "--interval-seconds",
                "0.05",
                "--progress-file",
                "{progress_file}",
            ],
            checkpoint_relative_path="checkpoint/counter.json",
            progress_relative_path="runtime/progress.json",
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path / "state",
        local_node_id="boston",
    )
    manager = ArtifactManager(registry)
    completion = CompletionManager(registry)

    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id="boston",
        repository_root=".",
        artifact_manager=manager,
        completion_manager=completion,
    )
    runtime.start("pause-task")
    checkpoint = registry.checkpoint_file("pause-task")
    wait_for_value(checkpoint, 3)

    paused_at = datetime.now(timezone.utc)
    runtime.pause(
        "pause-task",
        paused_at,
        paused_at + timedelta(seconds=30),
        datetime.now(timezone.utc) + timedelta(seconds=0.15),
        "test pause",
    )
    paused_value = checkpoint_value(checkpoint)
    time.sleep(0.12)
    assert checkpoint_value(checkpoint) == paused_value
    assert registry.get_state("pause-task").status == TaskStatus.PAUSED

    # New runtime object simulates a daemon restart. It has no Popen object,
    # but the persisted PID and wall-clock deadline are sufficient.
    restarted_runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id="boston",
        repository_root=".",
        artifact_manager=manager,
        completion_manager=completion,
    )
    pause_service = PauseService(
        local_node_id="boston",
        policy=PausePolicy(
            pause_seconds=0,
            idle_seconds=30,
            resume_seconds=0,
            max_pause_window_seconds=60,
            scan_interval_seconds=0.01,
        ),
        clock=MagellanClock(ClockPolicy(mode="wall")),
        registry=registry,
        runtime=restarted_runtime,
    )

    await asyncio.sleep(0.08)
    assert await pause_service.run_once() == 1
    assert registry.get_state("pause-task").status == TaskStatus.RUNNING
    wait_for_value(checkpoint, paused_value + 2)

    restarted_runtime.stop("pause-task")
