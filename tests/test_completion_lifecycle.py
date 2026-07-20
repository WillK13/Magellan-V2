import json
from datetime import datetime, timezone

from magellan.artifacts.manager import ArtifactManager
from magellan.models.types import TaskProfile
from magellan.runtime.completion import CompletionManager
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
    TaskStatus,
)


def completion_registry(tmp_path):
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="complete-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(
            module="example.module",
            completion_relative_path="runtime/completion.json",
            output_relative_directory="output",
        ),
    )

    return PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )


def write_completion_files(registry) -> None:
    output = registry.output_directory("complete-task")
    assert output is not None
    output.mkdir(parents=True)
    (output / "result.txt").write_text(
        "finished",
        encoding="utf-8",
    )

    marker = registry.completion_file("complete-task")
    assert marker is not None
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "format_version": 1,
                "task_id": "complete-task",
                "success": True,
                "completed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "details": {"steps": 10},
            }
        ),
        encoding="utf-8",
    )


def test_completion_manager_publishes_manifest(tmp_path) -> None:
    registry = completion_registry(tmp_path)
    write_completion_files(registry)

    manager = CompletionManager(registry)
    manifest = manager.finalize(
        "complete-task",
        exit_code=0,
    )

    state = registry.get_state("complete-task")

    assert state.status == TaskStatus.COMPLETED
    assert state.final_output_bytes == len(b"finished")
    assert state.final_output_manifest_sha256 is not None
    assert manifest.files[0].path == "result.txt"
    assert registry.count_owned("boston") == 0


def test_runtime_reconcile_classifies_natural_exit(
    tmp_path,
    monkeypatch,
) -> None:
    registry = completion_registry(tmp_path)
    write_completion_files(registry)
    registry.mark_running("complete-task", pid=999999)

    completion_manager = CompletionManager(registry)
    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id="boston",
        repository_root=tmp_path,
        artifact_manager=ArtifactManager(registry),
        completion_manager=completion_manager,
    )

    monkeypatch.setattr(
        "magellan.runtime.local_process.pid_is_alive",
        lambda _pid: False,
    )

    events = runtime.reconcile()

    assert len(events) == 1
    assert events[0].status == TaskStatus.COMPLETED
    assert registry.get_state(
        "complete-task"
    ).status == TaskStatus.COMPLETED
