import json

from magellan.models.types import TaskProfile
from magellan.runtime.checkpoint import CheckpointManager
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
)


def test_manifest_checkpoint_validates(tmp_path) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="manifest-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
        ),
        runtime=LocalProcessSpec(
            module="example.module",
            checkpoint_relative_path=(
                "checkpoint/complete.json"
            ),
            checkpoint_manifest_relative_path=(
                "complete.json"
            ),
        ),
    )

    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )

    checkpoint_directory = (
        registry.checkpoint_directory(
            "manifest-task"
        )
    )
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = checkpoint_directory / "weights.bin"
    payload.write_bytes(b"magellan")

    manifest = {
        "format_version": 1,
        "files": [
            {
                "path": "weights.bin",
                "size_bytes": payload.stat().st_size,
            }
        ],
    }

    (
        checkpoint_directory / "complete.json"
    ).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    manager = CheckpointManager(registry)
    summary = manager.validate("manifest-task")

    assert summary.file_count == 1
    assert summary.size_bytes > 0
