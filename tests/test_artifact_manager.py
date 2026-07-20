from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.models import (
    StaticArtifactSpec,
)
from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
)


def test_artifact_is_cached_and_staged(
    tmp_path,
) -> None:
    source = tmp_path / "source-dataset"
    source.mkdir()
    (source / "train.txt").write_text(
        "Magellan artifact test",
        encoding="utf-8",
    )

    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="artifact-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
        ),
        runtime=LocalProcessSpec(
            module="example.module",
        ),
        artifacts=[
            StaticArtifactSpec(
                artifact_id="dataset",
                kind="dataset",
                source_directory=str(source),
                target_relative_directory="dataset",
            )
        ],
    )

    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path / "state",
        local_node_id="boston",
    )

    manager = ArtifactManager(registry)

    bindings = manager.ensure_task_artifacts(
        "artifact-task"
    )

    assert len(bindings) == 1
    assert bindings[0].size_bytes > 0

    staged = (
        registry.artifacts_directory(
            "artifact-task"
        )
        / "dataset"
        / "train.txt"
    )

    assert staged.read_text(
        encoding="utf-8"
    ) == "Magellan artifact test"
