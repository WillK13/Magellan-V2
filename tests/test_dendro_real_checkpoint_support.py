from __future__ import annotations

import json

from magellan.models.types import TaskProfile
from magellan.runtime.checkpoint import CheckpointManager
from magellan.runtime.dendro import DendroProgressSynchronizer
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import (
    DendroCheckpointDiscoverySpec,
    DendroProgressSpec,
    DendroRuntimeOptions,
    LocalProcessSpec,
    TaskDefinition,
)


def definition() -> TaskDefinition:
    return TaskDefinition(
        profile=TaskProfile(
            task_id="real-dendro",
            workload_type="dendro-gr",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
        ),
        runtime=LocalProcessSpec(
            adapter="dendro",
            command=["mpirun", "-np", "2", "./bssnSolver"],
            resume_arguments=["--restore-step", "{checkpoint_step}"],
            checkpoint_relative_path="checkpoint/state.marker",
            checkpoint_manifest_relative_path="manifest.json",
            progress_relative_path="runtime/progress.json",
            dendro_options=DendroRuntimeOptions(
                checkpoint_discovery=DendroCheckpointDiscoverySpec(
                    file_globs=["*.dat"],
                    step_regex=r"step_(?P<step>\d+)",
                    rank_regex=r"rank_(?P<rank>\d+)",
                    expected_file_count=2,
                    expected_rank_count=2,
                    stability_seconds=0,
                ),
                progress=DendroProgressSpec(
                    step_regex=r"BSSN step=(?P<step>\d+)",
                    total_steps=100,
                ),
            ),
        ),
    )


def test_discovers_latest_complete_rank_checkpoint(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[definition()],
        state_root=tmp_path,
        local_node_id="boston",
    )
    directory = registry.checkpoint_directory("real-dendro")
    directory.mkdir(parents=True)
    for name in [
        "checkpoint_step_20_rank_0.dat",
        "checkpoint_step_20_rank_1.dat",
        "checkpoint_step_30_rank_0.dat",
    ]:
        (directory / name).write_text(name, encoding="utf-8")

    summary = CheckpointManager(registry).validate("real-dendro")
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert summary.checkpoint_step == 20
    assert summary.file_count == 2
    assert manifest["checkpoint_step"] == 20
    assert {item["rank"] for item in manifest["files"]} == {0, 1}
    assert all("sha256" in item for item in manifest["files"])


def test_dendro_log_parser_writes_standard_progress(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[definition()],
        state_root=tmp_path,
        local_node_id="boston",
    )
    log = registry.task_directory("real-dendro") / "logs/process.log"
    log.parent.mkdir(parents=True)
    log.write_text("BSSN step=10\nBSSN step=25\n", encoding="utf-8")

    assert DendroProgressSynchronizer(registry).refresh("real-dendro")
    payload = json.loads(
        registry.progress_file("real-dendro").read_text(encoding="utf-8")
    )

    assert payload["completed_units"] == 25
    assert payload["total_units"] == 100
    assert payload["details"]["source"] == "dendro_log_parser"


def test_dendro_clean_exit_can_synthesize_completion_marker(tmp_path) -> None:
    import sys
    import time

    from magellan.artifacts.manager import ArtifactManager
    from magellan.runtime.completion import CompletionManager
    from magellan.runtime.local_process import LocalProcessRuntime
    from magellan.state.task_models import DendroCompletionSpec, TaskStatus

    script = tmp_path / "solver.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)\n"
        "print('DENDRO FINISHED', flush=True)\n",
        encoding="utf-8",
    )
    task = TaskDefinition(
        profile=TaskProfile(
            task_id="dendro-completion",
            workload_type="dendro-gr",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
        ),
        runtime=LocalProcessSpec(
            adapter="dendro",
            command=[sys.executable, str(script)],
            arguments=["{output_directory}"],
            checkpoint_relative_path="checkpoint/state.marker",
            completion_relative_path="runtime/completion.json",
            output_relative_directory="output",
            dendro_options=DendroRuntimeOptions(
                completion=DendroCompletionSpec(
                    success_regex=r"DENDRO FINISHED",
                )
            ),
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[task],
        state_root=tmp_path / "state",
        local_node_id="boston",
    )
    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id="boston",
        repository_root=tmp_path,
        artifact_manager=ArtifactManager(registry),
        completion_manager=CompletionManager(registry),
    )
    runtime.start("dendro-completion")

    deadline = time.monotonic() + 5
    events = []
    while time.monotonic() < deadline and not events:
        events = runtime.reconcile()
        time.sleep(0.05)

    assert events
    assert events[0].status == TaskStatus.COMPLETED
    state = registry.get_state("dendro-completion")
    assert state.status == TaskStatus.COMPLETED
    marker = registry.completion_file("dendro-completion")
    assert marker is not None and marker.is_file()
