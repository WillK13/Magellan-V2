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


def test_discovers_latest_native_bssn_checkpoint_generation(tmp_path) -> None:
    task = definition().model_copy(deep=True)
    task.runtime.dendro_options.checkpoint_discovery = (
        DendroCheckpointDiscoverySpec(
            native_bssn_prefix="bssn_cp",
            expected_file_count=6,
            expected_rank_count=2,
            stability_seconds=0,
        )
    )
    registry = PersistentTaskRegistry(
        definitions=[task],
        state_root=tmp_path,
        local_node_id="boston",
    )
    directory = registry.checkpoint_directory("real-dendro")
    directory.mkdir(parents=True)

    def write_generation(generation: int, step: int) -> None:
        (directory / f"bssn_cp_{generation}_step.cp").write_text(
            json.dumps({"DENDRO_TS_STEP_CURRENT": step}),
            encoding="utf-8",
        )
        for rank in range(2):
            (directory / f"bssn_cp_{generation}_{rank}.var").write_bytes(
                f"var-{generation}-{rank}".encode()
            )
            (
                directory / f"bssn_cp_{generation}_octree_{rank}.oct"
            ).write_bytes(f"oct-{generation}-{rank}".encode())
        (
            directory
            / f"bssn_cp_aeh_solver_checkpt-cp{generation}.json"
        ).write_text("{}", encoding="utf-8")

    write_generation(0, 77)
    write_generation(1, 79)
    summary = CheckpointManager(registry).validate("real-dendro")
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert summary.checkpoint_step == 79
    assert summary.file_count == 6
    assert manifest["checkpoint_generation"] == 1
    assert manifest["checkpoint_step"] == 79
    assert {item.get("rank") for item in manifest["files"]} == {0, 1, None}
    assert all("sha256" in item for item in manifest["files"])
    assert all("_0_" not in item["path"] for item in manifest["files"] if item["path"].endswith((".var", ".oct")))


def test_real_bssn_launcher_renders_checkpoint_paths(tmp_path) -> None:
    from scripts.run_real_dendro_bssn import render_runtime_parameters

    template = tmp_path / "template.toml"
    template.write_text(
        "BSSN_RESTORE_SOLVER = 0\n"
        'BSSN_CHKPT_FILE_PREFIX = "cp/bssn_cp"\n'
        'BSSN_VTU_FILE_PREFIX = "vtu/bssn_gr"\n'
        'BSSN_PROFILE_FILE_PREFIX = "dat/dgr"\n',
        encoding="utf-8",
    )
    checkpoint = tmp_path / "task" / "checkpoint"
    output = tmp_path / "task" / "output"
    rendered = render_runtime_parameters(
        template_path=template,
        output_path=tmp_path / "task" / "runtime.toml",
        checkpoint_directory=checkpoint,
        output_directory=output,
        resume=True,
    )
    text = rendered.read_text(encoding="utf-8")
    assert "BSSN_RESTORE_SOLVER = 1" in text
    assert str(checkpoint / "bssn_cp") in text
    assert str(output / "vtu" / "bssn_gr") in text
    assert str(output / "dat" / "dgr") in text
