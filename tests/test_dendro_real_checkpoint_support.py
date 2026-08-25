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


def _write_native_generation(
    directory,
    *,
    generation: int,
    step: int,
    complete: bool = True,
) -> None:
    (directory / f"bssn_cp_{generation}_step.cp").write_text(
        json.dumps({"DENDRO_TS_STEP_CURRENT": step}),
        encoding="utf-8",
    )
    for rank in range(2):
        (directory / f"bssn_cp_{generation}_{rank}.var").write_bytes(
            f"var-{generation}-{rank}".encode()
        )
        if complete or rank == 0:
            (
                directory / f"bssn_cp_{generation}_octree_{rank}.oct"
            ).write_bytes(f"oct-{generation}-{rank}".encode())
    (
        directory / f"bssn_cp_aeh_solver_checkpt-cp{generation}.json"
    ).write_text("{}", encoding="utf-8")


def _native_progress_definition() -> TaskDefinition:
    task = definition().model_copy(deep=True)
    task.runtime.dendro_options.checkpoint_discovery = (
        DendroCheckpointDiscoverySpec(
            native_bssn_prefix="bssn_cp",
            expected_file_count=6,
            expected_rank_count=2,
            stability_seconds=0,
        )
    )
    task.runtime.dendro_options.progress.step_regex = (
        r"(?:checkpoint at step|step)[^\r\n0-9]*(?P<step>\d+)"
    )
    return task

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


def test_dendro_progress_uses_latest_complete_native_checkpoint(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[_native_progress_definition()],
        state_root=tmp_path,
        local_node_id="boston",
    )
    log = registry.task_directory("real-dendro") / "logs/process.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "[ETS] : Timestep 0 - starting\n"
        "Current Step: 0\tCurrent time: 0\n",
        encoding="utf-8",
    )
    directory = registry.checkpoint_directory("real-dendro")
    directory.mkdir(parents=True, exist_ok=True)
    # BSSN rotates cp0/cp1; generation number is not progress order.
    _write_native_generation(directory, generation=0, step=5)
    _write_native_generation(directory, generation=1, step=3)

    assert DendroProgressSynchronizer(registry).refresh("real-dendro")
    payload = json.loads(
        registry.progress_file("real-dendro").read_text(encoding="utf-8")
    )

    assert payload["completed_units"] == 5
    assert payload["details"]["source"] == "dendro_native_checkpoint"
    assert payload["details"]["observed_log_step"] == 0
    assert payload["details"]["observed_checkpoint_step"] == 5
    assert payload["details"]["checkpoint_generation"] == 0


def test_dendro_progress_never_regresses_to_older_checkpoint(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[_native_progress_definition()],
        state_root=tmp_path,
        local_node_id="boston",
    )
    log = registry.task_directory("real-dendro") / "logs/process.log"
    log.parent.mkdir(parents=True)
    log.write_text("Current Step: 0\n", encoding="utf-8")
    directory = registry.checkpoint_directory("real-dendro")
    directory.mkdir(parents=True, exist_ok=True)
    _write_native_generation(directory, generation=0, step=5)

    synchronizer = DendroProgressSynchronizer(registry)
    assert synchronizer.refresh("real-dendro")
    progress_path = registry.progress_file("real-dendro")
    first = progress_path.read_text(encoding="utf-8")

    for path in directory.glob("bssn_cp_0_*"):
        path.unlink()
    (directory / "bssn_cp_aeh_solver_checkpt-cp0.json").unlink()
    _write_native_generation(directory, generation=1, step=3)

    assert not synchronizer.refresh("real-dendro")
    assert progress_path.read_text(encoding="utf-8") == first


def test_dendro_progress_ignores_incomplete_newer_checkpoint(tmp_path) -> None:
    registry = PersistentTaskRegistry(
        definitions=[_native_progress_definition()],
        state_root=tmp_path,
        local_node_id="boston",
    )
    log = registry.task_directory("real-dendro") / "logs/process.log"
    log.parent.mkdir(parents=True)
    log.write_text("Current Step: 0\n", encoding="utf-8")
    directory = registry.checkpoint_directory("real-dendro")
    directory.mkdir(parents=True, exist_ok=True)
    _write_native_generation(directory, generation=0, step=3)
    _write_native_generation(
        directory, generation=1, step=7, complete=False
    )

    assert DendroProgressSynchronizer(registry).refresh("real-dendro")
    payload = json.loads(
        registry.progress_file("real-dendro").read_text(encoding="utf-8")
    )
    assert payload["completed_units"] == 3
    assert payload["details"]["checkpoint_generation"] == 0


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
        'BSSN_PROFILE_FILE_PREFIX = "dat/dgr"\n'
        'AEH_SAVE_DIR = "aeh"\n',
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
    assert str(output / "aeh") in text
    assert (output / "aeh").is_dir()


def test_dendro_log_parser_does_not_cross_lines_into_timestamp(tmp_path) -> None:
    task = definition().model_copy(deep=True)
    task.runtime.dendro_options.progress.step_regex = (
        r"(?:checkpoint at step|step)[^\r\n0-9]*(?P<step>\d+)"
    )
    registry = PersistentTaskRegistry(
        definitions=[task],
        state_root=tmp_path,
        local_node_id="boston",
    )
    log = registry.task_directory("real-dendro") / "logs/process.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "[BSSNCtx] writing checkpoint file: /tmp/bssn_cp_0_step.cp\n"
        "[2026-08-24 20:31:16.290] [dendro] checkpoint complete\n",
        encoding="utf-8",
    )

    assert not DendroProgressSynchronizer(registry).refresh("real-dendro")
    assert not registry.progress_file("real-dendro").exists()

    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "[2026-08-24 20:31:17.000] [dendro] "
            "checkpoint at step 13\n"
        )

    assert DendroProgressSynchronizer(registry).refresh("real-dendro")
    payload = json.loads(
        registry.progress_file("real-dendro").read_text(encoding="utf-8")
    )
    assert payload["completed_units"] == 13


def test_real_bssn_launcher_applies_resolution_and_time_overrides(tmp_path) -> None:
    from scripts.run_real_dendro_bssn import render_runtime_parameters

    template = tmp_path / "template.toml"
    template.write_text(
        "BSSN_RESTORE_SOLVER = 0\n"
        'BSSN_CHKPT_FILE_PREFIX = "cp/bssn_cp"\n'
        'BSSN_VTU_FILE_PREFIX = "vtu/bssn_gr"\n'
        'BSSN_PROFILE_FILE_PREFIX = "dat/dgr"\n'
        'AEH_SAVE_DIR = "aeh"\n'
        "BSSN_MAXDEPTH = 8\n"
        "BSSN_RK_TIME_END = 1.0\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "task" / "checkpoint"
    output = tmp_path / "task" / "output"
    rendered = render_runtime_parameters(
        template_path=template,
        output_path=tmp_path / "task" / "runtime.toml",
        checkpoint_directory=checkpoint,
        output_directory=output,
        resume=False,
        overrides={
            "BSSN_MAXDEPTH": "10",
            "BSSN_RK_TIME_END": "2.5",
        },
    )
    text = rendered.read_text(encoding="utf-8")
    assert "BSSN_MAXDEPTH = 10" in text
    assert "BSSN_RK_TIME_END = 2.5" in text
    assert "BSSN_RESTORE_SOLVER = 0" in text
