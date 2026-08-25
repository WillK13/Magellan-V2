import json
import sys
import time
from pathlib import Path

from magellan.artifacts.manager import ArtifactManager
from magellan.models.types import TaskProfile
from magellan.runtime.adapters import RuntimeAdapterRegistry
from magellan.runtime.completion import CompletionManager
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition, TaskStatus


def test_process_group_count_falls_back_to_posix_ps(monkeypatch) -> None:
    class BrokenProcfsSampler:
        def sample(self, _process_group_id: int):
            raise OSError("procfs unavailable")

    class ProcessResult:
        stdout = "  7\n 42\nnot-a-number\n 42\n"

    monkeypatch.setattr(
        "magellan.runtime.local_process.ProcfsProcessSampler",
        BrokenProcfsSampler,
    )
    monkeypatch.setattr(
        "magellan.runtime.local_process.subprocess.run",
        lambda *_args, **_kwargs: ProcessResult(),
    )

    assert LocalProcessRuntime._process_group_count(42) == 2


def test_adapter_registry_builds_command_and_dendro_resume(tmp_path) -> None:
    registry = RuntimeAdapterRegistry()
    command = registry.get("command").build_launch_plan(
        LocalProcessSpec(
            adapter="command",
            command=["echo"],
            arguments=["{task_id}"],
        ),
        lambda value: value.format(task_id="task-1"),
        tmp_path / "checkpoint",
        tmp_path / "checkpoint" / "state.json",
    )
    assert command.command == ["echo", "task-1"]
    assert command.resumed_from_checkpoint is False

    checkpoint = tmp_path / "checkpoint" / "state.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}", encoding="utf-8")
    dendro = registry.get("dendro").build_launch_plan(
        LocalProcessSpec(
            adapter="dendro",
            command=["solver"],
            arguments=["input.par"],
            resume_arguments=["--restore", "{checkpoint_file}"],
        ),
        lambda value: value.format(checkpoint_file=str(checkpoint)),
        checkpoint.parent,
        checkpoint,
    )
    assert dendro.command == [
        "solver",
        "input.par",
        "--restore",
        str(checkpoint),
    ]
    assert dendro.resumed_from_checkpoint is True
    assert dendro.environment["MAGELLAN_DENDRO_RESUME"] == "1"


def test_python_module_adapter_marks_transferred_checkpoint_as_resume(tmp_path) -> None:
    registry = RuntimeAdapterRegistry()
    spec = LocalProcessSpec(
        adapter="python_module",
        module="magellan.workloads.counter",
        arguments=["--checkpoint-file", "{checkpoint_file}"],
    )
    checkpoint = tmp_path / "checkpoint" / "state.json"

    first = registry.get("python_module").build_launch_plan(
        spec,
        lambda value: value.format(checkpoint_file=str(checkpoint)),
        checkpoint.parent,
        checkpoint,
    )
    assert first.resumed_from_checkpoint is False

    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('{"value": 7}', encoding="utf-8")
    second = registry.get("python_module").build_launch_plan(
        spec,
        lambda value: value.format(checkpoint_file=str(checkpoint)),
        checkpoint.parent,
        checkpoint,
    )
    assert second.resumed_from_checkpoint is True


def test_generic_command_runtime_records_launch_metadata(tmp_path) -> None:
    script = tmp_path / "command_workload.py"
    script.write_text(
        """
import json, pathlib, signal, sys, time
checkpoint = pathlib.Path(sys.argv[1])
ready = pathlib.Path(sys.argv[2])
checkpoint.parent.mkdir(parents=True, exist_ok=True)
ready.parent.mkdir(parents=True, exist_ok=True)
checkpoint.write_text(json.dumps({'value': 1}))
ready.write_text('{}')
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
while running:
    time.sleep(0.05)
""",
        encoding="utf-8",
    )
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="command-task",
            workload_type="command",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(
            adapter="command",
            command=[sys.executable, str(script)],
            arguments=["{checkpoint_file}", "{readiness_file}"],
            readiness_relative_path="runtime/ready.json",
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
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

    state = runtime.start("command-task")
    assert state.status == TaskStatus.RUNNING
    assert state.runtime_adapter == "command"
    assert state.process_group_id == state.pid
    assert state.launch_command[0] == sys.executable
    runtime.stop("command-task")


def test_dendro_runtime_process_tree_checkpoint_and_resume(tmp_path) -> None:
    source = Path(__file__).resolve().parents[1] / "magellan" / "workloads" / "dendro_mock.py"
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="dendro-task",
            workload_type="dendro-gr",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1,
        ),
        runtime=LocalProcessSpec(
            adapter="dendro",
            command=[sys.executable, str(source)],
            arguments=[
                "--checkpoint-file", "{checkpoint_file}",
                "--checkpoint-manifest", "{checkpoint_manifest_file}",
                "--progress-file", "{progress_file}",
                "--completion-file", "{completion_file}",
                "--readiness-file", "{readiness_file}",
                "--output-dir", "{output_directory}",
                "--world-size", "2",
                "--max-step", "1000",
                "--interval-seconds", "0.02",
                "--checkpoint-every", "2",
            ],
            resume_arguments=["--resume"],
            checkpoint_relative_path="checkpoint/state.json",
            checkpoint_manifest_relative_path="manifest.json",
            readiness_relative_path="runtime/ready.json",
            progress_relative_path="runtime/progress.json",
            completion_relative_path="runtime/completion.json",
            output_relative_directory="output",
            minimum_process_count=3,
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
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

    first = runtime.start("dendro-task")
    assert first.runtime_adapter == "dendro"
    assert first.resumed_from_checkpoint is False
    deadline = time.monotonic() + 5
    manifest = registry.checkpoint_manifest_file("dendro-task")
    assert manifest is not None
    while time.monotonic() < deadline and not manifest.is_file():
        time.sleep(0.05)
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["world_size"] == 2
    assert len(payload["files"]) == 3
    runtime.stop("dendro-task")

    second = runtime.start("dendro-task")
    assert second.resumed_from_checkpoint is True
    assert "--resume" in second.launch_command
    runtime.stop("dendro-task")
