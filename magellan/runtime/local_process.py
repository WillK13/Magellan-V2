from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)
from magellan.state.task_models import (
    TaskRuntimeState,
    TaskStatus,
)


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class LocalProcessRuntime:
    def __init__(
        self,
        registry: PersistentTaskRegistry,
        local_node_id: str,
        repository_root: str | Path,
    ) -> None:
        self._registry = registry
        self._local_node_id = local_node_id
        self._repository_root = Path(repository_root).resolve()
        self._processes: dict[str, subprocess.Popen] = {}

    def _render_argument(
        self,
        task_id: str,
        value: str,
    ) -> str:
        return value.format(
            task_id=task_id,
            checkpoint_file=str(
                self._registry.checkpoint_file(task_id)
            ),
            checkpoint_directory=str(
                self._registry.checkpoint_directory(task_id)
            ),
            task_directory=str(
                self._registry.task_directory(task_id)
            ),
            repository_root=str(self._repository_root),
        )

    def start(self, task_id: str) -> TaskRuntimeState:
        definition = self._registry.get_definition(task_id)
        state = self._registry.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot start {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if (
            state.status == TaskStatus.RUNNING
            and state.pid is not None
            and pid_is_alive(state.pid)
        ):
            return state

        task_directory = self._registry.task_directory(task_id)
        checkpoint_file = self._registry.checkpoint_file(task_id)

        task_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        logs_directory = task_directory / "logs"
        logs_directory.mkdir(parents=True, exist_ok=True)

        log_path = logs_directory / "process.log"

        arguments = [
            self._render_argument(task_id, argument)
            for argument in definition.runtime.arguments
        ]

        command = [
            sys.executable,
            "-m",
            definition.runtime.module,
            *arguments,
        ]

        environment = os.environ.copy()
        environment.update(definition.runtime.environment)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["MAGELLAN_TASK_ID"] = task_id
        environment["MAGELLAN_NODE_ID"] = self._local_node_id

        working_directory = (
            self._repository_root
            / definition.runtime.working_directory
        ).resolve()

        readiness_file = self._registry.readiness_file(task_id)

        if readiness_file is not None:
            readiness_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            readiness_file.unlink(missing_ok=True)

        log_file = log_path.open("a", encoding="utf-8")

        try:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_file.close()

        try:
            self._wait_for_readiness(
                task_id=task_id,
                process=process,
            )
        except Exception as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            error = f"{type(exc).__name__}: {exc}"
            self._registry.mark_failed(task_id, error)
            raise RuntimeError(error) from exc

        self._processes[task_id] = process

        state = self._registry.mark_running(
            task_id=task_id,
            pid=process.pid,
        )

        print(
            f"[runtime-start] task={task_id} "
            f"pid={process.pid} node={self._local_node_id}",
            flush=True,
        )

        return state

    def stop(self, task_id: str) -> TaskRuntimeState:
        definition = self._registry.get_definition(task_id)
        state = self._registry.get_state(task_id)

        if state.pid is None:
            return self._registry.mark_stopped(task_id)

        pid = state.pid
        process = self._processes.get(task_id)

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._processes.pop(task_id, None)
            return self._registry.mark_stopped(task_id)

        deadline = (
            time.monotonic()
            + definition.runtime.stop_timeout_seconds
        )

        if process is not None:
            remaining = max(0.0, deadline - time.monotonic())

            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        while pid_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)

        if pid_is_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self._processes.pop(task_id, None)

        state = self._registry.mark_stopped(task_id)

        print(
            f"[runtime-stop] task={task_id} "
            f"old_pid={pid} node={self._local_node_id}",
            flush=True,
        )

        return state

    def reconcile(self) -> None:
        for state in self._registry.all_states():
            if (
                state.owner_node_id != self._local_node_id
                or state.status != TaskStatus.RUNNING
            ):
                continue

            if state.pid is None or not pid_is_alive(state.pid):
                self._registry.mark_failed(
                    state.task_id,
                    "Persisted process is no longer running",
                )
    def _wait_for_readiness(
        self,
        task_id: str,
        process: subprocess.Popen,
    ) -> None:
        definition = self._registry.get_definition(task_id)
        readiness_file = self._registry.readiness_file(task_id)

        if readiness_file is None:
            time.sleep(0.25)

            if process.poll() is not None:
                raise RuntimeError(
                    f"Task {task_id} exited immediately "
                    f"with code {process.returncode}"
                )

            return

        deadline = (
            time.monotonic()
            + definition.runtime.readiness_timeout_seconds
        )

        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Task {task_id} exited before becoming ready "
                    f"with code {process.returncode}"
                )

            if readiness_file.is_file():
                return

            time.sleep(0.25)

        raise TimeoutError(
            f"Task {task_id} did not become ready within "
            f"{definition.runtime.readiness_timeout_seconds}s"
        )
