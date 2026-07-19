from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from magellan.artifacts.manager import ArtifactManager
from magellan.runtime.completion import (
    CompletionManager,
    CompletionValidationError,
)
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


def reap_child_if_exited(pid: int) -> bool:
    """Reap an exited child when this process is still its parent.

    This matters in tests and daemon-reconstruction scenarios where a new
    ``LocalProcessRuntime`` instance no longer owns the original ``Popen``
    object. On macOS, ``kill(pid, 0)`` continues to report a zombie child as
    present until somebody calls ``waitpid``. Without reaping it, shutdown
    waits for the full timeout and a subsequent process-group ``SIGKILL`` can
    fail with ``EPERM`` even though the workload has already exited.

    After a real daemon restart, the workload is no longer a child of the new
    daemon, so ``waitpid`` raises ``ChildProcessError`` and normal PID probing
    remains in effect.
    """

    try:
        waited_pid, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    except OSError:
        return False

    return waited_pid == pid


@dataclass(frozen=True)
class RuntimeReconcileEvent:
    task_id: str
    status: TaskStatus
    exit_code: int | None
    error: str | None = None


class LocalProcessRuntime:
    def __init__(
        self,
        registry: PersistentTaskRegistry,
        local_node_id: str,
        repository_root: str | Path,
        artifact_manager: ArtifactManager,
        completion_manager: CompletionManager,
    ) -> None:
        self._registry = registry
        self._local_node_id = local_node_id
        self._repository_root = Path(repository_root).resolve()
        self._processes: dict[str, subprocess.Popen] = {}
        self._artifact_manager = artifact_manager
        self._completion_manager = completion_manager

    def _render_argument(
        self,
        task_id: str,
        value: str,
    ) -> str:
        completion_file = self._registry.completion_file(task_id)
        output_directory = self._registry.output_directory(task_id)
        progress_file = self._registry.progress_file(task_id)

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
            artifacts_directory=str(
                self._registry.artifacts_directory(task_id)
            ),
            completion_file=(
                str(completion_file)
                if completion_file is not None
                else ""
            ),
            output_directory=(
                str(output_directory)
                if output_directory is not None
                else ""
            ),
            progress_file=(
                str(progress_file)
                if progress_file is not None
                else ""
            ),
        )

    def start(self, task_id: str) -> TaskRuntimeState:
        self._artifact_manager.ensure_task_artifacts(task_id)
        definition = self._registry.get_definition(task_id)
        state = self._registry.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot start {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if state.status == TaskStatus.COMPLETED:
            raise RuntimeError(
                f"Cannot start completed task {task_id}"
            )

        if (
            state.status == TaskStatus.RUNNING
            and state.pid is not None
            and pid_is_alive(state.pid)
        ):
            return state

        if (
            state.status == TaskStatus.PAUSED
            and state.pid is not None
            and pid_is_alive(state.pid)
        ):
            raise RuntimeError(
                f"Cannot start paused task {task_id}; resume it instead"
            )

        task_directory = self._registry.task_directory(task_id)
        checkpoint_file = self._registry.checkpoint_file(task_id)
        task_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        output_directory = self._registry.output_directory(task_id)
        if output_directory is not None:
            output_directory.mkdir(parents=True, exist_ok=True)

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
        progress_file = self._registry.progress_file(task_id)

        if progress_file is not None:
            progress_file.parent.mkdir(parents=True, exist_ok=True)

        if readiness_file is not None:
            readiness_file.parent.mkdir(parents=True, exist_ok=True)
            readiness_file.unlink(missing_ok=True)

        # A completion marker belongs to exactly one successful run. Failed
        # or interrupted runs never create it, and retries begin cleanly.
        completion_file = self._registry.completion_file(task_id)
        if completion_file is not None:
            completion_file.parent.mkdir(parents=True, exist_ok=True)
            completion_file.unlink(missing_ok=True)

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
            self._registry.mark_failed(
                task_id,
                error,
                exit_code=process.poll(),
            )
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

    def pause(
        self,
        task_id: str,
        paused_at_utc,
        resume_at_utc,
        resume_wall_at_utc,
        reason: str,
    ) -> TaskRuntimeState:
        state = self._registry.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot pause {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status == TaskStatus.PAUSED:
            return state
        if (
            state.status != TaskStatus.RUNNING
            or state.pid is None
            or not pid_is_alive(state.pid)
        ):
            raise RuntimeError(
                f"Cannot pause {task_id}; no live running process"
            )

        try:
            os.killpg(state.pid, signal.SIGSTOP)
        except ProcessLookupError as exc:
            self._registry.mark_failed(
                task_id,
                "Process disappeared while pausing",
            )
            raise RuntimeError(
                f"Process disappeared while pausing {task_id}"
            ) from exc

        paused = self._registry.mark_paused(
            task_id=task_id,
            paused_at_utc=paused_at_utc,
            resume_at_utc=resume_at_utc,
            resume_wall_at_utc=resume_wall_at_utc,
            reason=reason,
        )

        print(
            f"[runtime-pause] task={task_id} "
            f"pid={state.pid} resume_at={resume_at_utc.isoformat()} "
            f"node={self._local_node_id}",
            flush=True,
        )
        return paused

    def resume(self, task_id: str) -> TaskRuntimeState:
        state = self._registry.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot resume {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status == TaskStatus.RUNNING:
            return state
        if state.status != TaskStatus.PAUSED or state.pid is None:
            raise RuntimeError(
                f"Cannot resume {task_id}; status is "
                f"{state.status.value}"
            )
        if not pid_is_alive(state.pid):
            self._registry.mark_failed(
                task_id,
                "Paused process is no longer running",
            )
            raise RuntimeError(
                f"Paused process for {task_id} is no longer running"
            )

        os.killpg(state.pid, signal.SIGCONT)
        resumed = self._registry.mark_resumed(task_id)

        print(
            f"[runtime-resume] task={task_id} "
            f"pid={state.pid} node={self._local_node_id}",
            flush=True,
        )
        return resumed

    def stop(self, task_id: str) -> TaskRuntimeState:
        definition = self._registry.get_definition(task_id)
        state = self._registry.get_state(task_id)

        if state.status == TaskStatus.COMPLETED:
            return state

        if state.pid is None:
            return self._registry.mark_stopped(task_id)

        pid = state.pid
        process = self._processes.get(task_id)

        if state.status == TaskStatus.PAUSED:
            try:
                os.killpg(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._processes.pop(task_id, None)
            return self._registry.mark_stopped(task_id)

        deadline = (
            time.monotonic()
            + definition.runtime.stop_timeout_seconds
        )

        exited = False

        if process is not None:
            remaining = max(0.0, deadline - time.monotonic())

            try:
                process.wait(timeout=remaining)
                exited = True
            except subprocess.TimeoutExpired:
                pass
        else:
            # A reconstructed runtime has no Popen handle. If it is still
            # running in the same Python parent (as in the restart test), reap
            # the terminated child directly instead of mistaking its zombie
            # entry for a live process.
            exited = reap_child_if_exited(pid)

        while not exited and time.monotonic() < deadline:
            if process is None and reap_child_if_exited(pid):
                exited = True
                break
            if not pid_is_alive(pid):
                exited = True
                break
            time.sleep(0.1)

        if not exited:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                exited = True
            except PermissionError:
                # macOS may return EPERM for a process group whose leader has
                # already exited but has not yet been reaped. Only suppress the
                # error after proving that the recorded PID is gone/reaped.
                if reap_child_if_exited(pid) or not pid_is_alive(pid):
                    exited = True
                else:
                    raise

        if process is None:
            reap_child_if_exited(pid)

        self._processes.pop(task_id, None)
        state = self._registry.mark_stopped(task_id)

        print(
            f"[runtime-stop] task={task_id} "
            f"old_pid={pid} node={self._local_node_id}",
            flush=True,
        )

        return state

    def reconcile(self) -> list[RuntimeReconcileEvent]:
        events: list[RuntimeReconcileEvent] = []

        for state in self._registry.all_states():
            if (
                state.owner_node_id != self._local_node_id
                or state.status not in {
                    TaskStatus.RUNNING,
                    TaskStatus.PAUSED,
                }
            ):
                continue

            process = self._processes.get(state.task_id)
            exit_code = (
                process.poll()
                if process is not None
                else None
            )

            alive = (
                exit_code is None
                and state.pid is not None
                and pid_is_alive(state.pid)
            )

            if alive:
                continue

            self._processes.pop(state.task_id, None)

            if self._completion_manager.completion_marker_exists(
                state.task_id
            ):
                try:
                    manifest = self._completion_manager.finalize(
                        task_id=state.task_id,
                        exit_code=exit_code,
                    )
                except Exception as exc:
                    error = (
                        "Completion validation failed: "
                        f"{exc}"
                    )
                    self._registry.mark_failed(
                        state.task_id,
                        error,
                        exit_code=exit_code,
                    )
                    events.append(
                        RuntimeReconcileEvent(
                            task_id=state.task_id,
                            status=TaskStatus.FAILED,
                            exit_code=exit_code,
                            error=error,
                        )
                    )
                    continue

                print(
                    f"[runtime-complete] task={state.task_id} "
                    f"files={len(manifest.files)} "
                    f"bytes={manifest.total_size_bytes}",
                    flush=True,
                )

                events.append(
                    RuntimeReconcileEvent(
                        task_id=state.task_id,
                        status=TaskStatus.COMPLETED,
                        exit_code=exit_code,
                    )
                )
                continue

            error = (
                "Persisted process is no longer running "
                "and no valid completion marker was written"
            )
            self._registry.mark_failed(
                state.task_id,
                error,
                exit_code=exit_code,
            )
            events.append(
                RuntimeReconcileEvent(
                    task_id=state.task_id,
                    status=TaskStatus.FAILED,
                    exit_code=exit_code,
                    error=error,
                )
            )

        return events

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
            if readiness_file.is_file():
                return

            if process.poll() is not None:
                raise RuntimeError(
                    f"Task {task_id} exited before becoming ready "
                    f"with code {process.returncode}"
                )

            time.sleep(0.25)

        raise TimeoutError(
            f"Task {task_id} did not become ready within "
            f"{definition.runtime.readiness_timeout_seconds}s"
        )
