from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable

from magellan.models.types import TaskProfile
from magellan.state.task_models import (
    TaskDefinition,
    TaskRuntimeState,
    TaskStatus,
    utc_now,
)


_CAPACITY_STATUSES = {
    TaskStatus.STOPPED,
    TaskStatus.RUNNING,
    TaskStatus.MIGRATING,
    TaskStatus.RECOVERING,
    TaskStatus.FAILED,
}


class PersistentTaskRegistry:
    def __init__(
        self,
        definitions: Iterable[TaskDefinition],
        state_root: str | Path,
        local_node_id: str,
    ) -> None:
        self._state_root = Path(state_root)
        self._local_node_id = local_node_id
        self._lock = RLock()
        self._definitions: dict[str, TaskDefinition] = {}

        for definition in definitions:
            task_id = definition.profile.task_id

            if task_id in self._definitions:
                raise ValueError(f"Duplicate task ID: {task_id}")

            self._definitions[task_id] = definition
            self._initialize_state_if_missing(definition)

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
        state_root: str | Path,
        local_node_id: str,
    ) -> "PersistentTaskRegistry":
        definitions: list[TaskDefinition] = []

        for path_value in paths:
            path = Path(path_value)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Task definition does not exist: {path}"
                )

            raw = json.loads(path.read_text(encoding="utf-8"))
            definitions.append(TaskDefinition.model_validate(raw))

        return cls(
            definitions=definitions,
            state_root=state_root,
            local_node_id=local_node_id,
        )

    @property
    def state_root(self) -> Path:
        return self._state_root

    @property
    def local_node_id(self) -> str:
        return self._local_node_id

    def task_directory(self, task_id: str) -> Path:
        return self._state_root / "tasks" / task_id

    def checkpoint_directory(self, task_id: str) -> Path:
        definition = self.get_definition(task_id)
        relative_path = Path(
            definition.runtime.checkpoint_relative_path
        )
        return self.task_directory(task_id) / relative_path.parent

    def checkpoint_file(self, task_id: str) -> Path:
        definition = self.get_definition(task_id)
        return (
            self.task_directory(task_id)
            / definition.runtime.checkpoint_relative_path
        )

    def artifacts_directory(self, task_id: str) -> Path:
        return self.task_directory(task_id) / "artifacts"

    def completion_file(self, task_id: str) -> Path | None:
        definition = self.get_definition(task_id)
        relative = definition.runtime.completion_relative_path

        if relative is None:
            return None

        return self.task_directory(task_id) / relative

    def output_directory(self, task_id: str) -> Path | None:
        definition = self.get_definition(task_id)
        relative = definition.runtime.output_relative_directory

        if relative is None:
            return None

        return self.task_directory(task_id) / relative

    def final_output_manifest_file(self, task_id: str) -> Path:
        return (
            self.task_directory(task_id)
            / "final-output"
            / "manifest.json"
        )

    def set_artifact_digests(
        self,
        task_id: str,
        artifact_digests: dict[str, str],
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)
        state.artifact_digests = dict(artifact_digests)
        return self.set_state(state)

    def checkpoint_manifest_file(
        self,
        task_id: str,
    ) -> Path | None:
        definition = self.get_definition(task_id)
        relative = (
            definition.runtime.checkpoint_manifest_relative_path
        )

        if relative is None:
            return None

        return self.checkpoint_directory(task_id) / relative

    def readiness_file(self, task_id: str) -> Path | None:
        definition = self.get_definition(task_id)
        relative = definition.runtime.readiness_relative_path

        if relative is None:
            return None

        return self.task_directory(task_id) / relative

    def _state_path(self, task_id: str) -> Path:
        return self.task_directory(task_id) / "state.json"

    def _initialize_state_if_missing(
        self,
        definition: TaskDefinition,
    ) -> None:
        task_id = definition.profile.task_id
        state_path = self._state_path(task_id)

        self.task_directory(task_id).mkdir(
            parents=True,
            exist_ok=True,
        )

        if state_path.exists():
            return

        initial_owner = definition.profile.current_node_id
        status = (
            TaskStatus.STOPPED
            if initial_owner == self._local_node_id
            else TaskStatus.REMOTE
        )

        self._write_state(
            TaskRuntimeState(
                task_id=task_id,
                owner_node_id=initial_owner,
                status=status,
            )
        )

    def _write_state(self, state: TaskRuntimeState) -> None:
        state.updated_at_utc = utc_now()
        path = self._state_path(state.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                state.model_dump(mode="json"),
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def get_definition(self, task_id: str) -> TaskDefinition:
        try:
            return self._definitions[task_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(
                f"Unknown Magellan task definition: {task_id}"
            ) from exc

    def get_state(self, task_id: str) -> TaskRuntimeState:
        self.get_definition(task_id)

        with self._lock:
            path = self._state_path(task_id)
            raw = json.loads(path.read_text(encoding="utf-8"))
            return TaskRuntimeState.model_validate(raw)

    def all_states(self) -> list[TaskRuntimeState]:
        return [
            self.get_state(task_id)
            for task_id in sorted(self._definitions)
        ]

    def count_owned(self, node_id: str) -> int:
        return sum(
            state.owner_node_id == node_id
            and state.status in _CAPACITY_STATUSES
            for state in self.all_states()
        )

    def running_owned_task_ids(
        self,
        node_id: str,
    ) -> list[str]:
        return [
            state.task_id
            for state in self.all_states()
            if (
                state.owner_node_id == node_id
                and state.status == TaskStatus.RUNNING
            )
        ]

    def failed_owned_task_ids(
        self,
        node_id: str,
    ) -> list[str]:
        return [
            state.task_id
            for state in self.all_states()
            if (
                state.owner_node_id == node_id
                and state.status == TaskStatus.FAILED
            )
        ]

    def scoring_profile(
        self,
        task_id: str,
        checkpoint_bytes: int | None = None,
    ) -> TaskProfile:
        definition = self.get_definition(task_id)
        state = self.get_state(task_id)
        updates: dict = {
            "current_node_id": state.owner_node_id,
            "last_migration_at": state.last_migration_at_utc,
        }

        if checkpoint_bytes is not None:
            updates["checkpoint_bytes"] = checkpoint_bytes

        return definition.profile.model_copy(
            deep=True,
            update=updates,
        )

    def set_state(
        self,
        state: TaskRuntimeState,
    ) -> TaskRuntimeState:
        with self._lock:
            self._write_state(state)
            return state.model_copy(deep=True)

    def mark_running(
        self,
        task_id: str,
        pid: int,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot run {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if state.status == TaskStatus.COMPLETED:
            raise RuntimeError(
                f"Cannot start completed task {task_id}"
            )

        state.status = TaskStatus.RUNNING
        state.pid = pid
        state.last_error = None
        state.last_exit_code = None
        state.next_recovery_at_utc = None
        state.recovery_exhausted = False
        return self.set_state(state)

    def mark_stopped(self, task_id: str) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.status == TaskStatus.COMPLETED:
            return state

        state.status = TaskStatus.STOPPED
        state.pid = None
        state.last_exit_code = None
        return self.set_state(state)

    def mark_migrating(
        self,
        task_id: str,
        migration_id: str,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot migrate {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if state.status == TaskStatus.COMPLETED:
            raise RuntimeError(
                f"Cannot migrate completed task {task_id}"
            )

        state.status = TaskStatus.MIGRATING
        state.last_migration_id = migration_id
        return self.set_state(state)

    def mark_failed(
        self,
        task_id: str,
        error: str,
        exit_code: int | None = None,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)
        state.status = TaskStatus.FAILED
        state.pid = None
        state.last_error = error
        state.last_exit_code = exit_code
        state.last_failure_at_utc = utc_now()
        state.next_recovery_at_utc = None
        return self.set_state(state)

    def schedule_recovery(
        self,
        task_id: str,
        recover_at_utc: datetime,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.status != TaskStatus.FAILED:
            return state

        state.next_recovery_at_utc = recover_at_utc
        return self.set_state(state)

    def begin_recovery(self, task_id: str) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot recover {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if state.status != TaskStatus.FAILED:
            raise RuntimeError(
                f"Cannot recover {task_id}; status is "
                f"{state.status.value}"
            )

        state.status = TaskStatus.RECOVERING
        state.recovery_attempts += 1
        state.next_recovery_at_utc = None
        return self.set_state(state)

    def mark_recovery_exhausted(
        self,
        task_id: str,
        error: str,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)
        state.status = TaskStatus.FAILED
        state.pid = None
        state.last_error = error
        state.next_recovery_at_utc = None
        state.recovery_exhausted = True
        return self.set_state(state)

    def mark_completed(
        self,
        task_id: str,
        completed_at_utc: datetime,
        manifest_relative_path: str,
        manifest_sha256: str,
        output_bytes: int,
        exit_code: int | None,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot complete {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        state.status = TaskStatus.COMPLETED
        state.pid = None
        state.last_error = None
        state.last_exit_code = exit_code
        state.completed_at_utc = completed_at_utc
        state.final_output_manifest_relative_path = (
            manifest_relative_path
        )
        state.final_output_manifest_sha256 = manifest_sha256
        state.final_output_bytes = output_bytes
        state.next_recovery_at_utc = None
        state.recovery_exhausted = False
        return self.set_state(state)

    def claim_local(
        self,
        task_id: str,
        generation: int,
        migration_id: str,
        artifact_digests: dict[str, str],
        migration_at_utc: datetime,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if generation < state.generation:
            raise RuntimeError(
                f"Stale generation {generation}; "
                f"current generation is {state.generation}"
            )

        state.owner_node_id = self._local_node_id
        state.generation = generation
        state.status = TaskStatus.STOPPED
        state.pid = None
        state.last_migration_id = migration_id
        state.last_migration_at_utc = migration_at_utc
        state.last_error = None
        state.last_exit_code = None
        state.artifact_digests = dict(artifact_digests)
        return self.set_state(state)

    def mark_remote(
        self,
        task_id: str,
        owner_node_id: str,
        generation: int,
        migration_id: str,
        migration_at_utc: datetime,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if (
            state.generation == generation
            and state.owner_node_id == owner_node_id
            and state.status == TaskStatus.COMPLETED
        ):
            return state

        state.owner_node_id = owner_node_id
        state.generation = generation
        state.status = TaskStatus.REMOTE
        state.pid = None
        state.last_migration_id = migration_id
        state.last_migration_at_utc = migration_at_utc
        state.last_error = None
        state.last_exit_code = None
        return self.set_state(state)

    def restore_local_after_failure(
        self,
        task_id: str,
        generation: int,
        error: str,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)
        state.owner_node_id = self._local_node_id
        state.generation = generation
        state.status = TaskStatus.STOPPED
        state.pid = None
        state.last_error = error
        state.last_exit_code = None
        return self.set_state(state)

    def apply_ownership(
        self,
        task_id: str,
        owner_node_id: str,
        generation: int,
        migration_at_utc: datetime | None = None,
        status: TaskStatus | None = None,
        completed_at_utc: datetime | None = None,
        final_output_manifest_sha256: str | None = None,
        final_output_bytes: int | None = None,
    ) -> bool:
        state = self.get_state(task_id)

        if generation < state.generation:
            return False

        if (
            generation == state.generation
            and state.status == TaskStatus.COMPLETED
            and status != TaskStatus.COMPLETED
        ):
            return False

        state.owner_node_id = owner_node_id
        state.generation = generation

        if migration_at_utc is not None:
            state.last_migration_at_utc = migration_at_utc

        if status == TaskStatus.COMPLETED:
            state.status = TaskStatus.COMPLETED
            state.pid = None
            state.completed_at_utc = completed_at_utc
            state.final_output_manifest_sha256 = (
                final_output_manifest_sha256
            )
            if final_output_bytes is not None:
                state.final_output_bytes = final_output_bytes
        elif owner_node_id != self._local_node_id:
            state.status = TaskStatus.REMOTE
            state.pid = None

        self.set_state(state)
        return True

    def summaries(self) -> list[dict]:
        result = []

        for task_id in sorted(self._definitions):
            definition = self.get_definition(task_id)
            state = self.get_state(task_id)
            result.append(
                {
                    "definition": definition.model_dump(mode="json"),
                    "state": state.model_dump(mode="json"),
                }
            )

        return result
