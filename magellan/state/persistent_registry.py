from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterable

from magellan.models.types import TaskProfile, TaskResourceRequest
from magellan.state.task_models import (
    TaskAccountingSnapshot,
    TaskDefinition,
    TaskRuntimeState,
    TaskStatus,
    utc_now,
)


_CAPACITY_STATUSES = {
    TaskStatus.STOPPED,
    TaskStatus.RUNNING,
    TaskStatus.PAUSED,
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

    def progress_file(self, task_id: str) -> Path | None:
        definition = self.get_definition(task_id)
        relative = definition.runtime.progress_relative_path

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
                estimated_remaining_seconds=(
                    definition.profile.estimated_remaining_seconds
                ),
                accumulated_compute_cost_usd=(
                    definition.profile.accumulated_cost_usd
                ),
                accumulated_cost_usd=(
                    definition.profile.accumulated_cost_usd
                ),
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

    def register_definition(
        self,
        definition: TaskDefinition,
    ) -> bool:
        """Register a runtime-created task without restarting the daemon."""
        task_id = definition.profile.task_id
        with self._lock:
            existing = self._definitions.get(task_id)
            if existing is not None:
                if (
                    existing.model_dump(mode="python")
                    != definition.model_dump(mode="python")
                ):
                    raise RuntimeError(
                        f"Conflicting task definition for {task_id}"
                    )
                return False
            self._definitions[task_id] = definition.model_copy(deep=True)
            self._initialize_state_if_missing(definition)
            return True

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

    def owned_resource_requests(
        self,
        node_id: str,
    ) -> list[TaskResourceRequest]:
        requests: list[TaskResourceRequest] = []
        for state in self.all_states():
            if (
                state.owner_node_id != node_id
                or state.status not in _CAPACITY_STATUSES
            ):
                continue
            definition = self.get_definition(state.task_id)
            requests.append(
                definition.profile.resource_request.model_copy(
                    deep=True
                )
            )
        return requests

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

    def paused_owned_task_ids(
        self,
        node_id: str,
    ) -> list[str]:
        return [
            state.task_id
            for state in self.all_states()
            if (
                state.owner_node_id == node_id
                and state.status == TaskStatus.PAUSED
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
            "last_pause_at": state.last_pause_at_utc,
            "estimated_remaining_seconds": (
                state.estimated_remaining_seconds
                if state.estimated_remaining_seconds is not None
                else definition.profile.estimated_remaining_seconds
            ),
            "accumulated_cost_usd": state.accumulated_cost_usd,
        }

        if checkpoint_bytes is not None:
            updates["checkpoint_bytes"] = checkpoint_bytes

        return definition.profile.model_copy(
            deep=True,
            update=updates,
        )

    def _apply_accounting_snapshot(
        self,
        state: TaskRuntimeState,
        snapshot: TaskAccountingSnapshot,
    ) -> None:
        for name in TaskAccountingSnapshot.model_fields:
            setattr(state, name, getattr(snapshot, name))

    def accounting_snapshot(
        self,
        task_id: str,
    ) -> TaskAccountingSnapshot:
        return self.get_state(task_id).accounting_snapshot()

    def record_accounting(
        self,
        task_id: str,
        runtime_seconds: float = 0.0,
        paused_seconds: float = 0.0,
        migration_seconds: float = 0.0,
        compute_cost_usd: float = 0.0,
        transfer_cost_usd: float = 0.0,
        compute_carbon_grams: float = 0.0,
        transfer_carbon_grams: float = 0.0,
        last_accounted_at_utc: datetime | None = None,
        estimated_remaining_seconds: float | None = None,
        progress_completed_units: float | None = None,
        progress_total_units: float | None = None,
        progress_fraction: float | None = None,
        progress_rate_units_per_second: float | None = None,
        progress_updated_at_utc: datetime | None = None,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        state.accumulated_runtime_seconds += max(0.0, runtime_seconds)
        state.accumulated_paused_seconds += max(0.0, paused_seconds)
        state.accumulated_migration_seconds += max(0.0, migration_seconds)

        state.accumulated_compute_cost_usd += max(0.0, compute_cost_usd)
        state.accumulated_transfer_cost_usd += max(0.0, transfer_cost_usd)
        state.accumulated_cost_usd = (
            state.accumulated_compute_cost_usd
            + state.accumulated_transfer_cost_usd
        )

        state.accumulated_compute_carbon_grams += max(
            0.0, compute_carbon_grams
        )
        state.accumulated_transfer_carbon_grams += max(
            0.0, transfer_carbon_grams
        )
        state.accumulated_carbon_grams = (
            state.accumulated_compute_carbon_grams
            + state.accumulated_transfer_carbon_grams
        )

        if last_accounted_at_utc is not None:
            state.last_accounted_at_utc = last_accounted_at_utc
        if estimated_remaining_seconds is not None:
            state.estimated_remaining_seconds = max(
                0.0, estimated_remaining_seconds
            )
        if progress_completed_units is not None:
            state.progress_completed_units = progress_completed_units
        if progress_total_units is not None:
            state.progress_total_units = progress_total_units
        if progress_fraction is not None:
            state.progress_fraction = progress_fraction
        if progress_rate_units_per_second is not None:
            state.progress_rate_units_per_second = (
                progress_rate_units_per_second
            )
        if progress_updated_at_utc is not None:
            state.progress_updated_at_utc = progress_updated_at_utc

        return self.set_state(state)

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
        runtime_adapter: str | None = None,
        launch_command: list[str] | None = None,
        resumed_from_checkpoint: bool = False,
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

        now = utc_now()
        state.status = TaskStatus.RUNNING
        state.pid = pid
        state.process_group_id = pid
        if runtime_adapter is not None:
            state.runtime_adapter = runtime_adapter
        if launch_command is not None:
            state.launch_command = list(launch_command)
        state.resumed_from_checkpoint = resumed_from_checkpoint
        state.started_at_utc = state.started_at_utc or now
        state.last_accounted_at_utc = now
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
        state.last_error = None
        state.last_exit_code = None
        state.next_recovery_at_utc = None
        state.recovery_exhausted = False
        return self.set_state(state)

    def mark_paused(
        self,
        task_id: str,
        paused_at_utc: datetime,
        resume_at_utc: datetime,
        resume_wall_at_utc: datetime,
        reason: str,
    ) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot pause {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status != TaskStatus.RUNNING or state.pid is None:
            raise RuntimeError(
                f"Cannot pause {task_id}; status is "
                f"{state.status.value}"
            )

        state.status = TaskStatus.PAUSED
        state.paused_at_utc = paused_at_utc
        state.resume_at_utc = resume_at_utc
        state.resume_wall_at_utc = resume_wall_at_utc
        state.last_pause_at_utc = paused_at_utc
        state.pause_reason = reason
        state.pause_count += 1
        state.last_accounted_at_utc = utc_now()
        return self.set_state(state)

    def mark_resumed(self, task_id: str) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.owner_node_id != self._local_node_id:
            raise RuntimeError(
                f"Cannot resume {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status != TaskStatus.PAUSED or state.pid is None:
            raise RuntimeError(
                f"Cannot resume {task_id}; status is "
                f"{state.status.value}"
            )

        state.status = TaskStatus.RUNNING
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
        state.last_accounted_at_utc = utc_now()
        return self.set_state(state)

    def mark_stopped(self, task_id: str) -> TaskRuntimeState:
        state = self.get_state(task_id)

        if state.status == TaskStatus.COMPLETED:
            return state

        state.status = TaskStatus.STOPPED
        state.pid = None
        state.process_group_id = None
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
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
        state.process_group_id = None
        state.last_error = error
        state.last_exit_code = exit_code
        state.last_failure_at_utc = utc_now()
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
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
        state.process_group_id = None
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
        state.process_group_id = None
        state.last_error = None
        state.last_exit_code = exit_code
        state.completed_at_utc = completed_at_utc
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
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
        accounting: TaskAccountingSnapshot | None = None,
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
        state.process_group_id = None
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
        state.last_migration_id = migration_id
        state.last_migration_at_utc = migration_at_utc
        state.last_error = None
        state.last_exit_code = None
        state.artifact_digests = dict(artifact_digests)
        if accounting is not None:
            self._apply_accounting_snapshot(state, accounting)
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
        state.process_group_id = None
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
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
        state.process_group_id = None
        state.paused_at_utc = None
        state.resume_at_utc = None
        state.resume_wall_at_utc = None
        state.pause_reason = None
        state.last_error = error
        state.last_exit_code = None
        return self.set_state(state)

    def apply_ownership(
        self,
        task_id: str,
        owner_node_id: str,
        generation: int,
        migration_id: str | None = None,
        migration_at_utc: datetime | None = None,
        status: TaskStatus | None = None,
        completed_at_utc: datetime | None = None,
        final_output_manifest_sha256: str | None = None,
        final_output_bytes: int | None = None,
        accounting: TaskAccountingSnapshot | None = None,
    ) -> bool:
        state = self.get_state(task_id)

        if generation < state.generation:
            return False

        if generation == state.generation and owner_node_id != state.owner_node_id:
            return False

        if (
            generation == state.generation
            and state.status == TaskStatus.COMPLETED
            and status != TaskStatus.COMPLETED
        ):
            return False

        state.owner_node_id = owner_node_id
        state.generation = generation

        if accounting is not None:
            self._apply_accounting_snapshot(state, accounting)

        if migration_id is not None:
            state.last_migration_id = migration_id
        if migration_at_utc is not None:
            state.last_migration_at_utc = migration_at_utc

        if status == TaskStatus.COMPLETED:
            state.status = TaskStatus.COMPLETED
            state.pid = None
            state.process_group_id = None
            state.paused_at_utc = None
            state.resume_at_utc = None
            state.resume_wall_at_utc = None
            state.pause_reason = None
            state.completed_at_utc = completed_at_utc
            state.final_output_manifest_sha256 = (
                final_output_manifest_sha256
            )
            if final_output_bytes is not None:
                state.final_output_bytes = final_output_bytes
        elif owner_node_id != self._local_node_id:
            state.status = TaskStatus.REMOTE
            state.pid = None
            state.process_group_id = None
            state.paused_at_utc = None
            state.resume_at_utc = None
            state.resume_wall_at_utc = None
            state.pause_reason = None

        self.set_state(state)
        return True

    def ownership_updates(self):
        from magellan.migration.models import OwnershipUpdate

        updates = []
        for state in self.all_states():
            updates.append(
                OwnershipUpdate(
                    task_id=state.task_id,
                    owner_node_id=state.owner_node_id,
                    generation=state.generation,
                    last_migration_id=state.last_migration_id,
                    migration_at_utc=state.last_migration_at_utc,
                    artifact_digests=dict(state.artifact_digests),
                    status=(
                        state.status
                        if state.status == TaskStatus.COMPLETED
                        else None
                    ),
                    completed_at_utc=state.completed_at_utc,
                    final_output_manifest_sha256=(
                        state.final_output_manifest_sha256
                    ),
                    final_output_bytes=state.final_output_bytes,
                    accounting=state.accounting_snapshot(),
                )
            )
        return updates

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
