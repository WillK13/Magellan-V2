from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from magellan.artifacts.models import StaticArtifactSpec
from magellan.models.types import TaskProfile


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    MIGRATING = "migrating"
    RECOVERING = "recovering"
    REMOTE = "remote"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskAccountingSnapshot(BaseModel):
    estimated_remaining_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    accumulated_runtime_seconds: float = Field(default=0.0, ge=0)
    accumulated_paused_seconds: float = Field(default=0.0, ge=0)
    accumulated_migration_seconds: float = Field(default=0.0, ge=0)

    accumulated_compute_cost_usd: float = Field(default=0.0, ge=0)
    accumulated_transfer_cost_usd: float = Field(default=0.0, ge=0)
    accumulated_cost_usd: float = Field(default=0.0, ge=0)

    accumulated_compute_carbon_grams: float = Field(default=0.0, ge=0)
    accumulated_transfer_carbon_grams: float = Field(default=0.0, ge=0)
    accumulated_carbon_grams: float = Field(default=0.0, ge=0)

    progress_completed_units: float | None = Field(default=None, ge=0)
    progress_total_units: float | None = Field(default=None, gt=0)
    progress_fraction: float | None = Field(default=None, ge=0, le=1)
    progress_rate_units_per_second: float | None = Field(
        default=None,
        gt=0,
    )
    progress_updated_at_utc: datetime | None = None


class LocalProcessSpec(BaseModel):
    """Portable runtime contract for Python modules and local commands.

    ``python_module`` preserves the original Magellan V2 behavior. ``command``
    launches an arbitrary executable. ``dendro`` is a command runtime with an
    application-checkpoint restart contract and optional resume arguments.
    """

    adapter: Literal["python_module", "command", "dendro"] = "python_module"
    module: str | None = None
    command: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    resume_arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    working_directory: str = "."

    # The parent directory of this path is the checkpoint directory.
    checkpoint_relative_path: str = "checkpoint/state.json"

    # Optional checkpoint manifest relative to the checkpoint directory.
    checkpoint_manifest_relative_path: str | None = None

    # Optional readiness marker relative to the task directory.
    readiness_relative_path: str | None = None
    readiness_timeout_seconds: float = Field(default=30.0, gt=0)

    # Standardized workload progress record relative to the task directory.
    progress_relative_path: str | None = None

    # A workload writes this marker only after successful natural completion.
    completion_relative_path: str | None = None

    # Files in this directory are published through a content manifest.
    output_relative_directory: str | None = None

    stop_timeout_seconds: float = Field(default=10.0, gt=0)
    minimum_process_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_runtime(self) -> "LocalProcessSpec":
        if self.adapter == "python_module":
            if not self.module:
                raise ValueError("python_module runtime requires module")
            if self.command:
                raise ValueError("python_module runtime cannot define command")
        else:
            if not self.command:
                raise ValueError(f"{self.adapter} runtime requires command")
            if self.module is not None:
                raise ValueError(f"{self.adapter} runtime cannot define module")

        if self.adapter != "dendro" and self.resume_arguments:
            raise ValueError("resume_arguments are only valid for dendro runtime")

        values = {
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "readiness_relative_path": self.readiness_relative_path,
            "progress_relative_path": self.progress_relative_path,
            "completion_relative_path": self.completion_relative_path,
            "output_relative_directory": self.output_relative_directory,
            "working_directory": self.working_directory,
        }

        for name, raw in values.items():
            if raw is None:
                continue
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a safe relative path")

        if self.checkpoint_manifest_relative_path is not None:
            manifest = Path(self.checkpoint_manifest_relative_path)
            if manifest.is_absolute() or ".." in manifest.parts:
                raise ValueError(
                    "checkpoint_manifest_relative_path must be a safe relative path"
                )

        return self


class TaskDefinition(BaseModel):
    profile: TaskProfile
    runtime: LocalProcessSpec
    artifacts: list[StaticArtifactSpec] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_artifacts(self) -> "TaskDefinition":
        artifact_ids = [
            artifact.artifact_id
            for artifact in self.artifacts
        ]

        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(
                "Artifact IDs must be unique within a task"
            )

        return self


class TaskRuntimeState(BaseModel):
    task_id: str = Field(min_length=1)
    owner_node_id: str = Field(min_length=1)

    generation: int = Field(default=0, ge=0)
    status: TaskStatus
    pid: int | None = Field(default=None, ge=1)
    process_group_id: int | None = Field(default=None, ge=1)
    runtime_adapter: str | None = None
    launch_command: list[str] = Field(default_factory=list)
    resumed_from_checkpoint: bool = False

    last_migration_id: str | None = None

    # Stored in the scheduler's evaluation-clock domain.
    last_migration_at_utc: datetime | None = None
    last_pause_at_utc: datetime | None = None
    paused_at_utc: datetime | None = None
    resume_at_utc: datetime | None = None
    resume_wall_at_utc: datetime | None = None
    pause_reason: str | None = None
    pause_count: int = Field(default=0, ge=0)

    last_error: str | None = None
    last_exit_code: int | None = None
    last_failure_at_utc: datetime | None = None

    recovery_attempts: int = Field(default=0, ge=0)
    next_recovery_at_utc: datetime | None = None
    recovery_exhausted: bool = False

    completed_at_utc: datetime | None = None
    final_output_manifest_relative_path: str | None = None
    final_output_manifest_sha256: str | None = None
    final_output_bytes: int = Field(default=0, ge=0)

    created_at_utc: datetime = Field(default_factory=utc_now)
    updated_at_utc: datetime = Field(default_factory=utc_now)
    started_at_utc: datetime | None = None
    last_accounted_at_utc: datetime | None = None

    estimated_remaining_seconds: float | None = Field(default=None, ge=0)

    accumulated_runtime_seconds: float = Field(default=0.0, ge=0)
    accumulated_paused_seconds: float = Field(default=0.0, ge=0)
    accumulated_migration_seconds: float = Field(default=0.0, ge=0)

    accumulated_compute_cost_usd: float = Field(default=0.0, ge=0)
    accumulated_transfer_cost_usd: float = Field(default=0.0, ge=0)
    accumulated_cost_usd: float = Field(default=0.0, ge=0)

    accumulated_compute_carbon_grams: float = Field(default=0.0, ge=0)
    accumulated_transfer_carbon_grams: float = Field(default=0.0, ge=0)
    accumulated_carbon_grams: float = Field(default=0.0, ge=0)

    progress_completed_units: float | None = Field(default=None, ge=0)
    progress_total_units: float | None = Field(default=None, gt=0)
    progress_fraction: float | None = Field(default=None, ge=0, le=1)
    progress_rate_units_per_second: float | None = Field(
        default=None,
        gt=0,
    )
    progress_updated_at_utc: datetime | None = None

    artifact_digests: dict[str, str] = Field(
        default_factory=dict
    )

    def accounting_snapshot(self) -> TaskAccountingSnapshot:
        return TaskAccountingSnapshot(
            **{
                name: getattr(self, name)
                for name in TaskAccountingSnapshot.model_fields
            }
        )
