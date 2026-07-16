from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from magellan.models.types import TaskProfile

from magellan.artifacts.models import StaticArtifactSpec


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    MIGRATING = "migrating"
    REMOTE = "remote"
    FAILED = "failed"


class LocalProcessSpec(BaseModel):
    module: str = Field(min_length=1)
    arguments: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    working_directory: str = "."

    # The parent directory of this path is the checkpoint directory.
    checkpoint_relative_path: str = "checkpoint/state.json"

    # Optional checkpoint manifest relative to the checkpoint directory.
    checkpoint_manifest_relative_path: str | None = None

    # Optional readiness marker relative to the task directory.
    readiness_relative_path: str | None = None
    readiness_timeout_seconds: float = Field(default=30.0, gt=0)

    stop_timeout_seconds: float = Field(default=10.0, gt=0)


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

    last_migration_id: str | None = None

    # Stored in the scheduler's evaluation-clock domain.
    last_migration_at_utc: datetime | None = None

    last_error: str | None = None

    created_at_utc: datetime = Field(default_factory=utc_now)
    updated_at_utc: datetime = Field(default_factory=utc_now)
    artifact_digests: dict[str, str] = Field(
        default_factory=dict
    )
