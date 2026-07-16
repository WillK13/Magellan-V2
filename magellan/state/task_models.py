from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from magellan.models.types import TaskProfile


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
    checkpoint_relative_path: str = "checkpoint/state.json"
    stop_timeout_seconds: float = Field(default=10.0, gt=0)


class TaskDefinition(BaseModel):
    profile: TaskProfile
    runtime: LocalProcessSpec


class TaskRuntimeState(BaseModel):
    task_id: str = Field(min_length=1)
    owner_node_id: str = Field(min_length=1)

    generation: int = Field(default=0, ge=0)
    status: TaskStatus
    pid: int | None = Field(default=None, ge=1)

    last_migration_id: str | None = None
    last_error: str | None = None

    created_at_utc: datetime = Field(default_factory=utc_now)
    updated_at_utc: datetime = Field(default_factory=utc_now)
