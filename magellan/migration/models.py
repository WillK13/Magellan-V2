from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from magellan.artifacts.models import ArtifactBinding
from magellan.state.task_models import (
    TaskAccountingSnapshot,
    TaskRuntimeState,
    TaskStatus,
)




class MigrationRole(str, Enum):
    SOURCE = "source"
    DESTINATION = "destination"


class MigrationStatus(str, Enum):
    PREPARING = "preparing"
    TRANSFERRING = "transferring"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    ROLLED_BACK = "rolled_back"
    UNCERTAIN = "uncertain"


class MigrationRecord(BaseModel):
    migration_id: str = Field(min_length=1)
    bid_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)
    generation: int = Field(ge=1)
    migration_at_utc: datetime
    role: MigrationRole
    status: MigrationStatus
    original_state: TaskRuntimeState | None = None
    pid: int | None = Field(default=None, ge=1)
    error: str | None = None
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MigrationActivationRequest(BaseModel):
    migration_id: str = Field(min_length=1)
    bid_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    generation: int = Field(ge=1)
    migration_at_utc: datetime
    artifacts: list[ArtifactBinding] = Field(
        default_factory=list
    )
    accounting: TaskAccountingSnapshot | None = None


class MigrationActivationResponse(BaseModel):
    migration_id: str
    task_id: str
    destination_node_id: str
    generation: int

    activated: bool
    pid: int | None = None
    error: str | None = None


class OwnershipUpdate(BaseModel):
    task_id: str = Field(min_length=1)
    owner_node_id: str = Field(min_length=1)
    generation: int = Field(ge=0)

    last_migration_id: str | None = None
    migration_at_utc: datetime | None = None
    artifact_digests: dict[str, str] = Field(
        default_factory=dict
    )

    status: TaskStatus | None = None
    completed_at_utc: datetime | None = None
    final_output_manifest_sha256: str | None = None
    final_output_bytes: int | None = Field(default=None, ge=0)
    accounting: TaskAccountingSnapshot | None = None
