from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from magellan.artifacts.models import ArtifactBinding


class MigrationActivationRequest(BaseModel):
    migration_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    generation: int = Field(ge=1)
    migration_at_utc: datetime
    artifacts: list[ArtifactBinding] = Field(
        default_factory=list
    )


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

    migration_at_utc: datetime | None = None
    artifact_digests: dict[str, str] = Field(
        default_factory=dict
    )