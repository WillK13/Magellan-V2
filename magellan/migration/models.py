from __future__ import annotations

from pydantic import BaseModel, Field


class MigrationActivationRequest(BaseModel):
    migration_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    generation: int = Field(ge=1)


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
