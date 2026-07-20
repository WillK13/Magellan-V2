from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from magellan.artifacts.models import StaticArtifactSpec
from magellan.capabilities.models import TaskCompatibilityRequirements
from magellan.models.types import TaskProfile, TaskResourceRequest
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json_value(value: Any) -> Any:
    """Convert a model payload into a deterministic JSON-compatible value."""
    if isinstance(value, BaseModel):
        return _canonical_json_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _canonical_json_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def canonical_digest(value: BaseModel | dict) -> str:
    payload = _canonical_json_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class TaskTemplateProfile(BaseModel):
    workload_type: str = Field(min_length=1)
    power_kw: float = Field(gt=0)
    checkpoint_bytes: int = Field(default=0, ge=0)
    data_bytes: int = Field(default=0, ge=0)
    prestaged_node_ids: set[str] = Field(default_factory=set)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    accumulated_cost_usd: float = Field(default=0.0, ge=0)
    cost_cap_usd: float | None = Field(default=None, gt=0)
    priority: int = Field(default=0, ge=0, le=100)
    deadline_at_utc: datetime | None = None
    resource_request: TaskResourceRequest = Field(
        default_factory=TaskResourceRequest
    )
    compatibility: TaskCompatibilityRequirements = Field(
        default_factory=TaskCompatibilityRequirements
    )


class TaskDefinitionSubmission(BaseModel):
    definition_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    profile: TaskTemplateProfile
    runtime: LocalProcessSpec
    artifacts: list[StaticArtifactSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifacts(self) -> "TaskDefinitionSubmission":
        ids = [item.artifact_id for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("Artifact IDs must be unique within a definition")
        return self


class TaskDefinitionRecord(TaskDefinitionSubmission):
    revision: int = Field(ge=1)
    digest: str = Field(min_length=64, max_length=64)
    origin_node_id: str = Field(min_length=1)
    created_at_utc: datetime = Field(default_factory=utc_now)

    def materialize(self, run_id: str, owner_node_id: str) -> TaskDefinition:
        profile = TaskProfile(
            task_id=run_id,
            current_node_id=owner_node_id,
            **self.profile.model_dump(),
        )
        return TaskDefinition(
            profile=profile,
            runtime=self.runtime.model_copy(deep=True),
            artifacts=[item.model_copy(deep=True) for item in self.artifacts],
        )


class TaskRunSubmission(BaseModel):
    definition_id: str = Field(min_length=1)
    revision: int | None = Field(default=None, ge=1)
    initial_owner_node_id: str | None = Field(default=None, min_length=1)
    idempotency_key: str = Field(min_length=1, max_length=200)
    auto_start: bool = False
    labels: dict[str, str] = Field(default_factory=dict)


class TaskRunRecord(BaseModel):
    run_id: str = Field(min_length=1)
    definition_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    definition_digest: str = Field(min_length=64, max_length=64)
    origin_node_id: str = Field(min_length=1)
    initial_owner_node_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    request_digest: str = Field(min_length=64, max_length=64)
    auto_start: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    created_at_utc: datetime = Field(default_factory=utc_now)


class TaskRunView(BaseModel):
    run: TaskRunRecord
    state: dict


class TaskCatalogSnapshot(BaseModel):
    reporting_node_id: str = Field(min_length=1)
    generated_at_utc: datetime = Field(default_factory=utc_now)
    definitions: list[TaskDefinitionRecord] = Field(default_factory=list)
    runs: list[TaskRunRecord] = Field(default_factory=list)
