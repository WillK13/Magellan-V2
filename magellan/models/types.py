from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from magellan.capabilities.models import TaskCompatibilityRequirements


class ActionType(str, Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    MIGRATE = "migrate"


class TaskResourceRequest(BaseModel):
    cpu_cores: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_count: int = Field(default=0, ge=0)
    accelerator_type: str | None = None


class TaskProfile(BaseModel):
    task_id: str = Field(min_length=1)
    workload_type: str = Field(min_length=1)
    current_node_id: str = Field(min_length=1)

    # Measured or configured task properties.
    power_kw: float = Field(gt=0)
    checkpoint_bytes: int = Field(ge=0)
    data_bytes: int = Field(default=0, ge=0)

    # Nodes that already have the static dataset/model.
    prestaged_node_ids: set[str] = Field(default_factory=set)

    # Optional task-level constraints/state.
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    accumulated_cost_usd: float = Field(default=0.0, ge=0)
    cost_cap_usd: float | None = Field(default=None, gt=0)
    last_migration_at: datetime | None = None
    last_pause_at: datetime | None = None

    # Carried with task bids. The current slot arbiter does not yet enforce
    # these fields; the resource-aware auction milestone will.
    priority: int = Field(default=0, ge=0, le=100)
    deadline_at_utc: datetime | None = None
    resource_request: TaskResourceRequest = Field(
        default_factory=TaskResourceRequest
    )
    compatibility: TaskCompatibilityRequirements = Field(
        default_factory=TaskCompatibilityRequirements
    )


class RawActionEstimate(BaseModel):
    action: ActionType
    source_node_id: str
    destination_node_id: str | None = None

    time_seconds: float = Field(ge=0)
    carbon_grams: float = Field(ge=0)
    cost_usd: float = Field(ge=0)

    details: dict[str, Any] = Field(default_factory=dict)


class ScoredAction(RawActionEstimate):
    normalized_time: float = Field(ge=0, le=1)
    normalized_carbon: float = Field(ge=0, le=1)
    normalized_cost: float = Field(ge=0, le=1)
    score: float = Field(ge=0)


class DecisionResult(BaseModel):
    selected: ScoredAction
    ranked_actions: list[ScoredAction]
    reason: str
    policy_metadata: dict[str, Any] = Field(default_factory=dict)
