from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from magellan.capabilities.models import TaskCompatibilityRequirements

from magellan.models.types import ActionType, ScoredAction, TaskResourceRequest


class AuctionStrategy(str, Enum):
    LOWEST_SCORE = "lowest_score"
    SHORTEST_REMAINING = "shortest_remaining"
    LONGEST_REMAINING = "longest_remaining"
    CREDIT_FAIR = "credit_fair"
    HIGHEST_REGRET = "highest_regret"
    PRIORITY_DEADLINE = "priority_deadline"
    RESOURCE_EFFICIENCY = "resource_efficiency"


class BidStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    ACTIVATING = "activating"
    CONSUMED = "consumed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskBidContext(BaseModel):
    workload_type: str = Field(min_length=1)
    priority: int = Field(default=0, ge=0, le=100)
    deadline_at_utc: datetime | None = None
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    checkpoint_bytes: int = Field(default=0, ge=0)
    static_data_bytes: int = Field(default=0, ge=0)
    accumulated_cost_usd: float = Field(default=0.0, ge=0)
    cost_cap_usd: float | None = Field(default=None, gt=0)
    resource_request: TaskResourceRequest = Field(
        default_factory=TaskResourceRequest
    )

    # The best action available if this destination rejects the task.
    # A larger opportunity loss means rejection is more harmful.
    fallback_action: ActionType | None = None
    fallback_destination_node_id: str | None = None
    fallback_score: float | None = Field(default=None, ge=0)
    opportunity_loss: float = Field(default=0.0, ge=0)

    effective_power_kw: float | None = Field(default=None, gt=0)
    power_source: str | None = None
    power_confidence: float | None = Field(default=None, ge=0, le=1)
    telemetry_freshness: str | None = None
    compatibility: TaskCompatibilityRequirements = Field(
        default_factory=TaskCompatibilityRequirements
    )


class BidRequest(BaseModel):
    bid_id: str = Field(min_length=1)
    epoch_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    bidder_type: Literal["task"] = "task"
    task_context: TaskBidContext | None = None

    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    candidate: ScoredAction
    submitted_at_utc: datetime

    @model_validator(mode="after")
    def validate_migration_candidate(self) -> "BidRequest":
        if self.candidate.action != ActionType.MIGRATE:
            raise ValueError(
                "A bid candidate must represent a migrate action"
            )

        if self.candidate.source_node_id != self.source_node_id:
            raise ValueError(
                "Candidate source does not match bid source"
            )

        if (
            self.candidate.destination_node_id
            != self.destination_node_id
        ):
            raise ValueError(
                "Candidate destination does not match bid destination"
            )

        return self


class BidRecord(BidRequest):
    status: BidStatus = BidStatus.PENDING
    received_at_utc: datetime
    decided_at_utc: datetime | None = None
    decision_reason: str | None = None

    reservation_expires_at_utc: datetime | None = None
    activation_started_at_utc: datetime | None = None
    consumed_at_utc: datetime | None = None

    auction_strategy: AuctionStrategy | None = None
    auction_rank: int | None = Field(default=None, ge=1)
    auction_credit_before: float = Field(default=0.0, ge=0)
    auction_credit_after: float = Field(default=0.0, ge=0)
    resource_fit: bool | None = None
    compatibility_fit: bool | None = None
    compatibility_reasons: list[str] = Field(default_factory=list)
    auction_metrics: dict[
        str,
        float | int | str | bool | None,
    ] = Field(default_factory=dict)
