from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from magellan.models.types import ActionType, ScoredAction


class BidStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class BidRequest(BaseModel):
    bid_id: str = Field(min_length=1)
    epoch_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

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

        if (
            self.candidate.source_node_id
            != self.source_node_id
        ):
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
