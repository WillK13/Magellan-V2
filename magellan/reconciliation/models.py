from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from magellan.migration.models import OwnershipUpdate


class OwnershipSnapshot(BaseModel):
    reporting_node_id: str = Field(min_length=1)
    generated_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updates: list[OwnershipUpdate] = Field(default_factory=list)
