from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ProgressSnapshot(BaseModel):
    format_version: int = Field(default=1, ge=1)
    task_id: str = Field(min_length=1)
    completed_units: float = Field(ge=0)
    total_units: float | None = Field(default=None, gt=0)
    updated_at_utc: datetime
    node_id: str | None = None
    details: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_completed_units(self) -> "ProgressSnapshot":
        if (
            self.total_units is not None
            and self.completed_units > self.total_units
        ):
            raise ValueError(
                "completed_units cannot exceed total_units"
            )
        return self


def load_progress(path: Path, task_id: str) -> ProgressSnapshot | None:
    if not path.is_file():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))

    # Compatibility with workload records that use a UNIX timestamp.
    if "updated_at_utc" not in raw and "updated_at_unix" in raw:
        raw["updated_at_utc"] = datetime.fromtimestamp(
            float(raw["updated_at_unix"]),
            tz=timezone.utc,
        ).isoformat()

    snapshot = ProgressSnapshot.model_validate(raw)

    if snapshot.task_id != task_id:
        raise ValueError(
            f"Progress record task_id={snapshot.task_id} "
            f"does not match {task_id}"
        )

    return snapshot
