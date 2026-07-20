from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class LocalNodeState:
    node_id: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    active_task_ids: set[str] = field(default_factory=set)

    @property
    def active_tasks(self) -> int:
        return len(self.active_task_ids)
