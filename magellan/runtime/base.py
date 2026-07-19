from __future__ import annotations

from datetime import datetime
from typing import Protocol

from magellan.runtime.local_process import RuntimeReconcileEvent
from magellan.state.task_models import TaskRuntimeState


class TaskRuntime(Protocol):
    def start(self, task_id: str) -> TaskRuntimeState:
        ...

    def pause(
        self,
        task_id: str,
        paused_at_utc: datetime,
        resume_at_utc: datetime,
        resume_wall_at_utc: datetime,
        reason: str,
    ) -> TaskRuntimeState:
        ...

    def resume(self, task_id: str) -> TaskRuntimeState:
        ...

    def stop(self, task_id: str) -> TaskRuntimeState:
        ...

    def reconcile(self) -> list[RuntimeReconcileEvent]:
        ...
