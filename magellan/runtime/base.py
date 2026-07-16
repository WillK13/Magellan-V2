from __future__ import annotations

from typing import Protocol

from magellan.state.task_models import TaskRuntimeState


class TaskRuntime(Protocol):
    def start(self, task_id: str) -> TaskRuntimeState:
        ...

    def stop(self, task_id: str) -> TaskRuntimeState:
        ...

    def reconcile(self) -> None:
        ...
