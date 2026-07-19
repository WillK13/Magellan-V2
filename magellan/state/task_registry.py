from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from magellan.models.types import TaskProfile, TaskResourceRequest


class TaskRegistry:
    """In-memory registry of task definitions known to this peer."""

    def __init__(self, tasks: Iterable[TaskProfile]) -> None:
        self._tasks: dict[str, TaskProfile] = {}

        for task in tasks:
            if task.task_id in self._tasks:
                raise ValueError(
                    f"Duplicate task ID: {task.task_id}"
                )

            self._tasks[task.task_id] = task

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
    ) -> "TaskRegistry":
        tasks: list[TaskProfile] = []

        for path_value in paths:
            path = Path(path_value)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Task definition does not exist: {path}"
                )

            raw = json.loads(path.read_text(encoding="utf-8"))
            tasks.append(TaskProfile.model_validate(raw))

        return cls(tasks)

    def all_tasks(self) -> list[TaskProfile]:
        return [
            task.model_copy(deep=True)
            for task in self._tasks.values()
        ]

    def owned_tasks(self, node_id: str) -> list[TaskProfile]:
        return [
            task.model_copy(deep=True)
            for task in self._tasks.values()
            if task.current_node_id == node_id
        ]

    def count_owned(self, node_id: str) -> int:
        return sum(
            task.current_node_id == node_id
            for task in self._tasks.values()
        )

    def owned_resource_requests(
        self,
        node_id: str,
    ) -> list[TaskResourceRequest]:
        return [
            task.resource_request.model_copy(deep=True)
            for task in self._tasks.values()
            if task.current_node_id == node_id
        ]

    def get(self, task_id: str) -> TaskProfile:
        try:
            return self._tasks[task_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(
                f"Unknown Magellan task: {task_id}"
            ) from exc
