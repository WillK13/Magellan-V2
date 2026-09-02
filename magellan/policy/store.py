from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from magellan.policy.models import AdaptiveTaskPolicyState


class AdaptivePolicyStore:
    """Atomic durable storage for per-task adaptive policy state."""

    def __init__(self, state_root: str | Path) -> None:
        self.path = Path(state_root) / "control" / "adaptive-policy.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._states: dict[str, AdaptiveTaskPolicyState] = {}
        self._defer_depth: ContextVar[int] = ContextVar(
            f"adaptive-policy-store-defer-{id(self)}",
            default=0,
        )
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._states = {
            task_id: AdaptiveTaskPolicyState.model_validate(value)
            for task_id, value in raw.get("tasks", {}).items()
        }

    def _persist(self) -> None:
        payload = {
            "format_version": 1,
            "tasks": {
                task_id: state.model_dump(mode="json")
                for task_id, state in sorted(self._states.items())
            },
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix="adaptive-policy-",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _persist_or_defer(self) -> None:
        if self._defer_depth.get() > 0:
            self._dirty = True
            return
        self._persist()
        self._dirty = False

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Defer writes in this execution context and flush once on exit.

        The defer flag is context-local rather than process-global. Unrelated
        API/peer tasks using the same store therefore retain immediate durable
        persistence while a scheduler epoch batches its own repeated updates.
        Nested batches flush only when the outermost batch exits. If the body
        raises, completed in-memory updates are still flushed before the
        exception propagates.
        """
        token = self._defer_depth.set(self._defer_depth.get() + 1)
        try:
            yield
        finally:
            outermost = self._defer_depth.get() == 1
            self._defer_depth.reset(token)
            if outermost:
                with self._lock:
                    if self._dirty:
                        self._persist()
                        self._dirty = False

    def get(self, task_id: str) -> AdaptiveTaskPolicyState | None:
        with self._lock:
            state = self._states.get(task_id)
            return state.model_copy(deep=True) if state is not None else None

    def put(self, state: AdaptiveTaskPolicyState) -> AdaptiveTaskPolicyState:
        with self._lock:
            self._states[state.task_id] = state.model_copy(deep=True)
            self._persist_or_defer()
            return state.model_copy(deep=True)

    def merge(self, state: AdaptiveTaskPolicyState) -> bool:
        """Install a newer task policy snapshot received from a peer."""
        with self._lock:
            current = self._states.get(state.task_id)
            if current is not None:
                if state.decision_count < current.decision_count:
                    return False
                if (
                    state.decision_count == current.decision_count
                    and state.updated_at_utc <= current.updated_at_utc
                ):
                    return False
            self._states[state.task_id] = state.model_copy(deep=True)
            self._persist_or_defer()
            return True

    def delete(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._states:
                return False
            del self._states[task_id]
            self._persist_or_defer()
            return True

    def list_states(self) -> list[AdaptiveTaskPolicyState]:
        with self._lock:
            return [
                self._states[key].model_copy(deep=True)
                for key in sorted(self._states)
            ]
