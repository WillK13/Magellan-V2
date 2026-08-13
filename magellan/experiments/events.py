from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentEvent(BaseModel):
    """One append-only structured observation emitted by a Magellan daemon."""

    sequence: int = Field(ge=1)
    event_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    observed_at_utc: datetime = Field(default_factory=utc_now)
    trace_time_utc: datetime | None = None
    task_id: str | None = None
    generation: int | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class ExperimentEventJournal:
    """Durable JSONL event stream used to reconstruct experiment decisions.

    The journal is append-only. It deliberately lives beside the existing
    control-plane state so a daemon restart does not erase experiment evidence.
    """

    def __init__(self, state_root: str | Path, node_id: str) -> None:
        self._path = Path(state_root) / "control" / "experiment-events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)
        self._node_id = node_id
        self._lock = RLock()
        self._last_sequence = self._scan_last_sequence()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._last_sequence

    def _scan_last_sequence(self) -> int:
        last = 0
        with self._path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = ExperimentEvent.model_validate_json(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid experiment event journal line {line_number}: {exc}"
                    ) from exc
                last = max(last, event.sequence)
        return last

    def append(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        generation: int | None = None,
        trace_time_utc: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> ExperimentEvent:
        with self._lock:
            event = ExperimentEvent(
                sequence=self._last_sequence + 1,
                event_id=str(uuid4()),
                node_id=self._node_id,
                event_type=event_type,
                trace_time_utc=trace_time_utc,
                task_id=task_id,
                generation=generation,
                payload=payload or {},
            )
            encoded = json.dumps(
                event.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._last_sequence = event.sequence
            return event.model_copy(deep=True)

    def list_events(
        self,
        *,
        after_sequence: int = 0,
        task_id: str | None = None,
        event_type: str | None = None,
        limit: int = 10_000,
    ) -> list[ExperimentEvent]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")

        events: list[ExperimentEvent] = []
        with self._lock, self._path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                event = ExperimentEvent.model_validate_json(line)
                if event.sequence <= after_sequence:
                    continue
                if task_id is not None and event.task_id != task_id:
                    continue
                if event_type is not None and event.event_type != event_type:
                    continue
                events.append(event)
                if len(events) >= limit:
                    break
        return events
