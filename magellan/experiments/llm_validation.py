from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ResumeValidation:
    source_checkpoint_id: str
    destination_checkpoint_id: str
    source_completed_steps: int
    destination_ready_steps: int
    destination_progress_steps: int
    optimizer_state_loaded: bool

    @property
    def checkpoint_id_matches(self) -> bool:
        return self.source_checkpoint_id == self.destination_checkpoint_id

    @property
    def resumed_at_same_step(self) -> bool:
        return self.source_completed_steps == self.destination_ready_steps

    @property
    def progress_continued(self) -> bool:
        return self.destination_progress_steps > self.source_completed_steps

    @property
    def passed(self) -> bool:
        return (
            self.checkpoint_id_matches
            and self.resumed_at_same_step
            and self.progress_continued
            and self.optimizer_state_loaded
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_checkpoint_id": self.source_checkpoint_id,
            "destination_checkpoint_id": self.destination_checkpoint_id,
            "source_completed_steps": self.source_completed_steps,
            "destination_ready_steps": self.destination_ready_steps,
            "destination_progress_steps": self.destination_progress_steps,
            "optimizer_state_loaded": self.optimizer_state_loaded,
            "checkpoint_id_matches": self.checkpoint_id_matches,
            "resumed_at_same_step": self.resumed_at_same_step,
            "progress_continued": self.progress_continued,
            "passed": self.passed,
        }


def checkpoint_event_at_or_after(
    events: Iterable[dict[str, Any]],
    *,
    minimum_steps: int,
    reasons: set[str] | None = None,
) -> dict[str, Any] | None:
    allowed = reasons or {"periodic", "shutdown", "completion"}
    matching = [
        event
        for event in events
        if event.get("reason") in allowed
        and int(event.get("completed_steps", -1)) >= minimum_steps
    ]
    if not matching:
        return None
    return max(
        matching,
        key=lambda event: (
            int(event.get("completed_steps", -1)),
            str(event.get("recorded_at_utc", "")),
        ),
    )


def last_checkpoint_event(
    events: Iterable[dict[str, Any]],
    *,
    reason: str | None = None,
) -> dict[str, Any] | None:
    matching = [
        event
        for event in events
        if reason is None or event.get("reason") == reason
    ]
    if not matching:
        return None
    return matching[-1]


def build_resume_validation(
    *,
    source_checkpoint: dict[str, Any],
    destination_ready: dict[str, Any],
    destination_progress: dict[str, Any],
) -> ResumeValidation:
    source_checkpoint_id = source_checkpoint.get("checkpoint_id")
    destination_checkpoint_id = destination_ready.get("resumed_checkpoint_id")
    if not isinstance(source_checkpoint_id, str) or not source_checkpoint_id:
        raise ValueError("Source checkpoint is missing checkpoint_id")
    if not isinstance(destination_checkpoint_id, str) or not destination_checkpoint_id:
        raise ValueError("Destination readiness is missing resumed_checkpoint_id")

    details = destination_progress.get("details")
    if details is not None and not isinstance(details, dict):
        raise ValueError("Progress details must be an object when present")

    return ResumeValidation(
        source_checkpoint_id=source_checkpoint_id,
        destination_checkpoint_id=destination_checkpoint_id,
        source_completed_steps=int(source_checkpoint["completed_steps"]),
        destination_ready_steps=int(destination_ready["completed_steps"]),
        destination_progress_steps=int(destination_progress["completed_units"]),
        optimizer_state_loaded=bool(destination_ready.get("optimizer_state_loaded")),
    )
