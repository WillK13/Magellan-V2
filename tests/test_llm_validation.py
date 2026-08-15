from __future__ import annotations

from magellan.experiments.llm_validation import (
    build_resume_validation,
    checkpoint_event_at_or_after,
    last_checkpoint_event,
)


def test_checkpoint_event_selection() -> None:
    events = [
        {"reason": "startup", "completed_steps": 0, "recorded_at_utc": "1"},
        {"reason": "periodic", "completed_steps": 2, "recorded_at_utc": "2"},
        {"reason": "periodic", "completed_steps": 4, "recorded_at_utc": "3"},
        {"reason": "shutdown", "completed_steps": 4, "recorded_at_utc": "4"},
    ]

    selected = checkpoint_event_at_or_after(events, minimum_steps=3)
    assert selected is not None
    assert selected["reason"] == "shutdown"
    assert last_checkpoint_event(events, reason="periodic") == events[2]


def test_resume_validation_requires_exact_checkpoint_and_progress() -> None:
    validation = build_resume_validation(
        source_checkpoint={"checkpoint_id": "abc", "completed_steps": 5},
        destination_ready={
            "resumed_checkpoint_id": "abc",
            "completed_steps": 5,
            "optimizer_state_loaded": True,
        },
        destination_progress={"completed_units": 7, "details": {"loss": 1.2}},
    )

    assert validation.checkpoint_id_matches
    assert validation.resumed_at_same_step
    assert validation.progress_continued
    assert validation.optimizer_state_loaded
    assert validation.passed


def test_resume_validation_fails_on_checkpoint_mismatch() -> None:
    validation = build_resume_validation(
        source_checkpoint={"checkpoint_id": "abc", "completed_steps": 5},
        destination_ready={
            "resumed_checkpoint_id": "other",
            "completed_steps": 5,
            "optimizer_state_loaded": True,
        },
        destination_progress={"completed_units": 6},
    )

    assert not validation.checkpoint_id_matches
    assert not validation.passed
