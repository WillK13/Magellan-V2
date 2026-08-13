from __future__ import annotations

from datetime import datetime, timezone

from magellan.experiments.events import ExperimentEventJournal


def test_experiment_event_journal_is_append_only_and_survives_restart(tmp_path) -> None:
    journal = ExperimentEventJournal(tmp_path, "boston")
    first = journal.append(
        "scheduler_decision",
        task_id="task-1",
        generation=0,
        trace_time_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        payload={"selected": "continue"},
    )
    second = journal.append(
        "migration_completed",
        task_id="task-1",
        generation=1,
        payload={"destination_node_id": "france"},
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert journal.last_sequence == 2

    restarted = ExperimentEventJournal(tmp_path, "boston")
    assert restarted.last_sequence == 2
    assert [event.sequence for event in restarted.list_events()] == [1, 2]
    assert [
        event.event_type
        for event in restarted.list_events(task_id="task-1", after_sequence=1)
    ] == ["migration_completed"]


def test_experiment_event_journal_filters_by_type(tmp_path) -> None:
    journal = ExperimentEventJournal(tmp_path, "virginia")
    journal.append("scheduler_decision", task_id="a")
    journal.append("scheduler_decision", task_id="b")
    journal.append("migration_failed", task_id="a")

    events = journal.list_events(task_id="a", event_type="scheduler_decision")
    assert len(events) == 1
    assert events[0].node_id == "virginia"
