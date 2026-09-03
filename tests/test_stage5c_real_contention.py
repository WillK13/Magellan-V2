from __future__ import annotations

from magellan.experiments.stage5c import (
    STAGE5C_DESTINATION_ID,
    STAGE5C_SOURCE_IDS,
    is_resource_contention_rejection,
    is_successful_bid_status,
    stage5c_passes,
)


def _source_rows():
    return [
        {
            "source_node_id": source,
            "task_id": f"task-{source}",
            "daemon_git_sha": "a" * 40,
            "trigger_ok": True,
        }
        for source in STAGE5C_SOURCE_IDS
    ]


def _decision_rows():
    return [
        {
            "node_id": source,
            "task_id": f"task-{source}",
            "selected_action": "migrate",
            "selected_destination_node_id": STAGE5C_DESTINATION_ID,
        }
        for source in STAGE5C_SOURCE_IDS
    ]


def _bid_rows():
    rows = []
    for index, source in enumerate(STAGE5C_SOURCE_IDS):
        if index == 0:
            status = "consumed"
            reason = "Destination activation completed"
        else:
            status = "rejected"
            reason = "Insufficient unreserved CPU cores"
        rows.append(
            {
                "task_id": f"task-{source}",
                "source_node_id": source,
                "destination_node_id": STAGE5C_DESTINATION_ID,
                "status": status,
                "decision_reason": reason,
            }
        )
    return rows


def _final_rows():
    rows = [
        {
            "task_id": "resident",
            "role": "resident",
            "final_owner_node_id": STAGE5C_DESTINATION_ID,
        }
    ]
    for index, source in enumerate(STAGE5C_SOURCE_IDS):
        rows.append(
            {
                "task_id": f"task-{source}",
                "role": "challenger",
                "final_owner_node_id": (
                    STAGE5C_DESTINATION_ID if index == 0 else source
                ),
            }
        )
    return rows


def test_consumed_is_successful_bid_outcome() -> None:
    assert is_successful_bid_status("accepted")
    assert is_successful_bid_status("consumed")
    assert not is_successful_bid_status("rejected")


def test_resource_contention_reason_detection() -> None:
    assert is_resource_contention_rejection(
        {
            "status": "rejected",
            "decision_reason": "Insufficient unreserved CPU cores",
        }
    )
    assert not is_resource_contention_rejection(
        {
            "status": "rejected",
            "decision_reason": "Destination does not provide accelerator type",
        }
    )


def test_stage5c_passes_exactly_one_measured_admission() -> None:
    assert stage5c_passes(
        source_rows=_source_rows(),
        decision_rows=_decision_rows(),
        bid_rows=_bid_rows(),
        migration_rows=[
            {
                "task_id": f"task-{STAGE5C_SOURCE_IDS[0]}",
                "status": "completed",
            }
        ],
        final_rows=_final_rows(),
        ownership_ok=True,
        destination_id=STAGE5C_DESTINATION_ID,
        resident_task_id="resident",
        resident_cpu_cores=0.9972222179,
        benchmark_cpu_cores=0.9972222179,
        capacity_cpu_cores=2.0,
        expected_git_sha="a" * 40,
    )


def test_stage5c_rejects_two_successful_challengers() -> None:
    bids = _bid_rows()
    bids[1]["status"] = "consumed"
    assert not stage5c_passes(
        source_rows=_source_rows(),
        decision_rows=_decision_rows(),
        bid_rows=bids,
        migration_rows=[
            {
                "task_id": f"task-{STAGE5C_SOURCE_IDS[0]}",
                "status": "completed",
            },
            {
                "task_id": f"task-{STAGE5C_SOURCE_IDS[1]}",
                "status": "completed",
            },
        ],
        final_rows=_final_rows(),
        ownership_ok=True,
        destination_id=STAGE5C_DESTINATION_ID,
        resident_task_id="resident",
        resident_cpu_cores=0.9972222179,
        benchmark_cpu_cores=0.9972222179,
        capacity_cpu_cores=2.0,
        expected_git_sha="a" * 40,
    )
