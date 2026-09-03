from __future__ import annotations

from magellan.experiments.stage5b import (
    STAGE5B_SOURCE_IDS,
    active_task_ids,
    ownership_converged,
    stage5b_passes,
)


def test_active_task_ids_only_returns_live_states() -> None:
    payload = {
        "tasks": [
            {"state": {"task_id": "a", "status": "running"}},
            {"state": {"task_id": "b", "status": "completed"}},
            {"state": {"task_id": "c", "status": "paused"}},
            {"state": {"task_id": "d", "status": "remote"}},
        ]
    }
    assert active_task_ids(payload) == ["a", "c"]


def test_ownership_convergence_requires_all_reporters_to_match() -> None:
    snapshots = {
        node: {
            "updates": [
                {
                    "task_id": "task-1",
                    "owner_node_id": "ethiopia",
                    "generation": 1,
                    "status": "running",
                    "last_migration_id": "m1",
                }
            ]
        }
        for node in ["a", "b", "c"]
    }
    ok, rows = ownership_converged(snapshots, ["task-1"])
    assert ok
    assert len(rows) == 3
    snapshots["c"]["updates"][0]["owner_node_id"] = "france"
    ok, _ = ownership_converged(snapshots, ["task-1"])
    assert not ok


def test_stage5b_pass_requires_decisions_on_origins_and_multiple_bid_sources() -> None:
    sha = "a" * 40
    sources = [
        {
            "source_node_id": source,
            "task_id": f"task-{index}",
            "trigger_ok": True,
            "daemon_git_sha": sha,
        }
        for index, source in enumerate(STAGE5B_SOURCE_IDS)
    ]
    decisions = [
        {"node_id": row["source_node_id"], "task_id": row["task_id"]}
        for row in sources
    ]
    bids = [
        {
            "reporting_node_id": "ethiopia",
            "task_id": sources[0]["task_id"],
            "source_node_id": sources[0]["source_node_id"],
            "destination_node_id": "ethiopia",
        },
        {
            "reporting_node_id": "france",
            "task_id": sources[1]["task_id"],
            "source_node_id": sources[1]["source_node_id"],
            "destination_node_id": "france",
        },
    ]
    assert stage5b_passes(
        source_rows=sources,
        decision_rows=decisions,
        bid_rows=bids,
        migration_rows=[],
        ownership_ok=True,
        expected_git_sha=sha,
    )
    bids.pop()
    assert not stage5b_passes(
        source_rows=sources,
        decision_rows=decisions,
        bid_rows=bids,
        migration_rows=[],
        ownership_ok=True,
        expected_git_sha=sha,
    )
