from __future__ import annotations

from collections import defaultdict
from typing import Any


STAGE5B_SOURCE_IDS = (
    "boston",
    "california",
    "south-australia",
    "virginia",
)
ACTIVE_TASK_STATUSES = {
    "running",
    "paused",
    "migrating",
    "recovering",
}


def active_task_ids(task_payload: dict[str, Any]) -> list[str]:
    output = []
    for item in task_payload.get("tasks", []):
        state = item.get("state", {})
        if str(state.get("status")) in ACTIVE_TASK_STATUSES:
            task_id = state.get("task_id")
            if task_id:
                output.append(str(task_id))
    return sorted(output)


def ownership_for_task(snapshot: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    matches = [
        update
        for update in snapshot.get("updates", [])
        if update.get("task_id") == task_id
    ]
    if not matches:
        return None
    return max(matches, key=lambda update: int(update.get("generation", 0)))


def ownership_converged(
    snapshots: dict[str, dict[str, Any]],
    task_ids: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    passed = True
    for task_id in task_ids:
        values = []
        for reporting_node_id, snapshot in sorted(snapshots.items()):
            update = ownership_for_task(snapshot, task_id)
            rows.append(
                {
                    "task_id": task_id,
                    "reporting_node_id": reporting_node_id,
                    "owner_node_id": update.get("owner_node_id") if update else "",
                    "generation": update.get("generation") if update else "",
                    "status": update.get("status") if update else "",
                    "last_migration_id": update.get("last_migration_id") if update else "",
                }
            )
            if update is None:
                passed = False
                continue
            values.append(
                (
                    str(update.get("owner_node_id")),
                    int(update.get("generation", 0)),
                    str(update.get("status")),
                    str(update.get("last_migration_id") or ""),
                )
            )
        if len(values) != len(snapshots) or len(set(values)) != 1:
            passed = False
    return passed, rows


def stage5b_passes(
    *,
    source_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    bid_rows: list[dict[str, Any]],
    migration_rows: list[dict[str, Any]],
    ownership_ok: bool,
    expected_git_sha: str,
) -> bool:
    expected_sources = set(STAGE5B_SOURCE_IDS)
    if {str(row.get("source_node_id")) for row in source_rows} != expected_sources:
        return False
    if len(source_rows) != len(expected_sources):
        return False
    if not all(bool(row.get("trigger_ok")) for row in source_rows):
        return False
    if not all(str(row.get("daemon_git_sha")) == expected_git_sha for row in source_rows):
        return False

    task_origin = {
        str(row["task_id"]): str(row["source_node_id"])
        for row in source_rows
    }
    decision_nodes = defaultdict(set)
    for row in decision_rows:
        decision_nodes[str(row.get("task_id"))].add(str(row.get("node_id")))
    for task_id, origin in task_origin.items():
        if decision_nodes.get(task_id) != {origin}:
            return False

    bid_sources = {
        str(row.get("source_node_id"))
        for row in bid_rows
        if str(row.get("task_id")) in task_origin
    }
    if len(bid_sources) < 2:
        return False
    for row in bid_rows:
        if str(row.get("reporting_node_id")) != str(row.get("destination_node_id")):
            return False
        if str(row.get("source_node_id")) == str(row.get("destination_node_id")):
            return False

    if any(str(row.get("status")) == "failed" for row in migration_rows):
        return False
    if not ownership_ok:
        return False
    return True
