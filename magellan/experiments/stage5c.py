from __future__ import annotations

from typing import Any

STAGE5C_SOURCE_IDS = (
    "boston",
    "california",
    "south-australia",
    "virginia",
)
STAGE5C_DESTINATION_ID = "ethiopia"

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


def ownership_for_task(
    snapshot: dict[str, Any],
    task_id: str,
) -> dict[str, Any] | None:
    matches = [
        update
        for update in snapshot.get("updates", [])
        if update.get("task_id") == task_id
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: int(item.get("generation", 0)))


def ownership_converged(
    snapshots: dict[str, dict[str, Any]],
    task_ids: list[str],
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    ok = True
    for task_id in task_ids:
        values = []
        for reporting_node_id, snapshot in sorted(snapshots.items()):
            update = ownership_for_task(snapshot, task_id)
            rows.append(
                {
                    "reporting_node_id": reporting_node_id,
                    "task_id": task_id,
                    "owner_node_id": (
                        update.get("owner_node_id", "") if update else ""
                    ),
                    "generation": (
                        int(update.get("generation", 0)) if update else -1
                    ),
                    "status": update.get("status", "") if update else "",
                    "last_migration_id": (
                        update.get("last_migration_id") or "" if update else ""
                    ),
                }
            )
            if update is None:
                ok = False
                continue
            values.append(
                (
                    str(update.get("owner_node_id")),
                    int(update.get("generation", 0)),
                    str(update.get("status") or ""),
                    str(update.get("last_migration_id") or ""),
                )
            )
        if len(values) != len(snapshots) or len(set(values)) != 1:
            ok = False
    return ok, rows


def is_successful_bid_status(status: str) -> bool:
    # By collection time, a successfully activated reservation has normally
    # advanced from accepted -> consumed.
    return status in {"accepted", "consumed"}


def is_resource_contention_rejection(row: dict[str, Any]) -> bool:
    if str(row.get("status")) != "rejected":
        return False
    reason = str(row.get("decision_reason") or "").lower()
    return (
        "unreserved cpu" in reason
        or "resources are already owned or reserved" in reason
        or "task slots are already owned or reserved" in reason
    )


def stage5c_passes(
    *,
    source_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    bid_rows: list[dict[str, Any]],
    migration_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    ownership_ok: bool,
    destination_id: str,
    resident_task_id: str,
    resident_cpu_cores: float,
    benchmark_cpu_cores: float,
    capacity_cpu_cores: float,
    expected_git_sha: str,
) -> bool:
    if len(source_rows) != len(STAGE5C_SOURCE_IDS):
        return False
    if not all(bool(row.get("trigger_ok")) for row in source_rows):
        return False
    if not all(
        str(row.get("daemon_git_sha")) == expected_git_sha
        for row in source_rows
    ):
        return False

    by_task_decision = {
        str(row.get("task_id")): row
        for row in decision_rows
    }
    for source in source_rows:
        task_id = str(source["task_id"])
        decision = by_task_decision.get(task_id)
        if decision is None:
            return False
        if str(decision.get("node_id")) != str(source["source_node_id"]):
            return False
        if str(decision.get("selected_action")) != "migrate":
            return False
        if str(decision.get("selected_destination_node_id")) != destination_id:
            return False

    challenge_task_ids = {str(row["task_id"]) for row in source_rows}
    challenge_bids = [
        row for row in bid_rows
        if str(row.get("task_id")) in challenge_task_ids
        and str(row.get("destination_node_id")) == destination_id
    ]
    if len(challenge_bids) != len(STAGE5C_SOURCE_IDS):
        return False
    if {str(row.get("source_node_id")) for row in challenge_bids} != set(
        STAGE5C_SOURCE_IDS
    ):
        return False
    successful = [
        row for row in challenge_bids
        if is_successful_bid_status(str(row.get("status")))
    ]
    rejected = [
        row for row in challenge_bids
        if str(row.get("status")) == "rejected"
    ]
    if len(successful) != 1 or len(rejected) != len(STAGE5C_SOURCE_IDS) - 1:
        return False
    if not all(is_resource_contention_rejection(row) for row in rejected):
        return False

    completed = [
        row for row in migration_rows
        if str(row.get("status")) == "completed"
        and str(row.get("task_id")) in challenge_task_ids
    ]
    failed = [
        row for row in migration_rows
        if str(row.get("status")) == "failed"
        and str(row.get("task_id")) in challenge_task_ids
    ]
    if len(completed) != 1 or failed:
        return False

    if not ownership_ok:
        return False

    final_by_task = {str(row["task_id"]): row for row in final_rows}
    resident = final_by_task.get(resident_task_id)
    if resident is None or str(resident.get("final_owner_node_id")) != destination_id:
        return False

    migrated_to_destination = [
        row for row in final_rows
        if str(row.get("task_id")) in challenge_task_ids
        and str(row.get("final_owner_node_id")) == destination_id
    ]
    if len(migrated_to_destination) != 1:
        return False

    for source in source_rows:
        task_id = str(source["task_id"])
        final = final_by_task.get(task_id)
        if final is None:
            return False
        if task_id == str(migrated_to_destination[0]["task_id"]):
            continue
        if str(final.get("final_owner_node_id")) != str(source["source_node_id"]):
            return False

    # The frozen benchmark request must produce exactly one additional fit:
    # resident + one challenger <= capacity, resident + two challengers > capacity.
    if resident_cpu_cores + benchmark_cpu_cores > capacity_cpu_cores + 1e-9:
        return False
    if resident_cpu_cores + 2 * benchmark_cpu_cores <= capacity_cpu_cores + 1e-9:
        return False

    return True
