from __future__ import annotations

from typing import Any


STAGE5D_RING = (
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
    "boston",
)


def expected_hops() -> list[tuple[str, str]]:
    return list(zip(STAGE5D_RING[:-1], STAGE5D_RING[1:]))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def progress_is_monotonic(hop_rows: list[dict[str, Any]]) -> bool:
    observed: list[float] = []
    for row in hop_rows:
        before = _float_or_none(row.get("progress_before"))
        after = _float_or_none(row.get("progress_after"))
        if before is not None:
            observed.append(before)
        if after is not None:
            if before is not None and after + 1e-9 < before:
                return False
            observed.append(after)
    return bool(observed) and all(
        later + 1e-9 >= earlier
        for earlier, later in zip(observed, observed[1:])
    )


def stage5d_passes(
    *,
    hop_rows: list[dict[str, Any]],
    final_owner_node_id: str,
    final_generation: int,
    ownership_converged_final: bool,
    expected_git_sha: str,
) -> bool:
    hops = expected_hops()
    if len(hop_rows) != len(hops):
        return False

    seen_migration_ids: set[str] = set()
    for index, (row, (source, destination)) in enumerate(
        zip(hop_rows, hops), start=1
    ):
        if int(row.get("hop_index", -1)) != index:
            return False
        if str(row.get("source_node_id")) != source:
            return False
        if str(row.get("destination_node_id")) != destination:
            return False
        if str(row.get("source_daemon_git_sha")) != expected_git_sha:
            return False
        if str(row.get("destination_daemon_git_sha")) != expected_git_sha:
            return False
        if str(row.get("owner_before")) != source:
            return False
        if int(row.get("generation_before", -1)) != index - 1:
            return False
        if not bool(row.get("migrated")):
            return False
        if str(row.get("owner_after")) != destination:
            return False
        if int(row.get("generation_after", -1)) != index:
            return False
        if str(row.get("destination_status_after")) != "running":
            return False
        if not row.get("destination_pid_after"):
            return False
        if not bool(row.get("ownership_converged")):
            return False
        if str(row.get("source_record_role")) != "source":
            return False
        if str(row.get("source_record_status")) != "activated":
            return False
        if str(row.get("destination_record_role")) != "destination":
            return False
        if str(row.get("destination_record_status")) != "activated":
            return False
        if str(row.get("bid_status")) not in {"accepted", "consumed"}:
            return False
        migration_id = str(row.get("migration_id") or "")
        if not migration_id or migration_id in seen_migration_ids:
            return False
        seen_migration_ids.add(migration_id)
        if float(row.get("total_downtime_seconds") or 0) <= 0:
            return False

    if not progress_is_monotonic(hop_rows):
        return False
    if final_owner_node_id != STAGE5D_RING[0]:
        return False
    if final_generation != len(hops):
        return False
    if not ownership_converged_final:
        return False

    sources = {str(row["source_node_id"]) for row in hop_rows}
    destinations = {str(row["destination_node_id"]) for row in hop_rows}
    expected_nodes = set(STAGE5D_RING[:-1])
    return sources == expected_nodes and destinations == expected_nodes
