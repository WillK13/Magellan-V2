from __future__ import annotations

from collections import Counter
from typing import Any

from magellan.experiments.stage5c import (
    STAGE5C_DESTINATION_ID,
    STAGE5C_SOURCE_IDS,
    stage5c_passes,
)

STAGE5E3_DESTINATION_ID = STAGE5C_DESTINATION_ID
STAGE5E3_SOURCE_IDS = STAGE5C_SOURCE_IDS
BENCHMARK_CLASS_ID = "benchmark-json-medium"
EXPECTED_TASK_COUNT = 1 + len(STAGE5E3_SOURCE_IDS)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def witness_passes(
    rows: list[dict[str, Any]],
    *,
    expected_task_ids: set[str],
) -> bool:
    if len(rows) != len(expected_task_ids):
        return False
    if {str(row.get("task_id")) for row in rows} != expected_task_ids:
        return False
    for row in rows:
        if str(row.get("class_id")) != BENCHMARK_CLASS_ID:
            return False
        if str(row.get("status")) != "running":
            return False
        if not _truthy(row.get("live")):
            return False
        if int(float(row.get("process_count") or 0)) < 1:
            return False
        if str(row.get("process_state") or "")[:1].upper() in {"Z", "X"}:
            return False
        if float(row.get("memory_rss_mb") or 0.0) <= 0.0:
            return False
    return True


def resource_rows_pass(
    rows: list[dict[str, Any]],
    *,
    expected_owner_counts: dict[str, int],
    benchmark_cpu_cores: float,
    benchmark_memory_mb: int,
) -> bool:
    if {str(row.get("node_id")) for row in rows} != set(expected_owner_counts):
        return False
    for row in rows:
        node_id = str(row.get("node_id"))
        count = int(expected_owner_counts[node_id])
        if int(float(row.get("expected_owned_task_count") or 0)) != count:
            return False
        if int(float(row.get("actual_owned_task_count") or 0)) != count:
            return False
        expected_cpu = count * benchmark_cpu_cores
        actual_cpu = float(row.get("reserved_cpu_cores") or 0.0)
        if abs(actual_cpu - expected_cpu) > 1e-6:
            return False
        expected_memory = count * benchmark_memory_mb
        actual_memory = int(float(row.get("reserved_memory_mb") or 0))
        if actual_memory != expected_memory:
            return False
        if not _truthy(row.get("reservation_matches_expected")):
            return False
        if not _truthy(row.get("capacity_respected")):
            return False
        if float(row.get("resource_busy_fraction") or 0.0) > 1.0 + 1e-9:
            return False
    return True


def stage5e3_passes(
    *,
    source_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    bid_rows: list[dict[str, Any]],
    migration_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    ownership_ok: bool,
    resident_task_id: str,
    benchmark_cpu_cores: float,
    benchmark_memory_mb: int,
    capacity_cpu_cores: float,
    expected_git_sha: str,
    pre_witness_rows: list[dict[str, Any]],
    post_witness_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> bool:
    if not stage5c_passes(
        source_rows=source_rows,
        decision_rows=decision_rows,
        bid_rows=bid_rows,
        migration_rows=migration_rows,
        final_rows=final_rows,
        ownership_ok=ownership_ok,
        destination_id=STAGE5E3_DESTINATION_ID,
        resident_task_id=resident_task_id,
        resident_cpu_cores=benchmark_cpu_cores,
        benchmark_cpu_cores=benchmark_cpu_cores,
        capacity_cpu_cores=capacity_cpu_cores,
        expected_git_sha=expected_git_sha,
    ):
        return False

    task_ids = {resident_task_id, *(str(row["task_id"]) for row in source_rows)}
    if not witness_passes(pre_witness_rows, expected_task_ids=task_ids):
        return False
    if not witness_passes(post_witness_rows, expected_task_ids=task_ids):
        return False

    post_by_task = {str(row["task_id"]): row for row in post_witness_rows}
    final_by_task = {str(row["task_id"]): row for row in final_rows}
    for task_id in task_ids:
        if task_id not in final_by_task or task_id not in post_by_task:
            return False
        if str(post_by_task[task_id].get("node_id")) != str(
            final_by_task[task_id].get("final_owner_node_id")
        ):
            return False

    destination_rows = [
        row for row in post_witness_rows
        if str(row.get("node_id")) == STAGE5E3_DESTINATION_ID
    ]
    if len(destination_rows) != 2:
        return False
    if resident_task_id not in {str(row.get("task_id")) for row in destination_rows}:
        return False

    owner_counts = Counter(
        str(row.get("final_owner_node_id")) for row in final_rows
    )
    expected_owner_counts = {
        str(row.get("node_id")): 0 for row in resource_rows
    }
    if STAGE5E3_DESTINATION_ID not in expected_owner_counts:
        return False
    if not set(STAGE5E3_SOURCE_IDS).issubset(expected_owner_counts):
        return False
    expected_owner_counts[STAGE5E3_DESTINATION_ID] = 2
    migrated_source = None
    for source in source_rows:
        task_id = str(source["task_id"])
        final_owner = str(final_by_task[task_id].get("final_owner_node_id"))
        if final_owner == STAGE5E3_DESTINATION_ID:
            migrated_source = str(source["source_node_id"])
            break
    if migrated_source is None:
        return False
    for source_id in STAGE5E3_SOURCE_IDS:
        expected_owner_counts[source_id] = 0 if source_id == migrated_source else 1

    if any(owner_counts.get(node_id, 0) != count for node_id, count in expected_owner_counts.items()):
        return False

    if not resource_rows_pass(
        resource_rows,
        expected_owner_counts=expected_owner_counts,
        benchmark_cpu_cores=benchmark_cpu_cores,
        benchmark_memory_mb=benchmark_memory_mb,
    ):
        return False

    return True
