from __future__ import annotations

from scripts.run_stage5e3_real_process_contention import (
    definition_payload,
    parse_ps_sessions,
)
from magellan.experiments.stage5e3 import (
    BENCHMARK_CLASS_ID,
    STAGE5E3_DESTINATION_ID,
    STAGE5E3_SOURCE_IDS,
    stage5e3_passes,
    witness_passes,
)

SHA = "a" * 40
CPU = 0.9972222178769694
MEM = 13
ALL_NODES = [
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
]


def source_rows():
    return [
        {
            "source_node_id": source,
            "task_id": f"task-{source}",
            "daemon_git_sha": SHA,
            "trigger_ok": True,
        }
        for source in STAGE5E3_SOURCE_IDS
    ]


def decision_rows():
    return [
        {
            "node_id": source,
            "task_id": f"task-{source}",
            "selected_action": "migrate",
            "selected_destination_node_id": STAGE5E3_DESTINATION_ID,
        }
        for source in STAGE5E3_SOURCE_IDS
    ]


def bid_rows():
    rows = []
    for index, source in enumerate(STAGE5E3_SOURCE_IDS):
        rows.append(
            {
                "task_id": f"task-{source}",
                "source_node_id": source,
                "destination_node_id": STAGE5E3_DESTINATION_ID,
                "status": "consumed" if index == 0 else "rejected",
                "decision_reason": (
                    "Destination activation completed"
                    if index == 0
                    else "Insufficient unreserved CPU cores"
                ),
            }
        )
    return rows


def final_rows():
    rows = [
        {
            "task_id": "resident",
            "role": "resident",
            "origin_node_id": STAGE5E3_DESTINATION_ID,
            "final_owner_node_id": STAGE5E3_DESTINATION_ID,
        }
    ]
    for index, source in enumerate(STAGE5E3_SOURCE_IDS):
        rows.append(
            {
                "task_id": f"task-{source}",
                "role": "challenger",
                "origin_node_id": source,
                "final_owner_node_id": (
                    STAGE5E3_DESTINATION_ID if index == 0 else source
                ),
            }
        )
    return rows


def witness_rows(phase: str, *, after: bool):
    rows = [
        {
            "phase": phase,
            "task_id": "resident",
            "role": "resident",
            "origin_node_id": STAGE5E3_DESTINATION_ID,
            "node_id": STAGE5E3_DESTINATION_ID,
            "class_id": BENCHMARK_CLASS_ID,
            "status": "running",
            "pid": 100,
            "process_count": 1,
            "process_state": "RS",
            "cpu_utilization_percent": 80.0,
            "memory_rss_mb": 13.0,
            "live": True,
        }
    ]
    for index, source in enumerate(STAGE5E3_SOURCE_IDS):
        node_id = STAGE5E3_DESTINATION_ID if after and index == 0 else source
        rows.append(
            {
                "phase": phase,
                "task_id": f"task-{source}",
                "role": "challenger",
                "origin_node_id": source,
                "node_id": node_id,
                "class_id": BENCHMARK_CLASS_ID,
                "status": "running",
                "pid": 200 + index,
                "process_count": 1,
                "process_state": "RS",
                "cpu_utilization_percent": 70.0,
                "memory_rss_mb": 13.0,
                "live": True,
            }
        )
    return rows


def resource_rows():
    migrated_source = STAGE5E3_SOURCE_IDS[0]
    counts = {node: 0 for node in ALL_NODES}
    counts[STAGE5E3_DESTINATION_ID] = 2
    for source in STAGE5E3_SOURCE_IDS:
        if source != migrated_source:
            counts[source] = 1
    rows = []
    for node in ALL_NODES:
        count = counts[node]
        rows.append(
            {
                "node_id": node,
                "expected_owned_task_count": count,
                "actual_owned_task_count": count,
                "expected_reserved_cpu_cores": count * CPU,
                "reserved_cpu_cores": count * CPU,
                "expected_reserved_memory_mb": count * MEM,
                "reserved_memory_mb": count * MEM,
                "resource_busy_fraction": (count * CPU) / 2.0,
                "available_cpu_cores": 2.0 - count * CPU,
                "reservation_matches_expected": True,
                "capacity_respected": True,
            }
        )
    return rows


def migration_rows():
    return [
        {
            "task_id": f"task-{STAGE5E3_SOURCE_IDS[0]}",
            "status": "completed",
        }
    ]


def test_stage5e3_passes_real_process_contention() -> None:
    assert stage5e3_passes(
        source_rows=source_rows(),
        decision_rows=decision_rows(),
        bid_rows=bid_rows(),
        migration_rows=migration_rows(),
        final_rows=final_rows(),
        ownership_ok=True,
        resident_task_id="resident",
        benchmark_cpu_cores=CPU,
        benchmark_memory_mb=MEM,
        capacity_cpu_cores=2.0,
        expected_git_sha=SHA,
        pre_witness_rows=witness_rows("pre_auction", after=False),
        post_witness_rows=witness_rows("post_auction", after=True),
        resource_rows=resource_rows(),
    )


def test_stage5e3_rejects_nonphysical_resident() -> None:
    pre = witness_rows("pre_auction", after=False)
    pre[0]["live"] = False
    pre[0]["process_state"] = "Z"
    pre[0]["memory_rss_mb"] = 0.0
    assert not stage5e3_passes(
        source_rows=source_rows(),
        decision_rows=decision_rows(),
        bid_rows=bid_rows(),
        migration_rows=migration_rows(),
        final_rows=final_rows(),
        ownership_ok=True,
        resident_task_id="resident",
        benchmark_cpu_cores=CPU,
        benchmark_memory_mb=MEM,
        capacity_cpu_cores=2.0,
        expected_git_sha=SHA,
        pre_witness_rows=pre,
        post_witness_rows=witness_rows("post_auction", after=True),
        resource_rows=resource_rows(),
    )


def test_stage5e3_requires_two_real_processes_at_destination_after() -> None:
    post = witness_rows("post_auction", after=True)
    post[1]["node_id"] = STAGE5E3_SOURCE_IDS[0]
    assert not stage5e3_passes(
        source_rows=source_rows(),
        decision_rows=decision_rows(),
        bid_rows=bid_rows(),
        migration_rows=migration_rows(),
        final_rows=final_rows(),
        ownership_ok=True,
        resident_task_id="resident",
        benchmark_cpu_cores=CPU,
        benchmark_memory_mb=MEM,
        capacity_cpu_cores=2.0,
        expected_git_sha=SHA,
        pre_witness_rows=witness_rows("pre_auction", after=False),
        post_witness_rows=post,
        resource_rows=resource_rows(),
    )


def test_stage5e3_rejects_resource_ledger_mismatch() -> None:
    resources = resource_rows()
    destination = next(row for row in resources if row["node_id"] == STAGE5E3_DESTINATION_ID)
    destination["reserved_cpu_cores"] = CPU
    destination["reservation_matches_expected"] = False
    assert not stage5e3_passes(
        source_rows=source_rows(),
        decision_rows=decision_rows(),
        bid_rows=bid_rows(),
        migration_rows=migration_rows(),
        final_rows=final_rows(),
        ownership_ok=True,
        resident_task_id="resident",
        benchmark_cpu_cores=CPU,
        benchmark_memory_mb=MEM,
        capacity_cpu_cores=2.0,
        expected_git_sha=SHA,
        pre_witness_rows=witness_rows("pre_auction", after=False),
        post_witness_rows=witness_rows("post_auction", after=True),
        resource_rows=resources,
    )


def test_witness_rejects_zombie_or_zero_rss() -> None:
    rows = witness_rows("pre_auction", after=False)
    task_ids = {str(row["task_id"]) for row in rows}
    assert witness_passes(rows, expected_task_ids=task_ids)
    rows[2]["process_state"] = "Z"
    assert not witness_passes(rows, expected_task_ids=task_ids)
    rows = witness_rows("pre_auction", after=False)
    rows[2]["memory_rss_mb"] = 0.0
    assert not witness_passes(rows, expected_task_ids=task_ids)


def test_ps_session_parser_sums_process_group() -> None:
    sessions = parse_ps_sessions(
        " 101 101 S 1024 20.0\n"
        " 101 102 R 2048 30.0\n"
        " 200 200 S 4096 10.0\n"
    )
    assert sessions[101]["process_count"] == 2
    assert sessions[101]["process_state"] == "S"
    assert sessions[101]["memory_rss_mb"] == 3.0
    assert sessions[101]["cpu_utilization_percent"] == 50.0


def test_definition_is_real_benchmark_with_frozen_request() -> None:
    payload = definition_payload(
        "comparison",
        ALL_NODES,
        {
            "cpu_cores": CPU,
            "memory_mb": MEM,
            "gpu_count": 0,
            "accelerator_type": None,
        },
        iterations=1_000_000,
    )
    assert payload["runtime"]["module"] == "magellan.workloads.benchmark"
    assert payload["profile"]["resource_request"]["cpu_cores"] == CPU
    assert payload["profile"]["resource_request"]["memory_mb"] == MEM
    args = payload["runtime"]["arguments"]
    assert args[args.index("--benchmark") + 1] == "json"
    assert args[args.index("--size") + 1] == "medium"
    assert args[args.index("--iterations") + 1] == "1000000"
