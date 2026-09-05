from __future__ import annotations

import json
from pathlib import Path

from magellan.config.models import NodeResourceCapacity
from magellan.models.types import TaskResourceRequest
from magellan.experiments.stage5e2 import (
    BENCHMARK_CLASS_ID,
    DENDRO_CLASS_ID,
    EXPECTED_CLASS_COUNTS,
    LLM_CLASS_ID,
    STAGE5E2_LAYOUT,
    layout_class_counts,
    physical_definition,
    physical_task_specs,
    stage5e2_passes,
    validate_physical_layout,
)


def _requests() -> dict[str, TaskResourceRequest]:
    return {
        BENCHMARK_CLASS_ID: TaskResourceRequest(
            cpu_cores=0.9972222178769694,
            memory_mb=13,
            gpu_count=0,
        ),
        DENDRO_CLASS_ID: TaskResourceRequest(
            cpu_cores=1.7763255933,
            memory_mb=1379,
            gpu_count=0,
        ),
        LLM_CLASS_ID: TaskResourceRequest(
            cpu_cores=0.7630787943,
            memory_mb=1572,
            gpu_count=0,
        ),
    }


def _capacities() -> dict[str, NodeResourceCapacity]:
    return {
        node_id: NodeResourceCapacity(
            cpu_cores=2.0,
            memory_mb=16002,
            gpu_count=0,
            accelerator_types=[],
        )
        for node_id in STAGE5E2_LAYOUT
    }


def _signatures() -> dict[str, set[tuple[int, int, int]]]:
    valid = {(2, 0, 0), (1, 0, 1), (0, 0, 2), (0, 1, 0)}
    return {node_id: set(valid) for node_id in STAGE5E2_LAYOUT}


def test_stage5e2_layout_is_exact_stage4d2_umax_mix() -> None:
    assert len(physical_task_specs()) == 11
    assert layout_class_counts() == EXPECTED_CLASS_COUNTS == {
        BENCHMARK_CLASS_ID: 4,
        DENDRO_CLASS_ID: 3,
        LLM_CLASS_ID: 4,
    }
    assert STAGE5E2_LAYOUT["boston"] == (
        BENCHMARK_CLASS_ID,
        BENCHMARK_CLASS_ID,
    )
    assert STAGE5E2_LAYOUT["south-australia"] == (LLM_CLASS_ID, LLM_CLASS_ID)


def test_layout_uses_only_frozen_maximal_packings_and_8836_percent_cpu() -> None:
    rows = validate_physical_layout(
        capacities=_capacities(),
        requests=_requests(),
        maximal_signatures=_signatures(),
    )
    assert len(rows) == 7
    assert all(row["is_frozen_maximal_packing"] for row in rows)
    used = sum(float(row["expected_reserved_cpu_cores"]) for row in rows)
    assert abs(used - 12.370180828607879) < 1e-9
    assert abs(used / 14.0 - 0.8835843449005628) < 1e-9


def test_layout_rejects_nonmaximal_signature() -> None:
    signatures = _signatures()
    signatures["boston"].remove((2, 0, 0))
    try:
        validate_physical_layout(
            capacities=_capacities(),
            requests=_requests(),
            maximal_signatures=signatures,
        )
    except ValueError as exc:
        assert "not maximal" in str(exc)
    else:
        raise AssertionError("expected non-maximal layout rejection")


def test_physical_definitions_use_measured_requests_without_changing_workloads() -> None:
    requests = _requests()
    nodes = list(STAGE5E2_LAYOUT)
    dendro_template = json.loads(
        Path("config/submissions/dendro-bssn-template.json").read_text(encoding="utf-8")
    )

    benchmark = physical_definition(
        class_id=BENCHMARK_CLASS_ID,
        definition_id="stage5e2-benchmark",
        request=requests[BENCHMARK_CLASS_ID],
        node_ids=nodes,
        seed=42,
        benchmark_iterations=1_000_000,
        llm_model="experiment-assets/models/distilgpt2",
        dendro_template=dendro_template,
        dendro_solver="/home/WILL/dgr-build/BSSN_GR/bssnSolver",
        dendro_parameter_template="/home/WILL/q1-magellan-magellan.toml",
    )
    assert benchmark["runtime"]["module"] == "magellan.workloads.benchmark"
    assert benchmark["profile"]["resource_request"]["cpu_cores"] == requests[
        BENCHMARK_CLASS_ID
    ].cpu_cores

    llm = physical_definition(
        class_id=LLM_CLASS_ID,
        definition_id="stage5e2-llm",
        request=requests[LLM_CLASS_ID],
        node_ids=nodes,
        seed=43,
        benchmark_iterations=1_000_000,
        llm_model="experiment-assets/models/distilgpt2",
        dendro_template=dendro_template,
        dendro_solver="/home/WILL/dgr-build/BSSN_GR/bssnSolver",
        dendro_parameter_template="/home/WILL/q1-magellan-magellan.toml",
    )
    assert llm["runtime"]["module"] == "magellan.workloads.llm_train"
    assert llm["profile"]["resource_request"]["memory_mb"] == requests[
        LLM_CLASS_ID
    ].memory_mb
    assert "--checkpoint-every" in llm["runtime"]["arguments"]

    dendro = physical_definition(
        class_id=DENDRO_CLASS_ID,
        definition_id="stage5e2-dendro",
        request=requests[DENDRO_CLASS_ID],
        node_ids=nodes,
        seed=44,
        benchmark_iterations=1_000_000,
        llm_model="experiment-assets/models/distilgpt2",
        dendro_template=dendro_template,
        dendro_solver="/home/WILL/dgr-build/BSSN_GR/bssnSolver",
        dendro_parameter_template="/home/WILL/q1-magellan-magellan.toml",
    )
    assert dendro["runtime"]["adapter"] == "dendro"
    assert dendro["profile"]["resource_request"]["cpu_cores"] == requests[
        DENDRO_CLASS_ID
    ].cpu_cores
    assert dendro["profile"]["compatibility"]["requires_same_mpi_world_size"] is True


def _passing_rows() -> tuple[list[dict], list[dict]]:
    task_rows = []
    for spec in physical_task_specs():
        task_rows.append(
            {
                "task_index": spec.task_index,
                "task_id": f"run-{spec.task_index}",
                "node_id": spec.node_id,
                "class_id": spec.class_id,
                "launched": True,
                "steady_running": True,
                "telemetry_sample_count": 6,
                "cpu_sample_count": 5,
                "rss_sample_count": 6,
                "max_memory_rss_mb": 100,
                "progress_min": 1,
                "progress_max": 2,
                "cleanup_ok": True,
            }
        )
    node_rows = []
    for node_id in STAGE5E2_LAYOUT:
        node_rows.append(
            {
                "node_id": node_id,
                "is_frozen_maximal_packing": True,
                "all_tasks_running": True,
                "reservation_matches_expected": True,
                "capacity_respected": True,
                "max_resource_busy_fraction": 0.99,
                "actual_cpu_sample_count": 6,
                "actual_rss_sample_count": 6,
            }
        )
    return task_rows, node_rows


def test_stage5e2_passes_only_complete_physical_evidence() -> None:
    tasks, nodes = _passing_rows()
    assert stage5e2_passes(tasks, nodes)
    tasks[0]["cpu_sample_count"] = 0
    assert not stage5e2_passes(tasks, nodes)


def test_stage5e2_rejects_capacity_violation() -> None:
    tasks, nodes = _passing_rows()
    nodes[2]["capacity_respected"] = False
    assert not stage5e2_passes(tasks, nodes)
