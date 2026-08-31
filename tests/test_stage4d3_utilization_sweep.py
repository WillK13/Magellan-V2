from __future__ import annotations

from magellan.experiments.stage4d2 import LayoutTask
from magellan.experiments.stage4d3 import (
    LOAD_QUOTAS,
    achieved_cpu_fraction,
    build_nested_utilization_layouts,
    class_counts,
)
from magellan.models.types import TaskResourceRequest


REQ = {
    "benchmark-json-medium": TaskResourceRequest(cpu_cores=0.9972222178769694, memory_mb=13),
    "dendro-r9-t1p0": TaskResourceRequest(cpu_cores=1.7763255932687552, memory_mb=1379),
    "llm-distilgpt2": TaskResourceRequest(cpu_cores=0.7630787942682616, memory_mb=1572),
}


def task(task_id: str, class_id: str, node_id: str) -> LayoutTask:
    return LayoutTask(
        task_id=task_id,
        class_id=class_id,
        initial_node_id=node_id,
        resource_request=REQ[class_id],
    )


def maximal_layout() -> list[LayoutTask]:
    return [
        task("b1", "benchmark-json-medium", "boston"),
        task("b2", "benchmark-json-medium", "california"),
        task("b3", "benchmark-json-medium", "ethiopia"),
        task("b4", "benchmark-json-medium", "france"),
        task("d1", "dendro-r9-t1p0", "nepal"),
        task("d2", "dendro-r9-t1p0", "south-australia"),
        task("d3", "dendro-r9-t1p0", "virginia"),
        task("l1", "llm-distilgpt2", "boston"),
        task("l2", "llm-distilgpt2", "california"),
        task("l3", "llm-distilgpt2", "ethiopia"),
        task("l4", "llm-distilgpt2", "france"),
    ]


def test_nested_balanced_utilization_populations() -> None:
    layouts = build_nested_utilization_layouts(maximal_layout())

    assert list(layouts) == ["u25", "u50", "u75", "umax"]
    for load_id, layout in layouts.items():
        assert class_counts(layout) == LOAD_QUOTAS[load_id]

    ids = {load_id: {task.task_id for task in layout} for load_id, layout in layouts.items()}
    assert ids["u25"] <= ids["u50"] <= ids["u75"] <= ids["umax"]


def test_measured_cpu_requests_land_near_target_utilizations() -> None:
    layouts = build_nested_utilization_layouts(maximal_layout())

    assert abs(achieved_cpu_fraction(layouts["u25"], cluster_cpu_cores=14.0) - 0.25) < 0.02
    assert abs(achieved_cpu_fraction(layouts["u50"], cluster_cpu_cores=14.0) - 0.50) < 0.02
    assert abs(achieved_cpu_fraction(layouts["u75"], cluster_cpu_cores=14.0) - 0.75) < 0.02
    assert 0.85 < achieved_cpu_fraction(layouts["umax"], cluster_cpu_cores=14.0) < 0.90


def test_subset_selection_prefers_geographic_coverage() -> None:
    layouts = build_nested_utilization_layouts(maximal_layout())
    assert len({task.initial_node_id for task in layouts["u25"]}) == 3
    assert len({task.initial_node_id for task in layouts["u50"]}) >= 5
