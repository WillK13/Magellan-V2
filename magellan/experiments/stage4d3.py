from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Any, Iterable

from magellan.experiments.stage4d2 import (
    LOWEST_SCORE_POLICY,
    STATIC_POLICY,
    UNLIMITED_POLICY,
    LayoutTask,
)

BENCHMARK_CLASS = "benchmark-json-medium"
DENDRO_CLASS = "dendro-r9-t1p0"
LLM_CLASS = "llm-distilgpt2"
CLASS_ORDER = (BENCHMARK_CLASS, DENDRO_CLASS, LLM_CLASS)

SWEEP_POLICIES = (STATIC_POLICY, UNLIMITED_POLICY, LOWEST_SCORE_POLICY)
LOAD_ORDER = ("u25", "u50", "u75", "umax")

LOAD_QUOTAS: dict[str, dict[str, int]] = {
    "u25": {BENCHMARK_CLASS: 1, DENDRO_CLASS: 1, LLM_CLASS: 1},
    "u50": {BENCHMARK_CLASS: 2, DENDRO_CLASS: 2, LLM_CLASS: 2},
    "u75": {BENCHMARK_CLASS: 3, DENDRO_CLASS: 3, LLM_CLASS: 3},
    "umax": {BENCHMARK_CLASS: 4, DENDRO_CLASS: 3, LLM_CLASS: 4},
}

NOMINAL_TARGET_FRACTIONS = {
    "u25": 0.25,
    "u50": 0.50,
    "u75": 0.75,
    "umax": None,
}


def class_counts(layout: Iterable[LayoutTask]) -> dict[str, int]:
    counts = Counter(task.class_id for task in layout)
    return {class_id: int(counts.get(class_id, 0)) for class_id in CLASS_ORDER}


def layout_cpu_cores(layout: Iterable[LayoutTask]) -> float:
    return sum(float(task.resource_request.cpu_cores) for task in layout)


def _choose_quota_subset(
    parent: list[LayoutTask],
    quotas: dict[str, int],
) -> list[LayoutTask]:
    grouped: dict[str, list[LayoutTask]] = defaultdict(list)
    for task in parent:
        grouped[task.class_id].append(task)

    combination_sets = []
    for class_id in CLASS_ORDER:
        need = int(quotas[class_id])
        available = sorted(grouped[class_id], key=lambda task: task.task_id)
        if len(available) < need:
            raise ValueError(
                f"Cannot select {need} {class_id} tasks from {len(available)} available"
            )
        combination_sets.append(list(combinations(available, need)))

    parent_order = {task.task_id: index for index, task in enumerate(parent)}
    best: tuple[Any, ...] | None = None
    best_tasks: list[LayoutTask] | None = None

    for parts in product(*combination_sets):
        candidate = [task for part in parts for task in part]
        distinct_nodes = len({task.initial_node_id for task in candidate})
        sorted_ids = tuple(sorted(task.task_id for task in candidate))
        # Maximize geographic coverage first, then use task ids for deterministic tie-breaking.
        score = (-distinct_nodes, sorted_ids)
        if best is None or score < best:
            best = score
            best_tasks = candidate

    if best_tasks is None:
        raise RuntimeError("No quota-compatible utilization subset found")

    return sorted(best_tasks, key=lambda task: parent_order[task.task_id])


def build_nested_utilization_layouts(
    maximal_layout: list[LayoutTask],
) -> dict[str, list[LayoutTask]]:
    observed = class_counts(maximal_layout)
    if observed != LOAD_QUOTAS["umax"]:
        raise ValueError(
            f"Stage 4D.3 expects the frozen Stage 4D.2 4/3/4 maximal population, got {observed}"
        )

    u75 = _choose_quota_subset(maximal_layout, LOAD_QUOTAS["u75"])
    u50 = _choose_quota_subset(u75, LOAD_QUOTAS["u50"])
    u25 = _choose_quota_subset(u50, LOAD_QUOTAS["u25"])

    layouts = {
        "u25": u25,
        "u50": u50,
        "u75": u75,
        "umax": list(maximal_layout),
    }

    for load_id, layout in layouts.items():
        counts = class_counts(layout)
        if counts != LOAD_QUOTAS[load_id]:
            raise AssertionError(f"{load_id} quota mismatch: {counts}")
    return layouts


def achieved_cpu_fraction(
    layout: Iterable[LayoutTask],
    *,
    cluster_cpu_cores: float,
) -> float:
    if cluster_cpu_cores <= 0:
        raise ValueError("cluster_cpu_cores must be positive")
    return layout_cpu_cores(layout) / cluster_cpu_cores


def summarize_utilization_rows(
    scenario_rows: list[dict[str, Any]],
    load_case_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_by_scenario = {row["scenario_id"]: row for row in load_case_rows}
    output: list[dict[str, Any]] = []

    for load_id in LOAD_ORDER:
        for policy in SWEEP_POLICIES:
            subset = [
                row
                for row in scenario_rows
                if row["policy"] == policy
                and case_by_scenario[row["scenario_id"]]["load_id"] == load_id
            ]
            if not subset:
                continue
            cases = [case_by_scenario[row["scenario_id"]] for row in subset]
            attempts = sum(int(row["bid_attempts"]) for row in subset)
            rejects = sum(int(row["bid_rejections"]) for row in subset)
            task_count = sum(int(row["task_count"]) for row in subset)
            carbon_ratio = sum(float(row["carbon_ratio_vs_static"]) for row in subset) / len(subset)

            output.append(
                {
                    "load_id": load_id,
                    "policy": policy,
                    "scenario_count": len(subset),
                    "task_count_total": task_count,
                    "target_cpu_fraction": cases[0]["target_cpu_fraction"],
                    "achieved_initial_cpu_fraction_mean": (
                        sum(float(case["achieved_initial_cpu_fraction"]) for case in cases)
                        / len(cases)
                    ),
                    "achieved_initial_cpu_cores_mean": (
                        sum(float(case["achieved_initial_cpu_cores"]) for case in cases)
                        / len(cases)
                    ),
                    "makespan_seconds_mean": (
                        sum(float(row["makespan_seconds"]) for row in subset) / len(subset)
                    ),
                    "carbon_grams_mean": (
                        sum(float(row["carbon_grams"]) for row in subset) / len(subset)
                    ),
                    "cost_usd_mean": (
                        sum(float(row["cost_usd"]) for row in subset) / len(subset)
                    ),
                    "time_ratio_mean": (
                        sum(float(row["time_ratio_vs_static"]) for row in subset) / len(subset)
                    ),
                    "carbon_ratio_mean": carbon_ratio,
                    "carbon_savings_percent_mean": 100.0 * (1.0 - carbon_ratio),
                    "cost_ratio_mean": (
                        sum(float(row["cost_ratio_vs_static"]) for row in subset) / len(subset)
                    ),
                    "migrations_total": sum(int(row["migrations"]) for row in subset),
                    "bid_attempts_total": attempts,
                    "bid_accepts_total": sum(int(row["bid_accepts"]) for row in subset),
                    "bid_rejections_total": rejects,
                    "bid_rejection_rate": (rejects / attempts) if attempts else 0.0,
                    "tasks_migrated_total": sum(int(row["tasks_migrated"]) for row in subset),
                    "migration_rate_per_task": (
                        sum(int(row["migrations"]) for row in subset) / task_count
                        if task_count
                        else 0.0
                    ),
                    "distinct_nodes_visited_mean": (
                        sum(int(row["distinct_nodes_visited"]) for row in subset) / len(subset)
                    ),
                }
            )
    return output
