from __future__ import annotations

from magellan.config.loader import load_policy_config
from magellan.config.models import NodeResourceCapacity
from magellan.experiments.stage4b import WorkloadCalibration
from magellan.experiments.stage4d4 import (
    BENCHMARK_CLASS,
    STRATEGY_VALUES,
    cohort_specs,
    required_strategies,
    run_fixed_cohort,
    verify_single_measured_slot,
)
from magellan.models.types import TaskResourceRequest


def calibration() -> WorkloadCalibration:
    return WorkloadCalibration(
        class_id=BENCHMARK_CLASS,
        workload="benchmark",
        variant="json-medium",
        canonical_runtime_seconds=3600.0,
        power_kw=0.08,
        checkpoint_bytes=259,
        checkpoint_seconds=0.02,
        restore_seconds=0.04,
        migration_overhead_seconds=3.0,
    )


def test_measured_stage4d1_shape_leaves_exactly_one_benchmark_admission() -> None:
    capacity = NodeResourceCapacity(cpu_cores=2.0, memory_mb=16002, gpu_count=0)
    request = TaskResourceRequest(
        cpu_cores=0.9972222178769694,
        memory_mb=13,
        gpu_count=0,
    )
    result = verify_single_measured_slot(
        capacity=capacity,
        benchmark_request=request,
    )
    assert result["first_fits"] is True
    assert result["second_fits"] is False


def test_controlled_cohort_attributes_are_orthogonal() -> None:
    specs = cohort_specs(259200.0)
    by_id = {spec.task_id: spec for spec in specs}
    assert min(specs, key=lambda spec: spec.candidate_score).task_id == "task-a"
    assert min(specs, key=lambda spec: spec.remaining_seconds).task_id == "task-b"
    assert max(specs, key=lambda spec: spec.remaining_seconds).task_id == "task-e"
    assert max(specs, key=lambda spec: spec.opportunity_loss).task_id == "task-c"
    assert len(by_id) == 5


def test_all_required_production_strategies_rank_complete_fixed_cohort() -> None:
    strategies = required_strategies()
    assert set(strategies) == set(STRATEGY_VALUES)

    capacity = NodeResourceCapacity(cpu_cores=2.0, memory_mb=16002, gpu_count=0)
    request = TaskResourceRequest(
        cpu_cores=0.9972222178769694,
        memory_mb=13,
        gpu_count=0,
    )
    policy = load_policy_config("config/policy.prod.json")

    for value in STRATEGY_VALUES:
        rows, summary = run_fixed_cohort(
            strategy=strategies[value],
            capacity=capacity,
            benchmark_request=request,
            calibration=calibration(),
            policy=policy,
            target_seconds=259200.0,
            source_node_id="boston",
            destination_node_id="ethiopia",
        )
        assert summary["all_tasks_admitted"] is True
        assert len(summary["admission_order"].split("->")) == 5
        accepted = [row for row in rows if row["status"] == "accepted"]
        assert len(accepted) == 5
