from __future__ import annotations

import math
import statistics
import tempfile
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from magellan.bidding.models import AuctionStrategy, BidRecord, TaskBidContext
from magellan.bidding.ranking import rank_bids
from magellan.bidding.resources import ResourceLedger
from magellan.carbon.store import CarbonStore
from magellan.config.models import ClusterConfig, NodeResourceCapacity
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.stage4b import FrozenCalibrationGraph, WorkloadCalibration
from magellan.experiments.stage4e1 import ScaleTaskSpec
from magellan.models.types import ActionType, ScoredAction, TaskProfile
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task


@dataclass(frozen=True)
class BenchmarkTask:
    spec: ScaleTaskSpec
    profile: TaskProfile
    calibration: WorkloadCalibration
    graph: FrozenCalibrationGraph


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_tasks(
    *,
    specs: list[ScaleTaskSpec],
    calibrations: dict[str, WorkloadCalibration],
    runtime_scales: dict[str, float],
    node_slowdowns: dict[str, float],
    graphs: dict[str, FrozenCalibrationGraph],
    all_node_ids: set[str],
) -> list[BenchmarkTask]:
    output = []
    for spec in specs:
        calibration = calibrations[spec.class_id]
        remaining_work = calibration.scaled_work_seconds(runtime_scales[spec.class_id])
        slowdown = node_slowdowns[spec.home_node_id]
        profile = TaskProfile(
            task_id=spec.task_id,
            workload_type=calibration.workload or spec.class_id,
            current_node_id=spec.home_node_id,
            power_kw=calibration.power_kw,
            checkpoint_bytes=calibration.checkpoint_bytes,
            data_bytes=0,
            prestaged_node_ids=all_node_ids,
            estimated_remaining_seconds=remaining_work * slowdown,
            accumulated_cost_usd=0.0,
            cost_cap_usd=None,
            last_migration_at=None,
            last_pause_at=None,
            resource_request=spec.resource_request,
        )
        output.append(
            BenchmarkTask(
                spec=spec,
                profile=profile,
                calibration=calibration,
                graph=graphs[spec.class_id],
            )
        )
    return output


def _migration_candidate(decision) -> ScoredAction:
    migrations = [
        action
        for action in decision.ranked_actions
        if action.action == ActionType.MIGRATE
        and action.destination_node_id is not None
    ]
    if not migrations:
        raise RuntimeError("Control-plane benchmark task has no migration candidate")
    return min(migrations, key=lambda action: action.score)


def _bid_context(
    *,
    task: BenchmarkTask,
    candidate: ScoredAction,
    ranked_actions: list[ScoredAction],
) -> TaskBidContext:
    alternatives = [
        action
        for action in ranked_actions
        if not (
            action.action == candidate.action
            and action.source_node_id == candidate.source_node_id
            and action.destination_node_id == candidate.destination_node_id
        )
    ]
    fallback = min(alternatives, key=lambda action: action.score) if alternatives else None
    opportunity_loss = (
        max(0.0, fallback.score - candidate.score)
        if fallback is not None
        else 0.0
    )
    return TaskBidContext(
        workload_type=task.profile.workload_type,
        estimated_remaining_seconds=task.profile.estimated_remaining_seconds,
        checkpoint_bytes=task.calibration.checkpoint_bytes,
        static_data_bytes=0,
        accumulated_cost_usd=0.0,
        resource_request=task.spec.resource_request,
        fallback_action=fallback.action if fallback is not None else None,
        fallback_destination_node_id=(
            fallback.destination_node_id if fallback is not None else None
        ),
        fallback_score=fallback.score if fallback is not None else None,
        opportunity_loss=opportunity_loss,
        effective_power_kw=task.calibration.power_kw,
        power_source="stage4a3_frozen_profile",
    )


def make_adaptive_service(
    *,
    policy: ScoringPolicy,
    root: Path,
) -> AdaptivePolicyService:
    return AdaptivePolicyService(
        policy.adaptive,
        policy.weights,
        AdaptivePolicyStore(root),
    )


def execute_control_plane_epoch(
    *,
    tasks: list[BenchmarkTask],
    capacities: dict[str, NodeResourceCapacity],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
    adaptive_service: AdaptivePolicyService,
    auction_strategy: AuctionStrategy = AuctionStrategy.LOWEST_SCORE,
) -> dict[str, Any]:
    all_node_ids = {node.id for node in cluster.nodes}
    telemetry_confidence = float(policy.telemetry.cpu_power_confidence)

    decision_start = time.perf_counter_ns()
    decisions = []
    for task in tasks:
        decisions.append(
            (
                task,
                evaluate_task(
                    task=task.profile,
                    cluster=cluster,
                    policy=policy,
                    graph=task.graph,  # type: ignore[arg-type]
                    carbon_store=carbon_store,
                    at_utc=at_utc,
                    static_data_bytes_by_destination={
                        node_id: 0
                        for node_id in all_node_ids - {task.profile.current_node_id}
                    },
                    adaptive_service=adaptive_service,
                    telemetry_confidence=telemetry_confidence,
                    compatible_destination_ids=all_node_ids - {task.profile.current_node_id},
                ),
            )
        )
    decision_end = time.perf_counter_ns()

    auction_start = time.perf_counter_ns()
    bids_by_destination: dict[str, list[BidRecord]] = defaultdict(list)
    for index, (task, decision) in enumerate(decisions, start=1):
        candidate = _migration_candidate(decision)
        destination = candidate.destination_node_id
        if destination is None:
            raise RuntimeError("Migration candidate missing destination")
        context = _bid_context(
            task=task,
            candidate=candidate,
            ranked_actions=decision.ranked_actions,
        )
        bids_by_destination[destination].append(
            BidRecord(
                bid_id=f"stage4e2:{len(tasks)}:{index}",
                epoch_id=f"stage4e2:{len(tasks)}",
                task_id=task.spec.task_id,
                task_context=context,
                source_node_id=task.profile.current_node_id,
                destination_node_id=destination,
                candidate=candidate,
                submitted_at_utc=at_utc.to_pydatetime(warn=False),
                received_at_utc=at_utc.to_pydatetime(warn=False),
            )
        )

    ranked_bid_count = 0
    accepted_count = 0
    for destination_id in sorted(bids_by_destination):
        bids = bids_by_destination[destination_id]
        ranked = rank_bids(
            bids=bids,
            strategy=auction_strategy,
            credits={},
            node_resources=capacities[destination_id],
            policy=policy.auction,
            now_utc=at_utc.to_pydatetime(warn=False),
        )
        ledger = ResourceLedger.from_capacity(capacities[destination_id])
        for item in ranked:
            ranked_bid_count += 1
            request = item.bid.task_context.resource_request
            fits, _ = ledger.compatible(request)
            if fits:
                ledger.consume(request)
                accepted_count += 1
    auction_end = time.perf_counter_ns()

    return {
        "decision_wall_ns": decision_end - decision_start,
        "auction_wall_ns": auction_end - auction_start,
        "epoch_wall_ns": auction_end - decision_start,
        "decision_count": len(decisions),
        "bid_count": ranked_bid_count,
        "destination_count": len(bids_by_destination),
        "accepted_count": accepted_count,
    }


def benchmark_control_plane(
    *,
    tasks: list[BenchmarkTask],
    capacities: dict[str, NodeResourceCapacity],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
    repetitions: int,
    warmups: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if repetitions < 3:
        raise ValueError("repetitions must be at least 3")
    if warmups < 1:
        raise ValueError("warmups must be at least 1")

    samples: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="magellan-stage4e2-") as directory:
        root = Path(directory)

        cold_service = make_adaptive_service(
            policy=policy,
            root=root / "cold",
        )
        cold_cpu_start = time.process_time_ns()
        cold = execute_control_plane_epoch(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
            adaptive_service=cold_service,
        )
        cold_cpu_ns = time.process_time_ns() - cold_cpu_start

        for warmup in range(warmups):
            service = make_adaptive_service(
                policy=policy,
                root=root / f"warmup-{warmup}",
            )
            execute_control_plane_epoch(
                tasks=tasks,
                capacities=capacities,
                cluster=cluster,
                policy=policy,
                carbon_store=carbon_store,
                at_utc=at_utc,
                adaptive_service=service,
            )

        for repetition in range(repetitions):
            service = make_adaptive_service(
                policy=policy,
                root=root / f"sample-{repetition}",
            )
            cpu_start = time.process_time_ns()
            result = execute_control_plane_epoch(
                tasks=tasks,
                capacities=capacities,
                cluster=cluster,
                policy=policy,
                carbon_store=carbon_store,
                at_utc=at_utc,
                adaptive_service=service,
            )
            cpu_ns = time.process_time_ns() - cpu_start
            samples.append(
                {
                    "task_count": len(tasks),
                    "repetition": repetition + 1,
                    "decision_wall_ms": result["decision_wall_ns"] / 1e6,
                    "auction_wall_ms": result["auction_wall_ns"] / 1e6,
                    "epoch_wall_ms": result["epoch_wall_ns"] / 1e6,
                    "epoch_cpu_ms": cpu_ns / 1e6,
                    "bid_count": result["bid_count"],
                    "destination_count": result["destination_count"],
                    "accepted_count": result["accepted_count"],
                }
            )

        # Memory is measured separately so tracemalloc instrumentation does not
        # contaminate the latency samples above.
        memory_service = make_adaptive_service(
            policy=policy,
            root=root / "memory",
        )
        tracemalloc.start()
        baseline_current, _ = tracemalloc.get_traced_memory()
        memory_result = execute_control_plane_epoch(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
            adaptive_service=memory_service,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    decision_values = [row["decision_wall_ms"] for row in samples]
    auction_values = [row["auction_wall_ms"] for row in samples]
    epoch_values = [row["epoch_wall_ms"] for row in samples]
    cpu_values = [row["epoch_cpu_ms"] for row in samples]
    bid_count = int(samples[0]["bid_count"])
    median_epoch_ms = statistics.median(epoch_values)
    median_decision_ms = statistics.median(decision_values)
    median_auction_ms = statistics.median(auction_values)

    summary = {
        "task_count": len(tasks),
        "repetitions": repetitions,
        "warmups": warmups,
        "cold_epoch_wall_ms": cold["epoch_wall_ns"] / 1e6,
        "cold_epoch_cpu_ms": cold_cpu_ns / 1e6,
        "decision_wall_ms_median": median_decision_ms,
        "decision_wall_ms_p95": percentile(decision_values, 0.95),
        "decision_per_task_ms_median": median_decision_ms / len(tasks),
        "decision_tasks_per_second": (
            len(tasks) / (median_decision_ms / 1000.0)
            if median_decision_ms > 0
            else 0.0
        ),
        "auction_wall_ms_median": median_auction_ms,
        "auction_wall_ms_p95": percentile(auction_values, 0.95),
        "auction_per_bid_us_median": (
            median_auction_ms * 1000.0 / bid_count
            if bid_count
            else 0.0
        ),
        "auction_bids_per_second": (
            bid_count / (median_auction_ms / 1000.0)
            if median_auction_ms > 0
            else 0.0
        ),
        "epoch_wall_ms_median": median_epoch_ms,
        "epoch_wall_ms_p95": percentile(epoch_values, 0.95),
        "epoch_cpu_ms_median": statistics.median(cpu_values),
        "epoch_cpu_to_wall_ratio": (
            statistics.median(cpu_values) / median_epoch_ms
            if median_epoch_ms > 0
            else 0.0
        ),
        "epoch_tasks_per_second": (
            len(tasks) / (median_epoch_ms / 1000.0)
            if median_epoch_ms > 0
            else 0.0
        ),
        "bid_count": bid_count,
        "destination_count": int(samples[0]["destination_count"]),
        "accepted_count": int(samples[0]["accepted_count"]),
        "peak_incremental_tracemalloc_kb": max(
            0, peak - baseline_current
        ) / 1024.0,
        "memory_probe_epoch_wall_ms": memory_result["epoch_wall_ns"] / 1e6,
    }
    return summary, samples
