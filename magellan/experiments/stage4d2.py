from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from magellan.bidding.models import AuctionStrategy, BidRecord, TaskBidContext
from magellan.bidding.ranking import rank_bids
from magellan.bidding.resources import ResourceLedger, ResourceVector, sum_requests
from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig, NodeResourceCapacity
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.stage4b import (
    FrozenCalibrationGraph,
    WorkloadCalibration,
    _compute_segment,
    _realized_migration,
    replay_magellan_causal,
)
from magellan.models.types import ActionType, ScoredAction, TaskProfile, TaskResourceRequest
from magellan.models.utils import seconds_to_hours
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task


STATIC_POLICY = "static_initial_layout"
UNLIMITED_POLICY = "magellan_unlimited_reference"
LOWEST_SCORE_POLICY = "magellan_capacity_lowest_score"
CREDIT_FAIR_POLICY = "magellan_capacity_credit_fair"
CAPACITY_POLICIES = (LOWEST_SCORE_POLICY, CREDIT_FAIR_POLICY)
ALL_POLICIES = (STATIC_POLICY, UNLIMITED_POLICY, *CAPACITY_POLICIES)

# Every element is one of the maximal resource packings frozen by Stage 4D.1.
# Rotating this sequence over the seven nodes changes which site receives which
# workload mix without changing the global workload population.
CANONICAL_PACKING_SEQUENCE: tuple[tuple[str, ...], ...] = (
    ("benchmark-json-medium", "benchmark-json-medium"),
    ("benchmark-json-medium", "llm-distilgpt2"),
    ("dendro-r9-t1p0",),
    ("llm-distilgpt2", "llm-distilgpt2"),
    ("dendro-r9-t1p0",),
    ("benchmark-json-medium", "llm-distilgpt2"),
    ("dendro-r9-t1p0",),
)
CANONICAL_TASK_MIX = {
    "benchmark-json-medium": 4,
    "dendro-r9-t1p0": 3,
    "llm-distilgpt2": 4,
}


@dataclass(frozen=True)
class LayoutTask:
    task_id: str
    class_id: str
    initial_node_id: str
    resource_request: TaskResourceRequest


@dataclass
class CapacityTaskState:
    task_id: str
    class_id: str
    initial_node_id: str
    owner_node_id: str
    resource_request: TaskResourceRequest
    calibration: WorkloadCalibration
    remaining_work_seconds: float
    accumulated_carbon_grams: float = 0.0
    accumulated_cost_usd: float = 0.0
    compute_seconds: float = 0.0
    migration_seconds: float = 0.0
    paused_idle_seconds: float = 0.0
    pause_overhead_seconds: float = 0.0
    migrations: int = 0
    pauses: int = 0
    decision_count: int = 0
    bid_attempts: int = 0
    bid_accepts: int = 0
    bid_rejections: int = 0
    owner_path: list[str] = field(default_factory=list)
    last_migration_at: datetime | None = None
    last_pause_at: datetime | None = None
    blocked_until_utc: pd.Timestamp | None = None
    finished_at_utc: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.owner_path:
            self.owner_path = [self.owner_node_id]

    @property
    def completed(self) -> bool:
        return self.remaining_work_seconds <= 1e-9


def read_resource_model(
    stage4d1_bundle: str | Path,
) -> tuple[dict[str, NodeResourceCapacity], dict[str, TaskResourceRequest]]:
    import csv

    root = Path(stage4d1_bundle)
    with (root / "node_capacities.csv").open(encoding="utf-8", newline="") as handle:
        node_rows = list(csv.DictReader(handle))
    with (root / "workload_resource_requests.csv").open(encoding="utf-8", newline="") as handle:
        workload_rows = list(csv.DictReader(handle))

    capacities = {
        row["node_id"]: NodeResourceCapacity(
            cpu_cores=float(row["effective_cpu_cores"]),
            memory_mb=int(row["effective_memory_mb"]),
            gpu_count=int(row["effective_gpu_count"]),
            accelerator_types=set(),
        )
        for row in node_rows
    }
    requests = {
        row["class_id"]: TaskResourceRequest(
            cpu_cores=float(row["cpu_request_cores"]),
            memory_mb=int(row["memory_request_mb"]),
            gpu_count=int(row["gpu_request_count"]),
        )
        for row in workload_rows
    }
    return capacities, requests


def maximal_packing_signatures(stage4d1_bundle: str | Path) -> dict[str, set[tuple[int, int, int]]]:
    import csv

    root = Path(stage4d1_bundle)
    with (root / "maximal_packings.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
    for row in rows:
        output[row["node_id"]].add(
            (
                int(row["count_benchmark-json-medium"]),
                int(row["count_dendro-r9-t1p0"]),
                int(row["count_llm-distilgpt2"]),
            )
        )
    return dict(output)


def _packing_signature(classes: Iterable[str]) -> tuple[int, int, int]:
    values = list(classes)
    return (
        values.count("benchmark-json-medium"),
        values.count("dendro-r9-t1p0"),
        values.count("llm-distilgpt2"),
    )


def build_initial_layout(
    *,
    scenario_id: str,
    node_ids: list[str],
    requests: dict[str, TaskResourceRequest],
    rotation: int,
    maximal_signatures: dict[str, set[tuple[int, int, int]]] | None = None,
) -> list[LayoutTask]:
    if len(node_ids) != len(CANONICAL_PACKING_SEQUENCE):
        raise ValueError("Canonical Stage 4D.2 layout requires exactly seven nodes")
    output: list[LayoutTask] = []
    for node_index, node_id in enumerate(node_ids):
        packing = CANONICAL_PACKING_SEQUENCE[(node_index + rotation) % len(CANONICAL_PACKING_SEQUENCE)]
        signature = _packing_signature(packing)
        if maximal_signatures is not None and signature not in maximal_signatures.get(node_id, set()):
            raise ValueError(
                f"Canonical packing {signature} is not maximal for Stage 4D.1 node {node_id}"
            )
        per_class_seen: dict[str, int] = defaultdict(int)
        for class_id in packing:
            per_class_seen[class_id] += 1
            short = {
                "benchmark-json-medium": "bench",
                "dendro-r9-t1p0": "dendro",
                "llm-distilgpt2": "llm",
            }[class_id]
            output.append(
                LayoutTask(
                    task_id=(
                        f"{scenario_id}-{node_id}-{short}-{per_class_seen[class_id]}"
                    ),
                    class_id=class_id,
                    initial_node_id=node_id,
                    resource_request=requests[class_id],
                )
            )
    mix: dict[str, int] = defaultdict(int)
    for task in output:
        mix[task.class_id] += 1
    if dict(mix) != CANONICAL_TASK_MIX:
        raise AssertionError(f"Unexpected canonical task mix: {dict(mix)}")
    return output


def layout_rows(scenario_id: str, layout: list[LayoutTask]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario_id,
            "task_id": task.task_id,
            "class_id": task.class_id,
            "initial_node_id": task.initial_node_id,
            "cpu_request_cores": task.resource_request.cpu_cores,
            "memory_request_mb": task.resource_request.memory_mb,
            "gpu_request_count": task.resource_request.gpu_count,
        }
        for task in layout
    ]


def _task_context(
    *,
    task: CapacityTaskState,
    candidate: ScoredAction,
    ranked_actions: list[ScoredAction],
) -> TaskBidContext:
    alternatives = [
        action
        for action in ranked_actions
        if not (
            action.action == candidate.action
            and action.destination_node_id == candidate.destination_node_id
            and action.source_node_id == candidate.source_node_id
        )
    ]
    fallback = min(alternatives, key=lambda action: action.score) if alternatives else None
    opportunity_loss = (
        max(0.0, fallback.score - candidate.score)
        if fallback is not None
        else 0.0
    )
    return TaskBidContext(
        workload_type=task.calibration.workload or task.class_id,
        estimated_remaining_seconds=task.remaining_work_seconds,
        checkpoint_bytes=task.calibration.checkpoint_bytes,
        static_data_bytes=0,
        accumulated_cost_usd=task.accumulated_cost_usd,
        resource_request=task.resource_request,
        fallback_action=fallback.action if fallback is not None else None,
        fallback_destination_node_id=(
            fallback.destination_node_id if fallback is not None else None
        ),
        fallback_score=fallback.score if fallback is not None else None,
        opportunity_loss=opportunity_loss,
        effective_power_kw=task.calibration.power_kw,
        power_source="stage4a3_frozen_profile",
    )


def _occupancy(
    tasks: Iterable[CapacityTaskState],
) -> dict[str, list[CapacityTaskState]]:
    output: dict[str, list[CapacityTaskState]] = defaultdict(list)
    for task in tasks:
        if not task.completed:
            output[task.owner_node_id].append(task)
    return output


def _resource_vector(tasks: Iterable[CapacityTaskState]) -> ResourceVector:
    return sum_requests([task.resource_request for task in tasks])


def static_task_outcomes(
    *,
    layout: list[LayoutTask],
    calibrations: dict[str, WorkloadCalibration],
    runtime_scales: dict[str, float],
    node_slowdowns: dict[str, float],
    cluster: ClusterConfig,
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    scenario_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in layout:
        calibration = calibrations[task.class_id]
        remaining = calibration.scaled_work_seconds(runtime_scales[task.class_id])
        wall = remaining * node_slowdowns[task.initial_node_id]
        segment = _compute_segment(
            node=cluster.get_node(task.initial_node_id),
            carbon_store=carbon_store,
            start_utc=arrival_utc,
            seconds=wall,
            power_kw=calibration.power_kw,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "policy": STATIC_POLICY,
                "task_id": task.task_id,
                "class_id": task.class_id,
                "initial_node_id": task.initial_node_id,
                "final_node_id": task.initial_node_id,
                "completed": True,
                "completion_seconds": wall,
                "compute_seconds": wall,
                "migration_seconds": 0.0,
                "paused_idle_seconds": 0.0,
                "pause_overhead_seconds": 0.0,
                "carbon_grams": segment[0],
                "cost_usd": segment[1],
                "migrations": 0,
                "pauses": 0,
                "decision_count": 0,
                "bid_attempts": 0,
                "bid_accepts": 0,
                "bid_rejections": 0,
                "owner_path": task.initial_node_id,
            }
        )
    return rows


def unlimited_task_outcomes(
    *,
    layout: list[LayoutTask],
    calibrations: dict[str, WorkloadCalibration],
    runtime_scales: dict[str, float],
    node_slowdowns: dict[str, float],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    edge_rows: list[dict[str, str]],
    arrival_utc: pd.Timestamp,
    scenario_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in layout:
        calibration = calibrations[task.class_id]
        graph = FrozenCalibrationGraph(
            cluster=cluster,
            edge_rows=edge_rows,
            workload=calibration,
        )
        outcome = replay_magellan_causal(
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            graph=graph,
            arrival_utc=arrival_utc,
            runtime_scale=runtime_scales[task.class_id],
            start_node_id=task.initial_node_id,
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "policy": UNLIMITED_POLICY,
                "task_id": task.task_id,
                "class_id": task.class_id,
                "initial_node_id": task.initial_node_id,
                "final_node_id": outcome.final_node_id,
                "completed": outcome.completed,
                "completion_seconds": outcome.makespan_seconds,
                "compute_seconds": outcome.compute_seconds,
                "migration_seconds": outcome.migration_seconds,
                "paused_idle_seconds": outcome.paused_idle_seconds,
                "pause_overhead_seconds": outcome.pause_overhead_seconds,
                "carbon_grams": outcome.carbon_grams,
                "cost_usd": outcome.cost_usd,
                "migrations": outcome.migrations,
                "pauses": outcome.pauses,
                "decision_count": outcome.decision_count,
                "bid_attempts": outcome.migrations,
                "bid_accepts": outcome.migrations,
                "bid_rejections": 0,
                "owner_path": "->".join(outcome.owner_path),
            }
        )
    return rows


def _apply_pause(
    *,
    task: CapacityTaskState,
    now: pd.Timestamp,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    cluster: ClusterConfig,
    selected: ScoredAction,
) -> None:
    source = cluster.get_node(task.owner_node_id)
    idle_seconds = float(selected.details.get("idle_seconds", policy.pause.idle_seconds))
    pause_seconds = float(policy.pause.pause_seconds)
    resume_seconds = float(policy.pause.resume_seconds)
    pause_intensity = (
        carbon_store.average(source.id, now, pause_seconds)
        if pause_seconds > 0
        else 0.0
    )
    resume_start = now + pd.Timedelta(seconds=pause_seconds + idle_seconds)
    resume_intensity = (
        carbon_store.average(source.id, resume_start, resume_seconds)
        if resume_seconds > 0
        else 0.0
    )
    task.accumulated_carbon_grams += task.calibration.power_kw * source.pue * (
        seconds_to_hours(pause_seconds) * pause_intensity
        + seconds_to_hours(resume_seconds) * resume_intensity
    )
    elapsed = pause_seconds + idle_seconds + resume_seconds
    task.paused_idle_seconds += idle_seconds
    task.pause_overhead_seconds += pause_seconds + resume_seconds
    task.pauses += 1
    task.last_pause_at = now.to_pydatetime(warn=False)
    task.blocked_until_utc = now + pd.Timedelta(seconds=elapsed)


def replay_capacity_policy(
    *,
    policy_label: str,
    auction_strategy: AuctionStrategy,
    layout: list[LayoutTask],
    capacities: dict[str, NodeResourceCapacity],
    calibrations: dict[str, WorkloadCalibration],
    runtime_scales: dict[str, float],
    node_slowdowns: dict[str, float],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    edge_rows: list[dict[str, str]],
    arrival_utc: pd.Timestamp,
    scenario_id: str,
    max_elapsed_multiplier: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if policy_label not in CAPACITY_POLICIES:
        raise ValueError(f"Unsupported Stage 4D.2 capacity policy {policy_label}")
    epoch_seconds = float(cluster.epoch_seconds)
    target_seconds = max(
        calibrations[task.class_id].scaled_work_seconds(runtime_scales[task.class_id])
        for task in layout
    )
    maximum_rounds = int(math.ceil(target_seconds * max_elapsed_multiplier / epoch_seconds)) + 4
    graphs = {
        class_id: FrozenCalibrationGraph(
            cluster=cluster,
            edge_rows=edge_rows,
            workload=calibration,
        )
        for class_id, calibration in calibrations.items()
    }
    tasks = {
        item.task_id: CapacityTaskState(
            task_id=item.task_id,
            class_id=item.class_id,
            initial_node_id=item.initial_node_id,
            owner_node_id=item.initial_node_id,
            resource_request=item.resource_request,
            calibration=calibrations[item.class_id],
            remaining_work_seconds=calibrations[item.class_id].scaled_work_seconds(
                runtime_scales[item.class_id]
            ),
        )
        for item in layout
    }
    all_node_ids = {node.id for node in cluster.nodes}
    credits: dict[str, dict[str, float]] = {
        node_id: defaultdict(float) for node_id in all_node_ids
    }
    auction_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    now = as_utc_timestamp(arrival_utc)
    telemetry_confidence = float(policy.telemetry.cpu_power_confidence)

    with tempfile.TemporaryDirectory(prefix="magellan-stage4d2-") as directory:
        adaptive = AdaptivePolicyService(
            policy.adaptive,
            policy.weights,
            AdaptivePolicyStore(Path(directory)),
        )
        for round_index in range(1, maximum_rounds + 1):
            active = [task for task in tasks.values() if not task.completed]
            if not active:
                break

            occupancy_start = _occupancy(active)
            for node in cluster.nodes:
                used = _resource_vector(occupancy_start.get(node.id, []))
                cap = capacities[node.id]
                occupancy_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "policy": policy_label,
                        "round_index": round_index,
                        "at_utc": now.isoformat(),
                        "node_id": node.id,
                        "task_count": len(occupancy_start.get(node.id, [])),
                        "used_cpu_cores": used.cpu_cores,
                        "capacity_cpu_cores": cap.cpu_cores,
                        "used_memory_mb": used.memory_mb,
                        "capacity_memory_mb": cap.memory_mb,
                        "used_gpu_count": used.gpu_count,
                        "capacity_gpu_count": cap.gpu_count,
                    }
                )

            decisions: dict[str, Any] = {}
            bids_by_destination: dict[str, list[BidRecord]] = defaultdict(list)
            for task in sorted(active, key=lambda item: item.task_id):
                if task.blocked_until_utc is not None and now < task.blocked_until_utc:
                    continue
                slowdown = node_slowdowns[task.owner_node_id]
                profile = TaskProfile(
                    task_id=task.task_id,
                    workload_type=task.calibration.workload or task.class_id,
                    current_node_id=task.owner_node_id,
                    power_kw=task.calibration.power_kw,
                    checkpoint_bytes=task.calibration.checkpoint_bytes,
                    data_bytes=0,
                    prestaged_node_ids=all_node_ids,
                    estimated_remaining_seconds=task.remaining_work_seconds * slowdown,
                    accumulated_cost_usd=task.accumulated_cost_usd,
                    cost_cap_usd=None,
                    last_migration_at=task.last_migration_at,
                    last_pause_at=task.last_pause_at,
                    resource_request=task.resource_request,
                )
                decision = evaluate_task(
                    task=profile,
                    cluster=cluster,
                    policy=policy,
                    graph=graphs[task.class_id],  # type: ignore[arg-type]
                    carbon_store=carbon_store,
                    at_utc=now,
                    static_data_bytes_by_destination={
                        node_id: 0 for node_id in all_node_ids - {task.owner_node_id}
                    },
                    adaptive_service=adaptive,
                    telemetry_confidence=telemetry_confidence,
                    compatible_destination_ids=all_node_ids - {task.owner_node_id},
                )
                task.decision_count += 1
                decisions[task.task_id] = decision
                selected = decision.selected
                if selected.action != ActionType.MIGRATE:
                    continue
                destination = selected.destination_node_id
                if destination is None:
                    raise RuntimeError("Migration decision missing destination")
                task.bid_attempts += 1
                context = _task_context(
                    task=task,
                    candidate=selected,
                    ranked_actions=decision.ranked_actions,
                )
                bid_id = f"{scenario_id}:{policy_label}:{round_index}:{task.task_id}"
                bids_by_destination[destination].append(
                    BidRecord(
                        bid_id=bid_id,
                        epoch_id=f"{scenario_id}:{round_index}",
                        task_id=task.task_id,
                        task_context=context,
                        source_node_id=task.owner_node_id,
                        destination_node_id=destination,
                        candidate=selected,
                        submitted_at_utc=now.to_pydatetime(warn=False),
                        received_at_utc=now.to_pydatetime(warn=False),
                    )
                )

            accepted: dict[str, tuple[BidRecord, int, float, float]] = {}
            for destination_id in sorted(bids_by_destination):
                destination_bids = bids_by_destination[destination_id]
                used = _resource_vector(occupancy_start.get(destination_id, []))
                ledger = ResourceLedger.from_capacity(capacities[destination_id], used=used)
                ranked = rank_bids(
                    bids=destination_bids,
                    strategy=auction_strategy,
                    credits=dict(credits[destination_id]),
                    node_resources=capacities[destination_id],
                    policy=policy.auction,
                    now_utc=now.to_pydatetime(warn=False),
                )
                for rank, item in enumerate(ranked, start=1):
                    bid = item.bid
                    task = tasks[bid.task_id]
                    before = float(credits[destination_id].get(task.task_id, 0.0))
                    fits, reason = ledger.compatible(task.resource_request)
                    if fits:
                        ledger.consume(task.resource_request)
                        after = before * policy.auction.accepted_credit_decay
                        credits[destination_id][task.task_id] = after
                        task.bid_accepts += 1
                        accepted[task.task_id] = (bid, rank, before, after)
                        status = "accepted"
                        decision_reason = f"Selected by {auction_strategy.value}; destination resources reserved"
                    else:
                        after = min(
                            policy.auction.credit_max,
                            before + policy.auction.credit_increment,
                        )
                        credits[destination_id][task.task_id] = after
                        task.bid_rejections += 1
                        status = "rejected"
                        decision_reason = reason or "Destination resources unavailable"
                    auction_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "policy": policy_label,
                            "round_index": round_index,
                            "at_utc": now.isoformat(),
                            "destination_node_id": destination_id,
                            "task_id": task.task_id,
                            "class_id": task.class_id,
                            "source_node_id": task.owner_node_id,
                            "candidate_score": bid.candidate.score,
                            "auction_strategy": auction_strategy.value,
                            "auction_rank": rank,
                            "status": status,
                            "decision_reason": decision_reason,
                            "credit_before": before,
                            "credit_after": after,
                            "opportunity_loss": item.metrics["opportunity_loss"],
                            "dominant_resource_share": item.metrics["dominant_resource_share"],
                            "resource_efficiency": item.metrics["resource_efficiency"],
                            "requested_cpu_cores": task.resource_request.cpu_cores,
                            "requested_memory_mb": task.resource_request.memory_mb,
                            "requested_gpu_count": task.resource_request.gpu_count,
                        }
                    )

            migration_elapsed: dict[str, float] = {}
            for task_id, (bid, rank, credit_before, credit_after) in accepted.items():
                task = tasks[task_id]
                source_id = task.owner_node_id
                destination_id = bid.destination_node_id
                source = cluster.get_node(source_id)
                destination = cluster.get_node(destination_id)
                elapsed, carbon, cost, details = _realized_migration(
                    source=source,
                    destination=destination,
                    edge=graphs[task.class_id].edge(source_id, destination_id),
                    policy=policy,
                    carbon_store=carbon_store,
                    start_utc=now,
                    power_kw=task.calibration.power_kw,
                    checkpoint_bytes=task.calibration.checkpoint_bytes,
                )
                task.accumulated_carbon_grams += carbon
                task.accumulated_cost_usd += cost
                task.migration_seconds += elapsed
                task.migrations += 1
                task.last_migration_at = now.to_pydatetime(warn=False)
                task.owner_node_id = destination_id
                task.owner_path.append(destination_id)
                migration_elapsed[task_id] = elapsed
                migration_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "policy": policy_label,
                        "round_index": round_index,
                        "task_id": task.task_id,
                        "class_id": task.class_id,
                        "source_node_id": source_id,
                        "destination_node_id": destination_id,
                        "started_at_utc": now.isoformat(),
                        "finished_at_utc": (now + pd.Timedelta(seconds=elapsed)).isoformat(),
                        "migration_seconds": elapsed,
                        "migration_carbon_grams": carbon,
                        "migration_cost_usd": cost,
                        "candidate_score": bid.candidate.score,
                        "auction_rank": rank,
                        "credit_before": credit_before,
                        "credit_after": credit_after,
                        "remaining_boston_equivalent_seconds": task.remaining_work_seconds,
                        "transfer_model": details.get("transfer_model"),
                    }
                )

            round_end = now + pd.Timedelta(seconds=epoch_seconds)
            for task in active:
                if task.completed:
                    continue
                decision = decisions.get(task.task_id)
                if decision is None:
                    compute_start = max(now, task.blocked_until_utc or now)
                elif task.task_id in accepted:
                    compute_start = now + pd.Timedelta(seconds=migration_elapsed[task.task_id])
                elif decision.selected.action == ActionType.PAUSE:
                    _apply_pause(
                        task=task,
                        now=now,
                        policy=policy,
                        carbon_store=carbon_store,
                        cluster=cluster,
                        selected=decision.selected,
                    )
                    compute_start = max(now, task.blocked_until_utc or now)
                else:
                    # Continue, or a migration whose destination rejected the bid.
                    compute_start = now

                if compute_start >= round_end:
                    continue
                available_wall = (round_end - compute_start).total_seconds()
                slowdown = node_slowdowns[task.owner_node_id]
                wall = min(available_wall, task.remaining_work_seconds * slowdown)
                if wall <= 0:
                    continue
                segment = _compute_segment(
                    node=cluster.get_node(task.owner_node_id),
                    carbon_store=carbon_store,
                    start_utc=compute_start,
                    seconds=wall,
                    power_kw=task.calibration.power_kw,
                )
                task.accumulated_carbon_grams += segment[0]
                task.accumulated_cost_usd += segment[1]
                task.compute_seconds += wall
                task.remaining_work_seconds = max(
                    0.0,
                    task.remaining_work_seconds - wall / slowdown,
                )
                if task.completed:
                    task.finished_at_utc = compute_start + pd.Timedelta(seconds=wall)
            now = round_end
        else:
            remaining = [task.task_id for task in tasks.values() if not task.completed]
            raise RuntimeError(
                f"Stage 4D.2 replay exceeded {maximum_rounds} rounds; unfinished={remaining}"
            )

    task_rows: list[dict[str, Any]] = []
    for task in sorted(tasks.values(), key=lambda item: item.task_id):
        completion = (
            (task.finished_at_utc - as_utc_timestamp(arrival_utc)).total_seconds()
            if task.finished_at_utc is not None
            else math.nan
        )
        task_rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy_label,
                "task_id": task.task_id,
                "class_id": task.class_id,
                "initial_node_id": task.initial_node_id,
                "final_node_id": task.owner_node_id,
                "completed": task.completed,
                "completion_seconds": completion,
                "compute_seconds": task.compute_seconds,
                "migration_seconds": task.migration_seconds,
                "paused_idle_seconds": task.paused_idle_seconds,
                "pause_overhead_seconds": task.pause_overhead_seconds,
                "carbon_grams": task.accumulated_carbon_grams,
                "cost_usd": task.accumulated_cost_usd,
                "migrations": task.migrations,
                "pauses": task.pauses,
                "decision_count": task.decision_count,
                "bid_attempts": task.bid_attempts,
                "bid_accepts": task.bid_accepts,
                "bid_rejections": task.bid_rejections,
                "owner_path": "->".join(task.owner_path),
            }
        )
    return task_rows, auction_rows, migration_rows, occupancy_rows


def scenario_outcome_row(
    *,
    scenario_id: str,
    season: str,
    arrival_utc: str,
    policy: str,
    task_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completion = [float(row["completion_seconds"]) for row in task_rows]
    owner_nodes = set()
    for row in task_rows:
        owner_nodes.update(str(row["owner_path"]).split("->"))
    return {
        "scenario_id": scenario_id,
        "season": season,
        "arrival_utc": arrival_utc,
        "policy": policy,
        "task_count": len(task_rows),
        "completed_task_count": sum(bool(row["completed"]) for row in task_rows),
        "makespan_seconds": max(completion),
        "mean_completion_seconds": sum(completion) / len(completion),
        "carbon_grams": sum(float(row["carbon_grams"]) for row in task_rows),
        "cost_usd": sum(float(row["cost_usd"]) for row in task_rows),
        "migrations": sum(int(row["migrations"]) for row in task_rows),
        "pauses": sum(int(row["pauses"]) for row in task_rows),
        "bid_attempts": sum(int(row["bid_attempts"]) for row in task_rows),
        "bid_accepts": sum(int(row["bid_accepts"]) for row in task_rows),
        "bid_rejections": sum(int(row["bid_rejections"]) for row in task_rows),
        "tasks_migrated": sum(int(row["migrations"]) > 0 for row in task_rows),
        "distinct_nodes_visited": len(owner_nodes),
    }


def attach_static_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scenario[row["scenario_id"]][row["policy"]] = row
    output: list[dict[str, Any]] = []
    for row in rows:
        baseline = by_scenario[row["scenario_id"]][STATIC_POLICY]
        enriched = dict(row)
        enriched["time_ratio_vs_static"] = float(row["makespan_seconds"]) / float(
            baseline["makespan_seconds"]
        )
        enriched["carbon_ratio_vs_static"] = float(row["carbon_grams"]) / float(
            baseline["carbon_grams"]
        )
        enriched["cost_ratio_vs_static"] = float(row["cost_usd"]) / float(
            baseline["cost_usd"]
        )
        output.append(enriched)
    return output


def summarize_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for policy in ALL_POLICIES:
        subset = [row for row in rows if row["policy"] == policy]
        if not subset:
            continue
        output.append(
            {
                "policy": policy,
                "scenario_count": len(subset),
                "task_count_total": sum(int(row["task_count"]) for row in subset),
                "makespan_seconds_mean": sum(float(row["makespan_seconds"]) for row in subset) / len(subset),
                "carbon_grams_mean": sum(float(row["carbon_grams"]) for row in subset) / len(subset),
                "cost_usd_mean": sum(float(row["cost_usd"]) for row in subset) / len(subset),
                "time_ratio_mean": sum(float(row["time_ratio_vs_static"]) for row in subset) / len(subset),
                "carbon_ratio_mean": sum(float(row["carbon_ratio_vs_static"]) for row in subset) / len(subset),
                "cost_ratio_mean": sum(float(row["cost_ratio_vs_static"]) for row in subset) / len(subset),
                "migrations_total": sum(int(row["migrations"]) for row in subset),
                "bid_attempts_total": sum(int(row["bid_attempts"]) for row in subset),
                "bid_accepts_total": sum(int(row["bid_accepts"]) for row in subset),
                "bid_rejections_total": sum(int(row["bid_rejections"]) for row in subset),
                "tasks_migrated_total": sum(int(row["tasks_migrated"]) for row in subset),
            }
        )
    return output
