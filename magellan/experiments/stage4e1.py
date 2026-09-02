from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

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
)
from magellan.experiments.stage4d2 import (
    CREDIT_FAIR_POLICY,
    LOWEST_SCORE_POLICY,
)
from magellan.models.types import ActionType, ScoredAction, TaskProfile, TaskResourceRequest
from magellan.models.utils import seconds_to_hours
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task


STATIC_SCALE_POLICY = "static_resource_queue"
SCALE_POLICIES = (
    STATIC_SCALE_POLICY,
    LOWEST_SCORE_POLICY,
    CREDIT_FAIR_POLICY,
)
SCALE_SIZES = (25, 50, 100)
CLASS_SEQUENCE = (
    "benchmark-json-medium",
    "dendro-r9-t1p0",
    "llm-distilgpt2",
)


@dataclass(frozen=True)
class ScaleTaskSpec:
    task_id: str
    class_id: str
    home_node_id: str
    arrival_utc: pd.Timestamp
    resource_request: TaskResourceRequest


@dataclass
class ScaleTaskState:
    task_id: str
    class_id: str
    home_node_id: str
    arrival_utc: pd.Timestamp
    resource_request: TaskResourceRequest
    calibration: WorkloadCalibration
    remaining_work_seconds: float
    owner_node_id: str | None = None
    admitted_at_utc: pd.Timestamp | None = None
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

    @property
    def submitted(self) -> bool:
        return True

    @property
    def admitted(self) -> bool:
        return self.admitted_at_utc is not None

    @property
    def completed(self) -> bool:
        return self.remaining_work_seconds <= 1e-9

    @property
    def running(self) -> bool:
        return self.admitted and not self.completed and self.owner_node_id is not None


def build_scale_population(
    *,
    task_count: int,
    node_ids: list[str],
    requests: dict[str, TaskResourceRequest],
    start_utc: pd.Timestamp,
    arrival_window_seconds: float,
    epoch_seconds: float,
) -> list[ScaleTaskSpec]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if not node_ids:
        raise ValueError("node_ids cannot be empty")
    if arrival_window_seconds < 0:
        raise ValueError("arrival_window_seconds cannot be negative")
    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive")

    window_epochs = max(1, int(round(arrival_window_seconds / epoch_seconds)))
    output: list[ScaleTaskSpec] = []

    for index in range(task_count):
        class_id = CLASS_SEQUENCE[index % len(CLASS_SEQUENCE)]
        home_node_id = node_ids[index % len(node_ids)]
        arrival_epoch = min(
            window_epochs - 1,
            int(math.floor(index * window_epochs / task_count)),
        )
        output.append(
            ScaleTaskSpec(
                task_id=f"scale-{task_count:03d}-{index + 1:03d}",
                class_id=class_id,
                home_node_id=home_node_id,
                arrival_utc=as_utc_timestamp(start_utc)
                + pd.Timedelta(seconds=arrival_epoch * epoch_seconds),
                resource_request=requests[class_id],
            )
        )

    return output


def class_counts(specs: Iterable[ScaleTaskSpec]) -> dict[str, int]:
    counts = defaultdict(int)
    for task in specs:
        counts[task.class_id] += 1
    return {class_id: int(counts.get(class_id, 0)) for class_id in CLASS_SEQUENCE}


def node_counts(specs: Iterable[ScaleTaskSpec]) -> dict[str, int]:
    counts = defaultdict(int)
    for task in specs:
        counts[task.home_node_id] += 1
    return dict(counts)


def _occupancy(tasks: Iterable[ScaleTaskState]) -> dict[str, list[ScaleTaskState]]:
    output: dict[str, list[ScaleTaskState]] = defaultdict(list)
    for task in tasks:
        if task.running and task.owner_node_id is not None:
            output[task.owner_node_id].append(task)
    return output


def _resource_vector(tasks: Iterable[ScaleTaskState]) -> ResourceVector:
    return sum_requests([task.resource_request for task in tasks])


def _task_context(
    *,
    task: ScaleTaskState,
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


def _apply_pause(
    *,
    task: ScaleTaskState,
    now: pd.Timestamp,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    cluster: ClusterConfig,
    selected: ScoredAction,
) -> None:
    if task.owner_node_id is None:
        raise RuntimeError("Cannot pause an unadmitted Stage 4E.1 task")
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


def _event(
    *,
    scenario_id: str,
    policy_label: str,
    at_utc: pd.Timestamp,
    task: ScaleTaskState | ScaleTaskSpec,
    event_type: str,
    source_node_id: str | None = None,
    destination_node_id: str | None = None,
    details: str = "",
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "policy": policy_label,
        "at_utc": as_utc_timestamp(at_utc).isoformat(),
        "elapsed_seconds": (
            as_utc_timestamp(at_utc) - as_utc_timestamp(task.arrival_utc)
        ).total_seconds(),
        "task_id": task.task_id,
        "class_id": task.class_id,
        "event_type": event_type,
        "source_node_id": source_node_id or "",
        "destination_node_id": destination_node_id or "",
        "details": details,
    }


def _admit_queued(
    *,
    states: dict[str, ScaleTaskState],
    now: pd.Timestamp,
    capacities: dict[str, NodeResourceCapacity],
    cluster: ClusterConfig,
    scenario_id: str,
    policy_label: str,
    event_rows: list[dict[str, Any]],
) -> int:
    occupancy = _occupancy(states.values())
    admitted_count = 0

    for node in cluster.nodes:
        used = _resource_vector(occupancy.get(node.id, []))
        ledger = ResourceLedger.from_capacity(capacities[node.id], used=used)
        queued = sorted(
            (
                task
                for task in states.values()
                if not task.admitted
                and not task.completed
                and task.arrival_utc <= now
                and task.home_node_id == node.id
            ),
            key=lambda task: (task.arrival_utc, task.task_id),
        )

        # Feasibility-preserving FIFO scan: earlier tasks are considered first,
        # but a temporarily too-large head task does not force otherwise usable
        # measured resources to sit idle.
        for task in queued:
            fits, _ = ledger.compatible(task.resource_request)
            if not fits:
                continue
            ledger.consume(task.resource_request)
            task.owner_node_id = node.id
            task.admitted_at_utc = now
            task.owner_path = [node.id]
            occupancy[node.id].append(task)
            admitted_count += 1
            event_rows.append(
                _event(
                    scenario_id=scenario_id,
                    policy_label=policy_label,
                    at_utc=now,
                    task=task,
                    event_type="admitted",
                    destination_node_id=node.id,
                    details="source-side measured-resource admission",
                )
            )
    return admitted_count


def _percentile(values: list[float], fraction: float) -> float:
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


def _maximum_rounds(
    *,
    task_count: int,
    target_boston_seconds: float,
    arrival_window_seconds: float,
    epoch_seconds: float,
) -> int:
    # Conservative guard only. The real stop condition is all tasks completed.
    # Seven nodes can each run at least one frozen workload, so task_count/7 is
    # a safe coarse wave count; the 2.5 factor absorbs slow nodes, migration and
    # queue fragmentation without requiring an hours-long fixed horizon.
    horizon = (
        arrival_window_seconds
        + target_boston_seconds * math.ceil(task_count / 7) * 2.5
        + 6 * 3600
    )
    return int(math.ceil(horizon / epoch_seconds)) + 8


def replay_scale_policy(
    *,
    policy_label: str,
    auction_strategy: AuctionStrategy | None,
    specs: list[ScaleTaskSpec],
    capacities: dict[str, NodeResourceCapacity],
    calibrations: dict[str, WorkloadCalibration],
    runtime_scales: dict[str, float],
    node_slowdowns: dict[str, float],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    edge_rows: list[dict[str, str]],
    scenario_id: str,
    target_boston_seconds: float,
    arrival_window_seconds: float,
    progress: Callable[[str], None] | None = None,
    progress_every_rounds: int = 24,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if policy_label != STATIC_SCALE_POLICY and auction_strategy is None:
        raise ValueError("Magellan scaling policies require an auction strategy")

    epoch_seconds = float(cluster.epoch_seconds)
    start_utc = min(task.arrival_utc for task in specs)
    final_arrival_utc = max(task.arrival_utc for task in specs)
    maximum_rounds = _maximum_rounds(
        task_count=len(specs),
        target_boston_seconds=target_boston_seconds,
        arrival_window_seconds=arrival_window_seconds,
        epoch_seconds=epoch_seconds,
    )

    graphs = {
        class_id: FrozenCalibrationGraph(
            cluster=cluster,
            edge_rows=edge_rows,
            workload=calibration,
        )
        for class_id, calibration in calibrations.items()
    }
    states = {
        spec.task_id: ScaleTaskState(
            task_id=spec.task_id,
            class_id=spec.class_id,
            home_node_id=spec.home_node_id,
            arrival_utc=spec.arrival_utc,
            resource_request=spec.resource_request,
            calibration=calibrations[spec.class_id],
            remaining_work_seconds=calibrations[spec.class_id].scaled_work_seconds(
                runtime_scales[spec.class_id]
            ),
        )
        for spec in specs
    }
    all_node_ids = {node.id for node in cluster.nodes}
    credits: dict[str, dict[str, float]] = {
        node_id: defaultdict(float) for node_id in all_node_ids
    }
    auction_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    telemetry_confidence = float(policy.telemetry.cpu_power_confidence)
    now = as_utc_timestamp(start_utc)
    submitted_ids: set[str] = set()
    cpu_utilization_samples: list[float] = []
    max_queue_depth = 0

    with tempfile.TemporaryDirectory(prefix="magellan-stage4e1-") as directory:
        adaptive = AdaptivePolicyService(
            policy.adaptive,
            policy.weights,
            AdaptivePolicyStore(Path(directory)),
        )

        for round_index in range(1, maximum_rounds + 1):
            for task in sorted(states.values(), key=lambda item: item.task_id):
                if task.task_id in submitted_ids or task.arrival_utc > now:
                    continue
                submitted_ids.add(task.task_id)
                event_rows.append(
                    _event(
                        scenario_id=scenario_id,
                        policy_label=policy_label,
                        at_utc=task.arrival_utc,
                        task=task,
                        event_type="submitted",
                        destination_node_id=task.home_node_id,
                        details="deterministic source-region arrival",
                    )
                )

            _admit_queued(
                states=states,
                now=now,
                capacities=capacities,
                cluster=cluster,
                scenario_id=scenario_id,
                policy_label=policy_label,
                event_rows=event_rows,
            )

            running = [task for task in states.values() if task.running]
            queued = [
                task
                for task in states.values()
                if not task.admitted and task.arrival_utc <= now and not task.completed
            ]
            future = [task for task in states.values() if task.arrival_utc > now]
            max_queue_depth = max(max_queue_depth, len(queued))

            if not running and not queued and not future:
                break

            if (
                progress is not None
                and progress_every_rounds > 0
                and (round_index == 1 or round_index % progress_every_rounds == 0)
            ):
                elapsed_hours = (now - start_utc).total_seconds() / 3600.0
                progress(
                    f"round {round_index}/{maximum_rounds} "
                    f"simulated={elapsed_hours:.1f}h "
                    f"running={len(running)} queued={len(queued)} "
                    f"completed={sum(task.completed for task in states.values())}/{len(states)}"
                )

            occupancy_start = _occupancy(states.values())
            used_cluster_cpu = 0.0
            capacity_cluster_cpu = 0.0
            queue_by_home = defaultdict(int)
            for task in queued:
                queue_by_home[task.home_node_id] += 1

            for node in cluster.nodes:
                used = _resource_vector(occupancy_start.get(node.id, []))
                cap = capacities[node.id]
                used_cluster_cpu += used.cpu_cores
                capacity_cluster_cpu += cap.cpu_cores
                occupancy_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "policy": policy_label,
                        "round_index": round_index,
                        "at_utc": now.isoformat(),
                        "node_id": node.id,
                        "running_task_count": len(occupancy_start.get(node.id, [])),
                        "queued_home_task_count": int(queue_by_home.get(node.id, 0)),
                        "used_cpu_cores": used.cpu_cores,
                        "capacity_cpu_cores": cap.cpu_cores,
                        "used_memory_mb": used.memory_mb,
                        "capacity_memory_mb": cap.memory_mb,
                        "used_gpu_count": used.gpu_count,
                        "capacity_gpu_count": cap.gpu_count,
                    }
                )
            cpu_utilization_samples.append(
                used_cluster_cpu / capacity_cluster_cpu if capacity_cluster_cpu else 0.0
            )

            decisions: dict[str, Any] = {}
            bids_by_destination: dict[str, list[BidRecord]] = defaultdict(list)

            if policy_label != STATIC_SCALE_POLICY:
                for task in sorted(running, key=lambda item: item.task_id):
                    if task.blocked_until_utc is not None and now < task.blocked_until_utc:
                        continue
                    if task.owner_node_id is None:
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
                    bids_by_destination[destination].append(
                        BidRecord(
                            bid_id=f"{scenario_id}:{policy_label}:{round_index}:{task.task_id}",
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
            if auction_strategy is not None:
                for destination_id in sorted(bids_by_destination):
                    destination_bids = bids_by_destination[destination_id]
                    used = _resource_vector(occupancy_start.get(destination_id, []))
                    ledger = ResourceLedger.from_capacity(
                        capacities[destination_id],
                        used=used,
                    )
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
                        task = states[bid.task_id]
                        before = float(credits[destination_id].get(task.task_id, 0.0))
                        fits, reason = ledger.compatible(task.resource_request)
                        if fits:
                            ledger.consume(task.resource_request)
                            after = before * policy.auction.accepted_credit_decay
                            credits[destination_id][task.task_id] = after
                            task.bid_accepts += 1
                            accepted[task.task_id] = (bid, rank, before, after)
                            status = "accepted"
                            decision_reason = (
                                f"Selected by {auction_strategy.value}; "
                                "destination resources reserved"
                            )
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
                task = states[task_id]
                if task.owner_node_id is None:
                    raise RuntimeError("Accepted migration for unadmitted task")
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
                        "finished_at_utc": (
                            now + pd.Timedelta(seconds=elapsed)
                        ).isoformat(),
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
                event_rows.append(
                    _event(
                        scenario_id=scenario_id,
                        policy_label=policy_label,
                        at_utc=now,
                        task=task,
                        event_type="migration_start",
                        source_node_id=source_id,
                        destination_node_id=destination_id,
                        details=f"auction_rank={rank}",
                    )
                )
                event_rows.append(
                    _event(
                        scenario_id=scenario_id,
                        policy_label=policy_label,
                        at_utc=now + pd.Timedelta(seconds=elapsed),
                        task=task,
                        event_type="migration_finish",
                        source_node_id=source_id,
                        destination_node_id=destination_id,
                        details=f"migration_seconds={elapsed:.6f}",
                    )
                )

            # Accepted migrations immediately free their source reservations.
            # Fill those measured resources from source-side queues before the
            # compute portion of the same epoch.
            if accepted:
                _admit_queued(
                    states=states,
                    now=now,
                    capacities=capacities,
                    cluster=cluster,
                    scenario_id=scenario_id,
                    policy_label=policy_label,
                    event_rows=event_rows,
                )

            round_end = now + pd.Timedelta(seconds=epoch_seconds)
            running_after_auction = [task for task in states.values() if task.running]
            for task in running_after_auction:
                decision = decisions.get(task.task_id)
                if task.blocked_until_utc is not None and now < task.blocked_until_utc:
                    compute_start = max(now, task.blocked_until_utc)
                elif task.task_id in accepted:
                    compute_start = now + pd.Timedelta(
                        seconds=migration_elapsed[task.task_id]
                    )
                elif (
                    decision is not None
                    and decision.selected.action == ActionType.PAUSE
                ):
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
                    compute_start = now

                if compute_start >= round_end or task.owner_node_id is None:
                    continue
                available_wall = (round_end - compute_start).total_seconds()
                slowdown = node_slowdowns[task.owner_node_id]
                wall = min(
                    available_wall,
                    task.remaining_work_seconds * slowdown,
                )
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
                    event_rows.append(
                        _event(
                            scenario_id=scenario_id,
                            policy_label=policy_label,
                            at_utc=task.finished_at_utc,
                            task=task,
                            event_type="completed",
                            source_node_id=task.owner_node_id,
                            details="task drained from measured resource ledger",
                        )
                    )

            now = round_end
        else:
            unfinished = [
                task.task_id for task in states.values() if not task.completed
            ]
            raise RuntimeError(
                f"Stage 4E.1 exceeded {maximum_rounds} rounds; "
                f"unfinished={unfinished[:12]} total={len(unfinished)}"
            )

    task_rows: list[dict[str, Any]] = []
    for task in sorted(states.values(), key=lambda item: item.task_id):
        if task.admitted_at_utc is None or task.finished_at_utc is None:
            raise RuntimeError(f"Incomplete Stage 4E.1 task state: {task.task_id}")
        queue_wait = (task.admitted_at_utc - task.arrival_utc).total_seconds()
        completion_latency = (task.finished_at_utc - task.arrival_utc).total_seconds()
        task_rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy_label,
                "task_id": task.task_id,
                "class_id": task.class_id,
                "home_node_id": task.home_node_id,
                "arrival_utc": task.arrival_utc.isoformat(),
                "admitted_at_utc": task.admitted_at_utc.isoformat(),
                "queue_wait_seconds": queue_wait,
                "initial_node_id": task.owner_path[0],
                "final_node_id": task.owner_node_id,
                "completed": task.completed,
                "completion_latency_seconds": completion_latency,
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

    finish_times = [
        as_utc_timestamp(task.finished_at_utc)
        for task in states.values()
        if task.finished_at_utc is not None
    ]
    drain_seconds = (max(finish_times) - start_utc).total_seconds()
    queue_waits = [float(row["queue_wait_seconds"]) for row in task_rows]
    completion_latencies = [
        float(row["completion_latency_seconds"]) for row in task_rows
    ]

    run_summary = {
        "scenario_id": scenario_id,
        "policy": policy_label,
        "task_count": len(task_rows),
        "first_arrival_utc": start_utc.isoformat(),
        "final_arrival_utc": final_arrival_utc.isoformat(),
        "drain_seconds": drain_seconds,
        "throughput_tasks_per_hour": (
            len(task_rows) / seconds_to_hours(drain_seconds)
            if drain_seconds > 0
            else 0.0
        ),
        "mean_completion_latency_seconds": (
            sum(completion_latencies) / len(completion_latencies)
        ),
        "p95_completion_latency_seconds": _percentile(
            completion_latencies, 0.95
        ),
        "mean_queue_wait_seconds": sum(queue_waits) / len(queue_waits),
        "p95_queue_wait_seconds": _percentile(queue_waits, 0.95),
        "max_queue_wait_seconds": max(queue_waits),
        "max_queue_depth": max_queue_depth,
        "mean_cluster_cpu_utilization": (
            sum(cpu_utilization_samples) / len(cpu_utilization_samples)
            if cpu_utilization_samples
            else 0.0
        ),
        "carbon_grams": sum(float(row["carbon_grams"]) for row in task_rows),
        "cost_usd": sum(float(row["cost_usd"]) for row in task_rows),
        "migrations": sum(int(row["migrations"]) for row in task_rows),
        "pauses": sum(int(row["pauses"]) for row in task_rows),
        "bid_attempts": sum(int(row["bid_attempts"]) for row in task_rows),
        "bid_accepts": sum(int(row["bid_accepts"]) for row in task_rows),
        "bid_rejections": sum(int(row["bid_rejections"]) for row in task_rows),
        "tasks_migrated": sum(int(row["migrations"]) > 0 for row in task_rows),
        "distinct_nodes_visited": len(
            {
                node_id
                for row in task_rows
                for node_id in str(row["owner_path"]).split("->")
                if node_id
            }
        ),
    }
    return (
        task_rows,
        auction_rows,
        migration_rows,
        occupancy_rows,
        event_rows,
        run_summary,
    )


def attach_static_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_scenario[row["scenario_id"]][row["policy"]] = row

    output = []
    for row in rows:
        baseline = by_scenario[row["scenario_id"]][STATIC_SCALE_POLICY]
        enriched = dict(row)
        for metric, ratio_name in (
            ("drain_seconds", "drain_ratio_vs_static"),
            ("mean_completion_latency_seconds", "mean_completion_ratio_vs_static"),
            ("carbon_grams", "carbon_ratio_vs_static"),
            ("cost_usd", "cost_ratio_vs_static"),
            ("mean_queue_wait_seconds", "mean_queue_wait_ratio_vs_static"),
        ):
            base = float(baseline[metric])
            value = float(row[metric])
            enriched[ratio_name] = value / base if base > 0 else (1.0 if value == 0 else math.inf)
        output.append(enriched)
    return output


def per_class_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        groups[(row["scenario_id"], row["policy"], row["class_id"])].append(row)

    output = []
    for (scenario_id, policy, class_id), rows in sorted(groups.items()):
        waits = [float(row["queue_wait_seconds"]) for row in rows]
        latency = [float(row["completion_latency_seconds"]) for row in rows]
        output.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "class_id": class_id,
                "task_count": len(rows),
                "mean_queue_wait_seconds": sum(waits) / len(waits),
                "p95_queue_wait_seconds": _percentile(waits, 0.95),
                "mean_completion_latency_seconds": sum(latency) / len(latency),
                "p95_completion_latency_seconds": _percentile(latency, 0.95),
                "carbon_grams": sum(float(row["carbon_grams"]) for row in rows),
                "cost_usd": sum(float(row["cost_usd"]) for row in rows),
                "migrations": sum(int(row["migrations"]) for row in rows),
                "bid_rejections": sum(int(row["bid_rejections"]) for row in rows),
            }
        )
    return output
