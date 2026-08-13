from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.graph.topology import ClusterGraph, EdgeMetrics
from magellan.models.continue_model import estimate_continue
from magellan.models.types import ActionType, RawActionEstimate, TaskProfile
from magellan.models.utils import bytes_to_gb, seconds_to_hours, transfer_seconds
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task, score_actions


class ComparisonPolicy(str, Enum):
    BOSTON_STATIC = "boston_static"
    FRANCE_STATIC = "france_static"
    BEST_STATIC = "best_static"
    BEST_AT_DISPATCH = "best_at_dispatch"
    TEMPORAL_ONLY = "temporal_only"
    MAGELLAN_CAUSAL = "magellan_causal"


class ComparisonWorkload(BaseModel):
    name: str = Field(default="comparison-workload", min_length=1)
    duration_seconds: float = Field(gt=0)
    power_kw: float = Field(gt=0)
    checkpoint_bytes: int = Field(default=0, ge=0)
    static_data_bytes: int = Field(default=0, ge=0)
    start_node_id: str = Field(default="boston", min_length=1)
    cost_cap_usd: float | None = Field(default=None, gt=0)


class ReplayStep(BaseModel):
    index: int = Field(ge=1)
    action: str
    source_node_id: str
    destination_node_id: str | None = None
    started_at_utc: datetime
    finished_at_utc: datetime
    elapsed_seconds: float = Field(ge=0)
    compute_seconds: float = Field(default=0.0, ge=0)
    idle_seconds: float = Field(default=0.0, ge=0)
    migration_seconds: float = Field(default=0.0, ge=0)
    carbon_grams: float = Field(default=0.0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    remaining_seconds_after: float = Field(ge=0)
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PolicyOutcome(BaseModel):
    policy: str
    start_node_id: str
    final_node_id: str
    selected_initial_node_id: str | None = None
    completed: bool = True
    makespan_seconds: float = Field(ge=0)
    compute_seconds: float = Field(ge=0)
    paused_idle_seconds: float = Field(ge=0)
    pause_overhead_seconds: float = Field(ge=0)
    migration_seconds: float = Field(ge=0)
    carbon_grams: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    migrations: int = Field(ge=0)
    pauses: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    owner_path: list[str] = Field(default_factory=list)
    steps: list[ReplayStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SegmentMetrics:
    elapsed_seconds: float
    carbon_grams: float
    cost_usd: float


@dataclass(frozen=True)
class MigrationMetrics:
    elapsed_seconds: float
    carbon_grams: float
    cost_usd: float
    details: dict[str, Any]


def _compute_segment(
    *,
    node: NodeConfig,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    seconds: float,
    power_kw: float,
) -> SegmentMetrics:
    if seconds <= 0:
        return SegmentMetrics(0.0, 0.0, 0.0)
    intensity = carbon_store.average(node.id, start_utc, seconds)
    carbon = power_kw * node.pue * seconds_to_hours(seconds) * intensity
    cost = node.compute_price_usd_per_hour * seconds_to_hours(seconds)
    return SegmentMetrics(seconds, carbon, cost)


def _active_overhead_segment(
    *,
    node: NodeConfig,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    seconds: float,
    power_kw: float,
) -> SegmentMetrics:
    """Model pause/resume active overhead using Magellan's energy assumptions.

    The task-level monetary model does not charge pause/resume overhead as active
    compute, so the returned monetary cost is intentionally zero.
    """
    if seconds <= 0:
        return SegmentMetrics(0.0, 0.0, 0.0)
    intensity = carbon_store.average(node.id, start_utc, seconds)
    carbon = power_kw * node.pue * seconds_to_hours(seconds) * intensity
    return SegmentMetrics(seconds, carbon, 0.0)


def realized_migration(
    *,
    source: NodeConfig,
    destination: NodeConfig,
    edge: EdgeMetrics,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    power_kw: float,
    checkpoint_bytes: int,
    static_data_bytes: int = 0,
) -> MigrationMetrics:
    total_transfer_bytes = checkpoint_bytes + static_data_bytes
    transfer_duration = transfer_seconds(
        size_bytes=total_transfer_bytes,
        bandwidth_mbps=edge.bandwidth_mbps,
        latency_ms=edge.latency_ms,
    )
    checkpoint_seconds = (
        edge.checkpoint_seconds
        if edge.checkpoint_seconds is not None
        else policy.pause.pause_seconds
    )
    restore_seconds = (
        edge.restore_seconds
        if edge.restore_seconds is not None
        else policy.pause.resume_seconds
    )

    checkpoint_intensity = carbon_store.average(
        source.id,
        start_utc,
        checkpoint_seconds,
    )
    restore_start = start_utc + pd.Timedelta(
        seconds=checkpoint_seconds + transfer_duration
    )
    restore_intensity = carbon_store.average(
        destination.id,
        restore_start,
        restore_seconds,
    )

    source_carbon = (
        power_kw
        * source.pue
        * seconds_to_hours(checkpoint_seconds)
        * checkpoint_intensity
    )
    destination_carbon = (
        power_kw
        * destination.pue
        * seconds_to_hours(restore_seconds)
        * restore_intensity
    )

    transfer_size_gb = bytes_to_gb(total_transfer_bytes)
    network_energy_kwh = transfer_size_gb * (
        policy.migration.network_energy_kwh_per_gb_base
        + policy.migration.network_energy_kwh_per_gb_km * edge.distance_km
    )
    mean_network_intensity = (checkpoint_intensity + restore_intensity) / 2.0
    network_carbon = network_energy_kwh * mean_network_intensity
    transfer_cost = transfer_size_gb * source.egress_price_usd_per_gb

    elapsed = checkpoint_seconds + transfer_duration + restore_seconds
    return MigrationMetrics(
        elapsed_seconds=elapsed,
        carbon_grams=source_carbon + destination_carbon + network_carbon,
        cost_usd=transfer_cost,
        details={
            "checkpoint_seconds": checkpoint_seconds,
            "transfer_seconds": transfer_duration,
            "restore_seconds": restore_seconds,
            "checkpoint_bytes": checkpoint_bytes,
            "static_data_bytes": static_data_bytes,
            "total_transfer_bytes": total_transfer_bytes,
            "distance_km": edge.distance_km,
            "bandwidth_mbps": edge.bandwidth_mbps,
            "latency_ms": edge.latency_ms,
            "source_checkpoint_carbon_grams": source_carbon,
            "destination_restore_carbon_grams": destination_carbon,
            "network_carbon_grams": network_carbon,
            "transfer_cost_usd": transfer_cost,
        },
    )


def static_outcome(
    *,
    label: str,
    node: NodeConfig,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
) -> PolicyOutcome:
    segment = _compute_segment(
        node=node,
        carbon_store=carbon_store,
        start_utc=start_utc,
        seconds=workload.duration_seconds,
        power_kw=workload.power_kw,
    )
    step = ReplayStep(
        index=1,
        action="continue",
        source_node_id=node.id,
        destination_node_id=None,
        started_at_utc=start_utc.to_pydatetime(),
        finished_at_utc=(
            start_utc + pd.Timedelta(seconds=workload.duration_seconds)
        ).to_pydatetime(),
        elapsed_seconds=workload.duration_seconds,
        compute_seconds=workload.duration_seconds,
        carbon_grams=segment.carbon_grams,
        cost_usd=segment.cost_usd,
        remaining_seconds_after=0.0,
        reason="Static placement; no later scheduling decisions",
    )
    return PolicyOutcome(
        policy=label,
        start_node_id=node.id,
        final_node_id=node.id,
        selected_initial_node_id=node.id,
        makespan_seconds=workload.duration_seconds,
        compute_seconds=workload.duration_seconds,
        paused_idle_seconds=0.0,
        pause_overhead_seconds=0.0,
        migration_seconds=0.0,
        carbon_grams=segment.carbon_grams,
        cost_usd=segment.cost_usd,
        migrations=0,
        pauses=0,
        decision_count=0,
        owner_path=[node.id],
        steps=[step],
    )


def _static_candidates(
    *,
    cluster: ClusterConfig,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
) -> list[PolicyOutcome]:
    return [
        static_outcome(
            label="static_candidate",
            node=node,
            workload=workload,
            carbon_store=carbon_store,
            start_utc=start_utc,
        )
        for node in cluster.nodes
    ]


def _choose_weighted_static(
    outcomes: list[PolicyOutcome],
    policy: ScoringPolicy,
) -> tuple[PolicyOutcome, list[dict[str, Any]]]:
    raw = [
        RawActionEstimate(
            action=ActionType.CONTINUE,
            source_node_id=item.final_node_id,
            destination_node_id=None,
            time_seconds=item.makespan_seconds,
            carbon_grams=item.carbon_grams,
            cost_usd=item.cost_usd,
        )
        for item in outcomes
    ]
    ranked = score_actions(raw, policy)
    selected_id = ranked[0].source_node_id
    selected = next(item for item in outcomes if item.final_node_id == selected_id)
    ranking = [candidate.model_dump(mode="json") for candidate in ranked]
    return selected, ranking


def best_static_outcome(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
) -> PolicyOutcome:
    candidates = _static_candidates(
        cluster=cluster,
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start_utc,
    )
    selected, ranking = _choose_weighted_static(candidates, policy)
    result = selected.model_copy(deep=True)
    result.policy = ComparisonPolicy.BEST_STATIC.value
    result.metadata = {
        "knowledge": "clairvoyant_full_interval",
        "initial_placement_overhead": "free",
        "ranking": ranking,
        "candidate_metrics": [
            {
                "node_id": item.final_node_id,
                "carbon_grams": item.carbon_grams,
                "cost_usd": item.cost_usd,
                "makespan_seconds": item.makespan_seconds,
            }
            for item in candidates
        ],
    }
    return result


def best_at_dispatch_outcome(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
) -> PolicyOutcome:
    estimates: list[RawActionEstimate] = []
    for node in cluster.nodes:
        task = TaskProfile(
            task_id=f"dispatch-{workload.name}",
            workload_type="comparison",
            current_node_id=node.id,
            power_kw=workload.power_kw,
            checkpoint_bytes=workload.checkpoint_bytes,
            data_bytes=workload.static_data_bytes,
            estimated_remaining_seconds=workload.duration_seconds,
            cost_cap_usd=workload.cost_cap_usd,
        )
        estimates.append(
            estimate_continue(
                task=task,
                node=node,
                carbon_store=carbon_store,
                at_utc=start_utc,
                horizon_seconds=policy.horizon_seconds,
                forecast_policy=policy.carbon_forecast,
            )
        )

    ranked = score_actions(estimates, policy)
    selected_node_id = ranked[0].source_node_id
    selected_node = cluster.get_node(selected_node_id)
    outcome = static_outcome(
        label=ComparisonPolicy.BEST_AT_DISPATCH.value,
        node=selected_node,
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start_utc,
    )
    outcome.metadata = {
        "knowledge": "causal_submission_time_only",
        "initial_placement_overhead": "free",
        "dispatch_horizon_seconds": policy.horizon_seconds,
        "ranking": [item.model_dump(mode="json") for item in ranked],
    }
    return outcome


def _pause_metrics(
    *,
    node: NodeConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    idle_seconds: float,
    power_kw: float,
) -> tuple[SegmentMetrics, float, float]:
    pause = _active_overhead_segment(
        node=node,
        carbon_store=carbon_store,
        start_utc=start_utc,
        seconds=policy.pause.pause_seconds,
        power_kw=power_kw,
    )
    resume_start = start_utc + pd.Timedelta(
        seconds=policy.pause.pause_seconds + idle_seconds
    )
    resume = _active_overhead_segment(
        node=node,
        carbon_store=carbon_store,
        start_utc=resume_start,
        seconds=policy.pause.resume_seconds,
        power_kw=power_kw,
    )
    elapsed = policy.pause.pause_seconds + idle_seconds + policy.pause.resume_seconds
    combined = SegmentMetrics(
        elapsed_seconds=elapsed,
        carbon_grams=pause.carbon_grams + resume.carbon_grams,
        cost_usd=0.0,
    )
    return combined, idle_seconds, policy.pause.pause_seconds + policy.pause.resume_seconds


def replay_causal_policy(
    *,
    label: ComparisonPolicy,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    graph: ClusterGraph,
    start_utc: pd.Timestamp,
    max_decisions: int = 10_000,
    assume_static_prestaged: bool = True,
) -> PolicyOutcome:
    if label not in {ComparisonPolicy.TEMPORAL_ONLY, ComparisonPolicy.MAGELLAN_CAUSAL}:
        raise ValueError(f"Unsupported causal replay policy: {label}")

    current_time = as_utc_timestamp(start_utc)
    current_node_id = workload.start_node_id
    cluster.get_node(current_node_id)
    remaining = workload.duration_seconds
    accumulated_carbon = 0.0
    accumulated_cost = 0.0
    compute_seconds = 0.0
    paused_idle_seconds = 0.0
    pause_overhead_seconds = 0.0
    migration_seconds = 0.0
    migrations = 0
    pauses = 0
    last_migration_at: datetime | None = None
    last_pause_at: datetime | None = None
    owner_path = [current_node_id]
    steps: list[ReplayStep] = []

    all_node_ids = {node.id for node in cluster.nodes}
    prestaged = set(all_node_ids) if assume_static_prestaged else {current_node_id}

    with tempfile.TemporaryDirectory(prefix="magellan-replay-policy-") as directory:
        adaptive_service = AdaptivePolicyService(
            policy.adaptive,
            policy.weights,
            AdaptivePolicyStore(Path(directory)),
        )

        for decision_index in range(1, max_decisions + 1):
            if remaining <= 1e-9:
                break

            task = TaskProfile(
                task_id=f"replay-{label.value}-{workload.name}",
                workload_type="comparison",
                current_node_id=current_node_id,
                power_kw=workload.power_kw,
                checkpoint_bytes=workload.checkpoint_bytes,
                data_bytes=workload.static_data_bytes,
                prestaged_node_ids=prestaged,
                estimated_remaining_seconds=remaining,
                accumulated_cost_usd=accumulated_cost,
                cost_cap_usd=workload.cost_cap_usd,
                last_migration_at=last_migration_at,
                last_pause_at=last_pause_at,
            )
            compatible = (
                set()
                if label == ComparisonPolicy.TEMPORAL_ONLY
                else all_node_ids - {current_node_id}
            )
            static_data = {
                node_id: (0 if node_id in prestaged else workload.static_data_bytes)
                for node_id in compatible
            }
            decision = evaluate_task(
                task=task,
                cluster=cluster,
                policy=policy,
                graph=graph,
                carbon_store=carbon_store,
                at_utc=current_time,
                static_data_bytes_by_destination=static_data,
                adaptive_service=adaptive_service,
                telemetry_confidence=0.0,
                compatible_destination_ids=compatible,
            )
            selected = decision.selected
            step_start = current_time
            source_node = cluster.get_node(current_node_id)

            if selected.action == ActionType.CONTINUE:
                seconds = min(float(cluster.epoch_seconds), remaining)
                metrics = _compute_segment(
                    node=source_node,
                    carbon_store=carbon_store,
                    start_utc=current_time,
                    seconds=seconds,
                    power_kw=workload.power_kw,
                )
                current_time += pd.Timedelta(seconds=seconds)
                remaining = max(0.0, remaining - seconds)
                compute_seconds += seconds
                accumulated_carbon += metrics.carbon_grams
                accumulated_cost += metrics.cost_usd
                steps.append(
                    ReplayStep(
                        index=decision_index,
                        action="continue",
                        source_node_id=current_node_id,
                        started_at_utc=step_start.to_pydatetime(),
                        finished_at_utc=current_time.to_pydatetime(),
                        elapsed_seconds=seconds,
                        compute_seconds=seconds,
                        carbon_grams=metrics.carbon_grams,
                        cost_usd=metrics.cost_usd,
                        remaining_seconds_after=remaining,
                        reason=decision.reason,
                        details={
                            "decision": decision.model_dump(mode="json"),
                        },
                    )
                )
                continue

            if selected.action == ActionType.PAUSE:
                idle_seconds = float(selected.details.get("idle_seconds", 0.0))
                metrics, idle, overhead = _pause_metrics(
                    node=source_node,
                    policy=policy,
                    carbon_store=carbon_store,
                    start_utc=current_time,
                    idle_seconds=idle_seconds,
                    power_kw=workload.power_kw,
                )
                current_time += pd.Timedelta(seconds=metrics.elapsed_seconds)
                paused_idle_seconds += idle
                pause_overhead_seconds += overhead
                accumulated_carbon += metrics.carbon_grams
                last_pause_at = step_start.to_pydatetime()
                pauses += 1
                steps.append(
                    ReplayStep(
                        index=decision_index,
                        action="pause",
                        source_node_id=current_node_id,
                        started_at_utc=step_start.to_pydatetime(),
                        finished_at_utc=current_time.to_pydatetime(),
                        elapsed_seconds=metrics.elapsed_seconds,
                        idle_seconds=idle,
                        carbon_grams=metrics.carbon_grams,
                        cost_usd=0.0,
                        remaining_seconds_after=remaining,
                        reason=decision.reason,
                        details={
                            "pause_overhead_seconds": overhead,
                            "decision": decision.model_dump(mode="json"),
                        },
                    )
                )
                continue

            destination_id = selected.destination_node_id
            if destination_id is None:
                raise RuntimeError("Causal replay selected migration without destination")
            destination = cluster.get_node(destination_id)
            edge = graph.edge(current_node_id, destination_id)
            static_bytes = 0 if destination_id in prestaged else workload.static_data_bytes
            metrics = realized_migration(
                source=source_node,
                destination=destination,
                edge=edge,
                policy=policy,
                carbon_store=carbon_store,
                start_utc=current_time,
                power_kw=workload.power_kw,
                checkpoint_bytes=workload.checkpoint_bytes,
                static_data_bytes=static_bytes,
            )
            current_time += pd.Timedelta(seconds=metrics.elapsed_seconds)
            migration_seconds += metrics.elapsed_seconds
            accumulated_carbon += metrics.carbon_grams
            accumulated_cost += metrics.cost_usd
            migrations += 1
            last_migration_at = step_start.to_pydatetime()
            current_node_id = destination_id
            prestaged.add(destination_id)
            owner_path.append(destination_id)
            steps.append(
                ReplayStep(
                    index=decision_index,
                    action="migrate",
                    source_node_id=source_node.id,
                    destination_node_id=destination_id,
                    started_at_utc=step_start.to_pydatetime(),
                    finished_at_utc=current_time.to_pydatetime(),
                    elapsed_seconds=metrics.elapsed_seconds,
                    migration_seconds=metrics.elapsed_seconds,
                    carbon_grams=metrics.carbon_grams,
                    cost_usd=metrics.cost_usd,
                    remaining_seconds_after=remaining,
                    reason=decision.reason,
                    details={
                        **metrics.details,
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            )
        else:
            raise RuntimeError(
                f"Causal replay exceeded {max_decisions} decisions without completing"
            )

    makespan = (current_time - as_utc_timestamp(start_utc)).total_seconds()
    return PolicyOutcome(
        policy=label.value,
        start_node_id=workload.start_node_id,
        final_node_id=current_node_id,
        selected_initial_node_id=workload.start_node_id,
        completed=remaining <= 1e-9,
        makespan_seconds=makespan,
        compute_seconds=compute_seconds,
        paused_idle_seconds=paused_idle_seconds,
        pause_overhead_seconds=pause_overhead_seconds,
        migration_seconds=migration_seconds,
        carbon_grams=accumulated_carbon,
        cost_usd=accumulated_cost,
        migrations=migrations,
        pauses=pauses,
        decision_count=len(steps),
        owner_path=owner_path,
        steps=steps,
        metadata={
            "knowledge": "causal",
            "adaptive_policy_enabled": policy.adaptive.enabled,
            "telemetry_mode": "configured_edges_no_live_telemetry",
            "static_artifacts_prestaged": assume_static_prestaged,
            "decision_epoch_seconds": cluster.epoch_seconds,
        },
    )


def comparison_reference_scales(
    outcomes: list[PolicyOutcome],
) -> dict[str, float]:
    if not outcomes:
        raise ValueError("At least one outcome is required")
    return {
        "time_seconds": max(max(item.makespan_seconds for item in outcomes), 1e-12),
        "carbon_grams": max(max(item.carbon_grams for item in outcomes), 1e-12),
        "cost_usd": max(max(item.cost_usd for item in outcomes), 1e-12),
    }


def global_objective(
    outcome: PolicyOutcome,
    policy: ScoringPolicy,
    scales: dict[str, float],
) -> float:
    alpha, beta, gamma = policy.weights.normalized()
    return (
        alpha * outcome.makespan_seconds / scales["time_seconds"]
        + beta * outcome.carbon_grams / scales["carbon_grams"]
        + gamma * outcome.cost_usd / scales["cost_usd"]
    )


def outcome_row(outcome: PolicyOutcome) -> dict[str, Any]:
    return {
        "policy": outcome.policy,
        "start_node_id": outcome.start_node_id,
        "selected_initial_node_id": outcome.selected_initial_node_id,
        "final_node_id": outcome.final_node_id,
        "completed": outcome.completed,
        "makespan_seconds": outcome.makespan_seconds,
        "compute_seconds": outcome.compute_seconds,
        "paused_idle_seconds": outcome.paused_idle_seconds,
        "pause_overhead_seconds": outcome.pause_overhead_seconds,
        "migration_seconds": outcome.migration_seconds,
        "carbon_grams": outcome.carbon_grams,
        "cost_usd": outcome.cost_usd,
        "migrations": outcome.migrations,
        "pauses": outcome.pauses,
        "decision_count": outcome.decision_count,
        "owner_path": " -> ".join(outcome.owner_path),
    }


def ceil_steps(seconds: float, quantum_seconds: float) -> int:
    return max(1, int(math.ceil(seconds / quantum_seconds - 1e-12)))
