from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.comparison import (
    ComparisonWorkload,
    PolicyOutcome,
    ReplayStep,
    _compute_segment,
    ceil_steps,
    realized_migration,
)
from magellan.graph.topology import ClusterGraph


@dataclass(frozen=True)
class OracleState:
    elapsed_steps: int
    completed_work_steps: int
    node_id: str


@dataclass(frozen=True)
class OracleEdge:
    action: str
    source_node_id: str
    destination_node_id: str | None
    elapsed_steps: int
    elapsed_seconds: float
    compute_seconds: float
    idle_seconds: float
    migration_seconds: float
    carbon_grams: float
    cost_usd: float
    details: dict[str, Any]


def _incremental_objective(
    *,
    edge: OracleEdge,
    policy: ScoringPolicy,
    reference_time_seconds: float,
    reference_carbon_grams: float,
    reference_cost_usd: float,
) -> float:
    alpha, beta, gamma = policy.weights.normalized()
    return (
        alpha * edge.elapsed_seconds / max(reference_time_seconds, 1e-12)
        + beta * edge.carbon_grams / max(reference_carbon_grams, 1e-12)
        + gamma * edge.cost_usd / max(reference_cost_usd, 1e-12)
    )


def clairvoyant_oracle(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    workload: ComparisonWorkload,
    carbon_store: CarbonStore,
    graph: ClusterGraph,
    start_utc: pd.Timestamp,
    reference_time_seconds: float,
    reference_carbon_grams: float,
    reference_cost_usd: float,
    quantum_seconds: float = 900.0,
    max_elapsed_multiplier: float = 3.0,
    assume_static_prestaged: bool = True,
) -> PolicyOutcome:
    """Compute a discretized clairvoyant reference schedule.

    The oracle starts at the same node as Magellan and pays the same modeled
    migration overhead. It sees the entire future carbon trace. A WAIT edge is
    deliberately optimistic: the task is inactive for one quantum with zero
    task-attributed carbon/cost. This makes the result an intentionally optimistic reference,
    not a directly deployable scheduler.
    """
    if quantum_seconds <= 0:
        raise ValueError("quantum_seconds must be positive")
    if max_elapsed_multiplier < 1:
        raise ValueError("max_elapsed_multiplier must be >= 1")

    start = as_utc_timestamp(start_utc)
    cluster.get_node(workload.start_node_id)
    work_steps = int(math.ceil(workload.duration_seconds / quantum_seconds))
    max_elapsed_seconds = workload.duration_seconds * max_elapsed_multiplier
    max_elapsed_steps = int(math.ceil(max_elapsed_seconds / quantum_seconds))

    initial = OracleState(0, 0, workload.start_node_id)
    distances: dict[OracleState, float] = {initial: 0.0}
    metrics: dict[OracleState, tuple[float, float, float]] = {
        initial: (0.0, 0.0, 0.0)
    }
    previous: dict[OracleState, tuple[OracleState, OracleEdge]] = {}
    queue: list[tuple[float, int, OracleState]] = [(0.0, 0, initial)]
    serial = 1
    terminal: OracleState | None = None
    compute_cache: dict[tuple[int, str, float], Any] = {}
    migration_cache: dict[tuple[int, str, str], Any] = {}

    while queue:
        score, _, state = heapq.heappop(queue)
        if score > distances.get(state, float("inf")) + 1e-12:
            continue
        if state.completed_work_steps >= work_steps:
            terminal = state
            break
        if state.elapsed_steps >= max_elapsed_steps:
            continue

        now = start + pd.Timedelta(seconds=state.elapsed_steps * quantum_seconds)
        source = cluster.get_node(state.node_id)
        remaining_work = max(
            0.0,
            workload.duration_seconds - state.completed_work_steps * quantum_seconds,
        )
        compute_seconds = min(quantum_seconds, remaining_work)
        compute_key = (state.elapsed_steps, state.node_id, compute_seconds)
        compute_metrics = compute_cache.get(compute_key)
        if compute_metrics is None:
            compute_metrics = _compute_segment(
                node=source,
                carbon_store=carbon_store,
                start_utc=now,
                seconds=compute_seconds,
                power_kw=workload.power_kw,
            )
            compute_cache[compute_key] = compute_metrics
        continue_edge = OracleEdge(
            action="continue",
            source_node_id=state.node_id,
            destination_node_id=None,
            elapsed_steps=1,
            elapsed_seconds=compute_seconds,
            compute_seconds=compute_seconds,
            idle_seconds=0.0,
            migration_seconds=0.0,
            carbon_grams=compute_metrics.carbon_grams,
            cost_usd=compute_metrics.cost_usd,
            details={},
        )
        candidates: list[tuple[OracleState, OracleEdge]] = [
            (
                OracleState(
                    elapsed_steps=state.elapsed_steps + 1,
                    completed_work_steps=state.completed_work_steps + 1,
                    node_id=state.node_id,
                ),
                continue_edge,
            )
        ]

        wait_edge = OracleEdge(
            action="wait",
            source_node_id=state.node_id,
            destination_node_id=None,
            elapsed_steps=1,
            elapsed_seconds=quantum_seconds,
            compute_seconds=0.0,
            idle_seconds=quantum_seconds,
            migration_seconds=0.0,
            carbon_grams=0.0,
            cost_usd=0.0,
            details={"optimistic_inactive_wait": True},
        )
        candidates.append(
            (
                OracleState(
                    elapsed_steps=state.elapsed_steps + 1,
                    completed_work_steps=state.completed_work_steps,
                    node_id=state.node_id,
                ),
                wait_edge,
            )
        )

        for destination in graph.peers(state.node_id):
            edge_metrics = graph.edge(state.node_id, destination.id)
            migration_key = (state.elapsed_steps, state.node_id, destination.id)
            migration = migration_cache.get(migration_key)
            if migration is None:
                migration = realized_migration(
                    source=source,
                    destination=destination,
                    edge=edge_metrics,
                    policy=policy,
                    carbon_store=carbon_store,
                    start_utc=now,
                    power_kw=workload.power_kw,
                    checkpoint_bytes=workload.checkpoint_bytes,
                    static_data_bytes=(
                        0 if assume_static_prestaged else workload.static_data_bytes
                    ),
                )
                migration_cache[migration_key] = migration
            migration_steps = ceil_steps(migration.elapsed_seconds, quantum_seconds)
            candidates.append(
                (
                    OracleState(
                        elapsed_steps=state.elapsed_steps + migration_steps,
                        completed_work_steps=state.completed_work_steps,
                        node_id=destination.id,
                    ),
                    OracleEdge(
                        action="migrate",
                        source_node_id=state.node_id,
                        destination_node_id=destination.id,
                        elapsed_steps=migration_steps,
                        elapsed_seconds=migration_steps * quantum_seconds,
                        compute_seconds=0.0,
                        idle_seconds=0.0,
                        migration_seconds=migration.elapsed_seconds,
                        carbon_grams=migration.carbon_grams,
                        cost_usd=migration.cost_usd,
                        details={
                            **migration.details,
                            "actual_migration_seconds": migration.elapsed_seconds,
                            "quantized_elapsed_seconds": migration_steps * quantum_seconds,
                        },
                    ),
                )
            )

        for next_state, edge in candidates:
            if next_state.elapsed_steps > max_elapsed_steps:
                continue
            incremental = _incremental_objective(
                edge=edge,
                policy=policy,
                reference_time_seconds=reference_time_seconds,
                reference_carbon_grams=reference_carbon_grams,
                reference_cost_usd=reference_cost_usd,
            )
            candidate_score = score + incremental
            if candidate_score + 1e-12 >= distances.get(next_state, float("inf")):
                continue
            distances[next_state] = candidate_score
            prior_time, prior_carbon, prior_cost = metrics[state]
            metrics[next_state] = (
                prior_time + edge.elapsed_seconds,
                prior_carbon + edge.carbon_grams,
                prior_cost + edge.cost_usd,
            )
            previous[next_state] = (state, edge)
            heapq.heappush(queue, (candidate_score, serial, next_state))
            serial += 1

    if terminal is None:
        raise RuntimeError(
            "Oracle could not complete workload inside max elapsed horizon; "
            "increase max_elapsed_multiplier"
        )

    reversed_edges: list[OracleEdge] = []
    cursor = terminal
    while cursor != initial:
        parent, edge = previous[cursor]
        reversed_edges.append(edge)
        cursor = parent
    path_edges = list(reversed(reversed_edges))

    steps: list[ReplayStep] = []
    owner_path = [workload.start_node_id]
    step_start = start
    compute_seconds_total = 0.0
    idle_seconds_total = 0.0
    migration_seconds_total = 0.0
    migrations = 0
    pauses = 0
    completed_work = 0.0
    current_node_id = workload.start_node_id
    carbon_total = 0.0
    cost_total = 0.0

    for index, edge in enumerate(path_edges, start=1):
        step_finish = step_start + pd.Timedelta(seconds=edge.elapsed_seconds)
        completed_work += edge.compute_seconds
        compute_seconds_total += edge.compute_seconds
        idle_seconds_total += edge.idle_seconds
        migration_seconds_total += edge.migration_seconds
        carbon_total += edge.carbon_grams
        cost_total += edge.cost_usd
        if edge.action == "migrate":
            migrations += 1
            assert edge.destination_node_id is not None
            current_node_id = edge.destination_node_id
            owner_path.append(current_node_id)
        elif edge.action == "wait":
            pauses += 1
        steps.append(
            ReplayStep(
                index=index,
                action=edge.action,
                source_node_id=edge.source_node_id,
                destination_node_id=edge.destination_node_id,
                started_at_utc=step_start.to_pydatetime(),
                finished_at_utc=step_finish.to_pydatetime(),
                elapsed_seconds=edge.elapsed_seconds,
                compute_seconds=edge.compute_seconds,
                idle_seconds=edge.idle_seconds,
                migration_seconds=edge.migration_seconds,
                carbon_grams=edge.carbon_grams,
                cost_usd=edge.cost_usd,
                remaining_seconds_after=max(
                    0.0, workload.duration_seconds - completed_work
                ),
                reason="Clairvoyant shortest-path oracle",
                details=edge.details,
            )
        )
        step_start = step_finish

    makespan, _, _ = metrics[terminal]
    return PolicyOutcome(
        policy="clairvoyant_oracle",
        start_node_id=workload.start_node_id,
        final_node_id=current_node_id,
        selected_initial_node_id=workload.start_node_id,
        completed=True,
        makespan_seconds=makespan,
        compute_seconds=compute_seconds_total,
        paused_idle_seconds=idle_seconds_total,
        pause_overhead_seconds=0.0,
        migration_seconds=migration_seconds_total,
        carbon_grams=carbon_total,
        cost_usd=cost_total,
        migrations=migrations,
        pauses=pauses,
        decision_count=len(steps),
        owner_path=owner_path,
        steps=steps,
        metadata={
            "knowledge": "clairvoyant_full_trace",
            "initial_placement_overhead": "not_free_starts_at_submission_node",
            "oracle_type": "discretized_full_trace_reference",
            "quantum_seconds": quantum_seconds,
            "max_elapsed_multiplier": max_elapsed_multiplier,
            "optimistic_wait_zero_task_carbon_cost": True,
            "migration_overhead_mode": "magellan_physical_model",
            "static_artifacts_prestaged": assume_static_prestaged,
            "global_objective": {
                "reference_time_seconds": reference_time_seconds,
                "reference_carbon_grams": reference_carbon_grams,
                "reference_cost_usd": reference_cost_usd,
                "weights": {
                    "time": policy.weights.normalized()[0],
                    "carbon": policy.weights.normalized()[1],
                    "cost": policy.weights.normalized()[2],
                },
            },
            "objective_value": distances[terminal],
        },
    )
