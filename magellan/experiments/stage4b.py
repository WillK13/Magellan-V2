from __future__ import annotations

import csv
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.comparison import PolicyOutcome, ReplayStep
from magellan.experiments.measurement import summarize_samples
from magellan.graph.topology import EdgeMetrics, haversine_distance_km
from magellan.models.types import ActionType, RawActionEstimate, TaskProfile
from magellan.models.utils import bytes_to_gb, seconds_to_hours
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore
from magellan.scheduler.scoring import evaluate_task, score_actions


CORE_POLICIES = (
    "boston_static",
    "best_static",
    "gaia_carbon_time",
    "magellan_causal",
    "clairvoyant_spatiotemporal_static_oracle",
)
CORE_WORKLOADS = (
    "benchmark-json-medium",
    "dendro-r9-t1p0",
    "llm-distilgpt2",
)
DEFAULT_RUNTIME_SCALE = 60.0
DEFAULT_GAIA_QUANTUM_SECONDS = 3600.0
DEFAULT_GAIA_SHORT_LIMIT_SECONDS = 2 * 3600.0
DEFAULT_GAIA_SHORT_WAIT_SECONDS = 6 * 3600.0
DEFAULT_GAIA_LONG_WAIT_SECONDS = 24 * 3600.0
DEFAULT_ORACLE_WAIT_SECONDS = 24 * 3600.0
GAIA_POLICY_FLAGS = {
    "scheduling_policy": "carbon",
    "carbon_policy": "cst_average",
}


@dataclass(frozen=True)
class WorkloadCalibration:
    class_id: str
    workload: str
    variant: str
    canonical_runtime_seconds: float
    power_kw: float
    checkpoint_bytes: int
    checkpoint_seconds: float
    restore_seconds: float
    migration_overhead_seconds: float

    def scaled_work_seconds(self, runtime_scale: float) -> float:
        if runtime_scale <= 0:
            raise ValueError("runtime_scale must be positive")
        return self.canonical_runtime_seconds * runtime_scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "workload": self.workload,
            "variant": self.variant,
            "canonical_runtime_seconds": self.canonical_runtime_seconds,
            "power_kw": self.power_kw,
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_seconds": self.checkpoint_seconds,
            "restore_seconds": self.restore_seconds,
            "migration_overhead_seconds": self.migration_overhead_seconds,
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    class_id: str
    arrival_utc: pd.Timestamp


class FrozenCalibrationGraph:
    """Experiment-only graph backed by frozen Stage-4A calibration evidence."""

    def __init__(
        self,
        *,
        cluster: ClusterConfig,
        edge_rows: Iterable[dict[str, str]],
        workload: WorkloadCalibration,
    ) -> None:
        self._cluster = cluster
        self._workload = workload
        self._edges: dict[tuple[str, str], EdgeMetrics] = {}
        for row in edge_rows:
            source_id = str(row["source_node_id"])
            destination_id = str(row["destination_node_id"])
            source = cluster.get_node(source_id)
            destination = cluster.get_node(destination_id)
            steady = float(row["transfer_steady_bandwidth_mbps"])
            fixed = float(row["transfer_fixed_seconds"])
            if steady <= 0 or fixed < 0:
                raise ValueError(f"Invalid Stage 4A.1 transfer model for {source_id}->{destination_id}")
            measured_bandwidth = float(row.get("measured_bandwidth_median_mbps") or steady)
            measured_rtt = float(row.get("measured_rtt_median_ms") or 0.0)
            self._edges[(source_id, destination_id)] = EdgeMetrics(
                source_node_id=source_id,
                destination_node_id=destination_id,
                distance_km=haversine_distance_km(
                    source.latitude,
                    source.longitude,
                    destination.latitude,
                    destination.longitude,
                ),
                bandwidth_mbps=max(measured_bandwidth, 1e-9),
                latency_ms=max(measured_rtt, 0.0),
                bandwidth_source="measured_migration_transport_ema",
                latency_source="stage4a1_measured_rtt",
                checkpoint_seconds=workload.checkpoint_seconds,
                restore_seconds=workload.restore_seconds,
                migration_overhead_seconds=workload.migration_overhead_seconds,
                transfer_fixed_seconds=fixed,
                transfer_steady_bandwidth_mbps=steady,
                transfer_model_source="measured_migration_transport_affine_ema",
                calibration_source="stage4a2_frozen_workload_calibration",
            )

    def peers(self, node_id: str) -> list[NodeConfig]:
        return [node for node in self._cluster.nodes if node.id != node_id]

    def edge(self, source_node_id: str, destination_node_id: str) -> EdgeMetrics:
        try:
            return self._edges[(source_node_id, destination_node_id)]
        except KeyError as exc:
            raise ValueError(f"Missing frozen edge {source_node_id}->{destination_node_id}") from exc


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_stage4a1_edges(
    stage4a1_bundle: str | Path,
    stage4a1_summary: dict[str, Any],
) -> list[dict[str, str]]:
    network_relative = stage4a1_summary.get("network_bundle")
    if not network_relative:
        raise ValueError("Stage 4A.1 summary is missing network_bundle")
    edges_path = Path(stage4a1_bundle) / str(network_relative) / "edges.csv"
    if not edges_path.is_file():
        raise FileNotFoundError(edges_path)
    return read_csv(edges_path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _float_values(rows: Iterable[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in {None, ""}:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _matching_migration_rows(rows: list[dict[str, str]], class_id: str) -> list[dict[str, str]]:
    if class_id == "benchmark-json-medium":
        return [row for row in rows if row.get("workload") == "json" and row.get("variant") == "medium"]
    if class_id == "dendro-r9-t1p0":
        return [row for row in rows if row.get("workload") == "dendro" and row.get("variant") in {"r9-t1", "r9-t1p0"}]
    if class_id == "llm-distilgpt2":
        return [row for row in rows if row.get("workload") == "llm"]
    raise ValueError(f"No Stage 4B migration calibration selector for {class_id}")


def load_workload_calibrations(
    *,
    stage4a2_bundle: str | Path,
    stage4a3_bundle: str | Path,
    stage4a4_bundle: str | Path,
    class_ids: Iterable[str] = CORE_WORKLOADS,
) -> dict[str, WorkloadCalibration]:
    a2_rows = read_csv(Path(stage4a2_bundle) / "migration_samples.csv")
    a3_rows = {row["class_id"]: row for row in read_csv(Path(stage4a3_bundle) / "profile_classes.csv")}
    a4_rows = {row["class_id"]: row for row in read_csv(Path(stage4a4_bundle) / "static_classes.csv")}
    output: dict[str, WorkloadCalibration] = {}
    for class_id in class_ids:
        if class_id not in a3_rows or class_id not in a4_rows:
            raise ValueError(f"Missing Stage 4A.3/4A.4 class {class_id}")
        migration_rows = _matching_migration_rows(a2_rows, class_id)
        if not migration_rows:
            raise ValueError(f"Missing Stage 4A.2 migration rows for {class_id}")
        checkpoint_bytes = _float_values(migration_rows, "actual_checkpoint_bytes")
        checkpoint_seconds = _float_values(migration_rows, "actual_checkpoint_seconds")
        restore_seconds = _float_values(migration_rows, "actual_restore_seconds")
        overhead_seconds = _float_values(migration_rows, "actual_migration_overhead_seconds")
        if not checkpoint_bytes or not checkpoint_seconds or not restore_seconds:
            raise ValueError(f"Incomplete Stage 4A.2 migration calibration for {class_id}")
        power = float(a3_rows[class_id].get("power_median_kw") or 0.0)
        runtime = float(a4_rows[class_id].get("runtime_seconds_median") or 0.0)
        if power <= 0 or runtime <= 0:
            raise ValueError(f"Non-positive runtime/power calibration for {class_id}")
        output[class_id] = WorkloadCalibration(
            class_id=class_id,
            workload=str(a3_rows[class_id].get("workload") or ""),
            variant=str(a3_rows[class_id].get("variant") or ""),
            canonical_runtime_seconds=runtime,
            power_kw=power,
            checkpoint_bytes=int(round(median(checkpoint_bytes))),
            checkpoint_seconds=median(checkpoint_seconds),
            restore_seconds=median(restore_seconds),
            migration_overhead_seconds=median(overhead_seconds) if overhead_seconds else 0.0,
        )
    return output


def load_node_slowdowns(stage4a4_bundle: str | Path) -> dict[str, float]:
    values = {
        row["node_id"]: float(row["slowdown_vs_canonical"])
        for row in read_csv(Path(stage4a4_bundle) / "node_equivalence.csv")
    }
    if not values or any(value <= 0 for value in values.values()):
        raise ValueError("Stage 4A.4 node slowdown factors must be positive")
    if abs(values.get("boston", 0.0) - 1.0) > 1e-6:
        raise ValueError("Stage 4A.4 Boston slowdown factor must be 1.0")
    return values


def annual_scenarios(
    *,
    class_ids: Iterable[str] = CORE_WORKLOADS,
    year: int = 2024,
    months: Iterable[int] = range(1, 13),
) -> list[Scenario]:
    output: list[Scenario] = []
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}")
        for day, hour in ((5, 0), (20, 12)):
            arrival = pd.Timestamp(year=year, month=month, day=day, hour=hour, tz="UTC")
            for class_id in class_ids:
                output.append(
                    Scenario(
                        scenario_id=f"{arrival.strftime('%Y%m%dT%H%MZ')}-{class_id}",
                        class_id=class_id,
                        arrival_utc=arrival,
                    )
                )
    return output


def gaia_queue_parameters(
    calibrations: dict[str, WorkloadCalibration],
    *,
    runtime_scale: float,
    short_limit_seconds: float = DEFAULT_GAIA_SHORT_LIMIT_SECONDS,
    short_wait_seconds: float = DEFAULT_GAIA_SHORT_WAIT_SECONDS,
    long_wait_seconds: float = DEFAULT_GAIA_LONG_WAIT_SECONDS,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in calibrations.values():
        runtime = item.scaled_work_seconds(runtime_scale)
        grouped["short" if runtime <= short_limit_seconds else "long"].append(runtime)
    output: dict[str, dict[str, float]] = {}
    for queue, values in grouped.items():
        output[queue] = {
            "mean_runtime_seconds": sum(values) / len(values),
            "max_wait_seconds": short_wait_seconds if queue == "short" else long_wait_seconds,
            "class_count": float(len(values)),
        }
    return output


def _compute_segment(
    *,
    node: NodeConfig,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    seconds: float,
    power_kw: float,
) -> tuple[float, float]:
    if seconds <= 0:
        return 0.0, 0.0
    intensity = carbon_store.average(node.id, start_utc, seconds)
    carbon = power_kw * node.pue * seconds_to_hours(seconds) * intensity
    cost = node.compute_price_usd_per_hour * seconds_to_hours(seconds)
    return carbon, cost


def _static_outcome(
    *,
    label: str,
    node: NodeConfig,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
    start_wait_seconds: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> PolicyOutcome:
    runtime = calibration.scaled_work_seconds(runtime_scale) * node_slowdowns[node.id]
    execution_start = arrival_utc + pd.Timedelta(seconds=start_wait_seconds)
    carbon, cost = _compute_segment(
        node=node,
        carbon_store=carbon_store,
        start_utc=execution_start,
        seconds=runtime,
        power_kw=calibration.power_kw,
    )
    steps: list[ReplayStep] = []
    if start_wait_seconds > 0:
        steps.append(
            ReplayStep(
                index=1,
                action="wait_to_start",
                source_node_id=node.id,
                started_at_utc=arrival_utc.to_pydatetime(warn=False),
                finished_at_utc=execution_start.to_pydatetime(warn=False),
                elapsed_seconds=start_wait_seconds,
                idle_seconds=start_wait_seconds,
                remaining_seconds_after=runtime,
                reason="Submission-time deferral before uninterruptible execution",
            )
        )
    steps.append(
        ReplayStep(
            index=len(steps) + 1,
            action="continue",
            source_node_id=node.id,
            started_at_utc=execution_start.to_pydatetime(warn=False),
            finished_at_utc=(execution_start + pd.Timedelta(seconds=runtime)).to_pydatetime(warn=False),
            elapsed_seconds=runtime,
            compute_seconds=runtime,
            carbon_grams=carbon,
            cost_usd=cost,
            remaining_seconds_after=0.0,
            reason="Static uninterruptible execution",
        )
    )
    combined_metadata = dict(metadata or {})
    combined_metadata["submission_wait_seconds"] = start_wait_seconds
    return PolicyOutcome(
        policy=label,
        start_node_id=node.id,
        final_node_id=node.id,
        selected_initial_node_id=node.id,
        makespan_seconds=start_wait_seconds + runtime,
        compute_seconds=runtime,
        paused_idle_seconds=0.0,
        pause_overhead_seconds=0.0,
        migration_seconds=0.0,
        carbon_grams=carbon,
        cost_usd=cost,
        migrations=0,
        pauses=0,
        decision_count=1 if label != "boston_static" else 0,
        owner_path=[node.id],
        steps=steps,
        metadata=combined_metadata,
    )


def _rank_static_candidates(candidates: list[PolicyOutcome], policy: ScoringPolicy) -> tuple[PolicyOutcome, list[dict[str, Any]]]:
    estimates = [
        RawActionEstimate(
            action=ActionType.CONTINUE,
            source_node_id=item.final_node_id,
            destination_node_id=None,
            time_seconds=item.makespan_seconds,
            carbon_grams=item.carbon_grams,
            cost_usd=item.cost_usd,
            details={"submission_wait_seconds": item.metadata.get("submission_wait_seconds", 0.0)},
        )
        for item in candidates
    ]
    ranked = score_actions(estimates, policy)
    selected = ranked[0]
    selected_wait = float(selected.details.get("submission_wait_seconds", 0.0))
    outcome = next(
        item
        for item in candidates
        if item.final_node_id == selected.source_node_id
        and abs(float(item.metadata.get("submission_wait_seconds", 0.0)) - selected_wait) <= 1e-9
    )
    return outcome, [item.model_dump(mode="json") for item in ranked]


def best_static_outcome(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
) -> PolicyOutcome:
    candidates = [
        _static_outcome(
            label="static_candidate",
            node=node,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=arrival_utc,
            runtime_scale=runtime_scale,
        )
        for node in cluster.nodes
    ]
    selected, ranking = _rank_static_candidates(candidates, policy)
    result = selected.model_copy(deep=True)
    result.policy = "best_static"
    result.metadata = {
        **result.metadata,
        "knowledge": "clairvoyant_actual_execution_interval_at_arrival",
        "initial_placement_overhead": "free",
        "ranking": ranking,
    }
    return result


def gaia_carbon_time_score(*, immediate_carbon_grams: float, candidate_carbon_grams: float, wait_seconds: float, queue_mean_runtime_seconds: float) -> float:
    denominator = wait_seconds + queue_mean_runtime_seconds
    if denominator <= 0:
        raise ValueError("GAIA Carbon-Time denominator must be positive")
    return (immediate_carbon_grams - candidate_carbon_grams) / denominator


def gaia_carbon_time_outcome(
    *,
    boston: NodeConfig,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
    queue_parameters: dict[str, dict[str, float]],
    quantum_seconds: float = DEFAULT_GAIA_QUANTUM_SECONDS,
    short_limit_seconds: float = DEFAULT_GAIA_SHORT_LIMIT_SECONDS,
) -> PolicyOutcome:
    if quantum_seconds <= 0:
        raise ValueError("GAIA candidate quantum must be positive")
    actual_boston_runtime = calibration.scaled_work_seconds(runtime_scale) * node_slowdowns[boston.id]
    queue = "short" if actual_boston_runtime <= short_limit_seconds else "long"
    if queue not in queue_parameters:
        raise ValueError(f"Missing GAIA queue parameters for {queue}")
    mean_runtime = float(queue_parameters[queue]["mean_runtime_seconds"])
    max_wait = float(queue_parameters[queue]["max_wait_seconds"])
    immediate_carbon, _ = _compute_segment(
        node=boston,
        carbon_store=carbon_store,
        start_utc=arrival_utc,
        seconds=mean_runtime,
        power_kw=calibration.power_kw,
    )
    candidates: list[dict[str, float]] = []
    wait = 0.0
    while wait < max_wait - 1e-9 or abs(wait) <= 1e-9:
        candidate_start = arrival_utc + pd.Timedelta(seconds=wait)
        candidate_carbon, _ = _compute_segment(
            node=boston,
            carbon_store=carbon_store,
            start_utc=candidate_start,
            seconds=mean_runtime,
            power_kw=calibration.power_kw,
        )
        candidates.append(
            {
                "wait_seconds": wait,
                "estimated_carbon_grams": candidate_carbon,
                "cst": gaia_carbon_time_score(
                    immediate_carbon_grams=immediate_carbon,
                    candidate_carbon_grams=candidate_carbon,
                    wait_seconds=wait,
                    queue_mean_runtime_seconds=mean_runtime,
                ),
            }
        )
        wait += quantum_seconds
        if max_wait <= 0:
            break
    selected = max(candidates, key=lambda row: (row["cst"], -row["wait_seconds"]))
    return _static_outcome(
        label="gaia_carbon_time",
        node=boston,
        calibration=calibration,
        node_slowdowns=node_slowdowns,
        carbon_store=carbon_store,
        arrival_utc=arrival_utc,
        runtime_scale=runtime_scale,
        start_wait_seconds=selected["wait_seconds"],
        metadata={
            "knowledge": "perfect_future_carbon_within_gaia_wait_window",
            "gaia_queue": queue,
            "gaia_queue_mean_runtime_seconds": mean_runtime,
            "gaia_max_wait_seconds": max_wait,
            "gaia_quantum_seconds": quantum_seconds,
            "gaia_selected_cst": selected["cst"],
            "gaia_candidate_count": len(candidates),
            "gaia_policy_flags": GAIA_POLICY_FLAGS,
            "gaia_policy_reproduction": "Carbon-Time / cst_average",
        },
    )


def oracle_static_outcome(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
    max_wait_seconds: float = DEFAULT_ORACLE_WAIT_SECONDS,
    quantum_seconds: float = DEFAULT_GAIA_QUANTUM_SECONDS,
) -> PolicyOutcome:
    if max_wait_seconds < 0 or quantum_seconds <= 0:
        raise ValueError("Invalid oracle wait/quantum")
    candidates: list[PolicyOutcome] = []
    waits: list[float] = []
    value = 0.0
    while value <= max_wait_seconds + 1e-9:
        waits.append(min(value, max_wait_seconds))
        value += quantum_seconds
    for node in cluster.nodes:
        for wait in waits:
            candidates.append(
                _static_outcome(
                    label="oracle_candidate",
                    node=node,
                    calibration=calibration,
                    node_slowdowns=node_slowdowns,
                    carbon_store=carbon_store,
                    arrival_utc=arrival_utc,
                    runtime_scale=runtime_scale,
                    start_wait_seconds=wait,
                )
            )
    selected, ranking = _rank_static_candidates(candidates, policy)
    result = selected.model_copy(deep=True)
    result.policy = "clairvoyant_spatiotemporal_static_oracle"
    result.metadata = {
        **result.metadata,
        "knowledge": "clairvoyant_full_carbon_trace",
        "scope": "static placement plus submission-time deferral only; no migration",
        "max_wait_seconds": max_wait_seconds,
        "candidate_quantum_seconds": quantum_seconds,
        "candidate_count": len(candidates),
        "ranking": ranking,
    }
    return result


def _realized_migration(
    *,
    source: NodeConfig,
    destination: NodeConfig,
    edge: EdgeMetrics,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    start_utc: pd.Timestamp,
    power_kw: float,
    checkpoint_bytes: int,
) -> tuple[float, float, float, dict[str, Any]]:
    steady = edge.transfer_steady_bandwidth_mbps
    if steady is None or steady <= 0 or edge.transfer_model_source != "measured_migration_transport_affine_ema":
        raise ValueError("Stage 4B realized migration requires Stage 4A.1 affine transfer calibration")
    transfer_seconds = edge.transfer_fixed_seconds + checkpoint_bytes * 8.0 / (steady * 1_000_000.0)
    checkpoint_seconds = float(edge.checkpoint_seconds or 0.0)
    restore_seconds = float(edge.restore_seconds or 0.0)
    overhead_seconds = max(0.0, float(edge.migration_overhead_seconds))
    restore_start = start_utc + pd.Timedelta(seconds=checkpoint_seconds + transfer_seconds + overhead_seconds)
    source_intensity = carbon_store.average(source.id, start_utc, checkpoint_seconds) if checkpoint_seconds > 0 else carbon_store.value_at(source.id, start_utc)
    destination_intensity = carbon_store.average(destination.id, restore_start, restore_seconds) if restore_seconds > 0 else carbon_store.value_at(destination.id, restore_start)
    source_carbon = power_kw * source.pue * seconds_to_hours(checkpoint_seconds) * source_intensity
    destination_carbon = power_kw * destination.pue * seconds_to_hours(restore_seconds) * destination_intensity
    transfer_gb = bytes_to_gb(checkpoint_bytes)
    network_energy_kwh = transfer_gb * (
        policy.migration.network_energy_kwh_per_gb_base
        + policy.migration.network_energy_kwh_per_gb_km * edge.distance_km
    )
    network_carbon = network_energy_kwh * ((source_intensity + destination_intensity) / 2.0)
    transfer_cost = transfer_gb * source.egress_price_usd_per_gb
    elapsed = checkpoint_seconds + transfer_seconds + overhead_seconds + restore_seconds
    return elapsed, source_carbon + destination_carbon + network_carbon, transfer_cost, {
        "checkpoint_seconds": checkpoint_seconds,
        "transfer_seconds": transfer_seconds,
        "restore_seconds": restore_seconds,
        "migration_overhead_seconds": overhead_seconds,
        "checkpoint_bytes": checkpoint_bytes,
        "transfer_fixed_seconds": edge.transfer_fixed_seconds,
        "transfer_steady_bandwidth_mbps": steady,
        "transfer_model": "stage4a1_affine_migration_transport",
        "network_carbon_grams": network_carbon,
        "transfer_cost_usd": transfer_cost,
    }


def replay_magellan_causal(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    graph: FrozenCalibrationGraph,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
    start_node_id: str = "boston",
    max_decisions: int = 10_000,
) -> PolicyOutcome:
    current_time = as_utc_timestamp(arrival_utc)
    current_node_id = start_node_id
    remaining_work = calibration.scaled_work_seconds(runtime_scale)  # Boston-equivalent seconds.
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
    telemetry_confidence = float(policy.telemetry.cpu_power_confidence)

    with tempfile.TemporaryDirectory(prefix="magellan-stage4b-") as directory:
        adaptive_service = AdaptivePolicyService(policy.adaptive, policy.weights, AdaptivePolicyStore(Path(directory)))
        for decision_index in range(1, max_decisions + 1):
            if remaining_work <= 1e-9:
                break
            slowdown = node_slowdowns[current_node_id]
            task = TaskProfile(
                task_id=f"stage4b-{calibration.class_id}",
                workload_type=calibration.workload or "stage4b",
                current_node_id=current_node_id,
                power_kw=calibration.power_kw,
                checkpoint_bytes=calibration.checkpoint_bytes,
                data_bytes=0,
                prestaged_node_ids=all_node_ids,
                estimated_remaining_seconds=remaining_work * slowdown,
                accumulated_cost_usd=accumulated_cost,
                cost_cap_usd=None,
                last_migration_at=last_migration_at,
                last_pause_at=last_pause_at,
            )
            compatible = all_node_ids - {current_node_id}
            decision = evaluate_task(
                task=task,
                cluster=cluster,
                policy=policy,
                graph=graph,  # type: ignore[arg-type]
                carbon_store=carbon_store,
                at_utc=current_time,
                static_data_bytes_by_destination={node_id: 0 for node_id in compatible},
                adaptive_service=adaptive_service,
                telemetry_confidence=telemetry_confidence,
                compatible_destination_ids=compatible,
            )
            selected = decision.selected
            step_start = current_time
            source = cluster.get_node(current_node_id)

            if selected.action == ActionType.CONTINUE:
                wall_seconds = min(float(cluster.epoch_seconds), remaining_work * slowdown)
                work_done = wall_seconds / slowdown
                carbon, cost = _compute_segment(
                    node=source,
                    carbon_store=carbon_store,
                    start_utc=current_time,
                    seconds=wall_seconds,
                    power_kw=calibration.power_kw,
                )
                current_time += pd.Timedelta(seconds=wall_seconds)
                remaining_work = max(0.0, remaining_work - work_done)
                compute_seconds += wall_seconds
                accumulated_carbon += carbon
                accumulated_cost += cost
                steps.append(
                    ReplayStep(
                        index=decision_index,
                        action="continue",
                        source_node_id=current_node_id,
                        started_at_utc=step_start.to_pydatetime(warn=False),
                        finished_at_utc=current_time.to_pydatetime(warn=False),
                        elapsed_seconds=wall_seconds,
                        compute_seconds=wall_seconds,
                        carbon_grams=carbon,
                        cost_usd=cost,
                        remaining_seconds_after=remaining_work,
                        reason=decision.reason,
                        details={
                            "remaining_work_boston_equivalent_seconds": remaining_work,
                            "current_node_slowdown_factor": slowdown,
                            "decision": decision.model_dump(mode="json"),
                        },
                    )
                )
                continue

            if selected.action == ActionType.PAUSE:
                idle_seconds = float(selected.details.get("idle_seconds", 0.0))
                pause_seconds = float(policy.pause.pause_seconds)
                resume_seconds = float(policy.pause.resume_seconds)
                pause_intensity = carbon_store.average(source.id, current_time, pause_seconds) if pause_seconds > 0 else 0.0
                resume_start = current_time + pd.Timedelta(seconds=pause_seconds + idle_seconds)
                resume_intensity = carbon_store.average(source.id, resume_start, resume_seconds) if resume_seconds > 0 else 0.0
                carbon = calibration.power_kw * source.pue * (
                    seconds_to_hours(pause_seconds) * pause_intensity
                    + seconds_to_hours(resume_seconds) * resume_intensity
                )
                elapsed = pause_seconds + idle_seconds + resume_seconds
                current_time += pd.Timedelta(seconds=elapsed)
                paused_idle_seconds += idle_seconds
                pause_overhead_seconds += pause_seconds + resume_seconds
                accumulated_carbon += carbon
                last_pause_at = step_start.to_pydatetime(warn=False)
                pauses += 1
                steps.append(
                    ReplayStep(
                        index=decision_index,
                        action="pause",
                        source_node_id=current_node_id,
                        started_at_utc=step_start.to_pydatetime(warn=False),
                        finished_at_utc=current_time.to_pydatetime(warn=False),
                        elapsed_seconds=elapsed,
                        idle_seconds=idle_seconds,
                        carbon_grams=carbon,
                        cost_usd=0.0,
                        remaining_seconds_after=remaining_work,
                        reason=decision.reason,
                        details={"decision": decision.model_dump(mode="json")},
                    )
                )
                continue

            destination_id = selected.destination_node_id
            if destination_id is None:
                raise RuntimeError("Magellan selected migrate without a destination")
            destination = cluster.get_node(destination_id)
            edge = graph.edge(current_node_id, destination_id)
            elapsed, carbon, cost, migration_details = _realized_migration(
                source=source,
                destination=destination,
                edge=edge,
                policy=policy,
                carbon_store=carbon_store,
                start_utc=current_time,
                power_kw=calibration.power_kw,
                checkpoint_bytes=calibration.checkpoint_bytes,
            )
            current_time += pd.Timedelta(seconds=elapsed)
            migration_seconds += elapsed
            accumulated_carbon += carbon
            accumulated_cost += cost
            migrations += 1
            last_migration_at = step_start.to_pydatetime(warn=False)
            current_node_id = destination_id
            owner_path.append(destination_id)
            steps.append(
                ReplayStep(
                    index=decision_index,
                    action="migrate",
                    source_node_id=source.id,
                    destination_node_id=destination_id,
                    started_at_utc=step_start.to_pydatetime(warn=False),
                    finished_at_utc=current_time.to_pydatetime(warn=False),
                    elapsed_seconds=elapsed,
                    migration_seconds=elapsed,
                    carbon_grams=carbon,
                    cost_usd=cost,
                    remaining_seconds_after=remaining_work,
                    reason=decision.reason,
                    details={**migration_details, "decision": decision.model_dump(mode="json")},
                )
            )
        else:
            raise RuntimeError(f"Magellan replay exceeded {max_decisions} decisions")

    return PolicyOutcome(
        policy="magellan_causal",
        start_node_id=start_node_id,
        final_node_id=current_node_id,
        selected_initial_node_id=start_node_id,
        completed=remaining_work <= 1e-9,
        makespan_seconds=(current_time - as_utc_timestamp(arrival_utc)).total_seconds(),
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
            "knowledge": "causal_linear_trend_forecast_via_production_evaluate_task",
            "runtime_model": "boston_equivalent_work_scaled_by_stage4a4_node_slowdown",
            "decision_runtime_estimate": "current-node remaining wall time; destination slowdown is not injected into production scoring",
            "realized_transfer_model": "stage4a1_affine_migration_transport",
            "workload_migration_calibration": "stage4a2_checkpoint_restore_overhead",
            "telemetry_confidence": telemetry_confidence,
            "all_static_assets_prestaged": True,
        },
    )


def scenario_outcomes(
    *,
    scenario: Scenario,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    graph: FrozenCalibrationGraph,
    runtime_scale: float,
    gaia_queues: dict[str, dict[str, float]],
    gaia_quantum_seconds: float = DEFAULT_GAIA_QUANTUM_SECONDS,
    oracle_wait_seconds: float = DEFAULT_ORACLE_WAIT_SECONDS,
) -> list[PolicyOutcome]:
    boston = cluster.get_node("boston")
    return [
        _static_outcome(
            label="boston_static",
            node=boston,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
        ),
        best_static_outcome(
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
        ),
        gaia_carbon_time_outcome(
            boston=boston,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
            queue_parameters=gaia_queues,
            quantum_seconds=gaia_quantum_seconds,
        ),
        replay_magellan_causal(
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            graph=graph,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
        ),
        oracle_static_outcome(
            cluster=cluster,
            policy=policy,
            calibration=calibration,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            arrival_utc=scenario.arrival_utc,
            runtime_scale=runtime_scale,
            max_wait_seconds=oracle_wait_seconds,
            quantum_seconds=gaia_quantum_seconds,
        ),
    ]


def outcome_rows(
    *,
    scenario: Scenario,
    calibration: WorkloadCalibration,
    outcomes: list[PolicyOutcome],
    policy: ScoringPolicy,
) -> list[dict[str, Any]]:
    by_policy = {outcome.policy: outcome for outcome in outcomes}
    if set(by_policy) != set(CORE_POLICIES):
        raise ValueError(f"Scenario {scenario.scenario_id} did not produce the core policy set")
    baseline = by_policy["boston_static"]
    alpha, beta, gamma = policy.weights.normalized()
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        time_ratio = outcome.makespan_seconds / baseline.makespan_seconds
        carbon_ratio = outcome.carbon_grams / baseline.carbon_grams if baseline.carbon_grams > 0 else 0.0
        cost_ratio = outcome.cost_usd / baseline.cost_usd if baseline.cost_usd > 0 else 0.0
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "class_id": scenario.class_id,
                "workload": calibration.workload,
                "arrival_utc": scenario.arrival_utc.isoformat(),
                "policy": outcome.policy,
                "start_node_id": outcome.start_node_id,
                "final_node_id": outcome.final_node_id,
                "selected_initial_node_id": outcome.selected_initial_node_id,
                "completed": outcome.completed,
                "makespan_seconds": outcome.makespan_seconds,
                "compute_seconds": outcome.compute_seconds,
                "submission_wait_seconds": float(outcome.metadata.get("submission_wait_seconds", 0.0)),
                "paused_idle_seconds": outcome.paused_idle_seconds,
                "pause_overhead_seconds": outcome.pause_overhead_seconds,
                "migration_seconds": outcome.migration_seconds,
                "carbon_grams": outcome.carbon_grams,
                "cost_usd": outcome.cost_usd,
                "migrations": outcome.migrations,
                "pauses": outcome.pauses,
                "decision_count": outcome.decision_count,
                "owner_path": "->".join(outcome.owner_path),
                "time_ratio_vs_boston_static": time_ratio,
                "carbon_ratio_vs_boston_static": carbon_ratio,
                "cost_ratio_vs_boston_static": cost_ratio,
                "weighted_ratio_objective_vs_boston_static": alpha * time_ratio + beta * carbon_ratio + gamma * cost_ratio,
            }
        )
    return rows


def summarize_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    output: list[dict[str, Any]] = []
    for policy_name in CORE_POLICIES:
        items = grouped.get(policy_name, [])
        if not items:
            continue
        time_ratios = [float(item["time_ratio_vs_boston_static"]) for item in items]
        carbon_ratios = [float(item["carbon_ratio_vs_boston_static"]) for item in items]
        cost_ratios = [float(item["cost_ratio_vs_boston_static"]) for item in items]
        objectives = [float(item["weighted_ratio_objective_vs_boston_static"]) for item in items]
        output.append(
            {
                "policy": policy_name,
                "scenario_count": len(items),
                "makespan_seconds_mean": sum(float(item["makespan_seconds"]) for item in items) / len(items),
                "carbon_grams_mean": sum(float(item["carbon_grams"]) for item in items) / len(items),
                "cost_usd_mean": sum(float(item["cost_usd"]) for item in items) / len(items),
                "time_ratio_mean": sum(time_ratios) / len(time_ratios),
                "carbon_ratio_mean": sum(carbon_ratios) / len(carbon_ratios),
                "carbon_savings_percent_mean": 100.0 * (1.0 - sum(carbon_ratios) / len(carbon_ratios)),
                "cost_ratio_mean": sum(cost_ratios) / len(cost_ratios),
                "weighted_ratio_objective_mean": sum(objectives) / len(objectives),
                "migrations_total": sum(int(item["migrations"]) for item in items),
                "pauses_total": sum(int(item["pauses"]) for item in items),
            }
        )
    return output


def descriptive_policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy_name in CORE_POLICIES:
        items = [row for row in rows if row["policy"] == policy_name]
        if not items:
            continue
        result[policy_name] = {
            "scenario_count": len(items),
            "makespan_seconds": summarize_samples(float(item["makespan_seconds"]) for item in items).as_dict(),
            "carbon_grams": summarize_samples(float(item["carbon_grams"]) for item in items).as_dict(),
            "cost_usd": summarize_samples(float(item["cost_usd"]) for item in items).as_dict(),
            "carbon_ratio_vs_boston_static": summarize_samples(float(item["carbon_ratio_vs_boston_static"]) for item in items).as_dict(),
            "weighted_ratio_objective_vs_boston_static": summarize_samples(float(item["weighted_ratio_objective_vs_boston_static"]) for item in items).as_dict(),
        }
    return result
