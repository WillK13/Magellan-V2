from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from typing import Any, Iterable

import pandas as pd

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.comparison import PolicyOutcome, ReplayStep
from magellan.experiments.stage4b import (
    CORE_WORKLOADS,
    Scenario,
    WorkloadCalibration,
)
from magellan.models.utils import seconds_to_hours


TARGET_BOSTON_RUNTIME_SECONDS = 72.0 * 3600.0
LEADERSHIP_QUANTUM_SECONDS = 3600.0
DYNAMIC_POLICIES = ("boston_static", "magellan_causal")
SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}
SEASON_ORDER = ("winter", "spring", "summer", "fall")


MIGRATION_DIAGNOSTIC_FIELDS = [
    "scenario_id",
    "class_id",
    "migration_index",
    "source_node_id",
    "destination_node_id",
    "started_at_utc",
    "finished_at_utc",
    "migration_seconds",
    "migration_carbon_grams",
    "migration_cost_usd",
    "remaining_boston_equivalent_seconds",
    "source_intensity_departure_g_per_kwh",
    "destination_intensity_arrival_g_per_kwh",
    "source_scheduler_carbon_index",
    "destination_scheduler_carbon_index",
    "source_realized_work_carbon_index",
    "destination_realized_work_carbon_index",
    "scheduler_carbon_leader_at_departure",
    "realized_work_carbon_leader_at_departure",
    "destination_is_scheduler_carbon_leader",
    "destination_is_realized_work_carbon_leader",
    "predicted_migrate_carbon_grams",
    "predicted_continue_carbon_grams",
    "projected_carbon_savings_vs_continue_grams",
    "selected_score",
    "continue_score",
    "score_improvement_vs_continue",
    "clairvoyant_stay_source_carbon_grams",
    "clairvoyant_migrate_then_stay_carbon_grams",
    "clairvoyant_net_carbon_saved_grams",
    "clairvoyant_migration_beneficial",
]

RESIDENCE_FIELDS = [
    "scenario_id",
    "class_id",
    "node_id",
    "compute_seconds",
    "pause_seconds",
    "compute_fraction",
]



def runtime_scales_for_target(
    calibrations: dict[str, WorkloadCalibration],
    *,
    node_slowdowns: dict[str, float],
    target_boston_runtime_seconds: float = TARGET_BOSTON_RUNTIME_SECONDS,
) -> dict[str, float]:
    if target_boston_runtime_seconds <= 0:
        raise ValueError("target_boston_runtime_seconds must be positive")
    boston_slowdown = float(node_slowdowns.get("boston", 0.0))
    if boston_slowdown <= 0:
        raise ValueError("Boston slowdown factor must be positive")
    output: dict[str, float] = {}
    for class_id, calibration in calibrations.items():
        denominator = calibration.canonical_runtime_seconds * boston_slowdown
        if denominator <= 0:
            raise ValueError(f"Non-positive Boston runtime calibration for {class_id}")
        output[class_id] = target_boston_runtime_seconds / denominator
    return output




def boston_static_outcome(
    *,
    cluster: ClusterConfig,
    calibration: WorkloadCalibration,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    arrival_utc: pd.Timestamp,
    runtime_scale: float,
) -> PolicyOutcome:
    boston = cluster.get_node("boston")
    wall_seconds = calibration.scaled_work_seconds(runtime_scale) * float(node_slowdowns[boston.id])
    carbon = _compute_carbon(
        cluster=cluster,
        carbon_store=carbon_store,
        node_id=boston.id,
        start_utc=arrival_utc,
        seconds=wall_seconds,
        power_kw=calibration.power_kw,
    )
    cost = boston.compute_price_usd_per_hour * seconds_to_hours(wall_seconds)
    return PolicyOutcome(
        policy="boston_static",
        start_node_id=boston.id,
        final_node_id=boston.id,
        selected_initial_node_id=boston.id,
        completed=True,
        makespan_seconds=wall_seconds,
        compute_seconds=wall_seconds,
        paused_idle_seconds=0.0,
        pause_overhead_seconds=0.0,
        migration_seconds=0.0,
        carbon_grams=carbon,
        cost_usd=cost,
        migrations=0,
        pauses=0,
        decision_count=0,
        owner_path=[boston.id],
        metadata={"targeted_72h_static_reference": True},
    )


def dynamic_outcome_rows(
    *,
    scenario: Scenario,
    calibration: WorkloadCalibration,
    boston_static: PolicyOutcome,
    magellan: PolicyOutcome,
    policy: ScoringPolicy,
) -> list[dict[str, Any]]:
    alpha, beta, gamma = policy.weights.normalized()
    rows: list[dict[str, Any]] = []
    for outcome in (boston_static, magellan):
        time_ratio = outcome.makespan_seconds / boston_static.makespan_seconds
        carbon_ratio = outcome.carbon_grams / boston_static.carbon_grams if boston_static.carbon_grams > 0 else 0.0
        cost_ratio = outcome.cost_usd / boston_static.cost_usd if boston_static.cost_usd > 0 else 0.0
        rows.append({
            "scenario_id": scenario.scenario_id,
            "class_id": scenario.class_id,
            "workload": calibration.workload,
            "arrival_utc": scenario.arrival_utc.isoformat(),
            "policy": outcome.policy,
            "start_node_id": outcome.start_node_id,
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
            "owner_path": "->".join(outcome.owner_path),
            "time_ratio_vs_boston_static": time_ratio,
            "carbon_ratio_vs_boston_static": carbon_ratio,
            "cost_ratio_vs_boston_static": cost_ratio,
            "weighted_ratio_objective_vs_boston_static": alpha * time_ratio + beta * carbon_ratio + gamma * cost_ratio,
        })
    return rows


def candidate_window_summary(
    *,
    arrival_utc: pd.Timestamp,
    timeline_rows: list[dict[str, Any]],
    minimum_sustained_seconds: float,
    horizon_seconds: float = TARGET_BOSTON_RUNTIME_SECONDS,
    quantum_seconds: float = LEADERSHIP_QUANTUM_SECONDS,
) -> dict[str, Any]:
    if minimum_sustained_seconds < 0:
        raise ValueError("minimum_sustained_seconds must be non-negative")
    horizon_end = arrival_utc + pd.Timedelta(seconds=horizon_seconds)
    windows = _leadership_windows(
        timeline_rows,
        leader_field="scheduler_carbon_leader_id",
        quantum_seconds=quantum_seconds,
        horizon_end_utc=horizon_end,
    )
    sustained = [row for row in windows if float(row["duration_seconds"]) >= minimum_sustained_seconds]
    leader_path: list[str] = []
    sustained_hours: dict[str, float] = defaultdict(float)
    for row in sustained:
        leader = str(row["leader_id"])
        sustained_hours[leader] += float(row["duration_seconds"]) / 3600.0
        if not leader_path or leader_path[-1] != leader:
            leader_path.append(leader)
    return {
        "arrival_utc": arrival_utc.isoformat(),
        "season": SEASON_BY_MONTH[arrival_utc.month],
        "scheduler_leader_changes": leader_change_count(timeline_rows, "scheduler_carbon_leader_id"),
        "realized_work_leader_changes": leader_change_count(timeline_rows, "realized_work_carbon_leader_id"),
        "sustained_scheduler_leader_transitions": max(0, len(leader_path) - 1),
        "sustained_scheduler_unique_leaders": len(set(leader_path)),
        "sustained_scheduler_leader_path": "->".join(leader_path),
        "sustained_scheduler_leader_hours": ";".join(
            f"{node_id}:{hours:.1f}" for node_id, hours in sorted(sustained_hours.items())
        ),
        "minimum_sustained_seconds": minimum_sustained_seconds,
        "selected": False,
        "selection_rank_within_season": "",
    }


def select_crossover_arrivals(
    candidate_rows: list[dict[str, Any]],
    *,
    windows_per_season: int = 1,
) -> list[dict[str, Any]]:
    if windows_per_season <= 0:
        raise ValueError("windows_per_season must be positive")
    selected: list[dict[str, Any]] = []
    seasons_present = [season for season in SEASON_ORDER if any(row["season"] == season for row in candidate_rows)]
    for season in seasons_present:
        rows = [row for row in candidate_rows if row["season"] == season]
        rows.sort(
            key=lambda row: (
                -int(row["sustained_scheduler_leader_transitions"]),
                -int(row["sustained_scheduler_unique_leaders"]),
                -int(row["scheduler_leader_changes"]),
                str(row["arrival_utc"]),
            )
        )
        for rank, row in enumerate(rows[:windows_per_season], start=1):
            copy = dict(row)
            copy["selected"] = True
            copy["selection_rank_within_season"] = rank
            selected.append(copy)
    return selected


def selected_dynamic_scenarios(
    selected_rows: list[dict[str, Any]],
    *,
    class_ids: Iterable[str] = CORE_WORKLOADS,
) -> list[Scenario]:
    output: list[Scenario] = []
    for row in selected_rows:
        arrival = as_utc_timestamp(row["arrival_utc"])
        for class_id in class_ids:
            output.append(
                Scenario(
                    scenario_id=f"{arrival.strftime('%Y%m%dT%H%MZ')}-{class_id}",
                    class_id=class_id,
                    arrival_utc=arrival,
                )
            )
    return output

def _node_carbon_values(
    *,
    cluster: ClusterConfig,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    intensity: dict[str, float] = {}
    scheduler_index: dict[str, float] = {}
    realized_index: dict[str, float] = {}
    for node in cluster.nodes:
        value = float(carbon_store.value_at(node.id, at_utc))
        intensity[node.id] = value
        scheduler_index[node.id] = value * node.pue
        realized_index[node.id] = value * node.pue * float(node_slowdowns[node.id])
    return intensity, scheduler_index, realized_index



def leadership_timeline(
    *,
    scenario: Scenario,
    cluster: ClusterConfig,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
    horizon_seconds: float = TARGET_BOSTON_RUNTIME_SECONDS,
    quantum_seconds: float = LEADERSHIP_QUANTUM_SECONDS,
) -> list[dict[str, Any]]:
    if horizon_seconds <= 0 or quantum_seconds <= 0:
        raise ValueError("leadership horizon/quantum must be positive")
    sample_count = int(ceil(horizon_seconds / quantum_seconds))
    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        offset = index * quantum_seconds
        sampled_at = scenario.arrival_utc + pd.Timedelta(seconds=offset)
        intensity, scheduler_index, realized_index = _node_carbon_values(
            cluster=cluster,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            at_utc=sampled_at,
        )
        scheduler_leader = min(scheduler_index, key=scheduler_index.get)
        realized_leader = min(realized_index, key=realized_index.get)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "class_id": scenario.class_id,
                "sample_index": index + 1,
                "sampled_at_utc": sampled_at.isoformat(),
                "offset_seconds": offset,
                "scheduler_carbon_leader_id": scheduler_leader,
                "scheduler_carbon_leader_intensity_g_per_kwh": intensity[scheduler_leader],
                "scheduler_carbon_leader_index": scheduler_index[scheduler_leader],
                "realized_work_carbon_leader_id": realized_leader,
                "realized_work_carbon_leader_intensity_g_per_kwh": intensity[realized_leader],
                "realized_work_carbon_leader_index": realized_index[realized_leader],
            }
        )
    return rows



def leader_change_count(rows: Iterable[dict[str, Any]], field: str) -> int:
    values = [str(row[field]) for row in rows]
    return sum(left != right for left, right in zip(values, values[1:]))



def _leadership_windows(
    rows: list[dict[str, Any]],
    *,
    leader_field: str,
    quantum_seconds: float,
    horizon_end_utc: pd.Timestamp,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    windows: list[dict[str, Any]] = []
    start_index = 0
    current_leader = str(rows[0][leader_field])
    for index in range(1, len(rows) + 1):
        boundary = index == len(rows) or str(rows[index][leader_field]) != current_leader
        if not boundary:
            continue
        start = as_utc_timestamp(rows[start_index]["sampled_at_utc"])
        if index < len(rows):
            end = as_utc_timestamp(rows[index]["sampled_at_utc"])
        else:
            end = horizon_end_utc
        windows.append(
            {
                "leader_id": current_leader,
                "started_at_utc": start,
                "finished_at_utc": end,
                "duration_seconds": max(0.0, (end - start).total_seconds()),
                "sample_count": index - start_index,
            }
        )
        if index < len(rows):
            start_index = index
            current_leader = str(rows[index][leader_field])
    return windows



def _owner_at(outcome: PolicyOutcome, at_utc: pd.Timestamp) -> str:
    owner = outcome.start_node_id
    for step in outcome.steps:
        if step.action != "migrate" or step.destination_node_id is None:
            continue
        if as_utc_timestamp(step.finished_at_utc) <= at_utc:
            owner = step.destination_node_id
    return owner



def leadership_windows(
    *,
    scenario: Scenario,
    timeline_rows: list[dict[str, Any]],
    magellan: PolicyOutcome,
    minimum_opportunity_seconds: float,
    horizon_seconds: float = TARGET_BOSTON_RUNTIME_SECONDS,
    quantum_seconds: float = LEADERSHIP_QUANTUM_SECONDS,
) -> list[dict[str, Any]]:
    if minimum_opportunity_seconds < 0:
        raise ValueError("minimum_opportunity_seconds must be non-negative")
    horizon_end = scenario.arrival_utc + pd.Timedelta(seconds=horizon_seconds)
    magellan_end = scenario.arrival_utc + pd.Timedelta(seconds=magellan.makespan_seconds)
    relevant_end = min(horizon_end, magellan_end)
    migration_steps = [step for step in magellan.steps if step.action == "migrate"]
    output: list[dict[str, Any]] = []
    for leader_kind, field in (
        ("scheduler_carbon", "scheduler_carbon_leader_id"),
        ("realized_work_carbon", "realized_work_carbon_leader_id"),
    ):
        for index, window in enumerate(
            _leadership_windows(
                timeline_rows,
                leader_field=field,
                quantum_seconds=quantum_seconds,
                horizon_end_utc=horizon_end,
            ),
            start=1,
        ):
            start = max(as_utc_timestamp(window["started_at_utc"]), scenario.arrival_utc)
            end = min(as_utc_timestamp(window["finished_at_utc"]), relevant_end)
            duration = max(0.0, (end - start).total_seconds())
            if duration <= 0:
                continue
            owner = _owner_at(magellan, start)
            leader = str(window["leader_id"])
            is_opportunity = leader != owner and duration >= minimum_opportunity_seconds
            exploited = False
            for step in migration_steps:
                if step.destination_node_id != leader:
                    continue
                finished = as_utc_timestamp(step.finished_at_utc)
                if start <= finished < end:
                    exploited = True
                    break
            output.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "class_id": scenario.class_id,
                    "leader_kind": leader_kind,
                    "window_index": index,
                    "leader_node_id": leader,
                    "started_at_utc": start.isoformat(),
                    "finished_at_utc": end.isoformat(),
                    "duration_seconds": duration,
                    "owner_at_window_start": owner,
                    "minimum_opportunity_seconds": minimum_opportunity_seconds,
                    "is_cross_region_opportunity": is_opportunity,
                    "exploited_by_migration": bool(is_opportunity and exploited),
                    "ignored_opportunity": bool(is_opportunity and not exploited),
                }
            )
    return output



def _compute_carbon(
    *,
    cluster: ClusterConfig,
    carbon_store: CarbonStore,
    node_id: str,
    start_utc: pd.Timestamp,
    seconds: float,
    power_kw: float,
) -> float:
    if seconds <= 0:
        return 0.0
    node = cluster.get_node(node_id)
    intensity = float(carbon_store.average(node_id, start_utc, seconds))
    return power_kw * node.pue * seconds_to_hours(seconds) * intensity



def _decision_action(decision: dict[str, Any], action: str) -> dict[str, Any] | None:
    for item in decision.get("ranked_actions", []):
        value = item.get("action")
        if value == action or value == action.lower():
            return item
    return None



def migration_diagnostics(
    *,
    scenario: Scenario,
    magellan: PolicyOutcome,
    calibration: WorkloadCalibration,
    cluster: ClusterConfig,
    node_slowdowns: dict[str, float],
    carbon_store: CarbonStore,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    migration_index = 0
    for step in magellan.steps:
        if step.action != "migrate" or step.destination_node_id is None:
            continue
        migration_index += 1
        source_id = step.source_node_id
        destination_id = step.destination_node_id
        started = as_utc_timestamp(step.started_at_utc)
        finished = as_utc_timestamp(step.finished_at_utc)
        intensity, scheduler_index, realized_index = _node_carbon_values(
            cluster=cluster,
            node_slowdowns=node_slowdowns,
            carbon_store=carbon_store,
            at_utc=started,
        )
        destination_arrival_intensity = float(carbon_store.value_at(destination_id, finished))
        scheduler_leader = min(scheduler_index, key=scheduler_index.get)
        realized_leader = min(realized_index, key=realized_index.get)
        remaining_work = float(step.remaining_seconds_after)
        stay_seconds = remaining_work * float(node_slowdowns[source_id])
        destination_compute_seconds = remaining_work * float(node_slowdowns[destination_id])
        stay_carbon = _compute_carbon(
            cluster=cluster,
            carbon_store=carbon_store,
            node_id=source_id,
            start_utc=started,
            seconds=stay_seconds,
            power_kw=calibration.power_kw,
        )
        destination_compute_carbon = _compute_carbon(
            cluster=cluster,
            carbon_store=carbon_store,
            node_id=destination_id,
            start_utc=finished,
            seconds=destination_compute_seconds,
            power_kw=calibration.power_kw,
        )
        migrate_then_stay_carbon = float(step.carbon_grams) + destination_compute_carbon
        net_saved = stay_carbon - migrate_then_stay_carbon
        decision = step.details.get("decision") if isinstance(step.details, dict) else None
        if not isinstance(decision, dict):
            decision = {}
        selected = decision.get("selected") if isinstance(decision.get("selected"), dict) else {}
        continue_action = _decision_action(decision, "continue") or {}
        predicted_migrate = float(selected.get("carbon_grams") or 0.0)
        predicted_continue = float(continue_action.get("carbon_grams") or 0.0)
        selected_score = float(selected.get("score") or 0.0)
        continue_score = float(continue_action.get("score") or 0.0)
        output.append(
            {
                "scenario_id": scenario.scenario_id,
                "class_id": scenario.class_id,
                "migration_index": migration_index,
                "source_node_id": source_id,
                "destination_node_id": destination_id,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "migration_seconds": float(step.migration_seconds),
                "migration_carbon_grams": float(step.carbon_grams),
                "migration_cost_usd": float(step.cost_usd),
                "remaining_boston_equivalent_seconds": remaining_work,
                "source_intensity_departure_g_per_kwh": intensity[source_id],
                "destination_intensity_arrival_g_per_kwh": destination_arrival_intensity,
                "source_scheduler_carbon_index": scheduler_index[source_id],
                "destination_scheduler_carbon_index": scheduler_index[destination_id],
                "source_realized_work_carbon_index": realized_index[source_id],
                "destination_realized_work_carbon_index": realized_index[destination_id],
                "scheduler_carbon_leader_at_departure": scheduler_leader,
                "realized_work_carbon_leader_at_departure": realized_leader,
                "destination_is_scheduler_carbon_leader": destination_id == scheduler_leader,
                "destination_is_realized_work_carbon_leader": destination_id == realized_leader,
                "predicted_migrate_carbon_grams": predicted_migrate,
                "predicted_continue_carbon_grams": predicted_continue,
                "projected_carbon_savings_vs_continue_grams": predicted_continue - predicted_migrate,
                "selected_score": selected_score,
                "continue_score": continue_score,
                "score_improvement_vs_continue": continue_score - selected_score,
                "clairvoyant_stay_source_carbon_grams": stay_carbon,
                "clairvoyant_migrate_then_stay_carbon_grams": migrate_then_stay_carbon,
                "clairvoyant_net_carbon_saved_grams": net_saved,
                "clairvoyant_migration_beneficial": net_saved > 0,
            }
        )
    return output



def residence_rows(
    *,
    scenario: Scenario,
    magellan: PolicyOutcome,
) -> list[dict[str, Any]]:
    compute: dict[str, float] = defaultdict(float)
    paused: dict[str, float] = defaultdict(float)
    for step in magellan.steps:
        if step.action == "continue":
            compute[step.source_node_id] += float(step.compute_seconds)
        elif step.action == "pause":
            paused[step.source_node_id] += float(step.elapsed_seconds)
    total_compute = sum(compute.values())
    nodes = sorted(set(compute) | set(paused) | set(magellan.owner_path))
    return [
        {
            "scenario_id": scenario.scenario_id,
            "class_id": scenario.class_id,
            "node_id": node_id,
            "compute_seconds": compute[node_id],
            "pause_seconds": paused[node_id],
            "compute_fraction": compute[node_id] / total_compute if total_compute > 0 else 0.0,
        }
        for node_id in nodes
    ]



def dynamic_scenario_summary(
    *,
    scenario: Scenario,
    magellan: PolicyOutcome,
    timeline_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    migration_rows: list[dict[str, Any]],
    residence: list[dict[str, Any]],
    outcome_row: dict[str, Any],
) -> dict[str, Any]:
    scheduler_windows = [row for row in window_rows if row["leader_kind"] == "scheduler_carbon"]
    realized_windows = [row for row in window_rows if row["leader_kind"] == "realized_work_carbon"]
    scheduler_opportunities = [row for row in scheduler_windows if bool(row["is_cross_region_opportunity"])]
    realized_opportunities = [row for row in realized_windows if bool(row["is_cross_region_opportunity"])]
    return {
        "scenario_id": scenario.scenario_id,
        "class_id": scenario.class_id,
        "arrival_utc": scenario.arrival_utc.isoformat(),
        "completed": magellan.completed,
        "makespan_seconds": magellan.makespan_seconds,
        "carbon_grams": magellan.carbon_grams,
        "cost_usd": magellan.cost_usd,
        "carbon_ratio_vs_boston_static": float(outcome_row["carbon_ratio_vs_boston_static"]),
        "time_ratio_vs_boston_static": float(outcome_row["time_ratio_vs_boston_static"]),
        "cost_ratio_vs_boston_static": float(outcome_row["cost_ratio_vs_boston_static"]),
        "migrations": magellan.migrations,
        "pauses": magellan.pauses,
        "decision_count": magellan.decision_count,
        "distinct_nodes_visited": len(set(magellan.owner_path)),
        "owner_path": "->".join(magellan.owner_path),
        "multi_migration": magellan.migrations >= 2,
        "scheduler_carbon_leader_changes_72h": leader_change_count(timeline_rows, "scheduler_carbon_leader_id"),
        "realized_work_carbon_leader_changes_72h": leader_change_count(timeline_rows, "realized_work_carbon_leader_id"),
        "scheduler_opportunity_windows": len(scheduler_opportunities),
        "scheduler_opportunities_exploited": sum(bool(row["exploited_by_migration"]) for row in scheduler_opportunities),
        "scheduler_opportunities_ignored": sum(bool(row["ignored_opportunity"]) for row in scheduler_opportunities),
        "realized_work_opportunity_windows": len(realized_opportunities),
        "realized_work_opportunities_exploited": sum(bool(row["exploited_by_migration"]) for row in realized_opportunities),
        "realized_work_opportunities_ignored": sum(bool(row["ignored_opportunity"]) for row in realized_opportunities),
        "beneficial_migrations_clairvoyant_diagnostic": sum(bool(row["clairvoyant_migration_beneficial"]) for row in migration_rows),
        "migration_count_diagnostic": len(migration_rows),
        "compute_residence_nodes": len([row for row in residence if float(row["compute_seconds"]) > 0]),
    }



def aggregate_dynamic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("dynamic scenario rows must be non-empty")
    path_counts = Counter(str(row["owner_path"]) for row in rows)
    migration_counts = [int(row["migrations"]) for row in rows]
    total_migrations = sum(migration_counts)
    beneficial = sum(int(row["beneficial_migrations_clairvoyant_diagnostic"]) for row in rows)
    return {
        "scenario_count": len(rows),
        "magellan_migrations_total": total_migrations,
        "magellan_pauses_total": sum(int(row["pauses"]) for row in rows),
        "scenarios_zero_migrations": sum(value == 0 for value in migration_counts),
        "scenarios_one_migration": sum(value == 1 for value in migration_counts),
        "scenarios_multi_migration": sum(value >= 2 for value in migration_counts),
        "maximum_migrations_in_one_scenario": max(migration_counts),
        "dynamic_traversal_observed": any(value >= 2 for value in migration_counts),
        "distinct_owner_paths": len(path_counts),
        "most_common_owner_paths": [
            {"owner_path": path, "scenario_count": count}
            for path, count in path_counts.most_common(10)
        ],
        "scheduler_carbon_leader_changes_total": sum(int(row["scheduler_carbon_leader_changes_72h"]) for row in rows),
        "realized_work_carbon_leader_changes_total": sum(int(row["realized_work_carbon_leader_changes_72h"]) for row in rows),
        "scenarios_with_scheduler_carbon_leader_change": sum(int(row["scheduler_carbon_leader_changes_72h"]) > 0 for row in rows),
        "scenarios_with_realized_work_carbon_leader_change": sum(int(row["realized_work_carbon_leader_changes_72h"]) > 0 for row in rows),
        "scheduler_opportunity_windows_total": sum(int(row["scheduler_opportunity_windows"]) for row in rows),
        "scheduler_opportunities_exploited_total": sum(int(row["scheduler_opportunities_exploited"]) for row in rows),
        "scheduler_opportunities_ignored_total": sum(int(row["scheduler_opportunities_ignored"]) for row in rows),
        "realized_work_opportunity_windows_total": sum(int(row["realized_work_opportunity_windows"]) for row in rows),
        "realized_work_opportunities_exploited_total": sum(int(row["realized_work_opportunities_exploited"]) for row in rows),
        "realized_work_opportunities_ignored_total": sum(int(row["realized_work_opportunities_ignored"]) for row in rows),
        "clairvoyant_beneficial_migrations_total": beneficial,
        "clairvoyant_beneficial_migration_fraction": beneficial / total_migrations if total_migrations else None,
    }
