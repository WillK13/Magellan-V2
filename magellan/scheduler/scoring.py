from __future__ import annotations

from datetime import timezone

import pandas as pd

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.graph.topology import ClusterGraph
from magellan.models.continue_model import estimate_continue
from magellan.models.migrate_model import estimate_migrate
from magellan.models.pause_model import estimate_pause
from magellan.models.types import (
    ActionType,
    DecisionResult,
    RawActionEstimate,
    ScoredAction,
    TaskProfile,
)
from magellan.models.utils import minmax_normalize


def build_raw_actions(
    task: TaskProfile,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    graph: ClusterGraph,
    carbon_store: CarbonStore,
    at_utc: str | pd.Timestamp,
    static_data_bytes_by_destination: (
        dict[str, int] | None
    ) = None,
) -> list[RawActionEstimate]:
    now = as_utc_timestamp(at_utc)
    source = cluster.get_node(task.current_node_id)

    estimates: list[RawActionEstimate] = []

    estimates.append(
        estimate_continue(
            task=task,
            node=source,
            carbon_store=carbon_store,
            at_utc=now,
            horizon_seconds=policy.horizon_seconds,
        )
    )

    pause_estimate = estimate_pause(
        task=task,
        node=source,
        carbon_store=carbon_store,
        at_utc=now,
        horizon_seconds=policy.horizon_seconds,
        pause_policy=policy.pause,
    )

    if pause_estimate is not None:
        estimates.append(pause_estimate)

    for destination in graph.peers(source.id):
        edge = graph.edge(source.id, destination.id)
        
        static_data_bytes_override = None

        if static_data_bytes_by_destination is not None:
            static_data_bytes_override = (
                static_data_bytes_by_destination.get(
                    destination.id,
                    0,
                )
            )

        migration_estimate = estimate_migrate(
            task=task,
            source=source,
            destination=destination,
            edge=edge,
            carbon_store=carbon_store,
            at_utc=now,
            horizon_seconds=policy.horizon_seconds,
            pause_policy=policy.pause,
            migration_policy=policy.migration,
            static_data_bytes_override=(
                static_data_bytes_override
            ),
        )

        if (
            task.cost_cap_usd is not None
            and task.accumulated_cost_usd
            + migration_estimate.cost_usd
            > task.cost_cap_usd
        ):
            continue

        estimates.append(migration_estimate)

    return estimates


def score_actions(
    estimates: list[RawActionEstimate],
    policy: ScoringPolicy,
) -> list[ScoredAction]:
    if not estimates:
        raise ValueError("Cannot score an empty action list")

    normalized_times = minmax_normalize(
        [estimate.time_seconds for estimate in estimates]
    )
    normalized_carbons = minmax_normalize(
        [estimate.carbon_grams for estimate in estimates]
    )
    normalized_costs = minmax_normalize(
        [estimate.cost_usd for estimate in estimates]
    )

    alpha, beta, gamma = policy.weights.normalized()

    scored: list[ScoredAction] = []

    for index, estimate in enumerate(estimates):
        score = (
            alpha * normalized_times[index]
            + beta * normalized_carbons[index]
            + gamma * normalized_costs[index]
        )

        scored.append(
            ScoredAction(
                **estimate.model_dump(),
                normalized_time=normalized_times[index],
                normalized_carbon=normalized_carbons[index],
                normalized_cost=normalized_costs[index],
                score=score,
            )
        )

    return sorted(scored, key=lambda candidate: candidate.score)


def choose_action(
    task: TaskProfile,
    ranked_actions: list[ScoredAction],
    policy: ScoringPolicy,
    at_utc: str | pd.Timestamp,
) -> DecisionResult:
    if not ranked_actions:
        raise ValueError("No ranked actions were supplied")

    now = as_utc_timestamp(at_utc)

    local_actions = [
        action
        for action in ranked_actions
        if action.action in {
            ActionType.CONTINUE,
            ActionType.PAUSE,
        }
    ]

    if not local_actions:
        raise ValueError("At least one local action must exist")

    best_local = min(local_actions, key=lambda action: action.score)
    best_overall = ranked_actions[0]

    if best_overall.action == ActionType.PAUSE:
        continue_action = next(
            action
            for action in ranked_actions
            if action.action == ActionType.CONTINUE
        )

        if task.last_pause_at is not None:
            last_pause = pd.Timestamp(task.last_pause_at)
            if last_pause.tzinfo is None:
                last_pause = last_pause.tz_localize(timezone.utc)
            else:
                last_pause = last_pause.tz_convert("UTC")

            elapsed = (now - last_pause).total_seconds()
            if elapsed < policy.pause.min_pause_gap_seconds:
                return DecisionResult(
                    selected=continue_action,
                    ranked_actions=ranked_actions,
                    reason=(
                        "Pause has the lowest score, but the minimum "
                        "pause gap has not elapsed"
                    ),
                )

        return DecisionResult(
            selected=best_overall,
            ranked_actions=ranked_actions,
            reason="Pause has the lowest score",
        )

    if best_overall.action == ActionType.CONTINUE:
        return DecisionResult(
            selected=best_overall,
            ranked_actions=ranked_actions,
            reason="Continue has the lowest score",
        )

    if task.last_migration_at is not None:
        last_migration = pd.Timestamp(task.last_migration_at)

        if last_migration.tzinfo is None:
            last_migration = last_migration.tz_localize(timezone.utc)
        else:
            last_migration = last_migration.tz_convert("UTC")

        elapsed = (now - last_migration).total_seconds()

        if elapsed < policy.migration.min_migration_gap_seconds:
            return DecisionResult(
                selected=best_local,
                ranked_actions=ranked_actions,
                reason=(
                    "Migration has the lowest score, but the minimum "
                    "migration gap has not elapsed"
                ),
            )

    denominator = max(best_local.score, 1e-12)
    relative_improvement = (
        best_local.score - best_overall.score
    ) / denominator

    if (
        relative_improvement
        < policy.migration.required_improvement_fraction
    ):
        return DecisionResult(
            selected=best_local,
            ranked_actions=ranked_actions,
            reason=(
                "Migration improvement is below the required "
                "hysteresis threshold"
            ),
        )

    return DecisionResult(
        selected=best_overall,
        ranked_actions=ranked_actions,
        reason=(
            "Migration has the lowest score and passes migration "
            "gap and improvement checks"
        ),
    )


def evaluate_task(
    task: TaskProfile,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    graph: ClusterGraph,
    carbon_store: CarbonStore,
    at_utc: str | pd.Timestamp,
    static_data_bytes_by_destination: (
        dict[str, int] | None
    ) = None,
) -> DecisionResult:
    estimates = build_raw_actions(
        task=task,
        cluster=cluster,
        policy=policy,
        graph=graph,
        carbon_store=carbon_store,
        at_utc=at_utc,
        static_data_bytes_by_destination=(
            static_data_bytes_by_destination
        ),
    )

    ranked = score_actions(estimates, policy)

    return choose_action(
        task=task,
        ranked_actions=ranked,
        policy=policy,
        at_utc=at_utc,
    )
