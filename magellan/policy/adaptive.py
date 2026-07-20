from __future__ import annotations

from datetime import datetime, timezone

from magellan.config.policy_models import AdaptivePolicy, ObjectiveWeights
from magellan.models.types import ActionType, DecisionResult, RawActionEstimate, TaskProfile
from magellan.policy.models import (
    AdaptationSignals,
    AdaptiveDecisionContext,
    AdaptiveTaskPolicyState,
    MetricRangeSample,
    NormalizationBounds,
    PolicyDecisionRecord,
    WeightMultipliers,
    WeightVector,
)
from magellan.policy.store import AdaptivePolicyStore


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _weights_from_config(weights: ObjectiveWeights) -> WeightVector:
    time_weight, carbon_weight, cost_weight = weights.normalized()
    return WeightVector(
        time=time_weight,
        carbon=carbon_weight,
        cost=cost_weight,
    )


class AdaptivePolicyService:
    """Computes bounded task-local weights and rolling normalization."""

    def __init__(
        self,
        policy: AdaptivePolicy,
        baseline_weights: ObjectiveWeights,
        store: AdaptivePolicyStore,
    ) -> None:
        self.policy = policy
        self.store = store
        self._baseline = _weights_from_config(baseline_weights)

    def _state_for(self, task_id: str) -> AdaptiveTaskPolicyState:
        existing = self.store.get(task_id)
        if existing is not None:
            return existing
        return AdaptiveTaskPolicyState(
            task_id=task_id,
            baseline_weights=self._baseline,
            effective_weights=self._baseline,
        )

    def _append_ranges(
        self,
        state: AdaptiveTaskPolicyState,
        estimates: list[RawActionEstimate],
        observed_at: datetime,
    ) -> NormalizationBounds:
        values = {
            "time": [item.time_seconds for item in estimates],
            "carbon": [item.carbon_grams for item in estimates],
            "cost": [item.cost_usd for item in estimates],
        }
        for name, measurements in values.items():
            metric = getattr(state.normalization, name)
            metric.samples.append(
                MetricRangeSample(
                    minimum=min(measurements),
                    maximum=max(measurements),
                    observed_at_utc=observed_at,
                )
            )
            if len(metric.samples) > self.policy.rolling_window_epochs:
                metric.samples = metric.samples[
                    -self.policy.rolling_window_epochs :
                ]

        time_bounds = state.normalization.time.bounds()
        carbon_bounds = state.normalization.carbon.bounds()
        cost_bounds = state.normalization.cost.bounds()
        assert time_bounds is not None
        assert carbon_bounds is not None
        assert cost_bounds is not None
        return NormalizationBounds(
            time_min=time_bounds[0],
            time_max=time_bounds[1],
            carbon_min=carbon_bounds[0],
            carbon_max=carbon_bounds[1],
            cost_min=cost_bounds[0],
            cost_max=cost_bounds[1],
            source="rolling_window",
        )

    def _signals(
        self,
        task: TaskProfile,
        estimates: list[RawActionEstimate],
        at_utc: datetime,
        telemetry_confidence: float,
    ) -> AdaptationSignals:
        budget_slack: float | None = None
        cost_cap_exhausted = False
        if task.cost_cap_usd is not None:
            remaining_budget = max(
                0.0,
                task.cost_cap_usd - task.accumulated_cost_usd,
            )
            budget_slack = _clamp(
                remaining_budget / task.cost_cap_usd,
                0.0,
                1.0,
            )
            cost_cap_exhausted = remaining_budget <= 0

        deadline_slack_ratio: float | None = None
        deadline_at_risk = False
        if (
            task.deadline_at_utc is not None
            and task.estimated_remaining_seconds is not None
        ):
            deadline = task.deadline_at_utc
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            available = max(0.0, (deadline - at_utc).total_seconds())
            remaining = max(task.estimated_remaining_seconds, 1e-9)
            deadline_slack_ratio = available / remaining
            deadline_at_risk = deadline_slack_ratio <= 1.0

        local_carbons = [
            item.carbon_grams
            for item in estimates
            if item.action in {ActionType.CONTINUE, ActionType.PAUSE}
        ]
        migration_carbons = [
            item.carbon_grams
            for item in estimates
            if item.action == ActionType.MIGRATE
        ]
        carbon_opportunity = 0.0
        if local_carbons and migration_carbons:
            best_local = min(local_carbons)
            best_migration = min(migration_carbons)
            carbon_opportunity = _clamp(
                (best_local - best_migration) / max(best_local, 1e-9),
                0.0,
                1.0,
            )

        return AdaptationSignals(
            budget_slack_fraction=budget_slack,
            deadline_slack_ratio=deadline_slack_ratio,
            carbon_opportunity_fraction=carbon_opportunity,
            telemetry_confidence=_clamp(telemetry_confidence, 0.0, 1.0),
            cost_cap_exhausted=cost_cap_exhausted,
            deadline_at_risk=deadline_at_risk,
        )

    def _adapt(
        self,
        baseline: WeightVector,
        signals: AdaptationSignals,
    ) -> tuple[WeightMultipliers, WeightVector]:
        bound = self.policy.multiplier_bound_fraction
        lower = 1.0 - bound
        upper = 1.0 + bound

        time_pressure = 0.0
        if signals.deadline_slack_ratio is not None:
            time_pressure = _clamp(
                1.0 - signals.deadline_slack_ratio,
                0.0,
                1.0,
            )
            if signals.deadline_at_risk:
                time_pressure = 1.0

        cost_pressure = 0.0
        if signals.budget_slack_fraction is not None:
            cost_pressure = 1.0 - signals.budget_slack_fraction

        carbon_signal = (
            signals.carbon_opportunity_fraction
            * max(signals.telemetry_confidence, self.policy.confidence_floor)
        )

        multipliers = WeightMultipliers(
            time=_clamp(1.0 + bound * time_pressure, lower, upper),
            carbon=_clamp(1.0 + bound * carbon_signal, lower, upper),
            cost=_clamp(1.0 + bound * cost_pressure, lower, upper),
        )
        effective = WeightVector(
            time=baseline.time * multipliers.time,
            carbon=baseline.carbon * multipliers.carbon,
            cost=baseline.cost * multipliers.cost,
        ).normalized()
        return multipliers, effective

    def prepare(
        self,
        task: TaskProfile,
        estimates: list[RawActionEstimate],
        at_utc: datetime,
        *,
        telemetry_confidence: float = 0.0,
        hard_constraints: dict[str, bool | float | str | None] | None = None,
    ) -> AdaptiveDecisionContext:
        state = self._state_for(task.task_id)
        bounds = self._append_ranges(state, estimates, at_utc)
        signals = self._signals(
            task,
            estimates,
            at_utc,
            telemetry_confidence,
        )
        if self.policy.enabled:
            multipliers, effective = self._adapt(
                state.baseline_weights,
                signals,
            )
        else:
            multipliers = WeightMultipliers()
            effective = state.baseline_weights

        state.effective_weights = effective
        state.multipliers = multipliers
        state.signals = signals
        state.updated_at_utc = datetime.now(timezone.utc)
        self.store.put(state)

        return AdaptiveDecisionContext(
            task_id=task.task_id,
            baseline_weights=state.baseline_weights,
            effective_weights=effective,
            multipliers=multipliers,
            signals=signals,
            normalization_bounds=bounds,
            hard_constraints=hard_constraints or {},
        )

    def record_decision(
        self,
        decision: DecisionResult,
        context: AdaptiveDecisionContext,
        at_utc: datetime,
    ) -> AdaptiveTaskPolicyState:
        state = self._state_for(context.task_id)
        state.decision_count += 1
        record = PolicyDecisionRecord(
            task_id=context.task_id,
            decision_index=state.decision_count,
            evaluated_at_utc=at_utc,
            selected_action=decision.selected.action.value,
            selected_destination_node_id=(
                decision.selected.destination_node_id
            ),
            selected_score=decision.selected.score,
            baseline_weights=context.baseline_weights,
            effective_weights=context.effective_weights,
            multipliers=context.multipliers,
            signals=context.signals,
            normalization_bounds=context.normalization_bounds,
            hard_constraints=context.hard_constraints,
            reason=decision.reason,
        )
        state.last_decision = record
        state.decision_history.append(record)
        if len(state.decision_history) > self.policy.decision_history_limit:
            state.decision_history = state.decision_history[
                -self.policy.decision_history_limit :
            ]
        state.effective_weights = context.effective_weights
        state.multipliers = context.multipliers
        state.signals = context.signals
        state.updated_at_utc = datetime.now(timezone.utc)
        return self.store.put(state)

    def reset(self, task_id: str) -> bool:
        return self.store.delete(task_id)
