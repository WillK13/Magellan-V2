from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from magellan.config.policy_models import AdaptivePolicy, ObjectiveWeights
from magellan.models.types import ActionType, RawActionEstimate, TaskProfile
from magellan.policy.adaptive import AdaptivePolicyService
from magellan.policy.store import AdaptivePolicyStore


def estimates(
    local_carbon: float = 100.0,
    migration_carbon: float = 20.0,
    time_scale: float = 1.0,
) -> list[RawActionEstimate]:
    return [
        RawActionEstimate(
            action=ActionType.CONTINUE,
            source_node_id="boston",
            time_seconds=100 * time_scale,
            carbon_grams=local_carbon,
            cost_usd=1.0,
        ),
        RawActionEstimate(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=120 * time_scale,
            carbon_grams=migration_carbon,
            cost_usd=2.0,
        ),
    ]


def service(tmp_path, **policy_overrides) -> AdaptivePolicyService:
    return AdaptivePolicyService(
        policy=AdaptivePolicy(**policy_overrides),
        baseline_weights=ObjectiveWeights(
            time=0.25,
            carbon=0.5,
            cost=0.25,
        ),
        store=AdaptivePolicyStore(tmp_path),
    )


def profile(**updates) -> TaskProfile:
    values = {
        "task_id": "adaptive-task",
        "workload_type": "counter",
        "current_node_id": "boston",
        "power_kw": 0.1,
        "checkpoint_bytes": 100,
        "estimated_remaining_seconds": 3600.0,
        "accumulated_cost_usd": 0.0,
        "cost_cap_usd": 10.0,
    }
    values.update(updates)
    return TaskProfile(**values)


def test_budget_pressure_increases_cost_weight(tmp_path) -> None:
    adaptive = service(tmp_path)
    context = adaptive.prepare(
        profile(accumulated_cost_usd=9.5),
        estimates(),
        datetime.now(timezone.utc),
        telemetry_confidence=1.0,
    )

    assert context.multipliers.cost > 1.0
    assert context.effective_weights.cost > context.baseline_weights.cost
    assert sum(context.effective_weights.model_dump().values()) == pytest.approx(1)
    assert context.multipliers.cost <= 1.25


def test_deadline_risk_increases_time_weight(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    adaptive = service(tmp_path)
    context = adaptive.prepare(
        profile(deadline_at_utc=now + timedelta(seconds=1800)),
        estimates(),
        now,
        telemetry_confidence=1.0,
    )

    assert context.signals.deadline_at_risk is True
    assert context.multipliers.time == pytest.approx(1.25)
    assert context.effective_weights.time > context.baseline_weights.time


def test_carbon_opportunity_increases_carbon_weight(tmp_path) -> None:
    adaptive = service(tmp_path)
    context = adaptive.prepare(
        profile(),
        estimates(local_carbon=100, migration_carbon=10),
        datetime.now(timezone.utc),
        telemetry_confidence=1.0,
    )

    assert context.signals.carbon_opportunity_fraction == pytest.approx(0.9)
    assert context.multipliers.carbon > 1.0
    assert context.effective_weights.carbon > context.baseline_weights.carbon


def test_rolling_normalization_spans_multiple_epochs(tmp_path) -> None:
    adaptive = service(tmp_path, rolling_window_epochs=2)
    now = datetime.now(timezone.utc)
    first = adaptive.prepare(profile(), estimates(time_scale=1), now)
    second = adaptive.prepare(
        profile(),
        estimates(time_scale=10),
        now + timedelta(seconds=30),
    )

    assert first.normalization_bounds.time_min == pytest.approx(100)
    assert first.normalization_bounds.time_max == pytest.approx(120)
    assert second.normalization_bounds.time_min == pytest.approx(100)
    assert second.normalization_bounds.time_max == pytest.approx(1200)


def test_adaptive_policy_store_survives_restart(tmp_path) -> None:
    adaptive = service(tmp_path)
    now = datetime.now(timezone.utc)
    context = adaptive.prepare(profile(), estimates(), now)

    from magellan.models.types import DecisionResult, ScoredAction

    action = ScoredAction(
        **estimates()[0].model_dump(),
        normalized_time=0,
        normalized_carbon=1,
        normalized_cost=0,
        score=0.5,
    )
    adaptive.record_decision(
        DecisionResult(
            selected=action,
            ranked_actions=[action],
            reason="test",
        ),
        context,
        now,
    )

    restarted = AdaptivePolicyStore(tmp_path)
    state = restarted.get("adaptive-task")
    assert state is not None
    assert state.decision_count == 1
    assert state.last_decision is not None
    assert restarted.path.exists()


def test_carbon_forecast_confidence_dampens_carbon_adaptation(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    high = service(tmp_path / "high").prepare(
        profile(),
        estimates(local_carbon=100, migration_carbon=10),
        now,
        telemetry_confidence=1.0,
        carbon_forecast_confidence=1.0,
    )
    low = service(tmp_path / "low").prepare(
        profile(),
        estimates(local_carbon=100, migration_carbon=10),
        now,
        telemetry_confidence=1.0,
        carbon_forecast_confidence=0.1,
    )

    assert high.signals.carbon_forecast_confidence == 1.0
    assert low.signals.carbon_forecast_confidence == 0.1
    assert high.multipliers.carbon > low.multipliers.carbon
