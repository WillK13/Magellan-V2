from datetime import datetime, timezone

import pandas as pd

from magellan.config.policy_models import (
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
)
from magellan.models.types import (
    ActionType,
    ScoredAction,
    TaskProfile,
)
from magellan.scheduler.scoring import choose_action


def action(kind: ActionType, score: float) -> ScoredAction:
    return ScoredAction(
        action=kind,
        source_node_id="boston",
        destination_node_id=(
            "virginia" if kind == ActionType.MIGRATE else None
        ),
        time_seconds=1,
        carbon_grams=1,
        cost_usd=1,
        normalized_time=score,
        normalized_carbon=score,
        normalized_cost=score,
        score=score,
    )


def test_pause_gap_prevents_immediate_repause() -> None:
    now = pd.Timestamp("2024-01-01T00:01:00Z")
    task = TaskProfile(
        task_id="pause-gap",
        workload_type="test",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=0,
        last_pause_at=datetime(
            2024, 1, 1, 0, 0, 30, tzinfo=timezone.utc
        ),
    )
    policy = ScoringPolicy(
        horizon_seconds=60,
        weights=ObjectiveWeights(time=1, carbon=1, cost=1),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=10,
            resume_seconds=0,
            max_pause_window_seconds=120,
            min_pause_gap_seconds=60,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        clock=ClockPolicy(mode="wall"),
    )
    ranked = [
        action(ActionType.PAUSE, 0.1),
        action(ActionType.CONTINUE, 0.2),
        action(ActionType.MIGRATE, 0.3),
    ]

    result = choose_action(task, ranked, policy, now)
    assert result.selected.action == ActionType.CONTINUE
    assert "pause gap" in result.reason
