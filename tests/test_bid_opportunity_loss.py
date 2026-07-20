from magellan.daemon.scheduler_service import SchedulerService
from magellan.models.types import (
    ActionType,
    ScoredAction,
    TaskProfile,
)


def action(action_type, score, destination=None):
    return ScoredAction(
        action=action_type,
        source_node_id="boston",
        destination_node_id=destination,
        time_seconds=1,
        carbon_grams=1,
        cost_usd=1,
        normalized_time=0,
        normalized_carbon=0,
        normalized_cost=0,
        score=score,
    )


def test_task_bid_carries_second_choice_opportunity_loss() -> None:
    service = object.__new__(SchedulerService)
    task = TaskProfile(
        task_id="task",
        workload_type="counter",
        current_node_id="boston",
        power_kw=0.1,
        checkpoint_bytes=100,
        estimated_remaining_seconds=100,
    )
    france = action(ActionType.MIGRATE, 0.1, "france")
    continue_local = action(ActionType.CONTINUE, 0.7)
    virginia = action(ActionType.MIGRATE, 0.8, "virginia")

    context = service._task_bid_context(
        task,
        static_data_bytes=0,
        candidate=france,
        ranked_actions=[france, continue_local, virginia],
    )

    assert context.fallback_action == ActionType.CONTINUE
    assert context.fallback_score == 0.7
    assert context.opportunity_loss == 0.6
