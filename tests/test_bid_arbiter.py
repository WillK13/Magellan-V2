from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.arbiter import BidArbiter
from magellan.bidding.models import (
    BidRequest,
    BidStatus,
)
from magellan.bidding.store import BidStore
from magellan.models.types import (
    ActionType,
    ScoredAction,
)
from magellan.state.task_registry import TaskRegistry


def migration_candidate(
    score: float,
) -> ScoredAction:
    return ScoredAction(
        action=ActionType.MIGRATE,
        source_node_id="boston",
        destination_node_id="virginia",
        time_seconds=100,
        carbon_grams=10,
        cost_usd=1,
        normalized_time=0.5,
        normalized_carbon=0.5,
        normalized_cost=0.5,
        score=score,
        details={},
    )


@pytest.mark.asyncio
async def test_arbiter_accepts_lowest_score() -> None:
    store = BidStore()
    registry = TaskRegistry([])

    arbiter = BidArbiter(
        store=store,
        registry=registry,
        local_node_id="virginia",
        capacity=1,
        bid_window_seconds=1,
    )

    submitted_at = datetime.now(timezone.utc)

    first = BidRequest(
        bid_id="bid-high",
        epoch_id="epoch-1",
        task_id="task-high",
        source_node_id="boston",
        destination_node_id="virginia",
        candidate=migration_candidate(0.7),
        submitted_at_utc=submitted_at,
    )

    second = BidRequest(
        bid_id="bid-low",
        epoch_id="epoch-1",
        task_id="task-low",
        source_node_id="boston",
        destination_node_id="virginia",
        candidate=migration_candidate(0.2),
        submitted_at_utc=submitted_at,
    )

    await store.submit(first)
    await store.submit(second)

    future = datetime.now(timezone.utc) + timedelta(
        seconds=2
    )

    progressed = await arbiter.run_once(
        now_utc=future
    )

    assert progressed is True

    high_result = await store.get("bid-high")
    low_result = await store.get("bid-low")

    assert high_result is not None
    assert low_result is not None

    assert high_result.status == BidStatus.REJECTED
    assert low_result.status == BidStatus.ACCEPTED
