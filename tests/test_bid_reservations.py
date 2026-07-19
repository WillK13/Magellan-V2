from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.models import (
    BidRequest,
    BidStatus,
)
from magellan.bidding.store import BidStore
from magellan.models.types import ActionType, ScoredAction


def request() -> BidRequest:
    return BidRequest(
        bid_id="lease-1",
        epoch_id="epoch-1",
        task_id="task-1",
        source_node_id="boston",
        destination_node_id="virginia",
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=10,
            carbon_grams=1,
            cost_usd=0.1,
            normalized_time=0.1,
            normalized_carbon=0.1,
            normalized_cost=0.1,
            score=0.1,
        ),
    )


@pytest.mark.asyncio
async def test_accepted_bid_expires_and_releases_capacity() -> None:
    store = BidStore(reservation_ttl_seconds=5)
    bid = request()
    await store.submit(bid)

    decided_at = datetime.now(timezone.utc)
    accepted = await store.decide(
        bid.bid_id,
        BidStatus.ACCEPTED,
        "accepted",
        now_utc=decided_at,
    )

    assert accepted.reservation_expires_at_utc is not None
    assert await store.active_reservation_count() == 1

    expired = await store.expire_reservations(
        decided_at + timedelta(seconds=6)
    )

    assert len(expired) == 1
    assert expired[0].status == BidStatus.EXPIRED
    assert await store.active_reservation_count() == 0


@pytest.mark.asyncio
async def test_reservation_is_consumed_by_matching_activation() -> None:
    store = BidStore(reservation_ttl_seconds=30)
    bid = request()
    await store.submit(bid)
    await store.decide(
        bid.bid_id,
        BidStatus.ACCEPTED,
        "accepted",
    )

    activating = await store.begin_activation(
        bid_id=bid.bid_id,
        task_id=bid.task_id,
        source_node_id=bid.source_node_id,
        destination_node_id=bid.destination_node_id,
    )
    consumed = await store.consume(bid.bid_id)

    assert activating.status == BidStatus.ACTIVATING
    assert consumed.status == BidStatus.CONSUMED
    assert await store.active_reservation_count() == 0
