from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.models import BidRequest, BidStatus
from magellan.bidding.store import BidStore
from magellan.models.types import ActionType, ScoredAction


def make_bid() -> BidRequest:
    return BidRequest(
        bid_id="durable-bid",
        epoch_id="epoch",
        task_id="task",
        source_node_id="boston",
        destination_node_id="virginia",
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=1,
            carbon_grams=1,
            cost_usd=1,
            normalized_time=0,
            normalized_carbon=0,
            normalized_cost=0,
            score=0,
        ),
    )


@pytest.mark.asyncio
async def test_bid_and_reservation_survive_store_restart(tmp_path) -> None:
    state_file = tmp_path / "control" / "bids.json"
    store = BidStore(30, state_file=state_file)
    bid = make_bid()
    await store.submit(bid)
    accepted_at = datetime.now(timezone.utc)
    await store.decide(
        bid.bid_id,
        BidStatus.ACCEPTED,
        "accepted",
        now_utc=accepted_at,
    )

    restarted = BidStore(30, state_file=state_file)
    loaded = await restarted.get(bid.bid_id)

    assert loaded is not None
    assert loaded.status == BidStatus.ACCEPTED
    assert await restarted.active_reservation_count() == 1

    await restarted.expire_reservations(
        accepted_at + timedelta(seconds=31)
    )
    restarted_again = BidStore(30, state_file=state_file)
    expired = await restarted_again.get(bid.bid_id)
    assert expired is not None
    assert expired.status == BidStatus.EXPIRED
