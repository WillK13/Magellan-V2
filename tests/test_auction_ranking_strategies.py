from datetime import datetime, timedelta, timezone

from magellan.bidding.models import (
    AuctionStrategy,
    BidRecord,
    BidRequest,
    TaskBidContext,
)
from magellan.bidding.ranking import rank_bids
from magellan.config.models import NodeResourceCapacity
from magellan.config.policy_models import AuctionPolicy
from magellan.models.types import (
    ActionType,
    ScoredAction,
    TaskResourceRequest,
)


def bid(
    bid_id: str,
    *,
    score: float,
    remaining: float,
    opportunity_loss: float = 0.0,
    priority: int = 0,
    deadline=None,
    cpu: float = 1.0,
) -> BidRecord:
    now = datetime.now(timezone.utc)
    request = BidRequest(
        bid_id=bid_id,
        epoch_id="epoch",
        task_id=f"task-{bid_id}",
        source_node_id="boston",
        destination_node_id="virginia",
        task_context=TaskBidContext(
            workload_type="counter",
            estimated_remaining_seconds=remaining,
            opportunity_loss=opportunity_loss,
            priority=priority,
            deadline_at_utc=deadline,
            resource_request=TaskResourceRequest(
                cpu_cores=cpu,
            ),
        ),
        submitted_at_utc=now,
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
            score=score,
        ),
    )
    return BidRecord(
        **request.model_dump(),
        received_at_utc=now,
    )


def ordered_ids(
    bids,
    strategy: AuctionStrategy,
    credits=None,
    now=None,
):
    return [
        item.bid.bid_id
        for item in rank_bids(
            bids=bids,
            strategy=strategy,
            credits=credits or {},
            node_resources=NodeResourceCapacity(
                cpu_cores=4,
                memory_mb=8192,
                gpu_count=0,
            ),
            policy=AuctionPolicy(strategy=strategy.value),
            now_utc=now or datetime.now(timezone.utc),
        )
    ]


def test_shortest_and_longest_remaining_policies() -> None:
    bids = [
        bid("short", score=0.8, remaining=10),
        bid("long", score=0.1, remaining=100),
    ]
    assert ordered_ids(
        bids,
        AuctionStrategy.SHORTEST_REMAINING,
    )[0] == "short"
    assert ordered_ids(
        bids,
        AuctionStrategy.LONGEST_REMAINING,
    )[0] == "long"


def test_lowest_score_and_credit_fair_policies() -> None:
    bids = [
        bid("low", score=0.1, remaining=100),
        bid("credited", score=0.6, remaining=100),
    ]
    assert ordered_ids(
        bids,
        AuctionStrategy.LOWEST_SCORE,
    )[0] == "low"
    assert ordered_ids(
        bids,
        AuctionStrategy.CREDIT_FAIR,
        credits={"task-credited": 3.0},
    )[0] == "credited"


def test_highest_regret_prioritizes_worst_fallback() -> None:
    bids = [
        bid(
            "good-fallback",
            score=0.1,
            remaining=100,
            opportunity_loss=0.01,
        ),
        bid(
            "bad-fallback",
            score=0.2,
            remaining=100,
            opportunity_loss=0.6,
        ),
    ]
    assert ordered_ids(
        bids,
        AuctionStrategy.HIGHEST_REGRET,
    )[0] == "bad-fallback"


def test_priority_deadline_prioritizes_task_likely_to_miss() -> None:
    now = datetime.now(timezone.utc)
    bids = [
        bid(
            "comfortable",
            score=0.1,
            remaining=100,
            priority=10,
            deadline=now + timedelta(hours=4),
        ),
        bid(
            "urgent",
            score=0.5,
            remaining=3600,
            priority=80,
            deadline=now + timedelta(minutes=30),
        ),
    ]
    assert ordered_ids(
        bids,
        AuctionStrategy.PRIORITY_DEADLINE,
        now=now,
    )[0] == "urgent"


def test_resource_efficiency_prefers_more_value_per_core() -> None:
    bids = [
        bid(
            "large",
            score=0.2,
            remaining=100,
            opportunity_loss=0.4,
            cpu=4,
        ),
        bid(
            "small",
            score=0.3,
            remaining=100,
            opportunity_loss=0.35,
            cpu=1,
        ),
    ]
    assert ordered_ids(
        bids,
        AuctionStrategy.RESOURCE_EFFICIENCY,
    )[0] == "small"
