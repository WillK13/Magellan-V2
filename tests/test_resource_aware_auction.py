from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.arbiter import BidArbiter
from magellan.bidding.models import (
    AuctionStrategy,
    BidRequest,
    BidStatus,
    TaskBidContext,
)
from magellan.bidding.store import BidStore
from magellan.config.models import NodeResourceCapacity
from magellan.config.policy_models import AuctionPolicy
from magellan.models.types import (
    ActionType,
    ScoredAction,
    TaskResourceRequest,
)
from magellan.state.task_registry import TaskRegistry


def make_bid(
    bid_id: str,
    task_id: str,
    *,
    score: float,
    cpu: float = 1,
    memory: int = 0,
    gpu: int = 0,
    accelerator: str | None = None,
    opportunity_loss: float = 0.0,
) -> BidRequest:
    return BidRequest(
        bid_id=bid_id,
        epoch_id="epoch",
        task_id=task_id,
        source_node_id="boston",
        destination_node_id="virginia",
        task_context=TaskBidContext(
            workload_type="counter",
            estimated_remaining_seconds=100,
            opportunity_loss=opportunity_loss,
            resource_request=TaskResourceRequest(
                cpu_cores=cpu,
                memory_mb=memory,
                gpu_count=gpu,
                accelerator_type=accelerator,
            ),
        ),
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
            score=score,
        ),
    )


@pytest.mark.asyncio
async def test_resource_efficiency_packs_two_small_tasks() -> None:
    store = BidStore()
    arbiter = BidArbiter(
        store=store,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=3,
        bid_window_seconds=1,
        node_resources=NodeResourceCapacity(
            cpu_cores=4,
            memory_mb=4096,
            gpu_count=0,
        ),
        auction_policy=AuctionPolicy(
            strategy="resource_efficiency"
        ),
    )
    await store.submit(
        make_bid(
            "large",
            "large-task",
            score=0.1,
            cpu=4,
            opportunity_loss=0.2,
        )
    )
    await store.submit(
        make_bid(
            "small-a",
            "small-task-a",
            score=0.2,
            cpu=2,
            opportunity_loss=0.5,
        )
    )
    await store.submit(
        make_bid(
            "small-b",
            "small-task-b",
            score=0.2,
            cpu=2,
            opportunity_loss=0.5,
        )
    )

    await arbiter.run_once(
        datetime.now(timezone.utc) + timedelta(seconds=2)
    )

    large = await store.get("large")
    small_a = await store.get("small-a")
    small_b = await store.get("small-b")
    assert large is not None and large.status == BidStatus.REJECTED
    assert small_a is not None and small_a.status == BidStatus.ACCEPTED
    assert small_b is not None and small_b.status == BidStatus.ACCEPTED
    assert large.decision_reason == "Insufficient unreserved CPU cores"


@pytest.mark.asyncio
async def test_incompatible_gpu_request_is_rejected_without_credit() -> None:
    store = BidStore()
    arbiter = BidArbiter(
        store=store,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=2,
        bid_window_seconds=1,
        node_resources=NodeResourceCapacity(
            cpu_cores=4,
            memory_mb=4096,
            gpu_count=1,
            accelerator_types={"T4"},
        ),
    )
    await store.submit(
        make_bid(
            "gpu-bid",
            "gpu-task",
            score=0.1,
            gpu=1,
            accelerator="A100",
        )
    )
    await arbiter.run_once(
        datetime.now(timezone.utc) + timedelta(seconds=2)
    )
    result = await store.get("gpu-bid")
    assert result is not None
    assert result.status == BidStatus.REJECTED
    assert result.resource_fit is False
    assert await store.credit_for("gpu-task") == 0.0


@pytest.mark.asyncio
async def test_credit_fairness_survives_store_restart(tmp_path) -> None:
    state_file = tmp_path / "bids.json"
    store = BidStore(state_file=state_file)
    policy = AuctionPolicy(strategy="credit_fair")
    arbiter = BidArbiter(
        store=store,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=1,
        bid_window_seconds=1,
        auction_policy=policy,
    )
    await store.submit(make_bid("a1", "task-a", score=0.1))
    await store.submit(make_bid("b1", "task-b", score=0.2))
    await arbiter.run_once(
        datetime.now(timezone.utc) + timedelta(seconds=2)
    )
    first_a = await store.get("a1")
    first_b = await store.get("b1")
    assert first_a is not None and first_a.status == BidStatus.ACCEPTED
    assert first_b is not None and first_b.status == BidStatus.REJECTED
    assert await store.credit_for("task-b") == 1.0
    await store.cancel("a1", "test release")

    restarted = BidStore(state_file=state_file)
    restarted_arbiter = BidArbiter(
        store=restarted,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=1,
        bid_window_seconds=1,
        auction_policy=policy,
    )
    await restarted.submit(make_bid("a2", "task-a", score=0.1))
    await restarted.submit(make_bid("b2", "task-b", score=0.8))
    await restarted_arbiter.run_once(
        datetime.now(timezone.utc) + timedelta(seconds=2)
    )
    second_a = await restarted.get("a2")
    second_b = await restarted.get("b2")
    assert second_a is not None and second_a.status == BidStatus.REJECTED
    assert second_b is not None and second_b.status == BidStatus.ACCEPTED
    assert second_b.auction_strategy == AuctionStrategy.CREDIT_FAIR
    assert second_b.auction_credit_before == 1.0
    assert second_b.auction_credit_after == 0.0
