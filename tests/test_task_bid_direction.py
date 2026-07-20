from datetime import datetime, timezone

from magellan.bidding.models import BidRequest, TaskBidContext
from magellan.models.types import ActionType, ScoredAction, TaskResourceRequest


def test_bid_is_submitted_by_task_for_destination_capacity() -> None:
    bid = BidRequest(
        bid_id="bid-1",
        epoch_id="epoch-1",
        task_id="run-1",
        task_context=TaskBidContext(
            workload_type="llm",
            priority=20,
            estimated_remaining_seconds=3600,
            checkpoint_bytes=1024,
            static_data_bytes=2048,
            resource_request=TaskResourceRequest(
                cpu_cores=2,
                memory_mb=4096,
            ),
        ),
        source_node_id="boston",
        destination_node_id="france",
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="france",
            time_seconds=10,
            carbon_grams=1,
            cost_usd=1,
            normalized_time=0.1,
            normalized_carbon=0.0,
            normalized_cost=0.2,
            score=0.1,
        ),
        submitted_at_utc=datetime.now(timezone.utc),
    )

    assert bid.bidder_type == "task"
    assert bid.task_id == "run-1"
    assert bid.destination_node_id == "france"
    assert bid.task_context.priority == 20
    assert bid.task_context.resource_request.cpu_cores == 2
