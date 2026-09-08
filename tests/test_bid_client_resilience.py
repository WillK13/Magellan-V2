from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from magellan.bidding.client import BidClient
from magellan.bidding.models import BidRecord, BidRequest, BidStatus
from magellan.models.types import ActionType, ScoredAction


def bid_request() -> BidRequest:
    return BidRequest(
        bid_id="bid-1",
        epoch_id="epoch-1",
        task_id="task-1",
        source_node_id="boston",
        destination_node_id="ethiopia",
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="ethiopia",
            time_seconds=10,
            carbon_grams=1,
            cost_usd=0.1,
            normalized_time=0.1,
            normalized_carbon=0.1,
            normalized_cost=0.1,
            score=0.1,
        ),
    )


def bid_record(request: BidRequest, status: BidStatus) -> BidRecord:
    return BidRecord(
        **request.model_dump(),
        status=status,
        received_at_utc=datetime.now(timezone.utc),
    )


class Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class Cluster:
    request_timeout_seconds = 0.1
    bid_window_seconds = 1
    api_port = 8040

    @staticmethod
    def get_node(node_id: str):
        assert node_id == "ethiopia"
        return SimpleNamespace(internal_ip="10.0.0.2")


@pytest.mark.asyncio
async def test_submit_retries_lost_post_response_and_recovers_existing_bid(
    monkeypatch,
) -> None:
    request = bid_request()
    accepted = bid_record(request, BidStatus.ACCEPTED)

    class AsyncClient:
        post_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, json):
            type(self).post_calls += 1
            if type(self).post_calls == 1:
                # Model a response loss after the destination persisted the
                # bid. Retrying the same bid_id returns the existing record.
                raise httpx.ReadTimeout("response lost")
            return Response(accepted.model_dump(mode="json"))

        async def get(self, url):
            raise AssertionError("accepted retry should not need polling")

    monkeypatch.setattr(
        "magellan.bidding.client.httpx.AsyncClient",
        AsyncClient,
    )

    result = await BidClient(Cluster()).submit_and_wait(request)

    assert result.status == BidStatus.ACCEPTED
    assert AsyncClient.post_calls == 2


@pytest.mark.asyncio
async def test_submit_tolerates_transient_poll_failure(monkeypatch) -> None:
    request = bid_request()
    pending = bid_record(request, BidStatus.PENDING)
    accepted = bid_record(request, BidStatus.ACCEPTED)

    class AsyncClient:
        get_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, json):
            return Response(pending.model_dump(mode="json"))

        async def get(self, url):
            type(self).get_calls += 1
            if type(self).get_calls == 1:
                raise httpx.ReadTimeout("temporary poll timeout")
            return Response(accepted.model_dump(mode="json"))

    monkeypatch.setattr(
        "magellan.bidding.client.httpx.AsyncClient",
        AsyncClient,
    )

    result = await BidClient(Cluster()).submit_and_wait(request)

    assert result.status == BidStatus.ACCEPTED
    assert AsyncClient.get_calls == 2
