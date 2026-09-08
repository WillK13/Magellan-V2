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
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("GET", "http://test")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "response error",
                request=self.request,
                response=httpx.Response(
                    self.status_code,
                    request=self.request,
                ),
            )

    def json(self) -> dict:
        return self._payload


class Cluster:
    request_timeout_seconds = 0.1
    bid_window_seconds = 1
    reservation_renew_interval_seconds = 3
    api_port = 8040

    @staticmethod
    def get_node(node_id: str):
        assert node_id == "ethiopia"
        return SimpleNamespace(internal_ip="10.0.0.2")


def test_bid_control_timeout_and_deadline_are_longer_than_peer_timeout() -> None:
    class ProductionLikeCluster(Cluster):
        request_timeout_seconds = 5
        bid_window_seconds = 10
        reservation_renew_interval_seconds = 60

    client = BidClient(ProductionLikeCluster())

    assert client._control_request_timeout_seconds() == 35
    assert client._total_wait_seconds() == 75


@pytest.mark.asyncio
async def test_submit_recovers_lost_post_response_by_get_without_retry_storm(
    monkeypatch,
) -> None:
    request = bid_request()
    accepted = bid_record(request, BidStatus.ACCEPTED)

    class AsyncClient:
        post_calls = 0
        get_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, json):
            type(self).post_calls += 1
            raise httpx.ReadTimeout("response lost")

        async def get(self, url):
            type(self).get_calls += 1
            return Response(accepted.model_dump(mode="json"))

    monkeypatch.setattr(
        "magellan.bidding.client.httpx.AsyncClient",
        AsyncClient,
    )

    result = await BidClient(Cluster()).submit_and_wait(request)

    assert result.status == BidStatus.ACCEPTED
    assert AsyncClient.post_calls == 1
    assert AsyncClient.get_calls == 1


@pytest.mark.asyncio
async def test_submit_resubmits_once_after_destination_confirms_404(
    monkeypatch,
) -> None:
    request = bid_request()
    pending = bid_record(request, BidStatus.PENDING)
    accepted = bid_record(request, BidStatus.ACCEPTED)

    class AsyncClient:
        post_calls = 0
        get_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url, json):
            type(self).post_calls += 1
            if type(self).post_calls == 1:
                raise httpx.ReadTimeout("response lost before persistence")
            return Response(pending.model_dump(mode="json"))

        async def get(self, url):
            type(self).get_calls += 1
            if type(self).get_calls == 1:
                return Response({}, status_code=404)
            return Response(accepted.model_dump(mode="json"))

    monkeypatch.setattr(
        "magellan.bidding.client.httpx.AsyncClient",
        AsyncClient,
    )

    result = await BidClient(Cluster()).submit_and_wait(request)

    assert result.status == BidStatus.ACCEPTED
    assert AsyncClient.post_calls == 2
    assert AsyncClient.get_calls == 2


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
