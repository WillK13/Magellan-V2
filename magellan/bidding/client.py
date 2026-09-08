from __future__ import annotations

import asyncio
import time

import httpx

from magellan.bidding.models import (
    BidRecord,
    BidRequest,
    BidStatus,
)
from magellan.config.models import ClusterConfig


class BidClient:
    def __init__(self, cluster: ClusterConfig) -> None:
        self._cluster = cluster

    def _base_url(self, node_id: str) -> str:
        destination = self._cluster.get_node(node_id)
        return (
            f"http://{destination.internal_ip}:"
            f"{self._cluster.api_port}"
        )

    async def submit_and_wait(
        self,
        request: BidRequest,
    ) -> BidRecord:
        base_url = self._base_url(
            request.destination_node_id
        )
        timeout = httpx.Timeout(
            self._cluster.request_timeout_seconds
        )
        # Leave enough room for the auction window plus transient request
        # failures. The destination's BidStore is idempotent by bid_id, so
        # retrying a POST whose response was lost is safe and lets the source
        # recover an already-persisted/accepted bid instead of orphaning its
        # reservation.
        total_wait_seconds = (
            self._cluster.bid_window_seconds
            + 3 * self._cluster.request_timeout_seconds
            + 3
        )
        deadline = time.monotonic() + total_wait_seconds
        last_transport_error: Exception | None = None

        def retryable_http_error(exc: httpx.HTTPError) -> bool:
            if isinstance(exc, httpx.TransportError):
                return True
            if isinstance(exc, httpx.HTTPStatusError):
                return exc.response.status_code >= 500
            return False

        async with httpx.AsyncClient(timeout=timeout) as client:
            record: BidRecord | None = None

            while record is None:
                try:
                    response = await client.post(
                        f"{base_url}/bids",
                        json=request.model_dump(mode="json"),
                    )
                    response.raise_for_status()
                    record = BidRecord.model_validate(response.json())
                except httpx.HTTPError as exc:
                    if not retryable_http_error(exc):
                        raise
                    last_transport_error = exc
                    print(
                        f"[bid-submit-retry] bid={request.bid_id} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Timed out submitting bid {request.bid_id}; "
                            f"last transport error: {type(exc).__name__}: {exc}"
                        ) from exc
                    await asyncio.sleep(0.25)

            while record.status == BidStatus.PENDING:
                if time.monotonic() >= deadline:
                    detail = (
                        ""
                        if last_transport_error is None
                        else (
                            "; last transport error: "
                            f"{type(last_transport_error).__name__}: "
                            f"{last_transport_error}"
                        )
                    )
                    raise RuntimeError(
                        f"Timed out waiting for bid {request.bid_id}{detail}"
                    )

                await asyncio.sleep(0.25)
                try:
                    response = await client.get(
                        f"{base_url}/bids/{request.bid_id}"
                    )
                    response.raise_for_status()
                    record = BidRecord.model_validate(
                        response.json()
                    )
                except httpx.HTTPError as exc:
                    if not retryable_http_error(exc):
                        raise
                    last_transport_error = exc
                    print(
                        f"[bid-poll-retry] bid={request.bid_id} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

        return record

    async def renew(
        self,
        bid_id: str,
        destination_node_id: str,
    ) -> BidRecord:
        timeout = httpx.Timeout(
            self._cluster.request_timeout_seconds
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self._base_url(destination_node_id)}"
                f"/bids/{bid_id}/renew"
            )
            response.raise_for_status()

        return BidRecord.model_validate(response.json())

    async def cancel(
        self,
        bid_id: str,
        destination_node_id: str,
        reason: str,
    ) -> BidRecord:
        timeout = httpx.Timeout(
            self._cluster.request_timeout_seconds
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self._base_url(destination_node_id)}"
                f"/bids/{bid_id}/cancel",
                params={"reason": reason},
            )
            response.raise_for_status()

        return BidRecord.model_validate(response.json())
