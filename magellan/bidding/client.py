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
        total_wait_seconds = (
            self._cluster.bid_window_seconds
            + self._cluster.request_timeout_seconds
            + 3
        )
        deadline = time.monotonic() + total_wait_seconds

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/bids",
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            record = BidRecord.model_validate(response.json())

            while record.status == BidStatus.PENDING:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for bid "
                        f"{request.bid_id}"
                    )

                await asyncio.sleep(0.25)
                response = await client.get(
                    f"{base_url}/bids/{request.bid_id}"
                )
                response.raise_for_status()
                record = BidRecord.model_validate(
                    response.json()
                )

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
