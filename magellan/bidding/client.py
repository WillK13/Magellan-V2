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

    async def submit_and_wait(
        self,
        request: BidRequest,
    ) -> BidRecord:
        destination = self._cluster.get_node(
            request.destination_node_id
        )

        base_url = (
            f"http://{destination.internal_ip}:"
            f"{self._cluster.api_port}"
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

            record = BidRecord.model_validate(
                response.json()
            )

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
