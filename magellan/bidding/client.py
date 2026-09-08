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

    def _control_request_timeout_seconds(self) -> float:
        """Return the bounded timeout for auction-critical HTTP requests.

        General peer/telemetry requests use ``request_timeout_seconds``. Bids
        are different: the destination daemon can be CPU-starved by the very
        physical workload whose capacity it is arbitrating. Give the control
        RPC enough room to survive several normal request-timeout intervals
        plus one auction window without changing the global peer timeout.
        """
        return max(
            self._cluster.request_timeout_seconds,
            3 * self._cluster.bid_window_seconds
            + self._cluster.request_timeout_seconds,
        )

    def _total_wait_seconds(self) -> float:
        """Bound one bid from submission through an observable decision."""
        return max(
            self._control_request_timeout_seconds()
            + self._cluster.bid_window_seconds
            + self._cluster.request_timeout_seconds,
            self._cluster.reservation_renew_interval_seconds
            + self._cluster.bid_window_seconds
            + self._cluster.request_timeout_seconds,
        )

    @staticmethod
    def _retryable_http_error(exc: httpx.HTTPError) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    async def submit_and_wait(
        self,
        request: BidRequest,
    ) -> BidRecord:
        base_url = self._base_url(
            request.destination_node_id
        )
        timeout = httpx.Timeout(
            self._control_request_timeout_seconds()
        )
        deadline = time.monotonic() + self._total_wait_seconds()
        last_transport_error: Exception | None = None
        resubmitted_after_not_found = False

        async with httpx.AsyncClient(timeout=timeout) as client:
            record: BidRecord | None = None

            try:
                response = await client.post(
                    f"{base_url}/bids",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
                record = BidRecord.model_validate(response.json())
            except httpx.HTTPError as exc:
                if not self._retryable_http_error(exc):
                    raise
                last_transport_error = exc
                print(
                    f"[bid-submit-recover] bid={request.bid_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

            # A timed-out POST may already have been persisted by the
            # destination. Recover by reading the durable bid before sending
            # another copy. Only a confirmed 404 permits one idempotent
            # resubmission of the same bid_id.
            while record is None:
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
                        f"Timed out recovering bid {request.bid_id}{detail}"
                    )

                await asyncio.sleep(0.25)
                try:
                    response = await client.get(
                        f"{base_url}/bids/{request.bid_id}"
                    )
                    response.raise_for_status()
                    record = BidRecord.model_validate(response.json())
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        if not self._retryable_http_error(exc):
                            raise
                        last_transport_error = exc
                        print(
                            f"[bid-recover-retry] bid={request.bid_id} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        continue

                    if resubmitted_after_not_found:
                        continue

                    resubmitted_after_not_found = True
                    print(
                        f"[bid-resubmit-after-404] bid={request.bid_id}",
                        flush=True,
                    )
                    try:
                        response = await client.post(
                            f"{base_url}/bids",
                            json=request.model_dump(mode="json"),
                        )
                        response.raise_for_status()
                        record = BidRecord.model_validate(response.json())
                    except httpx.HTTPError as submit_exc:
                        if not self._retryable_http_error(submit_exc):
                            raise
                        last_transport_error = submit_exc
                        print(
                            f"[bid-submit-recover] bid={request.bid_id} "
                            f"error={type(submit_exc).__name__}: {submit_exc}",
                            flush=True,
                        )
                    continue
                except httpx.HTTPError as exc:
                    if not self._retryable_http_error(exc):
                        raise
                    last_transport_error = exc
                    print(
                        f"[bid-recover-retry] bid={request.bid_id} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )

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
                    if not self._retryable_http_error(exc):
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
