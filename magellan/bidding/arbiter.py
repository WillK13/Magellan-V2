from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Protocol

from magellan.bidding.models import BidStatus
from magellan.bidding.store import BidStore


class CapacityRegistry(Protocol):
    def count_owned(self, node_id: str) -> int:
        ...


class BidArbiter:
    """
    Collects bids during a fixed window and accepts the lowest
    scores up to the destination's available capacity. Accepted
    bids become expiring capacity reservations.
    """

    def __init__(
        self,
        store: BidStore,
        registry: CapacityRegistry,
        local_node_id: str,
        capacity: int,
        bid_window_seconds: float,
    ) -> None:
        self._store = store
        self._registry = registry
        self._local_node_id = local_node_id
        self._capacity = capacity
        self._bid_window_seconds = bid_window_seconds

    async def run_once(
        self,
        now_utc: datetime | None = None,
    ) -> bool:
        now = now_utc or datetime.now(timezone.utc)
        expired = await self._store.expire_reservations(now)

        for record in expired:
            print(
                f"[reservation-expired] node={self._local_node_id} "
                f"bid={record.bid_id} task={record.task_id}",
                flush=True,
            )

        pending = await self._store.pending_records()

        if not pending:
            return bool(expired)

        first_received = pending[0].received_at_utc
        window_end = first_received + timedelta(
            seconds=self._bid_window_seconds
        )

        if now < window_end:
            return bool(expired)

        window_bids = [
            bid
            for bid in pending
            if bid.received_at_utc <= window_end
        ]

        ranked = sorted(
            window_bids,
            key=lambda bid: (
                bid.candidate.score,
                bid.received_at_utc,
                bid.bid_id,
            ),
        )

        currently_owned = self._registry.count_owned(
            self._local_node_id
        )
        active_reservations = (
            await self._store.active_reservation_count()
        )
        available_slots = max(
            0,
            self._capacity
            - currently_owned
            - active_reservations,
        )

        winner_ids = {
            bid.bid_id
            for bid in ranked[:available_slots]
        }

        for bid in ranked:
            if bid.bid_id in winner_ids:
                status = BidStatus.ACCEPTED
                reason = (
                    "Lowest score within the bid window and "
                    "destination capacity is reserved"
                )
            else:
                status = BidStatus.REJECTED

                if available_slots == 0:
                    reason = (
                        "Destination has no unreserved capacity"
                    )
                else:
                    reason = (
                        "Another bid had a lower scheduling score"
                    )

            decided = await self._store.decide(
                bid_id=bid.bid_id,
                status=status,
                reason=reason,
                now_utc=now,
            )

            print(
                f"[arbiter] node={self._local_node_id} "
                f"bid={decided.bid_id} "
                f"task={decided.task_id} "
                f"source={decided.source_node_id} "
                f"score={decided.candidate.score:.6f} "
                f"status={decided.status.value} "
                f"expires={decided.reservation_expires_at_utc}",
                flush=True,
            )

        return True

    async def run(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            progressed = await self.run_once()

            if progressed:
                continue

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=0.25,
                )
            except asyncio.TimeoutError:
                pass
