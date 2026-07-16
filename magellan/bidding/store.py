from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from magellan.bidding.models import (
    BidRecord,
    BidRequest,
    BidStatus,
)


class BidStore:
    """Concurrency-safe in-memory bid storage for one daemon."""

    def __init__(self) -> None:
        self._records: dict[str, BidRecord] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        request: BidRequest,
    ) -> BidRecord:
        async with self._lock:
            existing = self._records.get(request.bid_id)

            if existing is not None:
                return existing.model_copy(deep=True)

            record = BidRecord(
                **request.model_dump(),
                status=BidStatus.PENDING,
                received_at_utc=datetime.now(timezone.utc),
            )

            self._records[record.bid_id] = record
            return record.model_copy(deep=True)

    async def get(
        self,
        bid_id: str,
    ) -> BidRecord | None:
        async with self._lock:
            record = self._records.get(bid_id)

            if record is None:
                return None

            return record.model_copy(deep=True)

    async def list_records(self) -> list[BidRecord]:
        async with self._lock:
            records = [
                record.model_copy(deep=True)
                for record in self._records.values()
            ]

        return sorted(
            records,
            key=lambda record: record.received_at_utc,
            reverse=True,
        )

    async def pending_records(self) -> list[BidRecord]:
        records = await self.list_records()

        return sorted(
            [
                record
                for record in records
                if record.status == BidStatus.PENDING
            ],
            key=lambda record: record.received_at_utc,
        )

    async def decide(
        self,
        bid_id: str,
        status: BidStatus,
        reason: str,
    ) -> BidRecord:
        if status == BidStatus.PENDING:
            raise ValueError(
                "A decision cannot set a bid back to pending"
            )

        async with self._lock:
            try:
                record = self._records[bid_id]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown bid: {bid_id}"
                ) from exc

            if record.status != BidStatus.PENDING:
                return record.model_copy(deep=True)

            record.status = status
            record.decided_at_utc = datetime.now(timezone.utc)
            record.decision_reason = reason

            return record.model_copy(deep=True)
