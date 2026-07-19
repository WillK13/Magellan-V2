from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from magellan.bidding.models import (
    BidRecord,
    BidRequest,
    BidStatus,
)


_ACTIVE_RESERVATION_STATUSES = {
    BidStatus.ACCEPTED,
    BidStatus.ACTIVATING,
}


class BidStore:
    """Concurrency-safe bid and capacity-lease storage for one daemon."""

    def __init__(
        self,
        reservation_ttl_seconds: float = 180.0,
        state_file: str | Path | None = None,
    ) -> None:
        self._state_file = Path(state_file) if state_file is not None else None
        self._records: dict[str, BidRecord] = {}
        self._lock = asyncio.Lock()
        self._reservation_ttl_seconds = reservation_ttl_seconds
        self._load()

    def _load(self) -> None:
        if self._state_file is None or not self._state_file.is_file():
            return
        raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        records = [BidRecord.model_validate(item) for item in raw]
        self._records = {record.bid_id: record for record in records}

    def _persist_locked(self) -> None:
        if self._state_file is None:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._state_file.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in self._records.values()],
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self._state_file)

    def _expire_locked(
        self,
        now_utc: datetime,
    ) -> list[BidRecord]:
        expired: list[BidRecord] = []

        for record in self._records.values():
            if record.status not in _ACTIVE_RESERVATION_STATUSES:
                continue

            expires = record.reservation_expires_at_utc

            if expires is None or expires > now_utc:
                continue

            record.status = BidStatus.EXPIRED
            record.decided_at_utc = now_utc
            record.decision_reason = (
                "Capacity reservation expired before activation completed"
            )
            expired.append(record.model_copy(deep=True))

        if expired:
            self._persist_locked()
        return expired

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
            self._persist_locked()
            return record.model_copy(deep=True)

    async def get(
        self,
        bid_id: str,
    ) -> BidRecord | None:
        async with self._lock:
            self._expire_locked(datetime.now(timezone.utc))
            record = self._records.get(bid_id)

            if record is None:
                return None

            return record.model_copy(deep=True)

    async def list_records(self) -> list[BidRecord]:
        async with self._lock:
            self._expire_locked(datetime.now(timezone.utc))
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

    async def active_reservation_count(self) -> int:
        async with self._lock:
            self._expire_locked(datetime.now(timezone.utc))
            return sum(
                record.status in _ACTIVE_RESERVATION_STATUSES
                for record in self._records.values()
            )

    async def expire_reservations(
        self,
        now_utc: datetime | None = None,
    ) -> list[BidRecord]:
        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            return self._expire_locked(now)

    async def decide(
        self,
        bid_id: str,
        status: BidStatus,
        reason: str,
        now_utc: datetime | None = None,
    ) -> BidRecord:
        if status not in {
            BidStatus.ACCEPTED,
            BidStatus.REJECTED,
        }:
            raise ValueError(
                "An arbiter decision must accept or reject a bid"
            )

        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            self._expire_locked(now)

            try:
                record = self._records[bid_id]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown bid: {bid_id}"
                ) from exc

            if record.status != BidStatus.PENDING:
                return record.model_copy(deep=True)

            record.status = status
            record.decided_at_utc = now
            record.decision_reason = reason

            if status == BidStatus.ACCEPTED:
                record.reservation_expires_at_utc = (
                    now
                    + timedelta(
                        seconds=self._reservation_ttl_seconds
                    )
                )

            self._persist_locked()
            return record.model_copy(deep=True)

    async def renew(
        self,
        bid_id: str,
        now_utc: datetime | None = None,
    ) -> BidRecord:
        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            self._expire_locked(now)
            record = self._records.get(bid_id)

            if record is None:
                raise KeyError(f"Unknown bid: {bid_id}")

            if record.status not in _ACTIVE_RESERVATION_STATUSES:
                raise RuntimeError(
                    f"Cannot renew bid {bid_id}; "
                    f"status={record.status.value}"
                )

            record.reservation_expires_at_utc = (
                now
                + timedelta(
                    seconds=self._reservation_ttl_seconds
                )
            )
            self._persist_locked()
            return record.model_copy(deep=True)

    async def begin_activation(
        self,
        bid_id: str,
        task_id: str,
        source_node_id: str,
        destination_node_id: str,
        now_utc: datetime | None = None,
    ) -> BidRecord:
        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            self._expire_locked(now)
            record = self._records.get(bid_id)

            if record is None:
                raise KeyError(f"Unknown bid: {bid_id}")

            expected = (
                record.task_id == task_id
                and record.source_node_id == source_node_id
                and record.destination_node_id
                == destination_node_id
            )

            if not expected:
                raise RuntimeError(
                    f"Reservation {bid_id} does not match "
                    "the activation request"
                )

            if record.status != BidStatus.ACCEPTED:
                raise RuntimeError(
                    f"Reservation {bid_id} is not activatable; "
                    f"status={record.status.value}"
                )

            record.status = BidStatus.ACTIVATING
            record.activation_started_at_utc = now
            record.decision_reason = (
                "Reservation claimed by destination activation"
            )
            self._persist_locked()
            return record.model_copy(deep=True)

    async def consume(
        self,
        bid_id: str,
        now_utc: datetime | None = None,
    ) -> BidRecord:
        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            record = self._records.get(bid_id)

            if record is None:
                raise KeyError(f"Unknown bid: {bid_id}")

            if record.status != BidStatus.ACTIVATING:
                raise RuntimeError(
                    f"Cannot consume bid {bid_id}; "
                    f"status={record.status.value}"
                )

            record.status = BidStatus.CONSUMED
            record.consumed_at_utc = now
            record.reservation_expires_at_utc = None
            record.decision_reason = (
                "Destination activation completed"
            )
            self._persist_locked()
            return record.model_copy(deep=True)

    async def cancel(
        self,
        bid_id: str,
        reason: str,
        now_utc: datetime | None = None,
    ) -> BidRecord:
        now = now_utc or datetime.now(timezone.utc)

        async with self._lock:
            record = self._records.get(bid_id)

            if record is None:
                raise KeyError(f"Unknown bid: {bid_id}")

            if record.status in {
                BidStatus.CONSUMED,
                BidStatus.REJECTED,
                BidStatus.EXPIRED,
                BidStatus.CANCELLED,
            }:
                return record.model_copy(deep=True)

            record.status = BidStatus.CANCELLED
            record.decided_at_utc = now
            record.reservation_expires_at_utc = None
            record.decision_reason = reason
            self._persist_locked()
            return record.model_copy(deep=True)
