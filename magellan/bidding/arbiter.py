from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Protocol

from magellan.capabilities.checker import check_compatibility
from magellan.capabilities.models import (
    NodeRuntimeCapabilities,
    TaskCompatibilityRequirements,
)
from magellan.bidding.models import (
    AuctionStrategy,
    BidRecord,
    BidStatus,
)
from magellan.bidding.ranking import rank_bids
from magellan.bidding.resources import (
    ResourceLedger,
    sum_requests,
)
from magellan.bidding.store import BidStore
from magellan.config.models import NodeResourceCapacity
from magellan.config.policy_models import AuctionPolicy
from magellan.models.types import TaskResourceRequest


class CapacityRegistry(Protocol):
    def count_owned(self, node_id: str) -> int:
        ...

    def owned_resource_requests(
        self,
        node_id: str,
    ) -> list[TaskResourceRequest]:
        ...


class BidArbiter:
    """Destination-local auction for task bids.

    Tasks compete for this node's scarce capacity; the node never bids
    for tasks. The selected policy ranks bids within a fixed window, while
    admission always enforces task slots and configured CPU, memory, GPU,
    and accelerator constraints.
    """

    def __init__(
        self,
        store: BidStore,
        registry: CapacityRegistry,
        local_node_id: str,
        capacity: int,
        bid_window_seconds: float,
        node_resources: NodeResourceCapacity | None = None,
        auction_policy: AuctionPolicy | None = None,
        node_capabilities: NodeRuntimeCapabilities | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._local_node_id = local_node_id
        self._capacity = capacity
        self._bid_window_seconds = bid_window_seconds
        self._node_resources = (
            node_resources or NodeResourceCapacity()
        )
        self._auction_policy = auction_policy or AuctionPolicy()
        self._node_capabilities = (
            node_capabilities or NodeRuntimeCapabilities()
        )
        self._strategy = AuctionStrategy(
            self._auction_policy.strategy
        )

    @property
    def strategy(self) -> AuctionStrategy:
        return self._strategy

    def _owned_requests(self) -> list[TaskResourceRequest]:
        method = getattr(
            self._registry,
            "owned_resource_requests",
            None,
        )
        if method is None:
            return []
        return method(self._local_node_id)

    @staticmethod
    def _request(record: BidRecord) -> TaskResourceRequest:
        if record.task_context is None:
            return TaskResourceRequest()
        return record.task_context.resource_request

    async def _available_resources(
        self,
    ) -> tuple[int, ResourceLedger]:
        currently_owned = self._registry.count_owned(
            self._local_node_id
        )
        reservations = await self._store.active_reservations()
        available_slots = max(
            0,
            self._capacity
            - currently_owned
            - len(reservations),
        )
        used_requests = self._owned_requests() + [
            self._request(record)
            for record in reservations
        ]
        ledger = ResourceLedger.from_capacity(
            self._node_resources,
            used=sum_requests(used_requests),
        )
        return available_slots, ledger

    async def status(self) -> dict:
        available_slots, ledger = await self._available_resources()
        return {
            "node_id": self._local_node_id,
            "strategy": self._strategy.value,
            "task_slot_capacity": self._capacity,
            "available_task_slots": available_slots,
            "resource_capacity": self._node_resources.model_dump(
                mode="json"
            ),
            "runtime_capabilities": self._node_capabilities.model_dump(
                mode="json"
            ),
            **ledger.snapshot(),
            "credits": await self._store.credits_snapshot(),
        }

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
        credits = await self._store.credits_snapshot()
        ranked = rank_bids(
            bids=window_bids,
            strategy=self._strategy,
            credits=credits,
            node_resources=self._node_resources,
            policy=self._auction_policy,
            now_utc=now,
        )
        available_slots, ledger = await self._available_resources()
        full_ledger = ResourceLedger.from_capacity(
            self._node_resources
        )

        accepted_ids: set[str] = set()
        competition_rejected_ids: set[str] = set()
        selected_task_ids: set[str] = set()

        for rank, item in enumerate(ranked, start=1):
            bid = item.bid
            request = self._request(bid)
            individually_feasible, infeasible_reason = (
                full_ledger.compatible(request)
            )
            requirements = (
                bid.task_context.compatibility
                if bid.task_context is not None
                else None
            )
            compatibility = check_compatibility(
                requirements or TaskCompatibilityRequirements(),
                self._node_capabilities,
            )
            status = BidStatus.REJECTED
            resource_fit = individually_feasible
            earns_credit = False

            if bid.task_id in selected_task_ids:
                reason = (
                    "Duplicate bid for a task already selected in "
                    "this auction window"
                )
            elif not compatibility.compatible:
                reason = "; ".join(compatibility.reasons)
            elif not individually_feasible:
                reason = infeasible_reason or (
                    "Task resource request is incompatible with "
                    "destination capacity"
                )
            elif available_slots <= 0:
                reason = (
                    "Destination task slots are already owned or reserved"
                )
                earns_credit = True
            else:
                fits_remaining, contention_reason = ledger.compatible(
                    request
                )
                if not fits_remaining:
                    reason = contention_reason or (
                        "Destination resources are already owned or reserved"
                    )
                    earns_credit = True
                else:
                    status = BidStatus.ACCEPTED
                    reason = (
                        f"Selected by {self._strategy.value} task auction; "
                        "destination resources are reserved"
                    )
                    available_slots -= 1
                    ledger.consume(request)
                    accepted_ids.add(bid.bid_id)
                    selected_task_ids.add(bid.task_id)

            if status == BidStatus.REJECTED and earns_credit:
                competition_rejected_ids.add(bid.bid_id)

            await self._store.decide(
                bid_id=bid.bid_id,
                status=status,
                reason=reason,
                now_utc=now,
                auction_strategy=self._strategy,
                auction_rank=rank,
                auction_credit_before=credits.get(
                    bid.task_id,
                    0.0,
                ),
                resource_fit=resource_fit,
                compatibility_fit=compatibility.compatible,
                compatibility_reasons=compatibility.reasons,
                auction_metrics={
                    **item.metrics,
                    "requested_cpu_cores": request.cpu_cores,
                    "requested_memory_mb": request.memory_mb,
                    "requested_gpu_count": request.gpu_count,
                },
            )

        await self._store.apply_credit_outcomes(
            accepted_bid_ids=accepted_ids,
            rejected_competition_bid_ids=(
                competition_rejected_ids
            ),
            increment=self._auction_policy.credit_increment,
            maximum=self._auction_policy.credit_max,
            accepted_decay=(
                self._auction_policy.accepted_credit_decay
            ),
        )

        for item in ranked:
            decided = await self._store.get(item.bid.bid_id)
            assert decided is not None
            print(
                f"[task-auction] node={self._local_node_id} "
                f"strategy={self._strategy.value} "
                f"rank={decided.auction_rank} "
                f"bid={decided.bid_id} "
                f"task={decided.task_id} "
                f"source={decided.source_node_id} "
                f"score={decided.candidate.score:.6f} "
                f"credit={decided.auction_credit_before:.2f}->"
                f"{decided.auction_credit_after:.2f} "
                f"status={decided.status.value} "
                f"reason={decided.decision_reason}",
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
