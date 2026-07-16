from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from magellan.bidding.client import BidClient
from magellan.bidding.models import BidRequest
from magellan.carbon.store import CarbonStore
from magellan.config.models import (
    ClusterConfig,
    NodeConfig,
)
from magellan.config.policy_models import ScoringPolicy
from magellan.graph.topology import ClusterGraph
from magellan.models.types import ActionType, TaskProfile
from magellan.runtime.clock import MagellanClock
from magellan.scheduler.scoring import evaluate_task
from magellan.state.task_registry import TaskRegistry


class SchedulerService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        policy: ScoringPolicy,
        graph: ClusterGraph,
        carbon_store: CarbonStore,
        clock: MagellanClock,
        registry: TaskRegistry,
        bid_client: BidClient,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._policy = policy
        self._graph = graph
        self._carbon_store = carbon_store
        self._clock = clock
        self._registry = registry
        self._bid_client = bid_client

    @staticmethod
    def _destination_label(
        task: TaskProfile,
        destination_node_id: str | None,
    ) -> str:
        return destination_node_id or task.current_node_id

    def _log_decision(
        self,
        task: TaskProfile,
        decision,
        trace_time,
    ) -> None:
        print(
            f"[epoch] node={self._local_node.id} "
            f"task={task.task_id} "
            f"trace_time={trace_time.isoformat()}",
            flush=True,
        )

        for rank, action in enumerate(
            decision.ranked_actions,
            start=1,
        ):
            destination = self._destination_label(
                task,
                action.destination_node_id,
            )

            print(
                f"[rank {rank}] "
                f"action={action.action.value} "
                f"destination={destination} "
                f"score={action.score:.6f} "
                f"time={action.time_seconds:.2f}s "
                f"carbon={action.carbon_grams:.6f}g "
                f"cost=${action.cost_usd:.6f}",
                flush=True,
            )

        selected_destination = self._destination_label(
            task,
            decision.selected.destination_node_id,
        )

        print(
            f"[selected] action="
            f"{decision.selected.action.value} "
            f"destination={selected_destination} "
            f"reason={decision.reason}",
            flush=True,
        )

    async def _evaluate_owned_task(
        self,
        task: TaskProfile,
        trace_time,
    ) -> None:
        decision = evaluate_task(
            task=task,
            cluster=self._cluster,
            policy=self._policy,
            graph=self._graph,
            carbon_store=self._carbon_store,
            at_utc=trace_time,
        )

        self._log_decision(
            task=task,
            decision=decision,
            trace_time=trace_time,
        )

        if decision.selected.action != ActionType.MIGRATE:
            print(
                f"[dry-run] task={task.task_id} "
                f"continues with local action "
                f"{decision.selected.action.value}",
                flush=True,
            )
            return

        destination_node_id = (
            decision.selected.destination_node_id
        )

        if destination_node_id is None:
            raise RuntimeError(
                "Selected migration has no destination"
            )

        epoch_id = (
            f"{task.task_id}:"
            f"{int(trace_time.timestamp())}"
        )

        bid = BidRequest(
            bid_id=str(uuid4()),
            epoch_id=epoch_id,
            task_id=task.task_id,
            source_node_id=self._local_node.id,
            destination_node_id=destination_node_id,
            candidate=decision.selected,
            submitted_at_utc=datetime.now(timezone.utc),
        )

        print(
            f"[bid-send] bid={bid.bid_id} "
            f"task={bid.task_id} "
            f"source={bid.source_node_id} "
            f"destination={bid.destination_node_id} "
            f"score={bid.candidate.score:.6f}",
            flush=True,
        )

        result = await self._bid_client.submit_and_wait(bid)

        print(
            f"[bid-result] bid={result.bid_id} "
            f"task={result.task_id} "
            f"status={result.status.value} "
            f"reason={result.decision_reason}",
            flush=True,
        )

        if result.status.value == "accepted":
            print(
                f"[dry-run] migration accepted for "
                f"task={task.task_id}, but no process or "
                f"ownership will move in this milestone",
                flush=True,
            )

    async def run_epoch(self) -> None:
        owned_tasks = self._registry.owned_tasks(
            self._local_node.id
        )

        if not owned_tasks:
            print(
                f"[scheduler] node={self._local_node.id} "
                f"is idle; listening for bids",
                flush=True,
            )
            return

        trace_time = self._clock.now()

        for task in owned_tasks:
            try:
                await self._evaluate_owned_task(
                    task=task,
                    trace_time=trace_time,
                )
            except Exception as exc:
                print(
                    f"[scheduler-error] node="
                    f"{self._local_node.id} "
                    f"task={task.task_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    async def run(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            epoch_started = time.monotonic()

            await self.run_epoch()

            elapsed = time.monotonic() - epoch_started

            sleep_seconds = max(
                0.0,
                self._cluster.epoch_seconds - elapsed,
            )

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=sleep_seconds,
                )
            except asyncio.TimeoutError:
                pass
