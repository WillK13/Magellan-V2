from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from magellan.bidding.client import BidClient
from magellan.bidding.models import (
    BidRequest,
    BidStatus,
)
from magellan.carbon.store import CarbonStore
from magellan.config.models import (
    ClusterConfig,
    NodeConfig,
)
from magellan.config.policy_models import ScoringPolicy
from magellan.graph.topology import ClusterGraph
from magellan.migration.service import MigrationService
from magellan.models.types import ActionType
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import (
    LocalProcessRuntime,
)
from magellan.scheduler.scoring import evaluate_task
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)


class SchedulerService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        policy: ScoringPolicy,
        graph: ClusterGraph,
        carbon_store: CarbonStore,
        clock: MagellanClock,
        registry: PersistentTaskRegistry,
        runtime: LocalProcessRuntime,
        bid_client: BidClient,
        migration_service: MigrationService,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._policy = policy
        self._graph = graph
        self._carbon_store = carbon_store
        self._clock = clock
        self._registry = registry
        self._runtime = runtime
        self._bid_client = bid_client
        self._migration_service = migration_service

    async def _evaluate_task(
        self,
        task_id: str,
        trace_time,
    ) -> None:
        task = self._registry.scoring_profile(task_id)

        decision = evaluate_task(
            task=task,
            cluster=self._cluster,
            policy=self._policy,
            graph=self._graph,
            carbon_store=self._carbon_store,
            at_utc=trace_time,
        )

        print(
            f"[epoch] node={self._local_node.id} "
            f"task={task_id} "
            f"trace_time={trace_time.isoformat()}",
            flush=True,
        )

        for rank, action in enumerate(
            decision.ranked_actions,
            start=1,
        ):
            destination = (
                action.destination_node_id
                or task.current_node_id
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

        selected = decision.selected
        destination = (
            selected.destination_node_id
            or task.current_node_id
        )

        print(
            f"[selected] action={selected.action.value} "
            f"destination={destination} "
            f"reason={decision.reason}",
            flush=True,
        )

        if selected.action != ActionType.MIGRATE:
            return

        if selected.destination_node_id is None:
            raise RuntimeError(
                "Selected migration has no destination"
            )

        bid = BidRequest(
            bid_id=str(uuid4()),
            epoch_id=(
                f"{task_id}:"
                f"{int(trace_time.timestamp())}"
            ),
            task_id=task_id,
            source_node_id=self._local_node.id,
            destination_node_id=selected.destination_node_id,
            candidate=selected,
            submitted_at_utc=datetime.now(timezone.utc),
        )

        print(
            f"[bid-send] bid={bid.bid_id} "
            f"task={task_id} "
            f"destination={bid.destination_node_id}",
            flush=True,
        )

        result = await self._bid_client.submit_and_wait(bid)

        print(
            f"[bid-result] bid={result.bid_id} "
            f"task={task_id} "
            f"status={result.status.value}",
            flush=True,
        )

        if result.status == BidStatus.ACCEPTED:
            await self._migration_service.migrate(
                task_id=task_id,
                destination_node_id=(
                    selected.destination_node_id
                ),
            )

    async def run_epoch(self) -> None:
        await asyncio.to_thread(
            self._runtime.reconcile
        )

        task_ids = self._registry.running_owned_task_ids(
            self._local_node.id
        )

        if not task_ids:
            print(
                f"[scheduler] node={self._local_node.id} "
                f"has no running owned tasks",
                flush=True,
            )
            return

        trace_time = self._clock.now()

        for task_id in task_ids:
            try:
                await self._evaluate_task(
                    task_id,
                    trace_time,
                )
            except Exception as exc:
                print(
                    f"[scheduler-error] node="
                    f"{self._local_node.id} "
                    f"task={task_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    async def run(
        self,
        stop_event: asyncio.Event,
    ) -> None:
        while not stop_event.is_set():
            started = time.monotonic()

            await self.run_epoch()

            elapsed = time.monotonic() - started

            delay = max(
                0.0,
                self._cluster.epoch_seconds - elapsed,
            )

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=delay,
                )
            except asyncio.TimeoutError:
                pass
