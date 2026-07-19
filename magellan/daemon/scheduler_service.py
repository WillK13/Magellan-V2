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
from magellan.migration.client import OwnershipBroadcaster
from magellan.migration.models import OwnershipUpdate
from magellan.migration.service import MigrationService
from magellan.models.types import ActionType
from magellan.runtime.accounting import RuntimeAccountingService
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import (
    LocalProcessRuntime,
)
from magellan.runtime.pause import PauseService
from magellan.scheduler.scoring import evaluate_task
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)
from magellan.state.task_models import TaskStatus

from magellan.runtime.checkpoint import (
    CheckpointManager,
    CheckpointValidationError,
)

from magellan.artifacts.prefetch import (
    ArtifactPrefetchService,
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
        checkpoint_manager: CheckpointManager,
        prefetch_service: ArtifactPrefetchService,
        broadcaster: OwnershipBroadcaster,
        pause_service: PauseService,
        accounting_service: RuntimeAccountingService,
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
        self._checkpoint_manager = checkpoint_manager
        self._prefetch_service = prefetch_service
        self._broadcaster = broadcaster
        self._pause_service = pause_service
        self._accounting_service = accounting_service
        self._broadcasted_completions: set[tuple[str, int, str | None]] = set()
        self._task_operation_locks: dict[str, asyncio.Lock] = {}

    def _task_operation_lock(self, task_id: str) -> asyncio.Lock:
        """Serialize scheduler and operator actions for one task."""
        return self._task_operation_locks.setdefault(
            task_id,
            asyncio.Lock(),
        )

    async def _evaluate_task(
        self,
        task_id: str,
        trace_time,
    ) -> None:
        async with self._task_operation_lock(task_id):
            state = self._registry.get_state(task_id)
            if (
                state.owner_node_id != self._local_node.id
                or state.status != TaskStatus.RUNNING
            ):
                print(
                    f"[scheduler-skip] task={task_id} "
                    f"owner={state.owner_node_id} "
                    f"status={state.status.value}",
                    flush=True,
                )
                return

            await self._evaluate_task_locked(task_id, trace_time)

    async def _evaluate_task_locked(
        self,
        task_id: str,
        trace_time,
    ) -> None:
        await asyncio.to_thread(
            self._accounting_service.settle_task,
            task_id,
            None,
            trace_time,
        )

        try:
            checkpoint_summary = await asyncio.to_thread(
                self._checkpoint_manager.validate,
                task_id,
            )
        except CheckpointValidationError as exc:
            print(
                f"[scheduler-skip] task={task_id} "
                f"checkpoint_not_ready={exc}",
                flush=True,
            )
            return

        task = self._registry.scoring_profile(
            task_id,
            checkpoint_bytes=checkpoint_summary.size_bytes,
        )

        print(
            f"[checkpoint-size] task={task_id} "
            f"bytes={checkpoint_summary.size_bytes} "
            f"files={checkpoint_summary.file_count}",
            flush=True,
        )

        static_data_bytes_by_destination: dict[str, int] = {}

        for destination in self._graph.peers(
            self._local_node.id
        ):
            missing_bytes = (
                await self._prefetch_service.missing_bytes(
                    task_id=task_id,
                    destination_node_id=destination.id,
                )
            )

            static_data_bytes_by_destination[
                destination.id
            ] = missing_bytes

            print(
                f"[artifact-plan] task={task_id} "
                f"destination={destination.id} "
                f"missing_bytes={missing_bytes}",
                flush=True,
            )

        decision = evaluate_task(
            task=task,
            cluster=self._cluster,
            policy=self._policy,
            graph=self._graph,
            carbon_store=self._carbon_store,
            at_utc=trace_time,
            static_data_bytes_by_destination=(
                static_data_bytes_by_destination
            ),
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

        if selected.action == ActionType.CONTINUE:
            return

        if selected.action == ActionType.PAUSE:
            await self._pause_service.pause(
                task_id=task_id,
                at_utc=trace_time,
                idle_seconds=float(
                    selected.details.get(
                        "idle_seconds",
                        self._policy.pause.idle_seconds,
                    )
                ),
                reason=decision.reason,
            )
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
                migration_at_utc=trace_time.to_pydatetime(),
                bid_id=result.bid_id,
            )

    async def request_pause(
        self,
        task_id: str,
        idle_seconds: float | None = None,
        reason: str = "Operator requested pause",
    ) -> dict:
        async with self._task_operation_lock(task_id):
            return await self._request_pause_locked(
                task_id=task_id,
                idle_seconds=idle_seconds,
                reason=reason,
            )

    async def _request_pause_locked(
        self,
        task_id: str,
        idle_seconds: float | None,
        reason: str,
    ) -> dict:
        state = self._registry.get_state(task_id)
        if state.owner_node_id != self._local_node.id:
            raise RuntimeError(
                f"Cannot pause {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status != TaskStatus.RUNNING:
            raise RuntimeError(
                f"Cannot pause {task_id}; status is "
                f"{state.status.value}"
            )

        requested_idle = (
            self._policy.pause.idle_seconds
            if idle_seconds is None
            else idle_seconds
        )
        if requested_idle < 0:
            raise ValueError("idle_seconds must be non-negative")
        if requested_idle > self._policy.pause.max_pause_window_seconds:
            raise ValueError(
                "idle_seconds exceeds max_pause_window_seconds"
            )

        trace_time = self._clock.now()
        await asyncio.to_thread(
            self._accounting_service.settle_task,
            task_id,
            None,
            trace_time,
        )
        paused = await self._pause_service.pause(
            task_id=task_id,
            at_utc=trace_time,
            idle_seconds=requested_idle,
            reason=reason,
        )
        return paused.model_dump(mode="json")

    async def request_resume(self, task_id: str) -> dict:
        async with self._task_operation_lock(task_id):
            return await self._request_resume_locked(task_id)

    async def _request_resume_locked(self, task_id: str) -> dict:
        state = self._registry.get_state(task_id)
        if state.owner_node_id != self._local_node.id:
            raise RuntimeError(
                f"Cannot resume {task_id}; owner is "
                f"{state.owner_node_id}"
            )
        if state.status != TaskStatus.PAUSED:
            raise RuntimeError(
                f"Cannot resume {task_id}; status is "
                f"{state.status.value}"
            )

        await asyncio.to_thread(
            self._accounting_service.settle_task,
            task_id,
        )
        resumed = await self._pause_service.resume(task_id)
        return resumed.model_dump(mode="json")

    async def request_migration(
        self,
        task_id: str,
        destination_node_id: str,
    ) -> dict:
        """Operator-triggered migration that still uses scoring and bidding."""
        async with self._task_operation_lock(task_id):
            return await self._request_migration_locked(
                task_id=task_id,
                destination_node_id=destination_node_id,
            )

    async def _request_migration_locked(
        self,
        task_id: str,
        destination_node_id: str,
    ) -> dict:
        state = self._registry.get_state(task_id)

        # A background scheduler epoch may have won the per-task lock and
        # completed this exact migration while the operator request waited.
        # Treat that outcome as idempotent success instead of submitting a
        # duplicate bid or reporting a misleading conflict.
        if state.owner_node_id == destination_node_id:
            return {
                "bid": None,
                "migrated": True,
                "already_migrated": True,
                "state": state.model_dump(mode="json"),
            }

        if state.owner_node_id != self._local_node.id:
            raise RuntimeError(
                f"Cannot migrate {task_id}; owner is "
                f"{state.owner_node_id}"
            )

        if state.status != TaskStatus.RUNNING:
            raise RuntimeError(
                f"Cannot migrate {task_id}; status is "
                f"{state.status.value}"
            )

        self._cluster.get_node(destination_node_id)

        if destination_node_id == self._local_node.id:
            raise ValueError(
                "Destination must differ from the local node"
            )

        checkpoint_summary = await asyncio.to_thread(
            self._checkpoint_manager.validate,
            task_id,
        )
        task = self._registry.scoring_profile(
            task_id,
            checkpoint_bytes=checkpoint_summary.size_bytes,
        )
        missing_bytes = await self._prefetch_service.missing_bytes(
            task_id=task_id,
            destination_node_id=destination_node_id,
        )
        trace_time = self._clock.now()
        decision = evaluate_task(
            task=task,
            cluster=self._cluster,
            policy=self._policy,
            graph=self._graph,
            carbon_store=self._carbon_store,
            at_utc=trace_time,
            static_data_bytes_by_destination={
                destination_node_id: missing_bytes,
            },
        )

        candidate = next(
            (
                action
                for action in decision.ranked_actions
                if (
                    action.action == ActionType.MIGRATE
                    and action.destination_node_id
                    == destination_node_id
                )
            ),
            None,
        )

        if candidate is None:
            raise RuntimeError(
                f"No feasible migration candidate for "
                f"{task_id} -> {destination_node_id}"
            )

        bid = BidRequest(
            bid_id=str(uuid4()),
            epoch_id=(
                f"operator:{task_id}:"
                f"{int(trace_time.timestamp())}"
            ),
            task_id=task_id,
            source_node_id=self._local_node.id,
            destination_node_id=destination_node_id,
            candidate=candidate,
            submitted_at_utc=datetime.now(timezone.utc),
        )
        result = await self._bid_client.submit_and_wait(bid)

        if result.status != BidStatus.ACCEPTED:
            return {
                "bid": result.model_dump(mode="json"),
                "migrated": False,
            }

        migrated = await self._migration_service.migrate(
            task_id=task_id,
            destination_node_id=destination_node_id,
            migration_at_utc=trace_time.to_pydatetime(),
            bid_id=result.bid_id,
        )

        return {
            "bid": result.model_dump(mode="json"),
            "migrated": migrated,
            "state": self._registry.get_state(
                task_id
            ).model_dump(mode="json"),
        }

    async def _broadcast_completed_states(self) -> None:
        for state in self._registry.all_states():
            if (
                state.owner_node_id != self._local_node.id
                or state.status != TaskStatus.COMPLETED
            ):
                continue

            key = (
                state.task_id,
                state.generation,
                state.final_output_manifest_sha256,
            )

            if key in self._broadcasted_completions:
                continue

            await self._broadcaster.broadcast(
                OwnershipUpdate(
                    task_id=state.task_id,
                    owner_node_id=state.owner_node_id,
                    generation=state.generation,
                    status=TaskStatus.COMPLETED,
                    completed_at_utc=state.completed_at_utc,
                    final_output_manifest_sha256=(
                        state.final_output_manifest_sha256
                    ),
                    final_output_bytes=state.final_output_bytes,
                    accounting=state.accounting_snapshot(),
                )
            )
            self._broadcasted_completions.add(key)

            print(
                f"[completion-broadcast] task={state.task_id} "
                f"owner={state.owner_node_id} "
                f"generation={state.generation}",
                flush=True,
            )

    async def run_epoch(self) -> None:
        await asyncio.to_thread(self._runtime.reconcile)
        await self._broadcast_completed_states()

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
