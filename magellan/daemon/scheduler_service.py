from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from magellan.capabilities.checker import check_compatibility
from magellan.bidding.client import BidClient
from magellan.bidding.models import (
    BidRequest,
    BidStatus,
    TaskBidContext,
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
from magellan.policy.adaptive import AdaptivePolicyService
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
from magellan.telemetry.service import TelemetryService
from magellan.experiments.events import ExperimentEventJournal
from magellan.submission.catalog import TaskCatalogStore

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
        telemetry_service: TelemetryService | None = None,
        adaptive_policy_service: AdaptivePolicyService | None = None,
        experiment_journal: ExperimentEventJournal | None = None,
        task_catalog: TaskCatalogStore | None = None,
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
        self._telemetry_service = telemetry_service
        self._adaptive_policy_service = adaptive_policy_service
        self._experiment_journal = experiment_journal
        self._task_catalog = task_catalog
        self._broadcasted_completions: set[tuple[str, int, str | None]] = set()
        self._task_operation_locks: dict[str, asyncio.Lock] = {}


    def _task_bid_context(
        self,
        task,
        static_data_bytes: int,
        candidate=None,
        ranked_actions=None,
    ) -> TaskBidContext:
        fallback = None
        if candidate is not None and ranked_actions is not None:
            alternatives = [
                action
                for action in ranked_actions
                if not (
                    action.action == candidate.action
                    and action.destination_node_id
                    == candidate.destination_node_id
                    and action.source_node_id
                    == candidate.source_node_id
                )
            ]
            if alternatives:
                fallback = min(
                    alternatives,
                    key=lambda action: action.score,
                )

        opportunity_loss = 0.0
        if fallback is not None and candidate is not None:
            opportunity_loss = max(
                0.0,
                fallback.score - candidate.score,
            )

        telemetry_view = None
        telemetry_service = getattr(self, "_telemetry_service", None)
        if telemetry_service is not None:
            telemetry_view = telemetry_service.store.task_view(
                task.task_id,
                task.power_kw,
                self._policy.telemetry.task_stale_after_seconds,
            )

        return TaskBidContext(
            workload_type=task.workload_type,
            priority=task.priority,
            deadline_at_utc=task.deadline_at_utc,
            estimated_remaining_seconds=task.estimated_remaining_seconds,
            checkpoint_bytes=task.checkpoint_bytes,
            static_data_bytes=static_data_bytes,
            accumulated_cost_usd=task.accumulated_cost_usd,
            cost_cap_usd=task.cost_cap_usd,
            resource_request=task.resource_request,
            compatibility=task.compatibility,
            fallback_action=(
                fallback.action if fallback is not None else None
            ),
            fallback_destination_node_id=(
                fallback.destination_node_id
                if fallback is not None
                else None
            ),
            fallback_score=(
                fallback.score if fallback is not None else None
            ),
            opportunity_loss=opportunity_loss,
            effective_power_kw=(
                telemetry_view.effective_power_kw
                if telemetry_view is not None
                else task.power_kw
            ),
            power_source=(
                telemetry_view.effective_power_source
                if telemetry_view is not None
                else "configured_fallback"
            ),
            power_confidence=(
                telemetry_view.power_confidence
                if telemetry_view is not None
                else None
            ),
            telemetry_freshness=(
                telemetry_view.freshness.value
                if telemetry_view is not None
                else "unavailable"
            ),
        )

    def _compatible_destinations(self, task) -> set[str]:
        compatible: set[str] = set()
        for destination in self._graph.peers(self._local_node.id):
            result = check_compatibility(
                task.compatibility,
                destination.capabilities,
            )
            if result.compatible:
                compatible.add(destination.id)
            else:
                print(
                    f"[compatibility-prune] task={task.task_id} "
                    f"destination={destination.id} "
                    f"reasons={'; '.join(result.reasons)}",
                    flush=True,
                )
        return compatible

    def _task_operation_lock(self, task_id: str) -> asyncio.Lock:
        """Serialize scheduler and operator actions for one task."""
        return self._task_operation_locks.setdefault(
            task_id,
            asyncio.Lock(),
        )

    def _background_scheduling_enabled(self, task_id: str) -> bool:
        """Return whether autonomous scheduler epochs may act on this run.

        Operator-only runs are useful for controlled measurements that still
        need the production pause/migration endpoints but must not be moved by
        an unrelated background scheduler epoch before the measurement trigger.
        """
        if self._task_catalog is None:
            return True
        try:
            run = self._task_catalog.get_run(task_id)
        except KeyError:
            return True
        return run.labels.get("scheduler_mode") != "operator_only"

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

            if not self._background_scheduling_enabled(task_id):
                print(
                    f"[scheduler-skip] task={task_id} "
                    "reason=operator_only",
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

        checkpoint_summary = None
        checkpoint_error: CheckpointValidationError | None = None
        try:
            checkpoint_summary = await asyncio.to_thread(
                self._checkpoint_manager.validate,
                task_id,
            )
        except CheckpointValidationError as exc:
            checkpoint_error = exc

        task = self._registry.scoring_profile(
            task_id,
            checkpoint_bytes=(
                checkpoint_summary.size_bytes
                if checkpoint_summary is not None
                else None
            ),
        )
        if self._telemetry_service is not None:
            task = self._telemetry_service.enrich_profile(task)

        static_data_bytes_by_destination: dict[str, int] = {}
        if checkpoint_summary is None:
            # A task may not have produced its first application checkpoint
            # yet. Continue and pause remain valid local decisions, while
            # migration is a hard-infeasible action until a complete
            # checkpoint exists. Do not skip the whole scheduler epoch.
            compatible_destination_ids: set[str] = set()
            print(
                f"[scheduler-local-only] task={task_id} "
                f"checkpoint_not_ready={checkpoint_error}",
                flush=True,
            )
        else:
            print(
                f"[checkpoint-size] task={task_id} "
                f"bytes={checkpoint_summary.size_bytes} "
                f"files={checkpoint_summary.file_count}",
                flush=True,
            )

            compatible_destination_ids = self._compatible_destinations(task)
            if (
                self._telemetry_service is not None
                and self._policy.telemetry.refresh_edges_before_decision
                and compatible_destination_ids
            ):
                await self._telemetry_service.ensure_edges_fresh(
                    compatible_destination_ids
                )
            for destination in self._graph.peers(
                self._local_node.id
            ):
                if destination.id not in compatible_destination_ids:
                    continue
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

        telemetry_confidence = 0.0
        if self._telemetry_service is not None:
            telemetry_view = self._telemetry_service.store.task_view(
                task.task_id,
                task.power_kw,
                self._policy.telemetry.task_stale_after_seconds,
            )
            telemetry_confidence = telemetry_view.power_confidence or 0.0

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
            adaptive_service=self._adaptive_policy_service,
            telemetry_confidence=telemetry_confidence,
            compatible_destination_ids=compatible_destination_ids,
        )

        print(
            f"[epoch] node={self._local_node.id} "
            f"task={task_id} "
            f"trace_time={trace_time.isoformat()}",
            flush=True,
        )

        if decision.policy_metadata:
            effective = decision.policy_metadata["effective_weights"]
            multipliers = decision.policy_metadata["multipliers"]
            print(
                f"[adaptive-policy] task={task_id} "
                f"weights=({effective['time']:.4f},"
                f"{effective['carbon']:.4f},"
                f"{effective['cost']:.4f}) "
                f"multipliers=({multipliers['time']:.3f},"
                f"{multipliers['carbon']:.3f},"
                f"{multipliers['cost']:.3f})",
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

        if self._experiment_journal is not None:
            state_snapshot = self._registry.get_state(task_id)
            trace_datetime = (
                trace_time.to_pydatetime()
                if hasattr(trace_time, "to_pydatetime")
                else trace_time
            )
            self._experiment_journal.append(
                "scheduler_decision",
                task_id=task_id,
                generation=state_snapshot.generation,
                trace_time_utc=trace_datetime,
                payload={
                    "task_profile": task.model_dump(mode="json"),
                    "state": state_snapshot.model_dump(mode="json"),
                    "checkpoint": (
                        {
                            "size_bytes": checkpoint_summary.size_bytes,
                            "file_count": checkpoint_summary.file_count,
                        }
                        if checkpoint_summary is not None
                        else None
                    ),
                    "static_data_bytes_by_destination": (
                        static_data_bytes_by_destination
                    ),
                    "decision": decision.model_dump(mode="json"),
                },
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
            task_context=self._task_bid_context(
                task,
                static_data_bytes_by_destination.get(
                    selected.destination_node_id,
                    0,
                ),
                candidate=selected,
                ranked_actions=decision.ranked_actions,
            ),
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

        definition = self._registry.get_definition(task_id)
        compatibility = check_compatibility(
            definition.profile.compatibility,
            self._cluster.get_node(destination_node_id).capabilities,
        )
        if not compatibility.compatible:
            raise RuntimeError(
                "Incompatible migration destination: "
                + "; ".join(compatibility.reasons)
            )

        if (
            self._telemetry_service is not None
            and self._policy.telemetry.refresh_edges_before_decision
        ):
            await self._telemetry_service.ensure_edges_fresh(
                {destination_node_id}
            )

        checkpoint_summary = await asyncio.to_thread(
            self._checkpoint_manager.validate,
            task_id,
        )
        task = self._registry.scoring_profile(
            task_id,
            checkpoint_bytes=checkpoint_summary.size_bytes,
        )
        if self._telemetry_service is not None:
            task = self._telemetry_service.enrich_profile(task)
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
            compatible_destination_ids={destination_node_id},
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
            task_context=self._task_bid_context(
                task,
                missing_bytes,
                candidate=candidate,
                ranked_actions=decision.ranked_actions,
            ),
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
                    adaptive_policy=(
                        self._adaptive_policy_service.store.get(
                            state.task_id
                        )
                        if self._adaptive_policy_service is not None
                        else None
                    ),
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
