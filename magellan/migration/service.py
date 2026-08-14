from __future__ import annotations

import asyncio
import shutil
import time
from datetime import datetime, timezone
from uuid import uuid4

from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.prefetch import ArtifactPrefetchService
from magellan.bidding.client import BidClient
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.client import (
    ActivationOutcomeUnknownError,
    MigrationClient,
    OwnershipBroadcaster,
)
from magellan.migration.journal import MigrationJournal
from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    MigrationRecord,
    MigrationRole,
    MigrationStatus,
    OwnershipUpdate,
)
from magellan.migration.transfer import RsyncCheckpointTransfer
from magellan.runtime.accounting import RuntimeAccountingService
from magellan.runtime.checkpoint import CheckpointManager
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.policy.store import AdaptivePolicyStore
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import TaskStatus
from magellan.telemetry.store import TelemetryStore
from magellan.experiments.events import ExperimentEventJournal


class MigrationService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        registry: PersistentTaskRegistry,
        runtime: LocalProcessRuntime,
        transfer: RsyncCheckpointTransfer,
        client: MigrationClient,
        broadcaster: OwnershipBroadcaster,
        checkpoint_manager: CheckpointManager,
        artifact_manager: ArtifactManager,
        prefetch_service: ArtifactPrefetchService,
        bid_client: BidClient,
        bid_store: BidStore,
        accounting_service: RuntimeAccountingService | None = None,
        journal: MigrationJournal | None = None,
        reconciliation_policy: ReconciliationPolicy | None = None,
        telemetry_store: TelemetryStore | None = None,
        adaptive_policy_store: AdaptivePolicyStore | None = None,
        experiment_journal: ExperimentEventJournal | None = None,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._registry = registry
        self._runtime = runtime
        self._transfer = transfer
        self._client = client
        self._broadcaster = broadcaster
        self._checkpoint_manager = checkpoint_manager
        self._artifact_manager = artifact_manager
        self._prefetch_service = prefetch_service
        self._bid_client = bid_client
        self._bid_store = bid_store
        self._accounting_service = accounting_service
        self._journal = journal or MigrationJournal(registry.state_root)
        self._reconciliation_policy = (
            reconciliation_policy or ReconciliationPolicy()
        )
        self._telemetry_store = telemetry_store
        self._adaptive_policy_store = adaptive_policy_store
        self._experiment_journal = experiment_journal
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._locks:
            self._locks[task_id] = asyncio.Lock()
        return self._locks[task_id]

    def get_record(self, migration_id: str) -> MigrationRecord | None:
        return self._journal.get(migration_id)

    def list_records(self) -> list[MigrationRecord]:
        return self._journal.list_records()

    def _put_record(self, record: MigrationRecord) -> MigrationRecord:
        record.updated_at_utc = datetime.now(timezone.utc)
        return self._journal.put(record)

    def _set_record_status(
        self,
        migration_id: str,
        status: MigrationStatus,
        *,
        pid: int | None = None,
        error: str | None = None,
    ) -> MigrationRecord | None:
        record = self._journal.get(migration_id)
        if record is None:
            return None
        record.status = status
        record.pid = pid
        record.error = error
        return self._put_record(record)

    async def _query_remote_outcome(
        self,
        destination_node_id: str,
        migration_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[MigrationRecord | None, bool]:
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._reconciliation_policy.activation_resolution_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        last_record: MigrationRecord | None = None
        destination_reached = False
        while time.monotonic() < deadline:
            try:
                last_record = await self._client.status(
                    destination_node_id, migration_id
                )
                destination_reached = True
            except ActivationOutcomeUnknownError:
                last_record = None

            if last_record is not None and last_record.status in {
                MigrationStatus.ACTIVATED,
                MigrationStatus.ROLLED_BACK,
            }:
                return last_record, destination_reached

            await asyncio.sleep(
                self._reconciliation_policy.activation_resolution_poll_seconds
            )
        return last_record, destination_reached

    async def _finalize_source_success(
        self,
        *,
        task_id: str,
        destination_node_id: str,
        migration_id: str,
        new_generation: int,
        migration_at_utc: datetime,
        artifact_bindings,
    ) -> None:
        self._registry.mark_remote(
            task_id=task_id,
            owner_node_id=destination_node_id,
            generation=new_generation,
            migration_id=migration_id,
            migration_at_utc=migration_at_utc,
        )
        await self._broadcaster.broadcast(
            OwnershipUpdate(
                task_id=task_id,
                owner_node_id=destination_node_id,
                generation=new_generation,
                last_migration_id=migration_id,
                migration_at_utc=migration_at_utc,
                artifact_digests={
                    binding.artifact_id: binding.digest
                    for binding in artifact_bindings
                },
                accounting=self._registry.accounting_snapshot(task_id),
                adaptive_policy=(
                    self._adaptive_policy_store.get(task_id)
                    if self._adaptive_policy_store is not None
                    else None
                ),
            )
        )
        self._set_record_status(
            migration_id, MigrationStatus.ACTIVATED
        )

    async def _rollback_source_record(
        self,
        record: MigrationRecord,
        reason: str,
    ) -> None:
        original = record.original_state
        if original is not None:
            restored = original.model_copy(deep=True)
            restored.owner_node_id = self._local_node.id
            restored.generation = max(0, record.generation - 1)
            restored.status = TaskStatus.STOPPED
            restored.pid = None
            restored.last_error = reason
            self._registry.set_state(restored)
        else:
            self._registry.restore_local_after_failure(
                task_id=record.task_id,
                generation=max(0, record.generation - 1),
                error=reason,
            )
        state = self._registry.get_state(record.task_id)
        if state.owner_node_id == self._local_node.id and state.pid is None:
            try:
                await asyncio.to_thread(self._runtime.start, record.task_id)
            except Exception as exc:
                self._registry.mark_failed(
                    record.task_id,
                    f"Durable migration rollback restart failed: {exc}",
                )
        self._set_record_status(
            record.migration_id,
            MigrationStatus.ROLLED_BACK,
            error=reason,
        )

    async def _reservation_heartbeat(
        self,
        stop_event: asyncio.Event,
        bid_id: str,
        destination_node_id: str,
    ) -> None:
        interval = (
            self._cluster.reservation_renew_interval_seconds
        )

        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=interval,
                )
                return
            except asyncio.TimeoutError:
                pass

            try:
                renewed = await self._bid_client.renew(
                    bid_id=bid_id,
                    destination_node_id=destination_node_id,
                )
                print(
                    f"[reservation-renewed] bid={bid_id} "
                    f"expires={renewed.reservation_expires_at_utc}",
                    flush=True,
                )
            except Exception as exc:
                # Activation remains the authoritative lease check. A
                # transient heartbeat failure is logged, not guessed.
                print(
                    f"[reservation-renew-warning] bid={bid_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    async def _cancel_remote_reservation(
        self,
        bid_id: str,
        destination_node_id: str,
        reason: str,
    ) -> None:
        try:
            await self._bid_client.cancel(
                bid_id=bid_id,
                destination_node_id=destination_node_id,
                reason=reason,
            )
        except Exception as exc:
            print(
                f"[reservation-cancel-warning] bid={bid_id} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )

    async def migrate(
        self,
        task_id: str,
        destination_node_id: str,
        migration_at_utc: datetime,
        bid_id: str,
    ) -> bool:
        async with self._lock_for(task_id):
            original_state = self._registry.get_state(task_id)

            if original_state.owner_node_id != self._local_node.id:
                print(
                    f"[migration-skip] task={task_id} "
                    f"owner={original_state.owner_node_id}",
                    flush=True,
                )
                return False

            migration_id = str(uuid4())
            new_generation = original_state.generation + 1
            artifact_bindings = []
            self._put_record(
                MigrationRecord(
                    migration_id=migration_id,
                    bid_id=bid_id,
                    task_id=task_id,
                    source_node_id=self._local_node.id,
                    destination_node_id=destination_node_id,
                    generation=new_generation,
                    migration_at_utc=migration_at_utc,
                    role=MigrationRole.SOURCE,
                    status=MigrationStatus.PREPARING,
                    original_state=original_state,
                )
            )
            heartbeat_stop = asyncio.Event()
            heartbeat_task = asyncio.create_task(
                self._reservation_heartbeat(
                    stop_event=heartbeat_stop,
                    bid_id=bid_id,
                    destination_node_id=destination_node_id,
                ),
                name=f"reservation-{bid_id}",
            )
            source_was_stopped = False
            activated = False
            checkpoint_seconds = 0.0
            transfer_seconds = 0.0
            transfer_setup_seconds = 0.0
            transfer_wall_seconds = 0.0
            activation_seconds = 0.0
            restore_seconds = 0.0

            try:
                missing_artifact_bytes = 0
                try:
                    missing_artifact_bytes = (
                        await self._prefetch_service.missing_bytes(
                            task_id=task_id,
                            destination_node_id=destination_node_id,
                        )
                    )
                    artifact_bindings = (
                        await self._prefetch_service.prefetch(
                            task_id=task_id,
                            destination_node_id=destination_node_id,
                            migration_id=migration_id,
                        )
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self._set_record_status(
                        migration_id,
                        MigrationStatus.ROLLED_BACK,
                        error=error,
                    )
                    print(
                        f"[prefetch-failed] task={task_id} "
                        f"destination={destination_node_id} "
                        f"error={error}",
                        flush=True,
                    )
                    return False

                if self._accounting_service is not None:
                    await asyncio.to_thread(
                        self._accounting_service.settle_task,
                        task_id,
                    )

                self._registry.mark_migrating(
                    task_id,
                    migration_id,
                )
                self._set_record_status(
                    migration_id, MigrationStatus.TRANSFERRING
                )
                downtime_started = time.monotonic()

                print(
                    f"[migration-start] task={task_id} "
                    f"source={self._local_node.id} "
                    f"destination={destination_node_id} "
                    f"generation={new_generation} bid={bid_id}",
                    flush=True,
                )

                checkpoint_started = time.monotonic()
                pre_checkpoint_seconds = checkpoint_started - downtime_started
                await asyncio.to_thread(
                    self._runtime.stop,
                    task_id,
                )
                source_was_stopped = True

                checkpoint_summary = await asyncio.to_thread(
                    self._checkpoint_manager.validate,
                    task_id,
                )
                checkpoint_finished = time.monotonic()
                checkpoint_seconds = checkpoint_finished - checkpoint_started

                print(
                    f"[checkpoint-valid] task={task_id} "
                    f"bytes={checkpoint_summary.size_bytes} "
                    f"files={checkpoint_summary.file_count}",
                    flush=True,
                )

                transfer_started = time.monotonic()
                post_checkpoint_seconds = max(
                    0.0, transfer_started - checkpoint_finished
                )
                transfer_result = await asyncio.to_thread(
                    self._transfer.send,
                    task_id,
                    destination_node_id,
                    migration_id,
                )
                transfer_finished = time.monotonic()
                transfer_call_wall_seconds = max(
                    0.0, transfer_finished - transfer_started
                )
                transfer_seconds = float(
                    getattr(
                        transfer_result,
                        "duration_seconds",
                        transfer_call_wall_seconds,
                    )
                )
                transfer_setup_seconds = float(
                    getattr(transfer_result, "setup_seconds", 0.0)
                )
                transfer_wall_seconds = float(
                    getattr(
                        transfer_result,
                        "wall_seconds",
                        transfer_call_wall_seconds,
                    )
                )
                post_transfer_started = transfer_finished
                observed_transfer_bytes = int(
                    getattr(
                        transfer_result,
                        "transfer_bytes",
                        checkpoint_summary.size_bytes,
                    )
                )
                if (
                    self._telemetry_store is not None
                    and observed_transfer_bytes > 0
                    and transfer_seconds > 0
                ):
                    self._telemetry_store.record_transfer(
                        self._local_node.id,
                        destination_node_id,
                        observed_transfer_bytes,
                        transfer_seconds,
                    )

                if self._accounting_service is not None:
                    await asyncio.to_thread(
                        self._accounting_service.record_migration,
                        task_id,
                        destination_node_id,
                        (
                            missing_artifact_bytes
                            + checkpoint_summary.size_bytes
                        ),
                        time.monotonic() - downtime_started,
                        migration_at_utc,
                    )

                activation_request = MigrationActivationRequest(
                    migration_id=migration_id,
                    bid_id=bid_id,
                    task_id=task_id,
                    source_node_id=self._local_node.id,
                    destination_node_id=destination_node_id,
                    generation=new_generation,
                    migration_at_utc=migration_at_utc,
                    artifacts=artifact_bindings,
                    accounting=self._registry.accounting_snapshot(
                        task_id
                    ),
                    adaptive_policy=(
                        self._adaptive_policy_store.get(task_id)
                        if self._adaptive_policy_store is not None
                        else None
                    ),
                )

                self._set_record_status(
                    migration_id, MigrationStatus.ACTIVATING
                )
                activation_started = time.monotonic()
                post_transfer_seconds = max(
                    0.0, activation_started - post_transfer_started
                )
                response = await self._client.activate(
                    activation_request
                )
                activation_finished = time.monotonic()
                activation_seconds = activation_finished - activation_started
                destination_activation_seconds = (
                    response.activation_wall_seconds or 0.0
                )
                restore_seconds = response.restore_wall_seconds or 0.0

                if not response.activated:
                    raise RuntimeError(
                        response.error
                        or "Destination rejected activation"
                    )

                activated = True
                total_downtime_seconds = (
                    activation_finished - downtime_started
                )
                activation_non_restore_seconds = max(
                    0.0, activation_seconds - restore_seconds
                )
                migration_overhead_seconds = max(
                    0.0,
                    total_downtime_seconds
                    - checkpoint_seconds
                    - transfer_seconds
                    - restore_seconds,
                )
                instrumented_wall_seconds = (
                    pre_checkpoint_seconds
                    + checkpoint_seconds
                    + post_checkpoint_seconds
                    + transfer_call_wall_seconds
                    + post_transfer_seconds
                    + activation_seconds
                )
                timing_residual_seconds = max(
                    0.0, total_downtime_seconds - instrumented_wall_seconds
                )
                if self._telemetry_store is not None:
                    self._telemetry_store.record_migration_calibration(
                        source_node_id=self._local_node.id,
                        destination_node_id=destination_node_id,
                        checkpoint_seconds=checkpoint_seconds,
                        transfer_seconds=transfer_seconds,
                        restore_seconds=restore_seconds,
                        activation_seconds=activation_seconds,
                        total_downtime_seconds=total_downtime_seconds,
                        transfer_bytes=(
                            missing_artifact_bytes
                            + checkpoint_summary.size_bytes
                        ),
                    )
                if self._experiment_journal is not None:
                    self._experiment_journal.append(
                        "migration_completed",
                        task_id=task_id,
                        generation=new_generation,
                        trace_time_utc=migration_at_utc,
                        payload={
                            "migration_id": migration_id,
                            "bid_id": bid_id,
                            "source_node_id": self._local_node.id,
                            "destination_node_id": destination_node_id,
                            "checkpoint_bytes": checkpoint_summary.size_bytes,
                            "missing_artifact_bytes": missing_artifact_bytes,
                            "checkpoint_transfer_bytes": observed_transfer_bytes,
                            "total_accounted_transfer_bytes": (
                                missing_artifact_bytes
                                + checkpoint_summary.size_bytes
                            ),
                            "pre_checkpoint_seconds": pre_checkpoint_seconds,
                            "checkpoint_seconds": checkpoint_seconds,
                            "post_checkpoint_seconds": post_checkpoint_seconds,
                            "transfer_setup_seconds": transfer_setup_seconds,
                            "transfer_seconds": transfer_seconds,
                            "transfer_wall_seconds": transfer_wall_seconds,
                            "transfer_call_wall_seconds": transfer_call_wall_seconds,
                            "post_transfer_seconds": post_transfer_seconds,
                            "restore_seconds": restore_seconds,
                            "activation_seconds": activation_seconds,
                            "destination_activation_seconds": (
                                destination_activation_seconds
                            ),
                            "activation_transport_seconds": max(
                                0.0,
                                activation_seconds
                                - destination_activation_seconds,
                            ),
                            "activation_non_restore_seconds": (
                                activation_non_restore_seconds
                            ),
                            "migration_overhead_seconds": (
                                migration_overhead_seconds
                            ),
                            "instrumented_wall_seconds": instrumented_wall_seconds,
                            "timing_residual_seconds": timing_residual_seconds,
                            "total_downtime_seconds": total_downtime_seconds,
                        },
                    )
                await self._finalize_source_success(
                    task_id=task_id,
                    destination_node_id=destination_node_id,
                    migration_id=migration_id,
                    new_generation=new_generation,
                    migration_at_utc=migration_at_utc,
                    artifact_bindings=artifact_bindings,
                )

                print(
                    f"[migration-complete] task={task_id} "
                    f"owner={destination_node_id} "
                    f"generation={new_generation}",
                    flush=True,
                )
                return True

            except ActivationOutcomeUnknownError as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._set_record_status(
                    migration_id,
                    MigrationStatus.UNCERTAIN,
                    error=error,
                )
                remote, destination_reached = await self._query_remote_outcome(
                    destination_node_id, migration_id
                )
                if remote is not None and remote.status == MigrationStatus.ACTIVATED:
                    activated = True
                    await self._finalize_source_success(
                        task_id=task_id,
                        destination_node_id=destination_node_id,
                        migration_id=migration_id,
                        new_generation=new_generation,
                        migration_at_utc=migration_at_utc,
                        artifact_bindings=artifact_bindings,
                    )
                    print(
                        f"[migration-resolved] task={task_id} "
                        "outcome=activated",
                        flush=True,
                    )
                    return True
                if (
                    remote is not None
                    and remote.status == MigrationStatus.ROLLED_BACK
                ) or (destination_reached and remote is None):
                    record = self._journal.get(migration_id)
                    if record is not None:
                        await self._rollback_source_record(
                            record,
                            "Destination confirmed activation did not commit",
                        )
                    return False

                self._registry.mark_recovery_exhausted(
                    task_id,
                    (
                        "Migration activation outcome remains unknown; "
                        "durable reconciliation will retry. "
                        f"{error}"
                    ),
                )
                print(
                    f"[migration-uncertain] task={task_id} "
                    f"error={error}",
                    flush=True,
                )
                return False

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[migration-failed] task={task_id} "
                    f"error={error}",
                    flush=True,
                )
                if self._experiment_journal is not None:
                    self._experiment_journal.append(
                        "migration_failed",
                        task_id=task_id,
                        generation=new_generation,
                        trace_time_utc=migration_at_utc,
                        payload={
                            "migration_id": migration_id,
                            "bid_id": bid_id,
                            "source_node_id": self._local_node.id,
                            "destination_node_id": destination_node_id,
                            "checkpoint_seconds": checkpoint_seconds,
                            "transfer_seconds": transfer_seconds,
                            "restore_seconds": restore_seconds,
                            "activation_seconds": activation_seconds,
                            "error": error,
                        },
                    )

                self._registry.restore_local_after_failure(
                    task_id=task_id,
                    generation=original_state.generation,
                    error=error,
                )

                if source_was_stopped:
                    try:
                        await asyncio.to_thread(
                            self._runtime.start,
                            task_id,
                        )
                    except Exception as restart_error:
                        self._registry.mark_failed(
                            task_id,
                            (
                                f"Migration failed: {error}; "
                                f"local restart failed: "
                                f"{restart_error}"
                            ),
                        )

                self._set_record_status(
                    migration_id,
                    MigrationStatus.ROLLED_BACK,
                    error=error,
                )
                return False

            finally:
                heartbeat_stop.set()
                await asyncio.gather(
                    heartbeat_task,
                    return_exceptions=True,
                )

                if not activated:
                    await self._cancel_remote_reservation(
                        bid_id=bid_id,
                        destination_node_id=destination_node_id,
                        reason=(
                            f"Source migration {migration_id} "
                            "did not complete"
                        ),
                    )

    async def activate_incoming(
        self,
        request: MigrationActivationRequest,
    ) -> MigrationActivationResponse:
        async with self._lock_for(request.task_id):
            return await self._activate_incoming_locked(request)

    async def _activate_incoming_locked(
        self,
        request: MigrationActivationRequest,
    ) -> MigrationActivationResponse:
        activation_started = time.monotonic()
        if request.destination_node_id != self._local_node.id:
            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=request.generation,
                activated=False,
                error=(
                    f"Request destination is "
                    f"{request.destination_node_id}, "
                    f"but local node is {self._local_node.id}"
                ),
            )

        state = self._registry.get_state(request.task_id)
        existing_record = self._journal.get(request.migration_id)

        # Idempotent retry after a successful activation/consumption.
        if (
            request.adaptive_policy is not None
            and self._adaptive_policy_store is not None
        ):
            self._adaptive_policy_store.merge(request.adaptive_policy)

        if (
            state.last_migration_id == request.migration_id
            and state.owner_node_id == self._local_node.id
            and state.generation == request.generation
        ):
            if existing_record is None:
                self._put_record(
                    MigrationRecord(
                        migration_id=request.migration_id,
                        bid_id=request.bid_id,
                        task_id=request.task_id,
                        source_node_id=request.source_node_id,
                        destination_node_id=request.destination_node_id,
                        generation=request.generation,
                        migration_at_utc=request.migration_at_utc,
                        role=MigrationRole.DESTINATION,
                        status=MigrationStatus.ACTIVATED,
                        pid=state.pid,
                    )
                )
            else:
                self._set_record_status(
                    request.migration_id,
                    MigrationStatus.ACTIVATED,
                    pid=state.pid,
                )
            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=state.generation,
                activated=True,
                pid=state.pid,
            )

        if (
            existing_record is not None
            and existing_record.status == MigrationStatus.ROLLED_BACK
        ):
            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=request.generation,
                activated=False,
                error=existing_record.error or "Activation previously rolled back",
            )

        original_state = state.model_copy(deep=True)
        if existing_record is None:
            self._put_record(
                MigrationRecord(
                    migration_id=request.migration_id,
                    bid_id=request.bid_id,
                    task_id=request.task_id,
                    source_node_id=request.source_node_id,
                    destination_node_id=request.destination_node_id,
                    generation=request.generation,
                    migration_at_utc=request.migration_at_utc,
                    role=MigrationRole.DESTINATION,
                    status=MigrationStatus.ACTIVATING,
                    original_state=original_state,
                )
            )
        else:
            self._set_record_status(
                request.migration_id, MigrationStatus.ACTIVATING
            )
        incoming_checkpoint = (
            self._registry.state_root
            / "incoming"
            / request.migration_id
            / request.task_id
            / "checkpoint"
        )
        local_checkpoint = self._registry.checkpoint_directory(
            request.task_id
        )
        backup = local_checkpoint.with_name(
            f"checkpoint.rollback-{request.migration_id}"
        )
        checkpoint_replaced = False
        runtime_started = False

        try:
            await self._bid_store.begin_activation(
                bid_id=request.bid_id,
                task_id=request.task_id,
                source_node_id=request.source_node_id,
                destination_node_id=request.destination_node_id,
            )

            if not incoming_checkpoint.is_dir():
                raise FileNotFoundError(
                    f"Incoming checkpoint not found: "
                    f"{incoming_checkpoint}"
                )

            local_checkpoint.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if backup.exists():
                shutil.rmtree(backup)

            if local_checkpoint.exists():
                local_checkpoint.rename(backup)

            shutil.copytree(
                incoming_checkpoint,
                local_checkpoint,
            )
            checkpoint_replaced = True

            checkpoint_summary = await asyncio.to_thread(
                self._checkpoint_manager.validate,
                request.task_id,
            )

            print(
                f"[incoming-checkpoint-valid] "
                f"task={request.task_id} "
                f"bytes={checkpoint_summary.size_bytes} "
                f"files={checkpoint_summary.file_count}",
                flush=True,
            )

            self._artifact_manager.stage_bindings(
                request.task_id,
                request.artifacts,
            )

            self._registry.claim_local(
                task_id=request.task_id,
                generation=request.generation,
                migration_id=request.migration_id,
                migration_at_utc=request.migration_at_utc,
                artifact_digests={
                    binding.artifact_id: binding.digest
                    for binding in request.artifacts
                },
                accounting=request.accounting,
            )

            restore_started = time.monotonic()
            runtime_state = await asyncio.to_thread(
                self._runtime.start,
                request.task_id,
            )
            restore_seconds = time.monotonic() - restore_started
            runtime_started = True

            if (
                request.adaptive_policy is not None
                and self._adaptive_policy_store is not None
            ):
                self._adaptive_policy_store.merge(
                    request.adaptive_policy
                )

            await self._bid_store.consume(request.bid_id)
            self._set_record_status(
                request.migration_id,
                MigrationStatus.ACTIVATED,
                pid=runtime_state.pid,
            )

            print(
                f"[migration-activated] "
                f"task={request.task_id} "
                f"node={self._local_node.id} "
                f"pid={runtime_state.pid} "
                f"generation={request.generation}",
                flush=True,
            )

            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

            shutil.rmtree(
                self._registry.state_root
                / "incoming"
                / request.migration_id,
                ignore_errors=True,
            )

            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=request.generation,
                activated=True,
                pid=runtime_state.pid,
                restore_wall_seconds=restore_seconds,
                activation_wall_seconds=(
                    time.monotonic() - activation_started
                ),
            )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            if runtime_started:
                try:
                    await asyncio.to_thread(
                        self._runtime.stop,
                        request.task_id,
                    )
                except Exception:
                    pass

            if checkpoint_replaced and local_checkpoint.exists():
                shutil.rmtree(local_checkpoint, ignore_errors=True)

            if backup.exists():
                backup.rename(local_checkpoint)

            self._registry.set_state(original_state)
            self._set_record_status(
                request.migration_id,
                MigrationStatus.ROLLED_BACK,
                error=error,
            )

            try:
                await self._bid_store.cancel(
                    request.bid_id,
                    reason=(
                        "Destination activation rolled back: "
                        f"{error}"
                    ),
                )
            except Exception:
                pass

            shutil.rmtree(
                self._registry.state_root
                / "incoming"
                / request.migration_id,
                ignore_errors=True,
            )

            print(
                f"[migration-activation-rollback] "
                f"task={request.task_id} error={error}",
                flush=True,
            )

            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=request.generation,
                activated=False,
                error=error,
            )

    async def _reconcile_destination_record(
        self, record: MigrationRecord
    ) -> None:
        state = self._registry.get_state(record.task_id)
        committed = (
            state.owner_node_id == self._local_node.id
            and state.generation == record.generation
            and state.last_migration_id == record.migration_id
            and state.status in {
                TaskStatus.RUNNING,
                TaskStatus.PAUSED,
                TaskStatus.COMPLETED,
            }
        )
        if committed:
            bid = await self._bid_store.get(record.bid_id)
            try:
                if bid is not None and bid.status.value == "accepted":
                    await self._bid_store.begin_activation(
                        record.bid_id,
                        record.task_id,
                        record.source_node_id,
                        record.destination_node_id,
                    )
                    bid = await self._bid_store.get(record.bid_id)
                if bid is not None and bid.status.value == "activating":
                    await self._bid_store.consume(record.bid_id)
            except Exception:
                pass
            self._set_record_status(
                record.migration_id,
                MigrationStatus.ACTIVATED,
                pid=state.pid,
            )
            return

        # An interrupted destination transaction that did not establish a
        # live local owner is rolled back from its durable original snapshot.
        if (
            state.last_migration_id == record.migration_id
            and state.owner_node_id == self._local_node.id
            and state.pid is not None
        ):
            try:
                await asyncio.to_thread(self._runtime.stop, record.task_id)
            except Exception:
                pass

        checkpoint = self._registry.checkpoint_directory(record.task_id)
        backup = checkpoint.with_name(
            f"checkpoint.rollback-{record.migration_id}"
        )
        if backup.exists():
            if checkpoint.exists():
                shutil.rmtree(checkpoint, ignore_errors=True)
            backup.rename(checkpoint)
        if record.original_state is not None:
            self._registry.set_state(record.original_state)
        try:
            await self._bid_store.cancel(
                record.bid_id,
                reason="Daemon restart rolled back incomplete activation",
            )
        except Exception:
            pass
        shutil.rmtree(
            self._registry.state_root / "incoming" / record.migration_id,
            ignore_errors=True,
        )
        self._set_record_status(
            record.migration_id,
            MigrationStatus.ROLLED_BACK,
            error="Daemon restart rolled back incomplete activation",
        )

    async def _reconcile_source_record(
        self, record: MigrationRecord
    ) -> None:
        state = self._registry.get_state(record.task_id)
        if (
            state.owner_node_id == record.destination_node_id
            and state.generation >= record.generation
        ):
            self._set_record_status(
                record.migration_id, MigrationStatus.ACTIVATED
            )
            return

        if (
            record.status == MigrationStatus.PREPARING
            and state.owner_node_id == self._local_node.id
            and state.status != TaskStatus.MIGRATING
        ):
            self._set_record_status(
                record.migration_id,
                MigrationStatus.ROLLED_BACK,
                error="Migration ended before source quiesce",
            )
            return

        remote, destination_reached = await self._query_remote_outcome(
            record.destination_node_id,
            record.migration_id,
            timeout_seconds=(
                self._reconciliation_policy.activation_resolution_poll_seconds
            ),
        )
        if remote is not None and remote.status == MigrationStatus.ACTIVATED:
            await self._finalize_source_success(
                task_id=record.task_id,
                destination_node_id=record.destination_node_id,
                migration_id=record.migration_id,
                new_generation=record.generation,
                migration_at_utc=record.migration_at_utc,
                artifact_bindings=[],
            )
            return
        if (
            remote is not None
            and remote.status == MigrationStatus.ROLLED_BACK
        ) or (destination_reached and remote is None):
            await self._rollback_source_record(
                record,
                "Durable reconciliation confirmed no destination activation",
            )
            return
        self._set_record_status(
            record.migration_id,
            MigrationStatus.UNCERTAIN,
            error="Destination outcome still unreachable",
        )

    async def reconcile_durable_state(self) -> int:
        repaired = 0
        terminal = {
            MigrationStatus.ACTIVATED,
            MigrationStatus.ROLLED_BACK,
        }
        for record in self._journal.list_records():
            if record.status in terminal:
                continue
            async with self._lock_for(record.task_id):
                before = self._journal.get(record.migration_id)
                if record.role == MigrationRole.DESTINATION:
                    await self._reconcile_destination_record(record)
                else:
                    await self._reconcile_source_record(record)
                after = self._journal.get(record.migration_id)
                if before != after:
                    repaired += 1
        if repaired:
            print(
                f"[migration-reconcile] node={self._local_node.id} "
                f"records={repaired}",
                flush=True,
            )
        return repaired
