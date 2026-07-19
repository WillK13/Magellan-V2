from __future__ import annotations

import asyncio
import shutil
from datetime import datetime
from uuid import uuid4

from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.prefetch import ArtifactPrefetchService
from magellan.bidding.client import BidClient
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.migration.client import (
    ActivationOutcomeUnknownError,
    MigrationClient,
    OwnershipBroadcaster,
)
from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    OwnershipUpdate,
)
from magellan.migration.transfer import RsyncCheckpointTransfer
from magellan.runtime.checkpoint import CheckpointManager
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry


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
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._locks:
            self._locks[task_id] = asyncio.Lock()
        return self._locks[task_id]

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

            try:
                try:
                    artifact_bindings = (
                        await self._prefetch_service.prefetch(
                            task_id=task_id,
                            destination_node_id=destination_node_id,
                            migration_id=migration_id,
                        )
                    )
                except Exception as exc:
                    print(
                        f"[prefetch-failed] task={task_id} "
                        f"destination={destination_node_id} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    return False

                self._registry.mark_migrating(
                    task_id,
                    migration_id,
                )

                print(
                    f"[migration-start] task={task_id} "
                    f"source={self._local_node.id} "
                    f"destination={destination_node_id} "
                    f"generation={new_generation} bid={bid_id}",
                    flush=True,
                )

                await asyncio.to_thread(
                    self._runtime.stop,
                    task_id,
                )
                source_was_stopped = True

                checkpoint_summary = await asyncio.to_thread(
                    self._checkpoint_manager.validate,
                    task_id,
                )

                print(
                    f"[checkpoint-valid] task={task_id} "
                    f"bytes={checkpoint_summary.size_bytes} "
                    f"files={checkpoint_summary.file_count}",
                    flush=True,
                )

                await asyncio.to_thread(
                    self._transfer.send,
                    task_id,
                    destination_node_id,
                    migration_id,
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
                )

                response = await self._client.activate(
                    activation_request
                )

                if not response.activated:
                    raise RuntimeError(
                        response.error
                        or "Destination rejected activation"
                    )

                activated = True
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
                        migration_at_utc=migration_at_utc,
                        artifact_digests={
                            binding.artifact_id: binding.digest
                            for binding in artifact_bindings
                        },
                    )
                )

                print(
                    f"[migration-complete] task={task_id} "
                    f"owner={destination_node_id} "
                    f"generation={new_generation}",
                    flush=True,
                )
                return True

            except ActivationOutcomeUnknownError as exc:
                # Never guess after a possibly-successful remote activation.
                # Restarting locally here could create split-brain execution.
                error = f"{type(exc).__name__}: {exc}"
                self._registry.mark_recovery_exhausted(
                    task_id,
                    (
                        "Migration activation outcome is unknown; "
                        "source remains stopped for safe reconciliation. "
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

        # Idempotent retry after a successful activation/consumption.
        if (
            state.last_migration_id == request.migration_id
            and state.owner_node_id == self._local_node.id
            and state.generation == request.generation
        ):
            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=state.generation,
                activated=True,
                pid=state.pid,
            )

        original_state = state.model_copy(deep=True)
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
            )

            runtime_state = await asyncio.to_thread(
                self._runtime.start,
                request.task_id,
            )
            runtime_started = True

            await self._bid_store.consume(request.bid_id)

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
