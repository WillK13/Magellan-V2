from __future__ import annotations

import asyncio
import shutil
#from pathlib import Path
from uuid import uuid4

from magellan.config.models import (
    ClusterConfig,
    NodeConfig,
)
from magellan.migration.client import (
    MigrationClient,
    OwnershipBroadcaster,
)
from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    OwnershipUpdate,
)
from magellan.migration.transfer import (
    RsyncCheckpointTransfer,
)
from magellan.runtime.local_process import (
    LocalProcessRuntime,
)
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)

from datetime import datetime

from magellan.runtime.checkpoint import CheckpointManager


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
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._registry = registry
        self._runtime = runtime
        self._transfer = transfer
        self._client = client
        self._broadcaster = broadcaster
        self._checkpoint_manager = checkpoint_manager

        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._locks:
            self._locks[task_id] = asyncio.Lock()

        return self._locks[task_id]

    async def migrate(
        self,
        task_id: str,
        destination_node_id: str,
        migration_at_utc: datetime,
    ) -> bool:
        async with self._lock_for(task_id):
            original_state = self._registry.get_state(task_id)

            if (
                original_state.owner_node_id
                != self._local_node.id
            ):
                print(
                    f"[migration-skip] task={task_id} "
                    f"owner={original_state.owner_node_id}",
                    flush=True,
                )
                return False

            migration_id = str(uuid4())
            new_generation = original_state.generation + 1

            self._registry.mark_migrating(
                task_id,
                migration_id,
            )

            try:
                print(
                    f"[migration-start] task={task_id} "
                    f"source={self._local_node.id} "
                    f"destination={destination_node_id} "
                    f"generation={new_generation}",
                    flush=True,
                )

                await asyncio.to_thread(
                    self._runtime.stop,
                    task_id,
                )

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
                    task_id=task_id,
                    source_node_id=self._local_node.id,
                    destination_node_id=destination_node_id,
                    generation=new_generation,
                    migration_at_utc=migration_at_utc,
                )

                response = await self._client.activate(
                    activation_request
                )

                if not response.activated:
                    raise RuntimeError(
                        response.error
                        or "Destination rejected activation"
                    )

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
                    )
                )

                print(
                    f"[migration-complete] task={task_id} "
                    f"owner={destination_node_id} "
                    f"generation={new_generation}",
                    flush=True,
                )

                return True

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

    async def activate_incoming(
        self,
        request: MigrationActivationRequest,
    ) -> MigrationActivationResponse:
        if (
            request.destination_node_id
            != self._local_node.id
        ):
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

        if (
            state.last_migration_id == request.migration_id
            and state.owner_node_id == self._local_node.id
            and state.pid is not None
        ):
            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=state.generation,
                activated=True,
                pid=state.pid,
            )

        incoming_checkpoint = (
            self._registry.state_root
            / "incoming"
            / request.migration_id
            / request.task_id
            / "checkpoint"
        )

        local_checkpoint = (
            self._registry.checkpoint_directory(
                request.task_id
            )
        )

        try:
            if not incoming_checkpoint.is_dir():
                raise FileNotFoundError(
                    f"Incoming checkpoint not found: "
                    f"{incoming_checkpoint}"
                )

            local_checkpoint.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            backup = local_checkpoint.with_name(
                "checkpoint.previous"
            )

            if backup.exists():
                shutil.rmtree(backup)

            if local_checkpoint.exists():
                local_checkpoint.rename(backup)

            shutil.copytree(
                incoming_checkpoint,
                local_checkpoint,
            )

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

            self._registry.claim_local(
                task_id=request.task_id,
                generation=request.generation,
                migration_id=request.migration_id,
                migration_at_utc=request.migration_at_utc,
            )

            runtime_state = await asyncio.to_thread(
                self._runtime.start,
                request.task_id,
            )

            print(
                f"[migration-activated] "
                f"task={request.task_id} "
                f"node={self._local_node.id} "
                f"pid={runtime_state.pid} "
                f"generation={request.generation}",
                flush=True,
            )

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

            self._registry.mark_failed(
                request.task_id,
                error,
            )

            return MigrationActivationResponse(
                migration_id=request.migration_id,
                task_id=request.task_id,
                destination_node_id=self._local_node.id,
                generation=request.generation,
                activated=False,
                error=error,
            )
