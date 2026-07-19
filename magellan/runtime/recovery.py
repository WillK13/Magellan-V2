from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from magellan.config.policy_models import RecoveryPolicy
from magellan.runtime.checkpoint import (
    CheckpointManager,
    CheckpointValidationError,
)
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import TaskStatus


class FailureRecoveryService:
    def __init__(
        self,
        local_node_id: str,
        policy: RecoveryPolicy,
        registry: PersistentTaskRegistry,
        runtime: LocalProcessRuntime,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._local_node_id = local_node_id
        self._policy = policy
        self._registry = registry
        self._runtime = runtime
        self._checkpoint_manager = checkpoint_manager
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, task_id: str) -> asyncio.Lock:
        if task_id not in self._locks:
            self._locks[task_id] = asyncio.Lock()
        return self._locks[task_id]

    def _backoff_seconds(self, attempts: int) -> float:
        return min(
            self._policy.max_backoff_seconds,
            self._policy.initial_backoff_seconds
            * (2 ** max(0, attempts)),
        )

    async def _recover_one(
        self,
        task_id: str,
        now_utc: datetime,
    ) -> bool:
        async with self._lock_for(task_id):
            state = self._registry.get_state(task_id)

            if (
                state.owner_node_id != self._local_node_id
                or state.status != TaskStatus.FAILED
                or state.recovery_exhausted
            ):
                return False

            if not self._policy.enabled:
                self._registry.mark_recovery_exhausted(
                    task_id,
                    "Automatic recovery is disabled",
                )
                return True

            if (
                state.recovery_attempts
                >= self._policy.max_restart_attempts
            ):
                self._registry.mark_recovery_exhausted(
                    task_id,
                    (
                        "Automatic recovery exhausted after "
                        f"{state.recovery_attempts} attempts: "
                        f"{state.last_error}"
                    ),
                )
                print(
                    f"[recovery-exhausted] task={task_id} "
                    f"attempts={state.recovery_attempts}",
                    flush=True,
                )
                return True

            if state.next_recovery_at_utc is None:
                recover_at = now_utc + timedelta(
                    seconds=self._backoff_seconds(
                        state.recovery_attempts
                    )
                )
                self._registry.schedule_recovery(
                    task_id,
                    recover_at,
                )
                print(
                    f"[recovery-scheduled] task={task_id} "
                    f"at={recover_at.isoformat()}",
                    flush=True,
                )
                return True

            if now_utc < state.next_recovery_at_utc:
                return False

            try:
                await asyncio.to_thread(
                    self._checkpoint_manager.validate,
                    task_id,
                )
            except CheckpointValidationError as exc:
                self._registry.mark_recovery_exhausted(
                    task_id,
                    f"No valid checkpoint for recovery: {exc}",
                )
                print(
                    f"[recovery-unavailable] task={task_id} "
                    f"error={exc}",
                    flush=True,
                )
                return True

            self._registry.begin_recovery(task_id)

            try:
                runtime_state = await asyncio.to_thread(
                    self._runtime.start,
                    task_id,
                )
            except Exception as exc:
                # LocalProcessRuntime persists FAILED on start failure.
                print(
                    f"[recovery-failed] task={task_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                return True

            print(
                f"[recovery-complete] task={task_id} "
                f"pid={runtime_state.pid} "
                f"attempt={runtime_state.recovery_attempts}",
                flush=True,
            )
            return True

    async def run_once(
        self,
        now_utc: datetime | None = None,
    ) -> bool:
        now = now_utc or datetime.now(timezone.utc)
        progressed = False

        for task_id in self._registry.failed_owned_task_ids(
            self._local_node_id
        ):
            progressed = (
                await self._recover_one(task_id, now)
                or progressed
            )

        return progressed

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.run_once()

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._policy.scan_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass
