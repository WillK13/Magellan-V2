from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

from magellan.config.policy_models import PausePolicy
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry


class PauseService:
    def __init__(
        self,
        local_node_id: str,
        policy: PausePolicy,
        clock: MagellanClock,
        registry: PersistentTaskRegistry,
        runtime: LocalProcessRuntime,
        accounting_service=None,
    ) -> None:
        self._local_node_id = local_node_id
        self._policy = policy
        self._clock = clock
        self._registry = registry
        self._runtime = runtime
        self._accounting_service = accounting_service

    async def pause(
        self,
        task_id: str,
        at_utc: datetime | pd.Timestamp,
        idle_seconds: float,
        reason: str,
    ):
        if idle_seconds < 0:
            raise ValueError("idle_seconds must be non-negative")

        pause_at = pd.Timestamp(at_utc)
        if pause_at.tzinfo is None:
            pause_at = pause_at.tz_localize("UTC")
        else:
            pause_at = pause_at.tz_convert("UTC")

        resume_at = pause_at + pd.Timedelta(seconds=idle_seconds)
        resume_wall_at = datetime.now(timezone.utc) + timedelta(
            seconds=self._clock.wall_seconds_for_evaluation_seconds(
                idle_seconds
            )
        )

        return await asyncio.to_thread(
            self._runtime.pause,
            task_id,
            pause_at.to_pydatetime(),
            resume_at.to_pydatetime(),
            resume_wall_at,
            reason,
        )

    async def resume(self, task_id: str):
        if self._accounting_service is not None:
            await asyncio.to_thread(
                self._accounting_service.settle_task,
                task_id,
            )

        return await asyncio.to_thread(
            self._runtime.resume,
            task_id,
        )

    async def run_once(
        self,
        at_utc: datetime | pd.Timestamp | None = None,
    ) -> int:
        now = pd.Timestamp(at_utc) if at_utc is not None else self._clock.now()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")

        resumed = 0

        for task_id in self._registry.paused_owned_task_ids(
            self._local_node_id
        ):
            state = self._registry.get_state(task_id)

            due = False

            if state.resume_wall_at_utc is not None:
                wall_deadline = state.resume_wall_at_utc
                if wall_deadline.tzinfo is None:
                    wall_deadline = wall_deadline.replace(
                        tzinfo=timezone.utc
                    )
                else:
                    wall_deadline = wall_deadline.astimezone(
                        timezone.utc
                    )
                due = datetime.now(timezone.utc) >= wall_deadline
            elif state.resume_at_utc is not None:
                resume_at = pd.Timestamp(state.resume_at_utc)
                if resume_at.tzinfo is None:
                    resume_at = resume_at.tz_localize("UTC")
                else:
                    resume_at = resume_at.tz_convert("UTC")
                due = now >= resume_at

            if not due:
                continue

            try:
                await self.resume(task_id)
                resumed += 1
            except Exception as exc:
                self._registry.mark_failed(
                    task_id,
                    f"Pause resume failed: {type(exc).__name__}: {exc}",
                )

        return resumed

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
