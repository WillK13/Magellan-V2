from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import AccountingPolicy, ScoringPolicy
from magellan.graph.topology import ClusterGraph
from magellan.models.utils import bytes_to_gb, seconds_to_hours
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import pid_is_alive
from magellan.runtime.progress import load_progress
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import TaskStatus


class RuntimeAccountingService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        policy: ScoringPolicy,
        graph: ClusterGraph,
        carbon_store: CarbonStore,
        clock: MagellanClock,
        registry: PersistentTaskRegistry,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._policy = policy
        self._accounting_policy: AccountingPolicy = policy.accounting
        self._graph = graph
        self._carbon_store = carbon_store
        self._clock = clock
        self._registry = registry

    def _as_wall_datetime(
        self,
        value: datetime | None,
    ) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _progress_update(
        self,
        task_id: str,
        now_wall: datetime,
    ) -> dict:
        progress_file = self._registry.progress_file(task_id)
        if progress_file is None:
            return {}

        snapshot = load_progress(progress_file, task_id)
        if snapshot is None:
            return {}

        state = self._registry.get_state(task_id)
        snapshot_time = self._as_wall_datetime(
            snapshot.updated_at_utc
        )

        # Ignore stale or duplicate progress records.
        if (
            state.progress_updated_at_utc is not None
            and snapshot_time
            <= self._as_wall_datetime(state.progress_updated_at_utc)
        ):
            return {}

        rate = state.progress_rate_units_per_second

        if (
            state.progress_completed_units is not None
            and state.progress_updated_at_utc is not None
        ):
            elapsed = (
                snapshot_time
                - self._as_wall_datetime(
                    state.progress_updated_at_utc
                )
            ).total_seconds()
            delta = (
                snapshot.completed_units
                - state.progress_completed_units
            )

            if elapsed > 0 and delta > 0:
                observed_rate = delta / elapsed
                if rate is None:
                    rate = observed_rate
                else:
                    alpha = self._accounting_policy.progress_ema_alpha
                    rate = (
                        alpha * observed_rate
                        + (1.0 - alpha) * rate
                    )

        progress_fraction = None
        estimated_remaining = state.estimated_remaining_seconds

        if snapshot.total_units is not None:
            progress_fraction = min(
                1.0,
                snapshot.completed_units / snapshot.total_units,
            )

            remaining_units = max(
                0.0,
                snapshot.total_units - snapshot.completed_units,
            )

            if remaining_units == 0:
                estimated_remaining = 0.0
            elif rate is not None and rate > 0:
                remaining_wall_seconds = remaining_units / rate
                estimated_remaining = (
                    self._clock.evaluation_seconds_for_wall_seconds(
                        remaining_wall_seconds
                    )
                )

        return {
            "estimated_remaining_seconds": estimated_remaining,
            "progress_completed_units": snapshot.completed_units,
            "progress_total_units": snapshot.total_units,
            "progress_fraction": progress_fraction,
            "progress_rate_units_per_second": rate,
            "progress_updated_at_utc": snapshot_time,
        }

    def settle_task(
        self,
        task_id: str,
        now_wall: datetime | None = None,
        trace_now: pd.Timestamp | datetime | None = None,
    ):
        now_wall = self._as_wall_datetime(now_wall)
        trace_time = (
            pd.Timestamp(trace_now)
            if trace_now is not None
            else self._clock.now()
        )
        if trace_time.tzinfo is None:
            trace_time = trace_time.tz_localize("UTC")
        else:
            trace_time = trace_time.tz_convert("UTC")

        state = self._registry.get_state(task_id)

        if state.owner_node_id != self._local_node.id:
            return state

        progress_updates = self._progress_update(
            task_id=task_id,
            now_wall=now_wall,
        )

        if state.last_accounted_at_utc is None:
            return self._registry.record_accounting(
                task_id=task_id,
                last_accounted_at_utc=now_wall,
                **progress_updates,
            )

        last_wall = self._as_wall_datetime(
            state.last_accounted_at_utc
        )
        elapsed_wall_seconds = max(
            0.0,
            (now_wall - last_wall).total_seconds(),
        )
        elapsed_eval_seconds = (
            self._clock.evaluation_seconds_for_wall_seconds(
                elapsed_wall_seconds
            )
        )

        runtime_seconds = 0.0
        paused_seconds = 0.0
        compute_cost = 0.0
        compute_carbon = 0.0

        if (
            state.status == TaskStatus.RUNNING
            and state.pid is not None
            and pid_is_alive(state.pid)
        ):
            runtime_seconds = elapsed_eval_seconds
            compute_hours = seconds_to_hours(
                elapsed_eval_seconds
            )
            compute_cost = (
                self._local_node.compute_price_usd_per_hour
                * compute_hours
            )

            if elapsed_eval_seconds > 0:
                period_start = trace_time - pd.Timedelta(
                    seconds=elapsed_eval_seconds
                )
                intensity = self._carbon_store.average(
                    self._local_node.id,
                    period_start,
                    elapsed_eval_seconds,
                )
                effective_power = (
                    self._registry.get_definition(
                        task_id
                    ).profile.power_kw
                    * self._local_node.pue
                )
                compute_carbon = (
                    effective_power
                    * compute_hours
                    * intensity
                )

        elif (
            state.status == TaskStatus.PAUSED
            and state.pid is not None
            and pid_is_alive(state.pid)
        ):
            paused_seconds = elapsed_eval_seconds

        return self._registry.record_accounting(
            task_id=task_id,
            runtime_seconds=runtime_seconds,
            paused_seconds=paused_seconds,
            compute_cost_usd=compute_cost,
            compute_carbon_grams=compute_carbon,
            last_accounted_at_utc=now_wall,
            **progress_updates,
        )

    def record_migration(
        self,
        task_id: str,
        destination_node_id: str,
        transfer_bytes: int,
        migration_wall_seconds: float,
        at_utc: datetime | pd.Timestamp,
    ):
        destination = self._cluster.get_node(destination_node_id)
        edge = self._graph.edge(
            self._local_node.id,
            destination_node_id,
        )
        transfer_gb = bytes_to_gb(transfer_bytes)
        transfer_cost = (
            transfer_gb
            * self._local_node.egress_price_usd_per_gb
        )

        network_energy_kwh = transfer_gb * (
            self._policy.migration.network_energy_kwh_per_gb_base
            + self._policy.migration.network_energy_kwh_per_gb_km
            * edge.distance_km
        )

        trace_time = pd.Timestamp(at_utc)
        if trace_time.tzinfo is None:
            trace_time = trace_time.tz_localize("UTC")
        else:
            trace_time = trace_time.tz_convert("UTC")

        source_intensity = self._carbon_store.value_at(
            self._local_node.id,
            trace_time,
        )
        destination_intensity = self._carbon_store.value_at(
            destination.id,
            trace_time,
        )
        network_carbon = network_energy_kwh * (
            source_intensity + destination_intensity
        ) / 2.0

        migration_seconds = (
            self._clock.evaluation_seconds_for_wall_seconds(
                migration_wall_seconds
            )
        )

        return self._registry.record_accounting(
            task_id=task_id,
            migration_seconds=migration_seconds,
            transfer_cost_usd=transfer_cost,
            transfer_carbon_grams=network_carbon,
            last_accounted_at_utc=datetime.now(timezone.utc),
        )

    async def run_once(self) -> int:
        processed = 0
        now_wall = datetime.now(timezone.utc)
        trace_now = self._clock.now()

        for state in self._registry.all_states():
            if (
                state.owner_node_id != self._local_node.id
                or state.status
                not in {TaskStatus.RUNNING, TaskStatus.PAUSED}
            ):
                continue

            await asyncio.to_thread(
                self.settle_task,
                state.task_id,
                now_wall,
                trace_now,
            )
            processed += 1

        return processed

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.run_once()

            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._accounting_policy.scan_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass
