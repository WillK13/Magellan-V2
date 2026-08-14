from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import TelemetryPolicy
from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import TaskStatus
from magellan.telemetry.models import ProcessMeasurement, TaskTelemetryRecord
from magellan.telemetry.power import RaplPowerReader
from magellan.telemetry.process import ProcfsProcessSampler
from magellan.telemetry.store import TelemetryStore


class ProcessSampler(Protocol):
    def sample(self, process_group_id: int) -> ProcessMeasurement: ...


class TelemetryService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        policy: TelemetryPolicy,
        registry: PersistentTaskRegistry,
        store: TelemetryStore,
        process_sampler: ProcessSampler | None = None,
        node_power_reader: RaplPowerReader | None = None,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._policy = policy
        self._registry = registry
        self._store = store
        self._sampler = process_sampler or ProcfsProcessSampler()
        self._node_power_reader = node_power_reader or RaplPowerReader()
        self._previous_cpu: dict[str, tuple[float, datetime]] = {}
        self._last_edge_probe_monotonic = 0.0
        self._last_bandwidth_probe_monotonic: dict[str, float] = {}
        self._bandwidth_probe_payload = bytes(self._policy.edge_bandwidth_probe_bytes)

    @property
    def store(self) -> TelemetryStore:
        return self._store

    @staticmethod
    def _directory_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )

    def sample_task(self, task_id: str) -> TaskTelemetryRecord:
        state = self._registry.get_state(task_id)
        definition = self._registry.get_definition(task_id)
        now = datetime.now(timezone.utc)
        existing = self._store.task_record(task_id) or TaskTelemetryRecord(
            task_id=task_id,
            node_id=self._local_node.id,
        )

        if state.pid is None or state.status not in {
            TaskStatus.RUNNING,
            TaskStatus.PAUSED,
        }:
            existing.pid = state.pid
            existing.process_count = 0
            existing.process_state = None
            existing.last_error = "Task has no live local process"
            return self._store.update_task(existing)

        try:
            measurement = self._sampler.sample(state.pid)
            cpu_percent: float | None = None
            previous = self._previous_cpu.get(task_id)
            if previous is not None:
                previous_cpu, previous_at = previous
                elapsed = max(0.0, (measurement.sampled_at_utc - previous_at).total_seconds())
                cpu_delta = max(0.0, measurement.cpu_time_seconds - previous_cpu)
                if elapsed > 0:
                    cpu_percent = cpu_delta / elapsed * 100.0
            self._previous_cpu[task_id] = (
                measurement.cpu_time_seconds,
                measurement.sampled_at_utc,
            )

            requested_cores = max(
                0.1,
                definition.profile.resource_request.cpu_cores,
            )
            utilization_fraction = 0.0
            if cpu_percent is not None:
                utilization_fraction = min(
                    1.0,
                    cpu_percent / (requested_cores * 100.0),
                )
            power_fraction = (
                self._policy.power_idle_fraction
                + (1.0 - self._policy.power_idle_fraction)
                * utilization_fraction
            )
            measured_power = (
                definition.profile.power_kw * power_fraction
                if cpu_percent is not None
                else definition.profile.power_kw
            )

            record = TaskTelemetryRecord(
                task_id=task_id,
                node_id=self._local_node.id,
                pid=state.pid,
                process_count=measurement.process_count,
                process_state=measurement.process_state,
                cpu_utilization_percent=cpu_percent,
                memory_rss_mb=measurement.memory_rss_mb,
                checkpoint_bytes=self._directory_bytes(
                    self._registry.checkpoint_directory(task_id)
                ),
                measured_power_kw=measured_power,
                power_source=(
                    "procfs_cpu_utilization_model"
                    if cpu_percent is not None
                    else "configured_fallback"
                ),
                power_confidence=(
                    self._policy.cpu_power_confidence
                    if cpu_percent is not None
                    else self._policy.fallback_power_confidence
                ),
                progress_rate_units_per_second=(
                    state.progress_rate_units_per_second
                ),
                estimated_remaining_seconds=state.estimated_remaining_seconds,
                sample_count=existing.sample_count + 1,
                last_sample_at_utc=now,
                last_error=None,
            )
        except Exception as exc:
            existing.pid = state.pid
            existing.last_error = f"{type(exc).__name__}: {exc}"
            return self._store.update_task(existing)

        return self._store.update_task(record)

    def _configured_edge_values(
        self, destination_node_id: str
    ) -> tuple[float, float]:
        configured = self._cluster.get_edge_override(
            self._local_node.id, destination_node_id
        )
        return (
            configured.bandwidth_mbps
            if configured is not None
            else self._cluster.default_bandwidth_mbps,
            configured.latency_ms
            if configured is not None
            else self._cluster.default_latency_ms,
        )

    def edge_view(self, destination_node_id: str):
        self._cluster.get_node(destination_node_id)
        bandwidth, latency = self._configured_edge_values(destination_node_id)
        return self._store.edge_view(
            self._local_node.id,
            destination_node_id,
            bandwidth,
            latency,
            self._policy.edge_stale_after_seconds,
        )

    def peer_ids(self) -> tuple[str, ...]:
        return tuple(
            node.id
            for node in self._cluster.nodes
            if node.id != self._local_node.id
        )

    def _bandwidth_probe_due(
        self, destination_node_id: str, *, force: bool
    ) -> bool:
        if force:
            return True
        view = self.edge_view(destination_node_id)
        if view.bandwidth_freshness.value != "fresh":
            return True
        last = self._last_bandwidth_probe_monotonic.get(destination_node_id)
        if last is None:
            # Persisted telemetry may still be fresh after a daemon restart.
            # Do not discard it, but refresh it on the next normal bandwidth
            # interval instead of forcing a burst immediately.
            age = view.bandwidth_age_seconds or 0.0
            return age >= self._policy.edge_bandwidth_probe_interval_seconds
        return (
            time.monotonic() - last
            >= self._policy.edge_bandwidth_probe_interval_seconds
        )

    async def probe_edge(
        self,
        destination_node_id: str,
        *,
        force_bandwidth: bool = False,
    ):
        """Refresh one directed edge using live peer measurements.

        The peer set comes from the current cluster membership.  Bandwidth is
        measured as an end-to-end upload sample; consumers must therefore not
        add RTT again when using ``measured_transfer_ema``.
        """
        if destination_node_id == self._local_node.id:
            raise ValueError("Destination must be a peer")
        destination = self._cluster.get_node(destination_node_id)
        timeout = httpx.Timeout(self._cluster.request_timeout_seconds)
        health_url = (
            f"http://{destination.internal_ip}:"
            f"{self._cluster.api_port}/health"
        )

        latency_started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(health_url)
                response.raise_for_status()
            latency_ms = (time.perf_counter() - latency_started) * 1000.0
            self._store.record_latency(
                self._local_node.id, destination_node_id, latency_ms
            )
        except Exception as exc:
            self._store.record_edge_failure(
                self._local_node.id,
                destination_node_id,
                f"{type(exc).__name__}: {exc}",
            )
            return self.edge_view(destination_node_id)

        if self._bandwidth_probe_due(
            destination_node_id, force=force_bandwidth
        ):
            upload_url = (
                f"http://{destination.internal_ip}:"
                f"{self._cluster.api_port}/telemetry/probe/upload"
            )
            transfer_started = time.perf_counter()
            try:
                # Use a fresh connection so the sample includes connection and
                # request/response effects, matching the end-to-end semantics
                # of observed migration-transfer throughput.
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        upload_url,
                        content=self._bandwidth_probe_payload,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                transfer_seconds = max(
                    1e-9, time.perf_counter() - transfer_started
                )
                received_bytes = int(payload.get("received_bytes", -1))
                if received_bytes != len(self._bandwidth_probe_payload):
                    raise RuntimeError(
                        "Peer bandwidth probe acknowledged "
                        f"{received_bytes} bytes; expected "
                        f"{len(self._bandwidth_probe_payload)}"
                    )
                self._store.record_transfer(
                    self._local_node.id,
                    destination_node_id,
                    len(self._bandwidth_probe_payload),
                    transfer_seconds,
                )
                self._last_bandwidth_probe_monotonic[
                    destination_node_id
                ] = time.monotonic()
            except Exception as exc:
                self._store.record_edge_failure(
                    self._local_node.id,
                    destination_node_id,
                    f"bandwidth_probe:{type(exc).__name__}: {exc}",
                )

        return self.edge_view(destination_node_id)

    async def ensure_edges_fresh(
        self, destination_node_ids: set[str] | list[str] | tuple[str, ...]
    ) -> dict[str, object]:
        """Refresh stale/unseen candidate edges before a scheduling decision."""
        unique = tuple(dict.fromkeys(destination_node_ids))
        stale: list[str] = []
        for destination_node_id in unique:
            view = self.edge_view(destination_node_id)
            if (
                view.latency_freshness.value != "fresh"
                or view.bandwidth_freshness.value != "fresh"
            ):
                stale.append(destination_node_id)

        for destination_node_id in stale:
            # Avoid self-contention from simultaneous bandwidth probes leaving
            # the same source NIC. Cold/stale refreshes are intentionally
            # serialized per source node.
            await self.probe_edge(
                destination_node_id, force_bandwidth=True
            )

        return {
            destination_node_id: self.edge_view(destination_node_id)
            for destination_node_id in unique
        }

    async def refresh_all_edges(self) -> dict[str, object]:
        """Force a live refresh for every peer in current cluster membership."""
        peer_ids = self.peer_ids()
        for peer_id in peer_ids:
            await self.probe_edge(peer_id, force_bandwidth=True)
        return {peer_id: self.edge_view(peer_id) for peer_id in peer_ids}

    def _apply_rapl_power(self, task_ids: list[str]) -> None:
        node_power_kw = self._node_power_reader.sample_power_kw()
        if node_power_kw is None or not task_ids:
            return
        records = [
            record
            for task_id in task_ids
            if (record := self._store.task_record(task_id)) is not None
        ]
        if not records:
            return
        weights = [max(0.0, record.cpu_utilization_percent or 0.0) for record in records]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(records)
            total = float(len(records))
        for record, weight in zip(records, weights, strict=True):
            record.measured_power_kw = max(1e-9, node_power_kw * weight / total)
            record.power_source = "rapl_cpu_share"
            record.power_confidence = 0.9
            self._store.update_task(record)

    async def run_once(self, force_edge_probe: bool = False) -> dict[str, int]:
        task_count = 0
        sampled_task_ids: list[str] = []
        for state in self._registry.all_states():
            if (
                state.owner_node_id == self._local_node.id
                and state.status in {TaskStatus.RUNNING, TaskStatus.PAUSED}
            ):
                await asyncio.to_thread(self.sample_task, state.task_id)
                sampled_task_ids.append(state.task_id)
                task_count += 1

        await asyncio.to_thread(self._apply_rapl_power, sampled_task_ids)

        now_monotonic = time.monotonic()
        probe_due = (
            force_edge_probe
            or now_monotonic - self._last_edge_probe_monotonic
            >= self._policy.edge_probe_interval_seconds
        )
        edge_count = 0
        if probe_due:
            for node in self._cluster.nodes:
                if node.id == self._local_node.id:
                    continue
                await self.probe_edge(node.id)
            edge_count = max(0, len(self._cluster.nodes) - 1)
            self._last_edge_probe_monotonic = now_monotonic

        return {"tasks": task_count, "edges": edge_count}

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self._policy.enabled:
            return
        while not stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._policy.task_scan_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    def enrich_profile(self, profile: TaskProfile) -> TaskProfile:
        view = self._store.task_view(
            profile.task_id,
            profile.power_kw,
            self._policy.task_stale_after_seconds,
        )
        checkpoint_bytes = (
            view.checkpoint_bytes
            if view.freshness.value == "fresh"
            and view.checkpoint_bytes is not None
            else profile.checkpoint_bytes
        )
        return profile.model_copy(
            deep=True,
            update={
                "power_kw": view.effective_power_kw,
                "checkpoint_bytes": checkpoint_bytes,
            },
        )
