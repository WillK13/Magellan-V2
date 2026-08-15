from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

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
        self._bandwidth_probe_lock = asyncio.Lock()
        self._ssh_user = os.getenv(
            "MAGELLAN_SSH_USER",
            os.getenv("USER", "WILL"),
        )
        self._probe_payload_paths: dict[int, Path] = {}

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
        if view.transfer_model_freshness.value == "unavailable":
            return True
        last = self._last_bandwidth_probe_monotonic.get(destination_node_id)
        if last is None:
            # Persisted telemetry survives daemon restarts. Refresh it on the
            # configured maintenance interval; decision-time freshness checks
            # still force an immediate probe when a stale edge matters.
            age = view.transfer_model_age_seconds or 0.0
            return age >= self._policy.edge_bandwidth_probe_interval_seconds
        return (
            time.monotonic() - last
            >= self._policy.edge_bandwidth_probe_interval_seconds
        )

    def _probe_payload_path(self, size_bytes: int) -> Path:
        cached = self._probe_payload_paths.get(size_bytes)
        if cached is not None and cached.is_file() and cached.stat().st_size == size_bytes:
            return cached

        root = (
            Path("/tmp")
            / "magellan-edge-probe-cache"
            / self._local_node.id
            / str(size_bytes)
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / "payload.bin"
        if not path.is_file() or path.stat().st_size != size_bytes:
            temporary = path.with_suffix(".tmp")
            remaining = size_bytes
            with temporary.open("wb") as handle:
                while remaining:
                    chunk = min(1024 * 1024, remaining)
                    handle.write(os.urandom(chunk))
                    remaining -= chunk
            os.replace(temporary, path)
        self._probe_payload_paths[size_bytes] = path
        return path

    def _run_rsync_bandwidth_probe(
        self, destination_node_id: str, size_bytes: int
    ) -> tuple[int, float]:
        """Measure the same rsync/SSH transfer component used by migration."""
        destination = self._cluster.get_node(destination_node_id)
        payload = self._probe_payload_path(size_bytes)
        source_directory = payload.parent
        remote_directory = (
            f"/tmp/magellan-edge-probe/{self._local_node.id}/"
            f"{uuid4().hex}"
        )
        target = f"{self._ssh_user}@{destination.internal_ip}"
        connect_timeout = max(1, int(self._cluster.request_timeout_seconds))
        ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={connect_timeout}",
        ]
        ssh_transport = (
            "ssh -o BatchMode=yes "
            "-o StrictHostKeyChecking=accept-new "
            f"-o ConnectTimeout={connect_timeout}"
        )
        timeout = self._policy.edge_bandwidth_probe_timeout_seconds

        subprocess.run(
            [
                "ssh",
                *ssh_options,
                target,
                f"mkdir -p {shlex.quote(remote_directory)}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        try:
            started = time.perf_counter()
            subprocess.run(
                [
                    "rsync",
                    "-az",
                    "--delete",
                    "-e",
                    ssh_transport,
                    f"{source_directory}/",
                    f"{target}:{remote_directory}/",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            duration = max(1e-9, time.perf_counter() - started)
        finally:
            subprocess.run(
                [
                    "ssh",
                    *ssh_options,
                    target,
                    f"rm -rf {shlex.quote(remote_directory)}",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        return size_bytes, duration

    def _run_ssh_stream_bandwidth_probe(
        self, destination_node_id: str
    ) -> tuple[int, float]:
        """Measure sustained source-to-peer SSH throughput for a bounded time.

        The remote command emits a readiness marker before the timer starts, so
        connection setup is excluded from the steady-state rate. A separate
        small rsync probe captures the fixed rsync/SSH transfer startup cost.
        """
        destination = self._cluster.get_node(destination_node_id)
        target = f"{self._ssh_user}@{destination.internal_ip}"
        connect_timeout = max(1, int(self._cluster.request_timeout_seconds))
        ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={connect_timeout}",
        ]
        timeout = self._policy.edge_bandwidth_probe_timeout_seconds
        target_seconds = self._policy.edge_bandwidth_probe_target_seconds
        max_bytes = self._policy.edge_bandwidth_probe_max_bytes

        process = subprocess.Popen(
            [
                "ssh",
                *ssh_options,
                target,
                "printf 'READY\\n'; cat >/dev/null",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None or process.stdin is None:
            process.kill()
            raise RuntimeError("Unable to open SSH probe pipes")

        ready = process.stdout.readline().strip()
        if ready != b"READY":
            stderr = b"" if process.stderr is None else process.stderr.read()
            process.kill()
            raise RuntimeError(
                "SSH stream probe did not become ready: "
                + stderr.decode(errors="replace").strip()
            )

        chunk = os.urandom(64 * 1024)
        sent = 0
        started = time.perf_counter()
        try:
            while sent < max_bytes:
                elapsed = time.perf_counter() - started
                if elapsed >= target_seconds:
                    break
                remaining = max_bytes - sent
                payload = chunk if remaining >= len(chunk) else chunk[:remaining]
                process.stdin.write(payload)
                sent += len(payload)
        except BrokenPipeError as exc:
            raise RuntimeError("SSH stream probe closed early") from exc
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

        duration = max(1e-9, time.perf_counter() - started)
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            stderr = b"" if process.stderr is None else process.stderr.read()
            raise RuntimeError(
                "SSH stream probe failed: "
                + stderr.decode(errors="replace").strip()
            )
        if sent <= 0:
            raise RuntimeError("SSH stream probe transferred no bytes")
        return sent, duration

    def _measure_migration_transport_model(
        self, destination_node_id: str
    ) -> tuple[tuple[int, float], tuple[int, float]]:
        """Measure fixed rsync startup plus sustained SSH transport rate."""
        small = self._run_rsync_bandwidth_probe(
            destination_node_id,
            self._policy.edge_bandwidth_probe_bytes,
        )
        stream = self._run_ssh_stream_bandwidth_probe(destination_node_id)
        return small, stream

    async def probe_edge(
        self,
        destination_node_id: str,
        *,
        force_bandwidth: bool = False,
    ):
        """Refresh one directed edge using live peer measurements.

        The peer set comes from current cluster membership. RTT is measured
        over HTTP. Migration transport combines a small rsync sample for fixed
        startup cost with a bounded sustained SSH stream for throughput. The
        resulting model is cached and periodically refreshed in the background.
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
            # One source daemon must not benchmark multiple outgoing links at
            # once: concurrent probes would measure self-contention instead of
            # edge throughput. The lock also serializes background, lazy, and
            # explicit preflight refreshes.
            async with self._bandwidth_probe_lock:
                if self._bandwidth_probe_due(
                    destination_node_id, force=force_bandwidth
                ):
                    try:
                        small, stream = await asyncio.to_thread(
                            self._measure_migration_transport_model,
                            destination_node_id,
                        )
                        self._store.record_transfer_model_stream(
                            self._local_node.id,
                            destination_node_id,
                            small[0],
                            small[1],
                            stream[0],
                            stream[1],
                            sample_source="ssh_stream_plus_rsync_setup_probe",
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
        """Explicitly refresh stale/unseen edges when an operator requests it.

        Normal scheduling reuses cached background telemetry; this method is
        retained for experiment preflight and deployments that explicitly opt
        into synchronous decision-time refresh.
        """
        unique = tuple(dict.fromkeys(destination_node_ids))
        stale: list[str] = []
        for destination_node_id in unique:
            view = self.edge_view(destination_node_id)
            if (
                view.latency_freshness.value != "fresh"
                or view.transfer_model_freshness.value != "fresh"
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
