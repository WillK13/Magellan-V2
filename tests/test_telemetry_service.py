from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import TelemetryPolicy
from magellan.models.types import TaskProfile, TaskResourceRequest
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition
from magellan.telemetry.models import ProcessMeasurement
from magellan.telemetry.service import TelemetryService
from magellan.telemetry.store import TelemetryStore


class SequenceSampler:
    def __init__(self, samples: list[ProcessMeasurement]) -> None:
        self.samples = samples

    def sample(self, _process_group_id: int) -> ProcessMeasurement:
        return self.samples.pop(0)


def node() -> NodeConfig:
    return NodeConfig(
        id="boston",
        name="Boston",
        vm_name="boston",
        zone="us-east1-c",
        internal_ip="10.0.0.1",
        carbon_region="Boston",
        dataset_file="unused.csv",
        latitude=42,
        longitude=-71,
        resources={"cpu_cores": 2, "memory_mb": 4096},
    )


def registry(tmp_path) -> PersistentTaskRegistry:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="task-1",
            workload_type="counter",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
            resource_request=TaskResourceRequest(cpu_cores=1, memory_mb=128),
        ),
        runtime=LocalProcessSpec(module="example.module"),
    )
    result = PersistentTaskRegistry([definition], tmp_path, "boston")
    result.mark_running("task-1", 999)
    checkpoint = result.checkpoint_directory("task-1")
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.bin").write_bytes(b"x" * 50)
    return result


def test_task_sampler_derives_cpu_power_memory_and_checkpoint(tmp_path) -> None:
    t0 = datetime.now(timezone.utc)
    sampler = SequenceSampler(
        [
            ProcessMeasurement(
                pid=999,
                process_count=1,
                process_state="R",
                cpu_time_seconds=10,
                memory_rss_mb=12,
                sampled_at_utc=t0,
            ),
            ProcessMeasurement(
                pid=999,
                process_count=2,
                process_state="S",
                cpu_time_seconds=10.5,
                memory_rss_mb=20,
                sampled_at_utc=t0 + timedelta(seconds=1),
            ),
        ]
    )
    store = TelemetryStore(tmp_path, ema_alpha=0.5)
    service = TelemetryService(
        local_node=node(),
        cluster=ClusterConfig(nodes=[node()]),
        policy=TelemetryPolicy(power_idle_fraction=0.2),
        registry=registry(tmp_path),
        store=store,
        process_sampler=sampler,
    )

    first = service.sample_task("task-1")
    assert first.cpu_utilization_percent is None
    assert first.measured_power_kw == pytest.approx(0.1)
    assert first.power_source == "configured_fallback"

    second = service.sample_task("task-1")
    assert second.cpu_utilization_percent == pytest.approx(50)
    assert second.measured_power_kw == pytest.approx(0.06)
    assert second.memory_rss_mb == pytest.approx(20)
    assert second.process_count == 2
    assert second.checkpoint_bytes == 50


def test_enrich_profile_uses_fresh_power_and_checkpoint(tmp_path) -> None:
    t0 = datetime.now(timezone.utc)
    store = TelemetryStore(tmp_path)
    reg = registry(tmp_path)
    service = TelemetryService(
        local_node=node(),
        cluster=ClusterConfig(nodes=[node()]),
        policy=TelemetryPolicy(task_stale_after_seconds=30),
        registry=reg,
        store=store,
        process_sampler=SequenceSampler([]),
    )
    from magellan.telemetry.models import TaskTelemetryRecord

    store.update_task(
        TaskTelemetryRecord(
            task_id="task-1",
            node_id="boston",
            measured_power_kw=0.07,
            power_source="procfs_cpu_utilization_model",
            power_confidence=0.75,
            checkpoint_bytes=987,
            sample_count=2,
            last_sample_at_utc=t0,
        )
    )
    enriched = service.enrich_profile(reg.scoring_profile("task-1"))
    assert enriched.power_kw == pytest.approx(0.07)
    assert enriched.checkpoint_bytes == 987


def peer_node(node_id: str, ip: str) -> NodeConfig:
    return NodeConfig(
        id=node_id,
        name=node_id.title(),
        vm_name=node_id,
        zone="test-zone",
        internal_ip=ip,
        carbon_region=node_id,
        dataset_file="unused.csv",
        latitude=40,
        longitude=-70,
        resources={"cpu_cores": 2, "memory_mb": 4096},
    )


@pytest.mark.asyncio
async def test_edge_refresh_is_topology_driven_and_lazy(tmp_path, monkeypatch) -> None:
    boston = node()
    virginia = peer_node("virginia", "10.0.0.2")
    california = peer_node("california", "10.0.0.3")
    cluster = ClusterConfig(nodes=[boston, virginia, california])
    store = TelemetryStore(tmp_path)
    service = TelemetryService(
        local_node=boston,
        cluster=cluster,
        policy=TelemetryPolicy(edge_stale_after_seconds=120),
        registry=registry(tmp_path),
        store=store,
        process_sampler=SequenceSampler([]),
    )

    store.record_latency("boston", "virginia", 20)
    store.record_transfer_model_pair(
        "boston", "virginia", 1_000_000, 0.2, 2_000_000, 0.3
    )

    called: list[tuple[str, bool]] = []

    async def fake_probe(destination_node_id: str, *, force_bandwidth: bool = False):
        called.append((destination_node_id, force_bandwidth))
        store.record_latency("boston", destination_node_id, 25)
        store.record_transfer_model_pair(
            "boston", destination_node_id, 1_000_000, 0.25, 2_000_000, 0.4
        )
        return service.edge_view(destination_node_id)

    monkeypatch.setattr(service, "probe_edge", fake_probe)

    views = await service.ensure_edges_fresh(["virginia", "california"])

    assert service.peer_ids() == ("virginia", "california")
    assert called == [("california", True)]
    assert views["virginia"].bandwidth_source == "measured_migration_transport_ema"
    assert views["california"].bandwidth_source == "measured_migration_transport_ema"


def test_cached_transport_probe_combines_small_rsync_and_bounded_stream(
    tmp_path, monkeypatch
) -> None:
    boston = node()
    virginia = peer_node("virginia", "10.0.0.2")
    service = TelemetryService(
        local_node=boston,
        cluster=ClusterConfig(nodes=[boston, virginia]),
        policy=TelemetryPolicy(
            edge_bandwidth_probe_bytes=1_048_576,
            edge_bandwidth_probe_max_bytes=32 * 1024 * 1024,
            edge_bandwidth_probe_target_seconds=3.0,
        ),
        registry=registry(tmp_path),
        store=TelemetryStore(tmp_path),
        process_sampler=SequenceSampler([]),
    )

    rsync_calls: list[tuple[str, int]] = []
    stream_calls: list[str] = []

    def fake_rsync(destination_node_id: str, size_bytes: int):
        rsync_calls.append((destination_node_id, size_bytes))
        return size_bytes, 0.35

    def fake_stream(destination_node_id: str):
        stream_calls.append(destination_node_id)
        return 12_000_000, 3.0

    monkeypatch.setattr(service, "_run_rsync_bandwidth_probe", fake_rsync)
    monkeypatch.setattr(service, "_run_ssh_stream_bandwidth_probe", fake_stream)

    small, stream = service._measure_migration_transport_model("virginia")

    assert rsync_calls == [("virginia", 1_048_576)]
    assert stream_calls == ["virginia"]
    assert small == (1_048_576, pytest.approx(0.35))
    assert stream == (12_000_000, pytest.approx(3.0))


def test_telemetry_defaults_reuse_cached_edges_between_background_refreshes() -> None:
    policy = TelemetryPolicy()
    assert policy.refresh_edges_before_decision is False
    assert (
        policy.edge_stale_after_seconds
        >= policy.edge_bandwidth_probe_interval_seconds
    )
