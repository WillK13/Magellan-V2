from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from magellan.config.models import ClusterConfig, NodeConfig, NetworkEdgeConfig
from magellan.config.policy_models import TelemetryPolicy
from magellan.graph.topology import ClusterGraph
from magellan.models.migrate_model import estimate_migrate
from magellan.models.types import TaskProfile
from magellan.config.policy_models import MigrationPolicy, PausePolicy
from magellan.telemetry.store import TelemetryStore


class ConstantCarbon:
    def average(self, *_args):
        return 100.0


def cluster() -> ClusterConfig:
    return ClusterConfig(
        nodes=[
            NodeConfig(
                id="boston",
                name="Boston",
                vm_name="boston",
                zone="a",
                internal_ip="10.0.0.1",
                carbon_region="Boston",
                dataset_file="unused",
                latitude=42,
                longitude=-71,
            ),
            NodeConfig(
                id="virginia",
                name="Virginia",
                vm_name="virginia",
                zone="b",
                internal_ip="10.0.0.2",
                carbon_region="Virginia",
                dataset_file="unused",
                latitude=37,
                longitude=-78,
            ),
        ],
        edges=[
            NetworkEdgeConfig(
                source_node_id="boston",
                destination_node_id="virginia",
                bandwidth_mbps=100,
                latency_ms=50,
            )
        ],
    )


def test_graph_prefers_fresh_measured_edge_and_calibration(tmp_path) -> None:
    store = TelemetryStore(tmp_path)
    store.record_latency("boston", "virginia", 12)
    store.record_transfer("boston", "virginia", 10_000_000, 2)
    store.record_migration_calibration(
        "boston",
        "virginia",
        checkpoint_seconds=3,
        transfer_seconds=2,
        restore_seconds=1,
        activation_seconds=1.5,
        total_downtime_seconds=7,
        transfer_bytes=10_000_000,
    )
    graph = ClusterGraph(cluster(), store, TelemetryPolicy())
    edge = graph.edge("boston", "virginia")
    assert edge.bandwidth_mbps == pytest.approx(40)
    assert edge.latency_ms == pytest.approx(12)
    assert edge.checkpoint_seconds == pytest.approx(3)
    assert edge.restore_seconds == pytest.approx(1)
    assert edge.bandwidth_source == "measured_transfer_ema"
    assert edge.calibration_source == "measured_migration_ema"

    estimate = estimate_migrate(
        task=TaskProfile(
            task_id="task",
            workload_type="counter",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=1_000_000,
            estimated_remaining_seconds=60,
        ),
        source=cluster().get_node("boston"),
        destination=cluster().get_node("virginia"),
        edge=edge,
        carbon_store=ConstantCarbon(),
        at_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        horizon_seconds=60,
        pause_policy=PausePolicy(
            pause_seconds=30,
            idle_seconds=0,
            resume_seconds=20,
            max_pause_window_seconds=60,
        ),
        migration_policy=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
    )
    assert estimate.details["checkpoint_seconds"] == pytest.approx(3)
    assert estimate.details["restore_seconds"] == pytest.approx(1)
    assert estimate.details["bandwidth_source"] == "measured_transfer_ema"


def test_graph_falls_back_when_edge_measurements_are_stale(tmp_path) -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    store = TelemetryStore(tmp_path)
    store.record_latency("boston", "virginia", 12, old)
    store.record_transfer("boston", "virginia", 10_000_000, 2, old)
    graph = ClusterGraph(
        cluster(),
        store,
        TelemetryPolicy(edge_stale_after_seconds=10),
    )
    edge = graph.edge("boston", "virginia")
    assert edge.bandwidth_mbps == pytest.approx(100)
    assert edge.latency_ms == pytest.approx(50)
    assert edge.bandwidth_source == "configured_fallback"
    assert edge.latency_source == "configured_fallback"
