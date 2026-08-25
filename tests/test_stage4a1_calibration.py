from __future__ import annotations

import csv

from magellan.config.models import NodeConfig, NodeResourceCapacity
from magellan.experiments.stage4a1 import summarize_network_bundle, validate_stage4a1_node


def node() -> NodeConfig:
    return NodeConfig(
        id="boston",
        name="Boston",
        vm_name="vm-boston",
        zone="us-east1-c",
        machine_type="e2-highmem-2",
        internal_ip="10.0.0.1",
        carbon_region="Boston_24H",
        dataset_file="Boston_24H.csv",
        latitude=42.0,
        longitude=-71.0,
        capacity=None,
        resources=NodeResourceCapacity(
            cpu_cores=2,
            memory_mb=16384,
            gpu_count=0,
        ),
    )


def test_stage4a1_node_preflight_accepts_final_hardware() -> None:
    errors = validate_stage4a1_node(
        node(),
        local_commit="abc123",
        health={
            "node_id": "boston",
            "carbon_metric": "lifecycle",
            "telemetry_state_file": "/tmp/runtime-state-gcp-measurement/control/telemetry.json",
            "owned_task_count": 0,
            "pending_bid_count": 0,
            "active_reservation_count": 0,
            "paused_task_count": 0,
        },
        capabilities={
            "ready": True,
            "drift": [],
            "observed": {"memory_mb": 16002},
        },
        auction={
            "reserved_cpu_cores": 0,
            "reserved_memory_mb": 0,
            "reserved_gpu_count": 0,
            "resource_busy_fraction": 0,
        },
        remote={
            "service_active": "active",
            "git_commit": "abc123",
            "git_status_porcelain": [],
            "machine_type": "e2-highmem-2",
            "instance_name": "vm-boston",
            "zone": "us-east1-c",
            "cpu_logical_count": 2,
            "memory_mb": 16002,
        },
        expected_machine_type="e2-highmem-2",
        expected_carbon_metric="lifecycle",
        expected_state_token="runtime-state-gcp-measurement",
    )
    assert errors == []


def test_stage4a1_node_preflight_rejects_busy_or_wrong_machine() -> None:
    errors = validate_stage4a1_node(
        node(),
        local_commit="abc123",
        health={
            "node_id": "boston",
            "carbon_metric": "lifecycle",
            "telemetry_state_file": "/tmp/runtime-state-gcp-measurement/control/telemetry.json",
            "owned_task_count": 1,
            "pending_bid_count": 0,
            "active_reservation_count": 0,
            "paused_task_count": 0,
        },
        capabilities={
            "ready": True,
            "drift": [],
            "observed": {"memory_mb": 16002},
        },
        auction={
            "reserved_cpu_cores": 1,
            "reserved_memory_mb": 512,
            "reserved_gpu_count": 0,
            "resource_busy_fraction": 0.5,
        },
        remote={
            "service_active": "active",
            "git_commit": "abc123",
            "git_status_porcelain": [],
            "machine_type": "e2-standard-2",
            "instance_name": "vm-boston",
            "zone": "us-east1-c",
            "cpu_logical_count": 2,
            "memory_mb": 16002,
        },
        expected_machine_type="e2-highmem-2",
        expected_carbon_metric="lifecycle",
        expected_state_token="runtime-state-gcp-measurement",
    )
    assert any("owned_task_count=1" in error for error in errors)
    assert any("reserved_cpu_cores=1" in error for error in errors)
    assert any("machine_type='e2-standard-2'" in error for error in errors)


def _write_csv(path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_stage4a1_network_summary_reports_edge_distributions(tmp_path) -> None:
    edges = [
        {
            "source_node_id": "a",
            "destination_node_id": "b",
            "measured_rtt_median_ms": 10,
            "measured_bandwidth_median_mbps": 100,
            "median_transfer_absolute_error_percent": 5,
            "measured_transfer_median_seconds": 1.0,
            "predicted_transfer_seconds": 1.05,
        },
        {
            "source_node_id": "b",
            "destination_node_id": "a",
            "measured_rtt_median_ms": 20,
            "measured_bandwidth_median_mbps": 50,
            "median_transfer_absolute_error_percent": 15,
            "measured_transfer_median_seconds": 2.0,
            "predicted_transfer_seconds": 2.3,
        },
    ]
    _write_csv(tmp_path / "edges.csv", list(edges[0]), edges)
    _write_csv(
        tmp_path / "rtt_samples.csv",
        ["source_node_id", "destination_node_id", "sample", "rtt_ms"],
        [
            {"source_node_id": "a", "destination_node_id": "b", "sample": 1, "rtt_ms": 10},
            {"source_node_id": "b", "destination_node_id": "a", "sample": 1, "rtt_ms": 20},
        ],
    )
    _write_csv(
        tmp_path / "bandwidth_samples.csv",
        ["source_node_id", "destination_node_id", "sample", "bandwidth_mbps"],
        [
            {"source_node_id": "a", "destination_node_id": "b", "sample": 1, "bandwidth_mbps": 100},
            {"source_node_id": "b", "destination_node_id": "a", "sample": 1, "bandwidth_mbps": 50},
        ],
    )

    summary = summarize_network_bundle(tmp_path)
    assert summary["directed_edge_count"] == 2
    assert summary["edge_rtt_ms"]["median"] == 15
    assert summary["edge_bandwidth_mbps"]["median"] == 75
    assert summary["absolute_transfer_prediction_error_percent"]["median"] == 10
    assert summary["worst_prediction_edges"][0]["edge"] == "b->a"
    assert summary["slowest_bandwidth_edges"][0]["edge"] == "b->a"
    assert summary["highest_rtt_edges"][0]["edge"] == "b->a"
