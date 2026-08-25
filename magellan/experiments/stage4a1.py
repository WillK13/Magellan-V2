from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from magellan.config.models import NodeConfig
from magellan.experiments.measurement import summarize_samples


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_stage4a1_node(
    node: NodeConfig,
    *,
    local_commit: str,
    health: dict[str, Any],
    capabilities: dict[str, Any],
    auction: dict[str, Any],
    remote: dict[str, Any],
    expected_machine_type: str | None,
    expected_carbon_metric: str,
    expected_state_token: str,
    memory_tolerance_fraction: float = 0.95,
) -> list[str]:
    """Return reproducibility/preflight errors for one Stage-4A.1 node."""

    errors: list[str] = []
    prefix = node.id

    if health.get("node_id") != node.id:
        errors.append(f"{prefix}: health node_id mismatch")
    if health.get("carbon_metric") != expected_carbon_metric:
        errors.append(
            f"{prefix}: carbon_metric={health.get('carbon_metric')!r}; "
            f"expected {expected_carbon_metric!r}"
        )
    state_file = str(health.get("telemetry_state_file", ""))
    if expected_state_token not in state_file:
        errors.append(
            f"{prefix}: telemetry state is not measurement-isolated: {state_file}"
        )

    for field in (
        "owned_task_count",
        "pending_bid_count",
        "active_reservation_count",
        "paused_task_count",
    ):
        if int(health.get(field, 0) or 0) != 0:
            errors.append(f"{prefix}: {field}={health.get(field)}; expected 0")

    if capabilities.get("ready") is not True:
        errors.append(f"{prefix}: capabilities ready=false")
    drift = capabilities.get("drift") or []
    if drift:
        errors.append(f"{prefix}: capability drift={drift}")

    for field in (
        "reserved_cpu_cores",
        "reserved_memory_mb",
        "reserved_gpu_count",
        "resource_busy_fraction",
    ):
        if _as_float(auction.get(field)) > 1e-9:
            errors.append(f"{prefix}: {field}={auction.get(field)}; expected 0")

    if remote.get("service_active") != "active":
        errors.append(
            f"{prefix}: magellan service={remote.get('service_active')!r}; expected 'active'"
        )
    if remote.get("git_commit") != local_commit:
        errors.append(
            f"{prefix}: git_commit={remote.get('git_commit')}; expected {local_commit}"
        )
    dirty = list(remote.get("git_status_porcelain") or [])
    if dirty:
        errors.append(f"{prefix}: tracked worktree changes={dirty}")

    configured_machine_type = node.machine_type
    actual_machine_type = remote.get("machine_type")
    if expected_machine_type and configured_machine_type != expected_machine_type:
        errors.append(
            f"{prefix}: configured machine_type={configured_machine_type!r}; "
            f"expected {expected_machine_type!r}"
        )
    if configured_machine_type and actual_machine_type != configured_machine_type:
        errors.append(
            f"{prefix}: GCP metadata machine_type={actual_machine_type!r}; "
            f"configured {configured_machine_type!r}"
        )
    if remote.get("instance_name") != node.vm_name:
        errors.append(
            f"{prefix}: instance_name={remote.get('instance_name')!r}; "
            f"configured {node.vm_name!r}"
        )
    if remote.get("zone") != node.zone:
        errors.append(
            f"{prefix}: zone={remote.get('zone')!r}; configured {node.zone!r}"
        )

    configured_cpu = node.resources.cpu_cores
    if configured_cpu is not None:
        remote_cpu = _as_float(remote.get("cpu_logical_count"))
        if remote_cpu + 1e-9 < configured_cpu:
            errors.append(
                f"{prefix}: observed cpu={remote_cpu}; configured {configured_cpu}"
            )

    configured_memory = node.resources.memory_mb
    if configured_memory is not None:
        minimum_memory = configured_memory * memory_tolerance_fraction
        remote_memory = _as_float(remote.get("memory_mb"))
        capability_memory = _as_float(
            (capabilities.get("observed") or {}).get("memory_mb")
        )
        if remote_memory < minimum_memory:
            errors.append(
                f"{prefix}: /proc memory={remote_memory:.0f} MB; "
                f"expected at least {minimum_memory:.0f} MB"
            )
        if capability_memory < minimum_memory:
            errors.append(
                f"{prefix}: capability memory={capability_memory:.0f} MB; "
                f"expected at least {minimum_memory:.0f} MB"
            )

    return errors


def summarize_network_bundle(bundle: str | Path) -> dict[str, Any]:
    """Build publication-friendly descriptive statistics for a WAN bundle."""

    root = Path(bundle)
    edges = _read_csv(root / "edges.csv")
    bandwidth_samples = _read_csv(root / "bandwidth_samples.csv")
    rtt_samples = _read_csv(root / "rtt_samples.csv")

    edge_rtts = [float(row["measured_rtt_median_ms"]) for row in edges]
    edge_bandwidths = [
        float(row["measured_bandwidth_median_mbps"]) for row in edges
    ]
    edge_errors = [
        float(row["median_transfer_absolute_error_percent"])
        for row in edges
        if row.get("median_transfer_absolute_error_percent") not in {None, ""}
    ]

    def pair(row: dict[str, str]) -> str:
        return f"{row['source_node_id']}->{row['destination_node_id']}"

    worst_prediction = sorted(
        edges,
        key=lambda row: float(row["median_transfer_absolute_error_percent"]),
        reverse=True,
    )[:5]
    slowest_bandwidth = sorted(
        edges,
        key=lambda row: float(row["measured_bandwidth_median_mbps"]),
    )[:5]
    highest_rtt = sorted(
        edges,
        key=lambda row: float(row["measured_rtt_median_ms"]),
        reverse=True,
    )[:5]

    return {
        "directed_edge_count": len(edges),
        "rtt_sample_count": len(rtt_samples),
        "bandwidth_sample_count": len(bandwidth_samples),
        "edge_rtt_ms": summarize_samples(edge_rtts).as_dict(),
        "edge_bandwidth_mbps": summarize_samples(edge_bandwidths).as_dict(),
        "absolute_transfer_prediction_error_percent": (
            summarize_samples(edge_errors).as_dict() if edge_errors else None
        ),
        "worst_prediction_edges": [
            {
                "edge": pair(row),
                "absolute_error_percent": float(
                    row["median_transfer_absolute_error_percent"]
                ),
                "measured_seconds": float(
                    row["measured_transfer_median_seconds"]
                ),
                "predicted_seconds": float(row["predicted_transfer_seconds"]),
            }
            for row in worst_prediction
        ],
        "slowest_bandwidth_edges": [
            {
                "edge": pair(row),
                "median_mbps": float(row["measured_bandwidth_median_mbps"]),
            }
            for row in slowest_bandwidth
        ],
        "highest_rtt_edges": [
            {
                "edge": pair(row),
                "median_rtt_ms": float(row["measured_rtt_median_ms"]),
            }
            for row in highest_rtt
        ],
    }
