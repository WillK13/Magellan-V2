from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import TYPE_CHECKING

from magellan.config.models import ClusterConfig
from magellan.telemetry.models import TelemetryFreshness

if TYPE_CHECKING:
    from magellan.config.policy_models import TelemetryPolicy
    from magellan.telemetry.store import TelemetryStore


@dataclass(frozen=True)
class EdgeMetrics:
    source_node_id: str
    destination_node_id: str
    distance_km: float
    bandwidth_mbps: float
    latency_ms: float
    bandwidth_source: str = "configured_fallback"
    latency_source: str = "configured_fallback"
    bandwidth_freshness: TelemetryFreshness = TelemetryFreshness.UNAVAILABLE
    latency_freshness: TelemetryFreshness = TelemetryFreshness.UNAVAILABLE
    checkpoint_seconds: float | None = None
    restore_seconds: float | None = None
    migration_overhead_seconds: float = 0.0
    transfer_fixed_seconds: float = 0.0
    transfer_steady_bandwidth_mbps: float | None = None
    transfer_model_source: str = "unavailable"
    calibration_source: str = "configured_fallback"


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    earth_radius_km = 6371.0

    lat_1 = radians(latitude_1)
    lon_1 = radians(longitude_1)
    lat_2 = radians(latitude_2)
    lon_2 = radians(longitude_2)

    delta_lat = lat_2 - lat_1
    delta_lon = lon_2 - lon_1

    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat_1) * cos(lat_2) * sin(delta_lon / 2) ** 2
    )

    return 2 * earth_radius_km * asin(sqrt(value))


class ClusterGraph:
    def __init__(
        self,
        cluster: ClusterConfig,
        telemetry_store: TelemetryStore | None = None,
        telemetry_policy: TelemetryPolicy | None = None,
    ) -> None:
        self._cluster = cluster
        self._telemetry_store = telemetry_store
        self._telemetry_policy = telemetry_policy

    def peers(self, node_id: str):
        return [node for node in self._cluster.nodes if node.id != node_id]

    def edge(
        self,
        source_node_id: str,
        destination_node_id: str,
    ) -> EdgeMetrics:
        source = self._cluster.get_node(source_node_id)
        destination = self._cluster.get_node(destination_node_id)

        override = self._cluster.get_edge_override(
            source_node_id,
            destination_node_id,
        )

        configured_bandwidth = (
            override.bandwidth_mbps
            if override
            else self._cluster.default_bandwidth_mbps
        )
        configured_latency = (
            override.latency_ms
            if override
            else self._cluster.default_latency_ms
        )

        bandwidth = configured_bandwidth
        latency = configured_latency
        bandwidth_source = "configured_fallback"
        latency_source = "configured_fallback"
        bandwidth_freshness = TelemetryFreshness.UNAVAILABLE
        latency_freshness = TelemetryFreshness.UNAVAILABLE
        checkpoint_seconds = None
        restore_seconds = None
        migration_overhead_seconds = 0.0
        transfer_fixed_seconds = 0.0
        transfer_steady_bandwidth_mbps = None
        transfer_model_source = "unavailable"
        calibration_source = "configured_fallback"

        if self._telemetry_store is not None and self._telemetry_policy is not None:
            view = self._telemetry_store.edge_view(
                source_node_id,
                destination_node_id,
                configured_bandwidth,
                configured_latency,
                self._telemetry_policy.edge_stale_after_seconds,
            )
            bandwidth = view.effective_bandwidth_mbps
            latency = view.effective_latency_ms
            bandwidth_source = view.bandwidth_source
            latency_source = view.latency_source
            bandwidth_freshness = view.bandwidth_freshness
            latency_freshness = view.latency_freshness
            transfer_fixed_seconds = view.effective_transfer_fixed_seconds
            transfer_steady_bandwidth_mbps = (
                view.effective_transfer_steady_bandwidth_mbps
            )
            transfer_model_source = view.transfer_model_source

            calibration = self._telemetry_store.calibration_view(
                source_node_id,
                destination_node_id,
                self._telemetry_policy.calibration_stale_after_seconds,
            )
            if calibration.freshness == TelemetryFreshness.FRESH:
                checkpoint_seconds = calibration.checkpoint_seconds_ema
                restore_seconds = calibration.restore_seconds_ema
                if (
                    calibration.total_downtime_seconds_ema is not None
                    and calibration.checkpoint_seconds_ema is not None
                    and calibration.transfer_seconds_ema is not None
                    and calibration.restore_seconds_ema is not None
                ):
                    migration_overhead_seconds = max(
                        0.0,
                        calibration.total_downtime_seconds_ema
                        - calibration.checkpoint_seconds_ema
                        - calibration.transfer_seconds_ema
                        - calibration.restore_seconds_ema,
                    )
                calibration_source = "measured_migration_ema"

        distance_km = haversine_distance_km(
            source.latitude,
            source.longitude,
            destination.latitude,
            destination.longitude,
        )

        return EdgeMetrics(
            source_node_id=source_node_id,
            destination_node_id=destination_node_id,
            distance_km=distance_km,
            bandwidth_mbps=bandwidth,
            latency_ms=latency,
            bandwidth_source=bandwidth_source,
            latency_source=latency_source,
            bandwidth_freshness=bandwidth_freshness,
            latency_freshness=latency_freshness,
            checkpoint_seconds=checkpoint_seconds,
            restore_seconds=restore_seconds,
            migration_overhead_seconds=migration_overhead_seconds,
            transfer_fixed_seconds=transfer_fixed_seconds,
            transfer_steady_bandwidth_mbps=transfer_steady_bandwidth_mbps,
            transfer_model_source=transfer_model_source,
            calibration_source=calibration_source,
        )
