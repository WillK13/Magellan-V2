from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from magellan.config.models import ClusterConfig


@dataclass(frozen=True)
class EdgeMetrics:
    source_node_id: str
    destination_node_id: str
    distance_km: float
    bandwidth_mbps: float
    latency_ms: float


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
    def __init__(self, cluster: ClusterConfig) -> None:
        self._cluster = cluster

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

        bandwidth_mbps = (
            override.bandwidth_mbps
            if override
            else self._cluster.default_bandwidth_mbps
        )
        latency_ms = (
            override.latency_ms
            if override
            else self._cluster.default_latency_ms
        )

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
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
        )
