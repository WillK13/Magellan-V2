from __future__ import annotations

from pydantic import BaseModel, Field, IPvAnyAddress


class NetworkEdgeConfig(BaseModel):
    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)
    bandwidth_mbps: float = Field(gt=0)
    latency_ms: float = Field(ge=0)


class NodeConfig(BaseModel):
    # Magellan identity
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    # Actual GCP identity
    vm_name: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    internal_ip: IPvAnyAddress

    # Logical datacenter identity
    carbon_region: str = Field(min_length=1)
    dataset_file: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    # Site properties
    capacity: int = Field(default=1, ge=1)
    pue: float = Field(default=1.2, ge=1)
    compute_price_usd_per_hour: float = Field(default=0.0, ge=0)
    egress_price_usd_per_gb: float = Field(default=0.0, ge=0)


class ClusterConfig(BaseModel):
    api_port: int = Field(default=8040, ge=1, le=65535)
    epoch_seconds: int = Field(default=900, ge=1)
    bid_window_seconds: int = Field(default=10, ge=1)
    request_timeout_seconds: float = Field(default=5.0, gt=0)

    default_bandwidth_mbps: float = Field(default=100.0, gt=0)
    default_latency_ms: float = Field(default=50.0, ge=0)

    nodes: list[NodeConfig]
    edges: list[NetworkEdgeConfig] = Field(default_factory=list)

    def get_node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown Magellan node: {node_id}")

    def get_edge_override(
        self,
        source_node_id: str,
        destination_node_id: str,
    ) -> NetworkEdgeConfig | None:
        for edge in self.edges:
            if (
                edge.source_node_id == source_node_id
                and edge.destination_node_id == destination_node_id
            ):
                return edge
        return None
