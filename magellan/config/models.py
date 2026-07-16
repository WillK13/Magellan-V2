from __future__ import annotations

from pydantic import BaseModel, Field, IPvAnyAddress


class NodeConfig(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    internal_ip: IPvAnyAddress
    carbon_region: str = Field(min_length=1)
    capacity: int = Field(default=1, ge=1)


class ClusterConfig(BaseModel):
    api_port: int = Field(default=8040, ge=1, le=65535)
    epoch_seconds: int = Field(default=900, ge=1)
    bid_window_seconds: int = Field(default=10, ge=1)
    request_timeout_seconds: float = Field(default=5.0, gt=0)
    nodes: list[NodeConfig]

    def get_node(self, node_id: str) -> NodeConfig:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown Magellan node: {node_id}")
