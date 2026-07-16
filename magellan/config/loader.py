from __future__ import annotations

import json
from pathlib import Path

from magellan.config.models import ClusterConfig


def load_cluster_config(path: str | Path) -> ClusterConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Cluster config does not exist: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    config = ClusterConfig.model_validate(raw)

    ids = [node.id for node in config.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Every node ID in cluster.json must be unique")

    ips = [str(node.internal_ip) for node in config.nodes]
    if len(ips) != len(set(ips)):
        raise ValueError("Every internal IP in cluster.json must be unique")

    return config
