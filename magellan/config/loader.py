from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy


ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_json_model(
    path: str | Path,
    model_type: type[ModelT],
) -> ModelT:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    try:
        raw = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {config_path}: {exc}"
        ) from exc

    return model_type.model_validate(raw)


def load_cluster_config(
    path: str | Path,
) -> ClusterConfig:
    config = _load_json_model(path, ClusterConfig)

    node_ids = [node.id for node in config.nodes]

    if len(node_ids) != len(set(node_ids)):
        raise ValueError(
            "Every Magellan node ID must be unique"
        )

    internal_ips = [
        str(node.internal_ip)
        for node in config.nodes
    ]

    if len(internal_ips) != len(set(internal_ips)):
        raise ValueError(
            "Every Magellan internal IP must be unique"
        )

    valid_node_ids = set(node_ids)

    for edge in config.edges:
        if edge.source_node_id not in valid_node_ids:
            raise ValueError(
                f"Edge has unknown source node: "
                f"{edge.source_node_id}"
            )

        if edge.destination_node_id not in valid_node_ids:
            raise ValueError(
                f"Edge has unknown destination node: "
                f"{edge.destination_node_id}"
            )

    return config


def load_policy_config(
    path: str | Path,
) -> ScoringPolicy:
    return _load_json_model(path, ScoringPolicy)
