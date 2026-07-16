from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from magellan.bidding.arbiter import BidArbiter
from magellan.bidding.client import BidClient
from magellan.bidding.store import BidStore
from magellan.carbon.store import CarbonStore
from magellan.config.loader import (
    load_cluster_config,
    load_policy_config,
)
from magellan.config.models import (
    ClusterConfig,
    NodeConfig,
)
from magellan.config.policy_models import ScoringPolicy
from magellan.daemon.scheduler_service import SchedulerService
from magellan.graph.topology import ClusterGraph
from magellan.runtime.clock import MagellanClock
from magellan.state.task_registry import TaskRegistry


@dataclass
class DaemonContext:
    cluster: ClusterConfig
    policy: ScoringPolicy
    local_node: NodeConfig
    graph: ClusterGraph
    carbon_store: CarbonStore
    clock: MagellanClock
    registry: TaskRegistry
    bid_store: BidStore
    bid_arbiter: BidArbiter
    bid_client: BidClient
    scheduler_service: SchedulerService


def _task_file_paths() -> list[Path]:
    raw = os.getenv(
        "MAGELLAN_TASK_FILES",
        "config/tasks/dev-llm.json",
    )

    paths = [
        Path(value.strip())
        for value in raw.split(",")
        if value.strip()
    ]

    if not paths:
        raise RuntimeError(
            "MAGELLAN_TASK_FILES did not contain any paths"
        )

    return paths


def build_daemon_context() -> DaemonContext:
    cluster_path = os.getenv(
        "MAGELLAN_CONFIG",
        "config/cluster.dev.json",
    )
    policy_path = os.getenv(
        "MAGELLAN_POLICY",
        "config/policy.dev.json",
    )
    datasets_path = os.getenv(
        "MAGELLAN_DATASETS",
        "datasets",
    )
    node_id = os.getenv(
        "MAGELLAN_NODE_ID",
        "",
    ).strip()

    if not node_id:
        raise RuntimeError(
            "MAGELLAN_NODE_ID must be set"
        )

    cluster = load_cluster_config(cluster_path)
    policy = load_policy_config(policy_path)

    try:
        local_node = cluster.get_node(node_id)
    except KeyError as exc:
        raise RuntimeError(str(exc)) from exc

    registry = TaskRegistry.from_files(
        _task_file_paths()
    )

    valid_node_ids = {
        node.id
        for node in cluster.nodes
    }

    for task in registry.all_tasks():
        if task.current_node_id not in valid_node_ids:
            raise RuntimeError(
                f"Task {task.task_id} references unknown owner "
                f"{task.current_node_id}"
            )

    graph = ClusterGraph(cluster)
    carbon_store = CarbonStore(
        cluster=cluster,
        datasets_directory=datasets_path,
    )
    clock = MagellanClock(policy.clock)

    bid_store = BidStore()
    bid_client = BidClient(cluster)

    bid_arbiter = BidArbiter(
        store=bid_store,
        registry=registry,
        local_node_id=local_node.id,
        capacity=local_node.capacity,
        bid_window_seconds=cluster.bid_window_seconds,
    )

    scheduler_service = SchedulerService(
        local_node=local_node,
        cluster=cluster,
        policy=policy,
        graph=graph,
        carbon_store=carbon_store,
        clock=clock,
        registry=registry,
        bid_client=bid_client,
    )

    return DaemonContext(
        cluster=cluster,
        policy=policy,
        local_node=local_node,
        graph=graph,
        carbon_store=carbon_store,
        clock=clock,
        registry=registry,
        bid_store=bid_store,
        bid_arbiter=bid_arbiter,
        bid_client=bid_client,
        scheduler_service=scheduler_service,
    )
