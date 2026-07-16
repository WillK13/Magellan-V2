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
from magellan.migration.client import (
    MigrationClient,
    OwnershipBroadcaster,
)
from magellan.migration.service import MigrationService
from magellan.migration.transfer import (
    RsyncCheckpointTransfer,
)
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import (
    LocalProcessRuntime,
)
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)

from magellan.runtime.checkpoint import CheckpointManager

from magellan.artifacts.client import ArtifactClient
from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.prefetch import (
    ArtifactPrefetchService,
)
from magellan.artifacts.transfer import (
    RsyncArtifactTransfer,
)


@dataclass
class DaemonContext:
    cluster: ClusterConfig
    policy: ScoringPolicy
    local_node: NodeConfig

    graph: ClusterGraph
    carbon_store: CarbonStore
    clock: MagellanClock

    registry: PersistentTaskRegistry
    runtime: LocalProcessRuntime

    bid_store: BidStore
    bid_arbiter: BidArbiter
    bid_client: BidClient

    migration_service: MigrationService
    scheduler_service: SchedulerService
    checkpoint_manager: CheckpointManager
    artifact_manager: ArtifactManager
    artifact_client: ArtifactClient
    prefetch_service: ArtifactPrefetchService


def _task_files() -> list[Path]:
    raw = os.getenv(
        "MAGELLAN_TASK_FILES",
        "config/tasks/dev-counter.json",
    )

    paths = [
        Path(item.strip())
        for item in raw.split(",")
        if item.strip()
    ]

    if not paths:
        raise RuntimeError(
            "MAGELLAN_TASK_FILES contained no paths"
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
    state_root = Path(
        os.getenv(
            "MAGELLAN_STATE_ROOT",
            "runtime-state",
        )
    ).resolve()
    repository_root = Path(
        os.getenv(
            "MAGELLAN_REPOSITORY_ROOT",
            ".",
        )
    ).resolve()
    remote_state_root = Path(
        os.getenv(
            "MAGELLAN_REMOTE_STATE_ROOT",
            str(state_root),
        )
    )
    ssh_user = os.getenv(
        "MAGELLAN_SSH_USER",
        os.getenv("USER", "WILL"),
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
    local_node = cluster.get_node(node_id)

    registry = PersistentTaskRegistry.from_files(
        paths=_task_files(),
        state_root=state_root,
        local_node_id=local_node.id,
    )

    artifact_manager = ArtifactManager(
        registry=registry,
    )

    artifact_client = ArtifactClient(
        cluster=cluster,
    )

    artifact_transfer = RsyncArtifactTransfer(
        cluster=cluster,
        manager=artifact_manager,
        ssh_user=ssh_user,
        remote_state_root=remote_state_root,
    )

    prefetch_service = ArtifactPrefetchService(
        manager=artifact_manager,
        client=artifact_client,
        transfer=artifact_transfer,
    )

    graph = ClusterGraph(cluster)

    carbon_store = CarbonStore(
        cluster=cluster,
        datasets_directory=datasets_path,
    )

    clock = MagellanClock(policy.clock)

    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id=local_node.id,
        repository_root=repository_root,
        artifact_manager=artifact_manager,
    )
    checkpoint_manager = CheckpointManager(
        registry=registry,
    )

    runtime.reconcile()

    bid_store = BidStore()
    bid_client = BidClient(cluster)

    bid_arbiter = BidArbiter(
        store=bid_store,
        registry=registry,
        local_node_id=local_node.id,
        capacity=local_node.capacity,
        bid_window_seconds=cluster.bid_window_seconds,
    )

    transfer = RsyncCheckpointTransfer(
        cluster=cluster,
        registry=registry,
        ssh_user=ssh_user,
        remote_state_root=remote_state_root,
    )

    migration_service = MigrationService(
        local_node=local_node,
        cluster=cluster,
        registry=registry,
        runtime=runtime,
        transfer=transfer,
        artifact_manager=artifact_manager,
        prefetch_service=prefetch_service,
        checkpoint_manager=checkpoint_manager,
        client=MigrationClient(
            cluster=cluster,
            activation_timeout_seconds=(
                policy.migration.activation_timeout_seconds
            ),
        ),
        broadcaster=OwnershipBroadcaster(
            cluster=cluster,
            local_node_id=local_node.id,
        ),
    )

    scheduler_service = SchedulerService(
        local_node=local_node,
        cluster=cluster,
        policy=policy,
        graph=graph,
        carbon_store=carbon_store,
        clock=clock,
        registry=registry,
        runtime=runtime,
        bid_client=bid_client,
        migration_service=migration_service,
        checkpoint_manager=checkpoint_manager,
        prefetch_service=prefetch_service,
    )

    return DaemonContext(
        cluster=cluster,
        policy=policy,
        local_node=local_node,
        graph=graph,
        carbon_store=carbon_store,
        clock=clock,
        registry=registry,
        runtime=runtime,
        bid_store=bid_store,
        bid_arbiter=bid_arbiter,
        bid_client=bid_client,
        migration_service=migration_service,
        scheduler_service=scheduler_service,
        checkpoint_manager=checkpoint_manager,
        artifact_manager=artifact_manager,
        prefetch_service=prefetch_service,
        artifact_client=artifact_client,
    )
