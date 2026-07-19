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
from magellan.migration.journal import MigrationJournal
from magellan.migration.service import MigrationService
from magellan.migration.transfer import (
    RsyncCheckpointTransfer,
)
from magellan.runtime.accounting import RuntimeAccountingService
from magellan.runtime.clock import MagellanClock
from magellan.runtime.local_process import (
    LocalProcessRuntime,
)
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)

from magellan.runtime.checkpoint import CheckpointManager
from magellan.runtime.completion import CompletionManager
from magellan.runtime.pause import PauseService
from magellan.runtime.recovery import FailureRecoveryService
from magellan.reconciliation.client import ReconciliationClient
from magellan.reconciliation.service import (
    DistributedReconciliationService,
)

from magellan.artifacts.client import ArtifactClient
from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.prefetch import (
    ArtifactPrefetchService,
)
from magellan.artifacts.transfer import (
    RsyncArtifactTransfer,
)
from magellan.submission.catalog import TaskCatalogStore
from magellan.submission.client import TaskCatalogClient
from magellan.submission.service import TaskSubmissionService


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
    completion_manager: CompletionManager
    recovery_service: FailureRecoveryService
    pause_service: PauseService
    accounting_service: RuntimeAccountingService
    artifact_manager: ArtifactManager
    artifact_client: ArtifactClient
    prefetch_service: ArtifactPrefetchService
    migration_journal: MigrationJournal
    reconciliation_service: DistributedReconciliationService
    task_catalog: TaskCatalogStore
    submission_service: TaskSubmissionService


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

    task_catalog = TaskCatalogStore(
        state_root=state_root,
        local_node_id=local_node.id,
    )

    registry = PersistentTaskRegistry.from_files(
        paths=_task_files(),
        state_root=state_root,
        local_node_id=local_node.id,
    )
    for dynamic_definition in task_catalog.materialized_definitions():
        registry.register_definition(dynamic_definition)

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

    completion_manager = CompletionManager(
        registry=registry,
    )

    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id=local_node.id,
        repository_root=repository_root,
        artifact_manager=artifact_manager,
        completion_manager=completion_manager,
    )
    checkpoint_manager = CheckpointManager(
        registry=registry,
    )

    accounting_service = RuntimeAccountingService(
        local_node=local_node,
        cluster=cluster,
        policy=policy,
        graph=graph,
        carbon_store=carbon_store,
        clock=clock,
        registry=registry,
    )

    pause_service = PauseService(
        local_node_id=local_node.id,
        policy=policy.pause,
        clock=clock,
        registry=registry,
        runtime=runtime,
        accounting_service=accounting_service,
    )

    runtime.reconcile()

    bid_store = BidStore(
        reservation_ttl_seconds=(
            cluster.reservation_ttl_seconds
        ),
        state_file=state_root / "control" / "bids.json",
    )
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

    broadcaster = OwnershipBroadcaster(
        cluster=cluster,
        local_node_id=local_node.id,
    )

    migration_journal = MigrationJournal(state_root)

    migration_service = MigrationService(
        local_node=local_node,
        cluster=cluster,
        registry=registry,
        runtime=runtime,
        transfer=transfer,
        artifact_manager=artifact_manager,
        prefetch_service=prefetch_service,
        checkpoint_manager=checkpoint_manager,
        bid_client=bid_client,
        bid_store=bid_store,
        accounting_service=accounting_service,
        journal=migration_journal,
        reconciliation_policy=policy.reconciliation,
        client=MigrationClient(
            cluster=cluster,
            activation_timeout_seconds=(
                policy.migration.activation_timeout_seconds
            ),
        ),
        broadcaster=broadcaster,
    )

    recovery_service = FailureRecoveryService(
        local_node_id=local_node.id,
        policy=policy.recovery,
        registry=registry,
        runtime=runtime,
        checkpoint_manager=checkpoint_manager,
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
        broadcaster=broadcaster,
        pause_service=pause_service,
        accounting_service=accounting_service,
    )

    submission_service = TaskSubmissionService(
        local_node=local_node,
        cluster=cluster,
        catalog=task_catalog,
        registry=registry,
        runtime=runtime,
    )

    reconciliation_service = DistributedReconciliationService(
        local_node_id=local_node.id,
        policy=policy.reconciliation,
        registry=registry,
        client=ReconciliationClient(
            cluster=cluster,
            local_node_id=local_node.id,
        ),
        migration_service=migration_service,
        catalog=task_catalog,
        catalog_client=TaskCatalogClient(
            cluster=cluster,
            local_node_id=local_node.id,
        ),
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
        completion_manager=completion_manager,
        recovery_service=recovery_service,
        pause_service=pause_service,
        accounting_service=accounting_service,
        artifact_manager=artifact_manager,
        prefetch_service=prefetch_service,
        artifact_client=artifact_client,
        migration_journal=migration_journal,
        reconciliation_service=reconciliation_service,
        task_catalog=task_catalog,
        submission_service=submission_service,
    )
