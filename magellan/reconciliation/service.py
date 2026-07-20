from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from magellan.config.policy_models import ReconciliationPolicy
from magellan.migration.service import MigrationService
from magellan.reconciliation.client import ReconciliationClient
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.submission.catalog import TaskCatalogStore
from magellan.submission.client import TaskCatalogClient
from magellan.policy.store import AdaptivePolicyStore


class DistributedReconciliationService:
    """Repairs task catalogs, missed ownership updates, and migrations."""

    def __init__(
        self,
        local_node_id: str,
        policy: ReconciliationPolicy,
        registry: PersistentTaskRegistry,
        client: ReconciliationClient,
        migration_service: MigrationService,
        catalog: TaskCatalogStore | None = None,
        catalog_client: TaskCatalogClient | None = None,
        adaptive_policy_store: AdaptivePolicyStore | None = None,
    ) -> None:
        self._local_node_id = local_node_id
        self._policy = policy
        self._registry = registry
        self._client = client
        self._migration_service = migration_service
        self._catalog = catalog
        self._catalog_client = catalog_client
        self._adaptive_policy_store = adaptive_policy_store
        self.last_completed_at_utc: datetime | None = None
        self.last_applied_updates = 0
        self.last_catalog_definitions = 0
        self.last_catalog_runs = 0

    async def _reconcile_catalog(self) -> tuple[int, int]:
        if self._catalog is None or self._catalog_client is None:
            return 0, 0
        definition_count = 0
        run_count = 0
        snapshots = await self._catalog_client.fetch_all()
        for snapshot in snapshots:
            added_definitions, added_runs = self._catalog.merge_snapshot(snapshot)
            definition_count += len(added_definitions)
            for run in added_runs:
                definition = self._catalog.materialize_run(run)
                self._registry.register_definition(definition)
                run_count += 1
        if definition_count or run_count:
            print(
                f"[catalog-reconcile-applied] node={self._local_node_id} "
                f"definitions={definition_count} runs={run_count}",
                flush=True,
            )
        return definition_count, run_count

    async def run_once(self) -> int:
        if not self._policy.enabled:
            return 0

        await self._migration_service.reconcile_durable_state()
        definitions, runs = await self._reconcile_catalog()
        applied = 0
        snapshots = await self._client.fetch_all()

        for snapshot in snapshots:
            for update in snapshot.updates:
                try:
                    if self._registry.apply_ownership(
                        task_id=update.task_id,
                        owner_node_id=update.owner_node_id,
                        generation=update.generation,
                        migration_id=update.last_migration_id,
                        migration_at_utc=update.migration_at_utc,
                        status=update.status,
                        completed_at_utc=update.completed_at_utc,
                        final_output_manifest_sha256=(
                            update.final_output_manifest_sha256
                        ),
                        final_output_bytes=update.final_output_bytes,
                        accounting=update.accounting,
                    ):
                        if update.artifact_digests:
                            self._registry.set_artifact_digests(
                                update.task_id,
                                update.artifact_digests,
                            )
                        if (
                            update.adaptive_policy is not None
                            and self._adaptive_policy_store is not None
                        ):
                            self._adaptive_policy_store.merge(
                                update.adaptive_policy
                            )
                        applied += 1
                except KeyError:
                    continue

        self.last_applied_updates = applied
        self.last_catalog_definitions = definitions
        self.last_catalog_runs = runs
        self.last_completed_at_utc = datetime.now(timezone.utc)
        if applied:
            print(
                f"[reconcile-applied] node={self._local_node_id} "
                f"updates={applied}",
                flush=True,
            )
        return applied + definitions + runs

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.run_once()
            except Exception as exc:
                print(
                    f"[reconcile-warning] node={self._local_node_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._policy.scan_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass
