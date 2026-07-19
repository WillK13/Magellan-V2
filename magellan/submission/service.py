from __future__ import annotations

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.runtime.local_process import LocalProcessRuntime
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.submission.catalog import TaskCatalogStore
from magellan.submission.models import (
    TaskDefinitionRecord,
    TaskDefinitionSubmission,
    TaskRunRecord,
    TaskRunSubmission,
    TaskRunView,
)


class TaskSubmissionService:
    def __init__(
        self,
        local_node: NodeConfig,
        cluster: ClusterConfig,
        catalog: TaskCatalogStore,
        registry: PersistentTaskRegistry,
        runtime: LocalProcessRuntime,
    ) -> None:
        self._local_node = local_node
        self._cluster = cluster
        self._catalog = catalog
        self._registry = registry
        self._runtime = runtime

    def submit_definition(
        self,
        submission: TaskDefinitionSubmission,
    ) -> TaskDefinitionRecord:
        record, created = self._catalog.register_definition(
            submission,
            origin_node_id=self._local_node.id,
        )
        print(
            f"[definition-{'created' if created else 'idempotent'}] "
            f"definition={record.definition_id}@{record.revision} "
            f"digest={record.digest}",
            flush=True,
        )
        return record

    def create_run(self, submission: TaskRunSubmission) -> TaskRunView:
        owner = submission.initial_owner_node_id or self._local_node.id
        self._cluster.get_node(owner)
        if submission.auto_start and owner != self._local_node.id:
            raise ValueError(
                "auto_start requires the submission peer to be the initial owner"
            )
        record, created = self._catalog.create_run(
            submission,
            owner_node_id=owner,
            origin_node_id=self._local_node.id,
        )
        definition = self._catalog.materialize_run(record)
        self._registry.register_definition(definition)

        if submission.auto_start and created:
            state = self._runtime.start(record.run_id)
        else:
            state = self._registry.get_state(record.run_id)

        print(
            f"[run-{'created' if created else 'idempotent'}] "
            f"run={record.run_id} definition={record.definition_id}@{record.revision} "
            f"owner={record.initial_owner_node_id}",
            flush=True,
        )
        return TaskRunView(
            run=record,
            state=state.model_dump(mode="json"),
        )

    def view_run(self, run_id: str) -> TaskRunView:
        record = self._catalog.get_run(run_id)
        state = self._registry.get_state(run_id)
        return TaskRunView(run=record, state=state.model_dump(mode="json"))

    def list_runs(self) -> list[TaskRunView]:
        return [self.view_run(item.run_id) for item in self._catalog.list_runs()]
