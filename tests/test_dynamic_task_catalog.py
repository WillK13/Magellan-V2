from __future__ import annotations

from pathlib import Path

import pytest

from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec
from magellan.submission.catalog import TaskCatalogStore
from magellan.submission.models import (
    TaskDefinitionSubmission,
    TaskRunSubmission,
    TaskTemplateProfile,
)


def definition(interval: str = "0.1") -> TaskDefinitionSubmission:
    return TaskDefinitionSubmission(
        definition_id="dynamic-counter",
        profile=TaskTemplateProfile(
            workload_type="counter",
            power_kw=0.1,
            checkpoint_bytes=1024,
            prestaged_node_ids={"boston", "virginia"},
            estimated_remaining_seconds=60,
            priority=7,
        ),
        runtime=LocalProcessSpec(
            module="magellan.workloads.counter",
            arguments=[
                "--checkpoint-file",
                "{checkpoint_file}",
                "--interval-seconds",
                interval,
            ],
            checkpoint_relative_path="checkpoint/counter.json",
        ),
    )


def test_definition_is_immutable_and_changed_payload_creates_revision(tmp_path) -> None:
    catalog = TaskCatalogStore(tmp_path, "boston")

    first, created = catalog.register_definition(definition())
    same, same_created = catalog.register_definition(definition())
    second, second_created = catalog.register_definition(definition("0.2"))

    assert created is True
    assert same_created is False
    assert same.revision == first.revision == 1
    assert same.digest == first.digest
    assert second_created is True
    assert second.revision == 2
    assert second.digest != first.digest


def test_run_submission_is_idempotent_and_rejects_key_reuse(tmp_path) -> None:
    catalog = TaskCatalogStore(tmp_path, "boston")
    catalog.register_definition(definition())
    request = TaskRunSubmission(
        definition_id="dynamic-counter",
        idempotency_key="client-request-1",
        initial_owner_node_id="boston",
    )

    first, created = catalog.create_run(request, owner_node_id="boston")
    second, second_created = catalog.create_run(request, owner_node_id="boston")

    assert created is True
    assert second_created is False
    assert second.run_id == first.run_id

    with pytest.raises(ValueError, match="different request"):
        catalog.create_run(
            request.model_copy(update={"labels": {"different": "true"}}),
            owner_node_id="boston",
        )


def test_catalog_and_dynamic_run_survive_restart(tmp_path) -> None:
    catalog = TaskCatalogStore(tmp_path, "boston")
    record, _ = catalog.register_definition(definition())
    run, _ = catalog.create_run(
        TaskRunSubmission(
            definition_id=record.definition_id,
            revision=record.revision,
            idempotency_key="restart-key",
        ),
        owner_node_id="boston",
    )

    restarted = TaskCatalogStore(tmp_path, "boston")
    restored_run = restarted.get_run(run.run_id)
    materialized = restarted.materialize_run(restored_run)

    assert restored_run == run
    assert materialized.profile.task_id == run.run_id
    assert materialized.profile.current_node_id == "boston"
    assert materialized.profile.priority == 7


def test_registry_accepts_runtime_created_definition(tmp_path) -> None:
    catalog = TaskCatalogStore(tmp_path, "boston")
    catalog.register_definition(definition())
    run, _ = catalog.create_run(
        TaskRunSubmission(
            definition_id="dynamic-counter",
            idempotency_key="registry-key",
        ),
        owner_node_id="boston",
    )
    registry = PersistentTaskRegistry(
        definitions=[],
        state_root=tmp_path,
        local_node_id="boston",
    )

    assert registry.register_definition(catalog.materialize_run(run)) is True
    assert registry.register_definition(catalog.materialize_run(run)) is False
    assert registry.get_state(run.run_id).owner_node_id == "boston"
    assert registry.get_state(run.run_id).status.value == "stopped"
