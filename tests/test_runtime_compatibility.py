from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.arbiter import BidArbiter
from magellan.bidding.models import BidRequest, BidStatus, TaskBidContext
from magellan.bidding.store import BidStore
from magellan.capabilities.checker import check_compatibility
from magellan.capabilities.models import (
    NodeRuntimeCapabilities,
    TaskCompatibilityRequirements,
)
from magellan.models.types import ActionType, ScoredAction
from magellan.state.task_registry import TaskRegistry


def requirements() -> TaskCompatibilityRequirements:
    return TaskCompatibilityRequirements(
        architectures={"x86_64"},
        operating_systems={"linux"},
        required_commands={"mpirun"},
        required_runtimes={"openmpi": ">=4,<5"},
        required_features={"mpi", "application-checkpoint"},
    )


def test_compatibility_reports_all_hard_failures() -> None:
    result = check_compatibility(
        requirements(),
        NodeRuntimeCapabilities(
            architecture="aarch64",
            operating_system="linux",
            commands={"python3"},
            runtimes={},
            features={"application-checkpoint"},
        ),
    )
    assert result.compatible is False
    assert any("architecture" in reason for reason in result.reasons)
    assert any("mpirun" in reason for reason in result.reasons)
    assert any("openmpi" in reason for reason in result.reasons)
    assert any("mpi" in reason for reason in result.reasons)


def compatible_bid() -> BidRequest:
    return BidRequest(
        bid_id="compatibility-bid",
        epoch_id="epoch",
        task_id="dendro-task",
        source_node_id="boston",
        destination_node_id="virginia",
        task_context=TaskBidContext(
            workload_type="dendro-gr",
            compatibility=requirements(),
        ),
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=1,
            carbon_grams=1,
            cost_usd=1,
            normalized_time=0,
            normalized_carbon=0,
            normalized_cost=0,
            score=0.1,
        ),
    )


@pytest.mark.asyncio
async def test_destination_rejects_incompatible_runtime_without_credit() -> None:
    store = BidStore()
    arbiter = BidArbiter(
        store=store,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=1,
        bid_window_seconds=1,
        node_capabilities=NodeRuntimeCapabilities(
            architecture="aarch64",
            operating_system="linux",
            commands={"python3"},
            features={"application-checkpoint"},
        ),
    )
    await store.submit(compatible_bid())
    await arbiter.run_once(
        datetime.now(timezone.utc) + timedelta(seconds=2)
    )
    record = await store.get("compatibility-bid")
    assert record is not None
    assert record.status == BidStatus.REJECTED
    assert record.compatibility_fit is False
    assert record.compatibility_reasons
    assert await store.credit_for("dendro-task") == 0


def test_local_runtime_rejects_incompatible_architecture(tmp_path) -> None:
    import sys

    from magellan.artifacts.manager import ArtifactManager
    from magellan.models.types import TaskProfile
    from magellan.runtime.completion import CompletionManager
    from magellan.runtime.local_process import LocalProcessRuntime
    from magellan.state.persistent_registry import PersistentTaskRegistry
    from magellan.state.task_models import LocalProcessSpec, TaskDefinition

    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="arm-task",
            workload_type="command",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=0,
            compatibility=TaskCompatibilityRequirements(
                architectures={"aarch64"},
            ),
        ),
        runtime=LocalProcessSpec(
            adapter="command",
            command=[sys.executable, "-c", "import time; time.sleep(1)"],
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )
    runtime = LocalProcessRuntime(
        registry=registry,
        local_node_id="boston",
        repository_root=tmp_path,
        artifact_manager=ArtifactManager(registry),
        completion_manager=CompletionManager(registry),
        node_capabilities=NodeRuntimeCapabilities(
            architecture="x86_64",
        ),
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        runtime.start("arm-task")
