from __future__ import annotations

from contextlib import contextmanager
from contextvars import Context
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from magellan.daemon.scheduler_service import SchedulerService
from magellan.policy.models import AdaptiveTaskPolicyState, WeightVector
from magellan.policy.store import AdaptivePolicyStore


def _state(task_id: str) -> AdaptiveTaskPolicyState:
    weights = WeightVector(time=0.25, carbon=0.5, cost=0.25)
    return AdaptiveTaskPolicyState(
        task_id=task_id,
        baseline_weights=weights,
        effective_weights=weights,
    )


def test_store_keeps_immediate_durability_outside_batch(tmp_path) -> None:
    store = AdaptivePolicyStore(tmp_path)

    with patch.object(store, "_persist", wraps=store._persist) as persist:
        store.put(_state("task-a"))
        store.put(_state("task-b"))

    assert persist.call_count == 2
    restarted = AdaptivePolicyStore(tmp_path)
    assert {state.task_id for state in restarted.list_states()} == {
        "task-a",
        "task-b",
    }


def test_store_batches_many_updates_into_one_durable_flush(tmp_path) -> None:
    store = AdaptivePolicyStore(tmp_path)

    with patch.object(store, "_persist", wraps=store._persist) as persist:
        with store.batch():
            store.put(_state("task-a"))
            store.put(_state("task-b"))
            store.put(_state("task-c"))
            assert persist.call_count == 0
        assert persist.call_count == 1

    restarted = AdaptivePolicyStore(tmp_path)
    assert {state.task_id for state in restarted.list_states()} == {
        "task-a",
        "task-b",
        "task-c",
    }


def test_nested_batch_flushes_only_once_and_exception_still_flushes(tmp_path) -> None:
    store = AdaptivePolicyStore(tmp_path)

    with patch.object(store, "_persist", wraps=store._persist) as persist:
        with pytest.raises(RuntimeError, match="boom"):
            with store.batch():
                store.put(_state("task-a"))
                with store.batch():
                    store.put(_state("task-b"))
                assert persist.call_count == 0
                raise RuntimeError("boom")

        assert persist.call_count == 1

    restarted = AdaptivePolicyStore(tmp_path)
    assert restarted.get("task-a") is not None
    assert restarted.get("task-b") is not None


def test_batch_deferral_is_context_local(tmp_path) -> None:
    store = AdaptivePolicyStore(tmp_path)

    with patch.object(store, "_persist", wraps=store._persist) as persist:
        with store.batch():
            store.put(_state("scheduler-task"))
            assert persist.call_count == 0

            # Simulate an unrelated API/peer execution context. It must retain
            # the store's default immediate-durability behavior even while the
            # scheduler context has deferred its own writes.
            Context().run(store.put, _state("peer-task"))
            assert persist.call_count == 1

        # The peer's immediate full-state snapshot already included the
        # scheduler task, so there is no dirty state left to flush again.
        assert persist.call_count == 1

    restarted = AdaptivePolicyStore(tmp_path)
    assert restarted.get("scheduler-task") is not None
    assert restarted.get("peer-task") is not None


@pytest.mark.asyncio
async def test_scheduler_batches_one_whole_local_epoch() -> None:
    events: list[str] = []

    class RecordingStore:
        @contextmanager
        def batch(self):
            events.append("batch-enter")
            try:
                yield
            finally:
                events.append("batch-exit")

    service = object.__new__(SchedulerService)
    service._runtime = SimpleNamespace(reconcile=lambda: None)
    service._broadcast_completed_states = AsyncMock()
    service._registry = SimpleNamespace(
        running_owned_task_ids=lambda _node_id: ["task-a", "task-b", "task-c"]
    )
    service._local_node = SimpleNamespace(id="boston")
    service._clock = SimpleNamespace(
        now=lambda: pd.Timestamp("2024-01-01T00:00:00Z")
    )
    service._adaptive_policy_service = SimpleNamespace(store=RecordingStore())
    service._evaluate_task = AsyncMock(
        side_effect=lambda task_id, _trace_time: events.append(task_id)
    )

    await service.run_epoch()

    assert events == [
        "batch-enter",
        "task-a",
        "task-b",
        "task-c",
        "batch-exit",
    ]
    assert service._evaluate_task.await_count == 3
