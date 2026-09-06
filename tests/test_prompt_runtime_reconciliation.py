import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from magellan.daemon.scheduler_service import SchedulerService
from magellan.runtime.local_process import RuntimeReconcileEvent
from magellan.state.task_models import TaskStatus


class _Registry:
    def __init__(self, *, running=None, paused=None):
        self.running = list(running or [])
        self.paused = list(paused or [])

    def running_owned_task_ids(self, _node_id):
        return list(self.running)

    def paused_owned_task_ids(self, _node_id):
        return list(self.paused)


class _Runtime:
    def __init__(self, events=None):
        self.events = dict(events or {})
        self.calls = []

    def reconcile_task(self, task_id):
        self.calls.append(task_id)
        return self.events.get(task_id)


def _service(registry, runtime):
    service = object.__new__(SchedulerService)
    service._local_node = SimpleNamespace(id="boston")
    service._registry = registry
    service._runtime = runtime
    service._task_operation_locks = {}
    service._broadcast_completed_states = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_prompt_reconciliation_handles_operator_only_lifecycle_without_scheduler():
    event = RuntimeReconcileEvent(
        task_id="operator-task",
        status=TaskStatus.COMPLETED,
        exit_code=0,
    )
    registry = _Registry(running=["operator-task"])
    runtime = _Runtime(events={"operator-task": event})
    service = _service(registry, runtime)

    events = await service.reconcile_runtime_once()

    assert events == [event]
    assert runtime.calls == ["operator-task"]
    service._broadcast_completed_states.assert_awaited_once()
    # The lifecycle path never consults the task catalog or the
    # scheduler_mode label. operator_only prevents policy decisions, not
    # completion/failure detection.
    assert not hasattr(service, "_task_catalog")


@pytest.mark.asyncio
async def test_prompt_reconciliation_shares_task_operation_lock():
    registry = _Registry(running=["locked-task"])
    runtime = _Runtime()
    service = _service(registry, runtime)

    operation_lock = service._task_operation_lock("locked-task")
    await operation_lock.acquire()
    pending = asyncio.create_task(service.reconcile_runtime_once())
    await asyncio.sleep(0)

    assert runtime.calls == []

    operation_lock.release()
    await pending

    assert runtime.calls == ["locked-task"]


@pytest.mark.asyncio
async def test_runtime_reconciliation_loop_uses_recovery_scan_cadence_not_epoch():
    service = object.__new__(SchedulerService)
    service._local_node = SimpleNamespace(id="boston")
    service._policy = SimpleNamespace(
        recovery=SimpleNamespace(scan_interval_seconds=0.01)
    )

    stop_event = asyncio.Event()
    calls = 0

    async def reconcile_once():
        nonlocal calls
        calls += 1
        stop_event.set()
        return []

    service.reconcile_runtime_once = reconcile_once

    await service.run_runtime_reconciliation(stop_event)

    assert calls == 1
