from magellan.models.types import TaskProfile
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)
from magellan.state.task_models import (
    LocalProcessSpec,
    TaskDefinition,
    TaskStatus,
)


def test_registry_persists_state(tmp_path) -> None:
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="counter-test",
            workload_type="counter",
            current_node_id="boston",
            power_kw=0.1,
            checkpoint_bytes=100,
        ),
        runtime=LocalProcessSpec(
            module="magellan.workloads.counter",
        ),
    )

    first = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )

    first.mark_running(
        "counter-test",
        pid=12345,
    )

    second = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )

    state = second.get_state("counter-test")

    assert state.owner_node_id == "boston"
    assert state.status == TaskStatus.RUNNING
    assert state.pid == 12345
