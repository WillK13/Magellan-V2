from magellan.models.types import TaskProfile
from magellan.state.task_registry import TaskRegistry


def test_registry_filters_owned_tasks() -> None:
    boston_task = TaskProfile(
        task_id="task-boston",
        workload_type="test",
        current_node_id="boston",
        power_kw=0.5,
        checkpoint_bytes=100,
    )

    virginia_task = TaskProfile(
        task_id="task-virginia",
        workload_type="test",
        current_node_id="virginia",
        power_kw=0.5,
        checkpoint_bytes=100,
    )

    registry = TaskRegistry(
        [boston_task, virginia_task]
    )

    assert registry.count_owned("boston") == 1
    assert registry.count_owned("virginia") == 1

    assert (
        registry.owned_tasks("boston")[0].task_id
        == "task-boston"
    )
