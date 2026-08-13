from __future__ import annotations

from magellan.workloads.counter import completion_target, ensure_checkpoint_padding


def test_checkpoint_padding_is_created_and_reused(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint" / "counter.json"
    padding = ensure_checkpoint_padding(checkpoint, 4096, chunk_bytes=257)
    assert padding is not None
    assert padding.name == "payload.bin"
    assert padding.stat().st_size == 4096
    before = padding.stat().st_mtime_ns

    again = ensure_checkpoint_padding(checkpoint, 4096, chunk_bytes=257)
    assert again == padding
    assert padding.stat().st_mtime_ns == before


def test_zero_padding_does_not_create_file(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint" / "counter.json"
    assert ensure_checkpoint_padding(checkpoint, 0) is None
    assert not (checkpoint.parent / "payload.bin").exists()


def test_completion_target_changes_only_after_migration() -> None:
    assert completion_target(
        current_value=10,
        configured_max_value=1000,
        current_node_id="boston",
        initial_node_id="boston",
        complete_after_migration_steps=5,
    ) == 1000
    assert completion_target(
        current_value=10,
        configured_max_value=1000,
        current_node_id="france",
        initial_node_id="boston",
        complete_after_migration_steps=5,
    ) == 15
