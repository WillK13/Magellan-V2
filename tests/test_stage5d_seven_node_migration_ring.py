from __future__ import annotations

from magellan.experiments.stage5d import (
    STAGE5D_RING,
    expected_hops,
    progress_is_monotonic,
    stage5d_passes,
)


def _hop_rows() -> list[dict]:
    rows = []
    for index, (source, destination) in enumerate(expected_hops(), start=1):
        rows.append(
            {
                "hop_index": index,
                "source_node_id": source,
                "destination_node_id": destination,
                "source_daemon_git_sha": "a" * 40,
                "destination_daemon_git_sha": "a" * 40,
                "owner_before": source,
                "generation_before": index - 1,
                "migrated": True,
                "owner_after": destination,
                "generation_after": index,
                "destination_status_after": "running",
                "destination_pid_after": 1000 + index,
                "progress_before": 10.0 + index,
                "progress_after": 11.0 + index,
                "source_record_role": "source",
                "source_record_status": "activated",
                "destination_record_role": "destination",
                "destination_record_status": "activated",
                "bid_status": "accepted",
                "migration_id": f"migration-{index}",
                "total_downtime_seconds": 1.0 + index,
                "ownership_converged": True,
            }
        )
    return rows


def test_ring_covers_every_node_as_source_and_destination() -> None:
    hops = expected_hops()
    assert len(hops) == 7
    assert {source for source, _ in hops} == set(STAGE5D_RING[:-1])
    assert {destination for _, destination in hops} == set(STAGE5D_RING[:-1])
    assert hops[0] == ("boston", "california")
    assert hops[-1] == ("virginia", "boston")


def test_progress_monotonic() -> None:
    assert progress_is_monotonic(_hop_rows())
    rows = _hop_rows()
    rows[4]["progress_after"] = 1.0
    assert not progress_is_monotonic(rows)


def test_stage5d_passes_complete_ring() -> None:
    assert stage5d_passes(
        hop_rows=_hop_rows(),
        final_owner_node_id="boston",
        final_generation=7,
        ownership_converged_final=True,
        expected_git_sha="a" * 40,
    )


def test_stage5d_rejects_generation_skip() -> None:
    rows = _hop_rows()
    rows[3]["generation_after"] = 5
    assert not stage5d_passes(
        hop_rows=rows,
        final_owner_node_id="boston",
        final_generation=7,
        ownership_converged_final=True,
        expected_git_sha="a" * 40,
    )
