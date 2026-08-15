from magellan.experiments.migration_matrix import (
    row_is_calibrated,
    summarize_migration_rows,
)


def _row(
    *, edge: tuple[str, str], size: int, sample: int, calibrated: bool
) -> dict[str, object]:
    source, destination = edge
    return {
        "source_node_id": source,
        "destination_node_id": destination,
        "requested_payload_bytes": size,
        "sample": sample,
        "final_status": "completed",
        "candidate_calibration_source": (
            "measured_migration_ema" if calibrated else "configured_fallback"
        ),
        "candidate_transfer_model": "affine_migration_transport",
        "transfer_absolute_error_percent": 10.0 if calibrated else 90.0,
        "downtime_absolute_error_percent": 5.0 if calibrated else 5000.0,
        "checkpoint_absolute_error_percent": 2.0,
        "restore_absolute_error_percent": 3.0,
        "actual_transfer_seconds": 2.0,
        "actual_downtime_seconds": 3.0,
    }


def test_row_is_calibrated_requires_measured_workload_and_live_transfer_model() -> None:
    row = _row(edge=("a", "b"), size=10, sample=2, calibrated=True)
    assert row_is_calibrated(row)
    row["candidate_transfer_model"] = "configured_fallback"
    assert not row_is_calibrated(row)


def test_summary_keeps_cold_samples_but_excludes_them_from_headline_accuracy() -> None:
    rows = [
        _row(edge=("a", "b"), size=10, sample=1, calibrated=False),
        _row(edge=("a", "b"), size=10, sample=2, calibrated=True),
        _row(edge=("a", "b"), size=100, sample=1, calibrated=True),
    ]
    summary = summarize_migration_rows(rows)

    assert summary["total_sample_count"] == 3
    assert summary["calibrated_sample_count"] == 2
    assert summary["cold_or_uncalibrated_sample_count"] == 1
    assert (
        summary["overall_calibrated"]["transfer_absolute_error_percent"]["median"]
        == 10.0
    )
    assert (
        summary["overall_calibrated"]["downtime_absolute_error_percent"]["median"]
        == 5.0
    )
    assert len(summary["cases"]) == 2
