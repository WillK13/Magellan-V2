from __future__ import annotations

import csv
from pathlib import Path

import pytest

from magellan.experiments.stage4a2 import (
    select_representative_edges,
    summarize_migration_accuracy,
    summarize_profile_samples,
)


def _write_edges(path: Path) -> None:
    fieldnames = [
        "source_node_id",
        "destination_node_id",
        "measured_bandwidth_median_mbps",
        "measured_rtt_median_ms",
    ]
    rows = [
        {"source_node_id": "a", "destination_node_id": "b", "measured_bandwidth_median_mbps": 10, "measured_rtt_median_ms": 200},
        {"source_node_id": "b", "destination_node_id": "a", "measured_bandwidth_median_mbps": 100, "measured_rtt_median_ms": 80},
        {"source_node_id": "a", "destination_node_id": "c", "measured_bandwidth_median_mbps": 50, "measured_rtt_median_ms": 120},
        {"source_node_id": "c", "destination_node_id": "a", "measured_bandwidth_median_mbps": 55, "measured_rtt_median_ms": 110},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_select_representative_edges_uses_bandwidth_extremes_and_median(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    _write_edges(path)
    selected = select_representative_edges(path)
    assert [edge.role for edge in selected] == ["short", "medium", "long"]
    assert selected[0].edge == "b->a"
    assert selected[-1].edge == "a->b"
    assert selected[1].edge in {"a->c", "c->a"}


def test_profile_summary_reports_measured_resource_distributions() -> None:
    summary = summarize_profile_samples(
        [
            {"cpu_utilization_percent": 50, "memory_rss_mb": 100, "checkpoint_bytes": 10},
            {"cpu_utilization_percent": 100, "memory_rss_mb": 200, "checkpoint_bytes": 20},
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["cpu_utilization_percent"]["median"] == pytest.approx(75)
    assert summary["memory_rss_mb"]["maximum"] == pytest.approx(200)
    assert summary["checkpoint_bytes"]["median"] == pytest.approx(15)


def test_migration_summary_does_not_encode_acceptance_thresholds() -> None:
    summary = summarize_migration_accuracy(
        [
            {
                "predicted_checkpoint_seconds": 1,
                "actual_checkpoint_seconds": 2,
                "predicted_transfer_seconds": 4,
                "actual_transfer_seconds": 4,
                "predicted_restore_seconds": 3,
                "actual_restore_seconds": 2,
                "predicted_downtime_seconds": 8,
                "actual_downtime_seconds": 10,
            }
        ]
    )
    assert summary["checkpoint_absolute_error_percent"]["median"] == pytest.approx(50)
    assert summary["transfer_absolute_error_percent"]["median"] == pytest.approx(0)
    assert summary["restore_absolute_error_percent"]["median"] == pytest.approx(50)
    assert summary["downtime_absolute_error_percent"]["median"] == pytest.approx(20)
    assert "passed" not in summary
