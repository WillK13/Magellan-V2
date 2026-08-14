from __future__ import annotations

import pytest

from magellan.experiments.measurement import (
    absolute_percent_error,
    directed_edge_pairs,
    percentile,
    predict_transfer_seconds,
    signed_percent_error,
    summarize_samples,
)


def test_sample_summary_is_deterministic() -> None:
    summary = summarize_samples([10.0, 20.0, 30.0, 40.0])
    assert summary.count == 4
    assert summary.minimum == 10.0
    assert summary.mean == 25.0
    assert summary.median == 25.0
    assert summary.p95 == pytest.approx(38.5)
    assert summary.maximum == 40.0
    assert summary.standard_deviation > 0
    assert summary.coefficient_of_variation > 0


def test_percent_error_and_transfer_prediction() -> None:
    assert signed_percent_error(12.0, 10.0) == pytest.approx(20.0)
    assert absolute_percent_error(8.0, 10.0) == pytest.approx(20.0)
    assert signed_percent_error(1.0, 0.0) is None
    assert predict_transfer_seconds(
        size_bytes=125_000_000,
        bandwidth_mbps=100.0,
        latency_ms=20.0,
    ) == pytest.approx(10.02)
    assert predict_transfer_seconds(
        size_bytes=125_000_000,
        bandwidth_mbps=100.0,
        latency_ms=20.0,
        bandwidth_is_end_to_end=True,
    ) == pytest.approx(10.0)


def test_directed_edge_pairs_cover_complete_mesh() -> None:
    pairs = directed_edge_pairs(["a", "b", "c"])
    assert len(pairs) == 6
    assert ("a", "a") not in pairs
    assert ("a", "b") in pairs
    assert ("b", "a") in pairs


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 95)


def test_directed_edge_pairs_scale_with_membership() -> None:
    seven = [f"n{i}" for i in range(7)]
    eight = [f"n{i}" for i in range(8)]

    assert len(directed_edge_pairs(seven)) == 42
    assert len(directed_edge_pairs(eight)) == 56
    assert all(
        source != destination
        for source, destination in directed_edge_pairs(eight)
    )
