from __future__ import annotations

from magellan.experiments.stage4e2 import percentile


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0


def test_percentile_handles_singleton_and_empty() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([7.0], 0.95) == 7.0
