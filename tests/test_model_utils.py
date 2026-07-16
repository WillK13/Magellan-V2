import pytest

from magellan.models.utils import (
    minmax_normalize,
    transfer_seconds,
)


def test_minmax_normalize() -> None:
    result = minmax_normalize([10.0, 20.0, 30.0])

    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_normalize_equal_values() -> None:
    assert minmax_normalize([5.0, 5.0]) == [0.0, 0.0]


def test_transfer_seconds_has_correct_units() -> None:
    # One decimal GB = 8 gigabits.
    # At 1,000 megabits/s, payload time is 8 seconds.
    result = transfer_seconds(
        size_bytes=1_000_000_000,
        bandwidth_mbps=1000,
        latency_ms=50,
    )

    assert result == pytest.approx(8.05)
