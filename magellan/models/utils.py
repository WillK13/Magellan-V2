from __future__ import annotations

from collections.abc import Sequence


BYTES_PER_GB = 1_000_000_000


def seconds_to_hours(seconds: float) -> float:
    return seconds / 3600.0


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / BYTES_PER_GB


def transfer_seconds(
    size_bytes: int,
    bandwidth_mbps: float,
    latency_ms: float,
) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")

    payload_bits = size_bytes * 8
    bandwidth_bits_per_second = bandwidth_mbps * 1_000_000

    return (
        payload_bits / bandwidth_bits_per_second
        + latency_ms / 1000.0
    )


def minmax_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0.0 for _ in values]

    span = maximum - minimum
    return [(value - minimum) / span for value in values]
