from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable

from magellan.models.utils import transfer_seconds


@dataclass(frozen=True)
class SampleSummary:
    count: int
    minimum: float
    mean: float
    median: float
    p95: float
    maximum: float
    standard_deviation: float
    coefficient_of_variation: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "maximum": self.maximum,
            "standard_deviation": self.standard_deviation,
            "coefficient_of_variation": self.coefficient_of_variation,
        }


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("At least one sample is required")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be in [0, 100]")
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values: Iterable[float]) -> SampleSummary:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("At least one sample is required")
    if any(value < 0 or not math.isfinite(value) for value in samples):
        raise ValueError("Samples must be finite and non-negative")

    average = mean(samples)
    deviation = pstdev(samples) if len(samples) > 1 else 0.0
    return SampleSummary(
        count=len(samples),
        minimum=min(samples),
        mean=average,
        median=median(samples),
        p95=percentile(samples, 95),
        maximum=max(samples),
        standard_deviation=deviation,
        coefficient_of_variation=(deviation / average if average > 0 else 0.0),
    )


def signed_percent_error(predicted: float, actual: float) -> float | None:
    if actual <= 0:
        return None
    return (predicted - actual) / actual * 100.0


def absolute_percent_error(predicted: float, actual: float) -> float | None:
    value = signed_percent_error(predicted, actual)
    return None if value is None else abs(value)


def predict_transfer_seconds(
    *,
    size_bytes: int,
    bandwidth_mbps: float,
    latency_ms: float,
) -> float:
    return transfer_seconds(
        size_bytes=size_bytes,
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
    )


def directed_edge_pairs(node_ids: Iterable[str]) -> list[tuple[str, str]]:
    ids = list(node_ids)
    return [(source, destination) for source in ids for destination in ids if source != destination]
