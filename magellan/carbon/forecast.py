from __future__ import annotations

from datetime import datetime
from math import sqrt
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field

from magellan.config.policy_models import CarbonForecastPolicy


class CarbonForecastEstimate(BaseModel):
    node_id: str = Field(min_length=1)
    average_g_per_kwh: float = Field(ge=0)
    current_g_per_kwh: float = Field(ge=0)
    source: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    freshness: str
    history_points: int = Field(ge=0)
    forecast_start_utc: datetime
    forecast_horizon_seconds: float = Field(ge=0)
    generated_at_utc: datetime
    latest_sample_at_utc: datetime | None = None
    sample_age_seconds: float | None = Field(default=None, ge=0)
    slope_g_per_kwh_per_hour: float | None = None
    residual_rmse: float | None = Field(default=None, ge=0)
    clamped: bool = False
    details: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict
    )


class CarbonForecastProvider(Protocol):
    def forecast(
        self,
        *,
        node_id: str,
        series: pd.Series,
        observed_at_utc: pd.Timestamp,
        forecast_start_utc: pd.Timestamp,
        duration_seconds: float,
        policy: CarbonForecastPolicy,
    ) -> CarbonForecastEstimate:
        ...


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _linear_fit(
    timestamps: list[pd.Timestamp],
    values: list[float],
) -> tuple[float, float, float]:
    """Return intercept at latest sample, slope/hour, and residual RMSE."""
    latest = timestamps[-1]
    xs = [
        (timestamp - latest).total_seconds() / 3600.0
        for timestamp in timestamps
    ]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    slope = 0.0
    if denominator > 0:
        slope = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(xs, values, strict=True)
        ) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(xs, values, strict=True)
    ]
    rmse = sqrt(sum(value * value for value in residuals) / len(residuals))
    return intercept, slope, rmse


class LinearTrendForecastProvider:
    """Bounded short-horizon regression with conservative fallbacks."""

    name = "linear_trend"

    def _fallback(
        self,
        *,
        node_id: str,
        policy: CarbonForecastPolicy,
        observed_at: pd.Timestamp,
        forecast_start: pd.Timestamp,
        duration_seconds: float,
        source: str,
        value: float,
        confidence: float,
        freshness: str,
        history_points: int,
        latest_sample_at: pd.Timestamp | None,
        sample_age_seconds: float | None,
    ) -> CarbonForecastEstimate:
        return CarbonForecastEstimate(
            node_id=node_id,
            average_g_per_kwh=max(0.0, value),
            current_g_per_kwh=max(0.0, value),
            source=source,
            confidence=max(0.0, min(1.0, confidence)),
            freshness=freshness,
            history_points=history_points,
            forecast_start_utc=forecast_start.to_pydatetime(warn=False),
            forecast_horizon_seconds=duration_seconds,
            generated_at_utc=observed_at.to_pydatetime(warn=False),
            latest_sample_at_utc=(
                latest_sample_at.to_pydatetime(warn=False)
                if latest_sample_at is not None
                else None
            ),
            sample_age_seconds=sample_age_seconds,
            details={
                "provider": policy.provider,
                "fallback": True,
            },
        )

    def forecast(
        self,
        *,
        node_id: str,
        series: pd.Series,
        observed_at_utc: pd.Timestamp,
        forecast_start_utc: pd.Timestamp,
        duration_seconds: float,
        policy: CarbonForecastPolicy,
    ) -> CarbonForecastEstimate:
        observed_at = _utc(observed_at_utc)
        forecast_start = _utc(forecast_start_utc)
        duration_seconds = max(0.0, float(duration_seconds))

        # The source series is sorted. Use index search instead of building a
        # full-length boolean mask for every scheduler forecast. Stage 4D can
        # issue tens of thousands of forecasts against the same annual trace;
        # this preserves the exact trailing-history semantics while making the
        # lookup O(log n) rather than O(n).
        history_end = int(series.index.searchsorted(observed_at, side="right"))
        history_start = max(0, history_end - policy.history_points)
        history = series.iloc[history_start:history_end]
        if history.empty:
            fallback = policy.configured_fallback_g_per_kwh
            if fallback is None:
                raise ValueError(
                    f"No carbon samples or configured fallback for {node_id}"
                )
            return self._fallback(
                node_id=node_id,
                policy=policy,
                observed_at=observed_at,
                forecast_start=forecast_start,
                duration_seconds=duration_seconds,
                source="configured_fallback",
                value=fallback,
                confidence=policy.fallback_confidence,
                freshness="unavailable",
                history_points=0,
                latest_sample_at=None,
                sample_age_seconds=None,
            )

        latest_at = history.index[-1]
        latest_value = float(history.iloc[-1])
        age_seconds = max(0.0, (observed_at - latest_at).total_seconds())
        freshness = (
            "fresh"
            if age_seconds <= policy.stale_after_seconds
            else "stale"
        )
        freshness_factor = max(
            0.0,
            1.0 - age_seconds / policy.stale_after_seconds,
        )

        if freshness == "stale":
            if policy.configured_fallback_g_per_kwh is not None:
                return self._fallback(
                    node_id=node_id,
                    policy=policy,
                    observed_at=observed_at,
                    forecast_start=forecast_start,
                    duration_seconds=duration_seconds,
                    source="configured_fallback",
                    value=policy.configured_fallback_g_per_kwh,
                    confidence=policy.fallback_confidence,
                    freshness="stale",
                    history_points=len(history),
                    latest_sample_at=latest_at,
                    sample_age_seconds=age_seconds,
                )
            return self._fallback(
                node_id=node_id,
                policy=policy,
                observed_at=observed_at,
                forecast_start=forecast_start,
                duration_seconds=duration_seconds,
                source="persistence",
                value=latest_value,
                confidence=0.0,
                freshness="stale",
                history_points=len(history),
                latest_sample_at=latest_at,
                sample_age_seconds=age_seconds,
            )

        if policy.provider == "persistence" or len(history) < policy.minimum_points:
            point_factor = min(1.0, len(history) / policy.minimum_points)
            return self._fallback(
                node_id=node_id,
                policy=policy,
                observed_at=observed_at,
                forecast_start=forecast_start,
                duration_seconds=duration_seconds,
                source="persistence",
                value=latest_value,
                confidence=policy.persistence_confidence
                * point_factor
                * freshness_factor,
                freshness="fresh",
                history_points=len(history),
                latest_sample_at=latest_at,
                sample_age_seconds=age_seconds,
            )

        timestamps = list(history.index)
        values = [float(value) for value in history.to_list()]
        intercept, raw_slope, residual_rmse = _linear_fit(timestamps, values)
        slope_limit = policy.maximum_change_per_hour
        slope = max(-slope_limit, min(slope_limit, raw_slope))
        slope_was_clamped = slope != raw_slope

        recent_min = min(values)
        recent_max = max(values)
        lower = max(0.0, recent_min - policy.maximum_change_per_hour)
        upper = recent_max + policy.maximum_change_per_hour

        start_offset_hours = (
            forecast_start - latest_at
        ).total_seconds() / 3600.0
        horizon_hours = duration_seconds / 3600.0
        if duration_seconds == 0:
            offsets = [start_offset_hours]
        else:
            sample_count = max(
                2,
                int(duration_seconds // policy.forecast_sample_seconds) + 1,
            )
            offsets = [
                start_offset_hours
                + horizon_hours * index / (sample_count - 1)
                for index in range(sample_count)
            ]

        predicted: list[float] = []
        value_was_clamped = False
        for offset in offsets:
            raw_value = intercept + slope * offset
            bounded = max(lower, min(upper, raw_value))
            if bounded != raw_value:
                value_was_clamped = True
            predicted.append(max(0.0, bounded))

        dynamic_scale = max(recent_max - recent_min, latest_value, 1.0)
        fit_factor = max(0.0, 1.0 - residual_rmse / dynamic_scale)
        point_factor = min(1.0, len(history) / policy.history_points)
        clamp_factor = 0.75 if (slope_was_clamped or value_was_clamped) else 1.0
        confidence = (
            point_factor
            * fit_factor
            * freshness_factor
            * clamp_factor
        )
        confidence = max(policy.confidence_floor, min(1.0, confidence))

        return CarbonForecastEstimate(
            node_id=node_id,
            average_g_per_kwh=sum(predicted) / len(predicted),
            current_g_per_kwh=latest_value,
            source=self.name,
            confidence=confidence,
            freshness="fresh",
            history_points=len(history),
            forecast_start_utc=forecast_start.to_pydatetime(warn=False),
            forecast_horizon_seconds=duration_seconds,
            generated_at_utc=observed_at.to_pydatetime(warn=False),
            latest_sample_at_utc=latest_at.to_pydatetime(warn=False),
            sample_age_seconds=age_seconds,
            slope_g_per_kwh_per_hour=slope,
            residual_rmse=residual_rmse,
            clamped=slope_was_clamped or value_was_clamped,
            details={
                "provider": policy.provider,
                "recent_min_g_per_kwh": recent_min,
                "recent_max_g_per_kwh": recent_max,
                "lower_bound_g_per_kwh": lower,
                "upper_bound_g_per_kwh": upper,
                "raw_slope_g_per_kwh_per_hour": raw_slope,
            },
        )


def forecast_or_average(
    carbon_store,
    *,
    node_id: str,
    observed_at_utc: pd.Timestamp,
    forecast_start_utc: pd.Timestamp,
    duration_seconds: float,
    policy: CarbonForecastPolicy | None,
) -> CarbonForecastEstimate:
    """Use the forecast API while preserving test/third-party store support."""
    if (
        policy is not None
        and policy.enabled
        and hasattr(carbon_store, "forecast")
    ):
        return carbon_store.forecast(
            node_id=node_id,
            observed_at_utc=observed_at_utc,
            forecast_start_utc=forecast_start_utc,
            duration_seconds=duration_seconds,
            policy=policy,
        )

    value = carbon_store.average(
        node_id,
        forecast_start_utc,
        duration_seconds,
    )
    timestamp = _utc(observed_at_utc)
    start = _utc(forecast_start_utc)
    return CarbonForecastEstimate(
        node_id=node_id,
        average_g_per_kwh=float(value),
        current_g_per_kwh=float(value),
        source="legacy_average",
        confidence=0.0,
        freshness="unavailable",
        history_points=0,
        forecast_start_utc=start.to_pydatetime(warn=False),
        forecast_horizon_seconds=duration_seconds,
        generated_at_utc=timestamp.to_pydatetime(warn=False),
    )
