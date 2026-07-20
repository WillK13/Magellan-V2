from __future__ import annotations

import pandas as pd

from magellan.carbon.forecast import forecast_or_average
from magellan.carbon.store import CarbonStore
from magellan.config.models import NodeConfig
from magellan.config.policy_models import (
    CarbonForecastPolicy,
    PausePolicy,
)
from magellan.models.types import (
    ActionType,
    RawActionEstimate,
    TaskProfile,
)
from magellan.models.utils import seconds_to_hours


def estimate_pause(
    task: TaskProfile,
    node: NodeConfig,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
    horizon_seconds: float,
    pause_policy: PausePolicy,
    idle_seconds: float | None = None,
    forecast_policy: CarbonForecastPolicy | None = None,
) -> RawActionEstimate | None:
    compute_seconds = horizon_seconds

    if task.estimated_remaining_seconds is not None:
        compute_seconds = min(
            compute_seconds,
            task.estimated_remaining_seconds,
        )

    selected_idle_seconds = (
        pause_policy.idle_seconds
        if idle_seconds is None
        else float(idle_seconds)
    )

    if selected_idle_seconds < 0:
        raise ValueError("idle_seconds must be non-negative")

    if (
        selected_idle_seconds + compute_seconds
        > pause_policy.max_pause_window_seconds
    ):
        return None

    pause_forecast = forecast_or_average(
        carbon_store,
        node_id=node.id,
        observed_at_utc=at_utc,
        forecast_start_utc=at_utc,
        duration_seconds=pause_policy.pause_seconds,
        policy=forecast_policy,
    )

    resume_start = at_utc + pd.Timedelta(
        seconds=pause_policy.pause_seconds + selected_idle_seconds
    )
    resume_forecast = forecast_or_average(
        carbon_store,
        node_id=node.id,
        observed_at_utc=at_utc,
        forecast_start_utc=resume_start,
        duration_seconds=pause_policy.resume_seconds,
        policy=forecast_policy,
    )

    compute_start = resume_start + pd.Timedelta(
        seconds=pause_policy.resume_seconds
    )
    compute_forecast = forecast_or_average(
        carbon_store,
        node_id=node.id,
        observed_at_utc=at_utc,
        forecast_start_utc=compute_start,
        duration_seconds=compute_seconds,
        policy=forecast_policy,
    )

    effective_power_kw = task.power_kw * node.pue

    pause_carbon_grams = (
        effective_power_kw
        * seconds_to_hours(pause_policy.pause_seconds)
        * pause_forecast.average_g_per_kwh
    )
    resume_carbon_grams = (
        effective_power_kw
        * seconds_to_hours(pause_policy.resume_seconds)
        * resume_forecast.average_g_per_kwh
    )
    compute_carbon_grams = (
        effective_power_kw
        * seconds_to_hours(compute_seconds)
        * compute_forecast.average_g_per_kwh
    )

    # Preserve the original Magellan assumption that task idle time
    # is not charged as active task compute.
    cost_usd = (
        node.compute_price_usd_per_hour
        * seconds_to_hours(compute_seconds)
    )

    total_time_seconds = (
        pause_policy.pause_seconds
        + selected_idle_seconds
        + pause_policy.resume_seconds
        + compute_seconds
    )

    return RawActionEstimate(
        action=ActionType.PAUSE,
        source_node_id=node.id,
        destination_node_id=None,
        time_seconds=total_time_seconds,
        carbon_grams=(
            pause_carbon_grams
            + resume_carbon_grams
            + compute_carbon_grams
        ),
        cost_usd=cost_usd,
        details={
            "pause_seconds": pause_policy.pause_seconds,
            "idle_seconds": selected_idle_seconds,
            "pause_duration_seconds": selected_idle_seconds,
            "resume_seconds": pause_policy.resume_seconds,
            "compute_seconds": compute_seconds,
            "pause_carbon_grams": pause_carbon_grams,
            "resume_carbon_grams": resume_carbon_grams,
            "compute_carbon_grams": compute_carbon_grams,
            "pause_carbon_intensity_g_per_kwh": (
                pause_forecast.average_g_per_kwh
            ),
            "resume_carbon_intensity_g_per_kwh": (
                resume_forecast.average_g_per_kwh
            ),
            "compute_carbon_intensity_g_per_kwh": (
                compute_forecast.average_g_per_kwh
            ),
            "carbon_confidence": min(
                pause_forecast.confidence,
                resume_forecast.confidence,
                compute_forecast.confidence,
            ),
            "pause_carbon_forecast": pause_forecast.model_dump(mode="json"),
            "resume_carbon_forecast": resume_forecast.model_dump(mode="json"),
            "compute_carbon_forecast": compute_forecast.model_dump(mode="json"),
        },
    )
