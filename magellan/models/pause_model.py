from __future__ import annotations

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.models import NodeConfig
from magellan.config.policy_models import PausePolicy
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
) -> RawActionEstimate | None:
    compute_seconds = horizon_seconds

    if task.estimated_remaining_seconds is not None:
        compute_seconds = min(
            compute_seconds,
            task.estimated_remaining_seconds,
        )

    if (
        pause_policy.idle_seconds + compute_seconds
        > pause_policy.max_pause_window_seconds
    ):
        return None

    overhead_seconds = (
        pause_policy.pause_seconds
        + pause_policy.resume_seconds
    )

    overhead_carbon_intensity = carbon_store.average(
        node.id,
        at_utc,
        overhead_seconds,
    )

    compute_start = at_utc + pd.Timedelta(
        seconds=(
            pause_policy.pause_seconds
            + pause_policy.idle_seconds
            + pause_policy.resume_seconds
        )
    )

    compute_carbon_intensity = carbon_store.average(
        node.id,
        compute_start,
        compute_seconds,
    )

    effective_power_kw = task.power_kw * node.pue

    overhead_carbon_grams = (
        effective_power_kw
        * seconds_to_hours(overhead_seconds)
        * overhead_carbon_intensity
    )

    compute_carbon_grams = (
        effective_power_kw
        * seconds_to_hours(compute_seconds)
        * compute_carbon_intensity
    )

    # Preserve the original Magellan assumption that task idle time
    # is not charged as active task compute.
    cost_usd = (
        node.compute_price_usd_per_hour
        * seconds_to_hours(compute_seconds)
    )

    total_time_seconds = (
        pause_policy.pause_seconds
        + pause_policy.idle_seconds
        + pause_policy.resume_seconds
        + compute_seconds
    )

    return RawActionEstimate(
        action=ActionType.PAUSE,
        source_node_id=node.id,
        destination_node_id=None,
        time_seconds=total_time_seconds,
        carbon_grams=(
            overhead_carbon_grams
            + compute_carbon_grams
        ),
        cost_usd=cost_usd,
        details={
            "pause_seconds": pause_policy.pause_seconds,
            "idle_seconds": pause_policy.idle_seconds,
            "resume_seconds": pause_policy.resume_seconds,
            "compute_seconds": compute_seconds,
            "overhead_carbon_intensity_g_per_kwh": (
                overhead_carbon_intensity
            ),
            "compute_carbon_intensity_g_per_kwh": (
                compute_carbon_intensity
            ),
        },
    )
