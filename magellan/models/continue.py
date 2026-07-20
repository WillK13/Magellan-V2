from __future__ import annotations

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.models import NodeConfig
from magellan.models.types import (
    ActionType,
    RawActionEstimate,
    TaskProfile,
)
from magellan.models.utils import seconds_to_hours


def estimate_continue(
    task: TaskProfile,
    node: NodeConfig,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
    horizon_seconds: float,
) -> RawActionEstimate:
    compute_seconds = horizon_seconds

    if task.estimated_remaining_seconds is not None:
        compute_seconds = min(
            compute_seconds,
            task.estimated_remaining_seconds,
        )

    carbon_intensity = carbon_store.average(
        node.id,
        at_utc,
        compute_seconds,
    )

    compute_hours = seconds_to_hours(compute_seconds)
    effective_power_kw = task.power_kw * node.pue

    carbon_grams = (
        effective_power_kw
        * compute_hours
        * carbon_intensity
    )

    cost_usd = (
        node.compute_price_usd_per_hour
        * compute_hours
    )

    return RawActionEstimate(
        action=ActionType.CONTINUE,
        source_node_id=node.id,
        destination_node_id=None,
        time_seconds=compute_seconds,
        carbon_grams=carbon_grams,
        cost_usd=cost_usd,
        details={
            "compute_seconds": compute_seconds,
            "carbon_intensity_g_per_kwh": carbon_intensity,
            "effective_power_kw": effective_power_kw,
        },
    )
