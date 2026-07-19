from __future__ import annotations

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.models import NodeConfig
from magellan.config.policy_models import (
    MigrationPolicy,
    PausePolicy,
)
from magellan.graph.topology import EdgeMetrics
from magellan.models.types import (
    ActionType,
    RawActionEstimate,
    TaskProfile,
)
from magellan.models.utils import (
    bytes_to_gb,
    seconds_to_hours,
    transfer_seconds,
)


def estimate_migrate(
    task: TaskProfile,
    source: NodeConfig,
    destination: NodeConfig,
    edge: EdgeMetrics,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
    horizon_seconds: float,
    pause_policy: PausePolicy,
    migration_policy: MigrationPolicy,
    static_data_bytes_override: int | None = None,
) -> RawActionEstimate:
    compute_seconds = horizon_seconds

    if task.estimated_remaining_seconds is not None:
        compute_seconds = min(
            compute_seconds,
            task.estimated_remaining_seconds,
        )

    if static_data_bytes_override is not None:
        static_data_bytes = (
            static_data_bytes_override
        )
    else:
        static_data_bytes = (
            0
            if destination.id in task.prestaged_node_ids
            else task.data_bytes
    )

    total_transfer_bytes = (
        task.checkpoint_bytes
        + static_data_bytes
    )

    transfer_duration_seconds = transfer_seconds(
        size_bytes=total_transfer_bytes,
        bandwidth_mbps=edge.bandwidth_mbps,
        latency_ms=edge.latency_ms,
    )

    checkpoint_seconds = (
        edge.checkpoint_seconds
        if edge.checkpoint_seconds is not None
        else pause_policy.pause_seconds
    )
    restore_seconds = (
        edge.restore_seconds
        if edge.restore_seconds is not None
        else pause_policy.resume_seconds
    )

    arrival_time = at_utc + pd.Timedelta(
        seconds=(
            checkpoint_seconds
            + transfer_duration_seconds
            + restore_seconds
        )
    )

    source_overhead_seconds = checkpoint_seconds + restore_seconds

    source_intensity = carbon_store.average(
        source.id,
        at_utc,
        source_overhead_seconds,
    )

    destination_intensity = carbon_store.average(
        destination.id,
        arrival_time,
        compute_seconds,
    )

    source_effective_power_kw = task.power_kw * source.pue
    destination_effective_power_kw = task.power_kw * destination.pue

    source_overhead_carbon_grams = (
        source_effective_power_kw
        * seconds_to_hours(source_overhead_seconds)
        * source_intensity
    )

    destination_compute_carbon_grams = (
        destination_effective_power_kw
        * seconds_to_hours(compute_seconds)
        * destination_intensity
    )

    transfer_size_gb = bytes_to_gb(total_transfer_bytes)

    network_energy_kwh = transfer_size_gb * (
        migration_policy.network_energy_kwh_per_gb_base
        + migration_policy.network_energy_kwh_per_gb_km
        * edge.distance_km
    )

    mean_network_intensity = (
        source_intensity + destination_intensity
    ) / 2.0

    network_carbon_grams = (
        network_energy_kwh
        * mean_network_intensity
    )

    compute_cost_usd = (
        destination.compute_price_usd_per_hour
        * seconds_to_hours(compute_seconds)
    )

    transfer_cost_usd = (
        transfer_size_gb
        * source.egress_price_usd_per_gb
    )

    total_time_seconds = (
        checkpoint_seconds
        + transfer_duration_seconds
        + restore_seconds
        + compute_seconds
    )

    return RawActionEstimate(
        action=ActionType.MIGRATE,
        source_node_id=source.id,
        destination_node_id=destination.id,
        time_seconds=total_time_seconds,
        carbon_grams=(
            source_overhead_carbon_grams
            + destination_compute_carbon_grams
            + network_carbon_grams
        ),
        cost_usd=(
            compute_cost_usd
            + transfer_cost_usd
        ),
        details={
            "distance_km": edge.distance_km,
            "bandwidth_mbps": edge.bandwidth_mbps,
            "latency_ms": edge.latency_ms,
            "bandwidth_source": edge.bandwidth_source,
            "latency_source": edge.latency_source,
            "checkpoint_seconds": checkpoint_seconds,
            "restore_seconds": restore_seconds,
            "calibration_source": edge.calibration_source,
            "checkpoint_bytes": task.checkpoint_bytes,
            "static_data_bytes": static_data_bytes,
            "total_transfer_bytes": total_transfer_bytes,
            "transfer_seconds": transfer_duration_seconds,
            "transfer_size_gb": transfer_size_gb,
            "source_carbon_intensity_g_per_kwh": source_intensity,
            "destination_carbon_intensity_g_per_kwh": (
                destination_intensity
            ),
            "network_energy_kwh": network_energy_kwh,
            "network_carbon_grams": network_carbon_grams,
            "compute_cost_usd": compute_cost_usd,
            "transfer_cost_usd": transfer_cost_usd,
            "arrival_time_utc": arrival_time.isoformat(),
        },
    )
