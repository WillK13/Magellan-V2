from __future__ import annotations

import pandas as pd

from magellan.carbon.forecast import CarbonForecastEstimate
from magellan.config.models import NodeConfig
from magellan.config.policy_models import (
    CarbonForecastPolicy,
    MigrationPolicy,
    PausePolicy,
)
from magellan.graph.topology import EdgeMetrics
from magellan.models.migrate_model import estimate_migrate
from magellan.models.types import TaskProfile


class NodeCarbonStore:
    def forecast(
        self,
        *,
        node_id,
        observed_at_utc,
        forecast_start_utc,
        duration_seconds,
        policy,
    ):
        del policy
        value = 100.0 if node_id == "source" else 1000.0
        observed = pd.Timestamp(observed_at_utc)
        start = pd.Timestamp(forecast_start_utc)
        return CarbonForecastEstimate(
            node_id=node_id,
            average_g_per_kwh=value,
            current_g_per_kwh=value,
            source="test",
            confidence=1,
            freshness="fresh",
            history_points=8,
            forecast_start_utc=start.to_pydatetime(),
            forecast_horizon_seconds=duration_seconds,
            generated_at_utc=observed.to_pydatetime(),
        )


def node(node_id: str, ip: str) -> NodeConfig:
    return NodeConfig(
        id=node_id,
        name=node_id,
        vm_name=node_id,
        zone="zone",
        internal_ip=ip,
        carbon_region=node_id,
        dataset_file=f"{node_id}.csv",
        latitude=0,
        longitude=0,
        pue=1,
        compute_price_usd_per_hour=0,
    )


def test_restore_carbon_is_charged_at_destination() -> None:
    estimate = estimate_migrate(
        task=TaskProfile(
            task_id="migration-carbon",
            workload_type="test",
            current_node_id="source",
            power_kw=1,
            checkpoint_bytes=0,
            estimated_remaining_seconds=0,
        ),
        source=node("source", "10.0.0.1"),
        destination=node("destination", "10.0.0.2"),
        edge=EdgeMetrics(
            source_node_id="source",
            destination_node_id="destination",
            distance_km=0,
            bandwidth_mbps=100,
            latency_ms=0,
            checkpoint_seconds=3600,
            restore_seconds=3600,
        ),
        carbon_store=NodeCarbonStore(),
        at_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        horizon_seconds=0,
        pause_policy=PausePolicy(
            pause_seconds=1,
            idle_seconds=0,
            resume_seconds=1,
            max_pause_window_seconds=10,
        ),
        migration_policy=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        forecast_policy=CarbonForecastPolicy(),
    )

    assert estimate.details["source_checkpoint_carbon_grams"] == 100
    assert estimate.details["destination_restore_carbon_grams"] == 1000
    assert estimate.carbon_grams == 1100
