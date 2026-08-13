from __future__ import annotations

from pathlib import Path

import pandas as pd

from magellan.carbon.store import CARBON_COLUMN, TIME_COLUMN, CarbonStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.comparison import ComparisonWorkload, static_outcome
from magellan.experiments.oracle import clairvoyant_oracle
from magellan.graph.topology import ClusterGraph


def node(node_id: str, file: str, ip: str, longitude: float) -> NodeConfig:
    return NodeConfig(
        id=node_id,
        name=node_id,
        vm_name=f"vm-{node_id}",
        zone="test",
        internal_ip=ip,
        carbon_region=node_id,
        dataset_file=file,
        latitude=0,
        longitude=longitude,
        pue=1,
        compute_price_usd_per_hour=0.05,
        egress_price_usd_per_gb=0,
    )


def test_oracle_starts_at_submission_node_and_can_migrate(tmp_path: Path) -> None:
    cluster = ClusterConfig(
        epoch_seconds=600,
        default_bandwidth_mbps=10_000,
        default_latency_ms=0,
        nodes=[
            node("boston", "b.csv", "10.0.0.1", 0),
            node("france", "f.csv", "10.0.0.2", 1),
        ],
    )
    timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="1h")
    pd.DataFrame({TIME_COLUMN: timestamps, CARBON_COLUMN: [1000.0] * 12}).to_csv(
        tmp_path / "b.csv", index=False
    )
    pd.DataFrame({TIME_COLUMN: timestamps, CARBON_COLUMN: [1.0] * 12}).to_csv(
        tmp_path / "f.csv", index=False
    )
    carbon = CarbonStore(cluster, tmp_path)
    policy = ScoringPolicy.model_validate(
        {
            "horizon_seconds": 600,
            "weights": {"time": 0.05, "carbon": 0.9, "cost": 0.05},
            "pause": {
                "pause_seconds": 1,
                "idle_seconds": 600,
                "resume_seconds": 1,
                "max_pause_window_seconds": 3600,
            },
            "migration": {
                "min_migration_gap_seconds": 0,
                "required_improvement_fraction": 0,
                "network_energy_kwh_per_gb_base": 0,
                "network_energy_kwh_per_gb_km": 0,
            },
            "adaptive": {"enabled": False},
            "carbon_forecast": {"enabled": False},
            "clock": {"mode": "trace", "trace_start_utc": "2024-01-01T00:00:00Z"},
        }
    )
    workload = ComparisonWorkload(
        duration_seconds=1800,
        power_kw=0.1,
        checkpoint_bytes=0,
        start_node_id="boston",
    )
    start = pd.Timestamp("2024-01-01T01:00:00Z")
    boston_static = static_outcome(
        label="boston_static",
        node=cluster.get_node("boston"),
        workload=workload,
        carbon_store=carbon,
        start_utc=start,
    )

    oracle = clairvoyant_oracle(
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon,
        graph=ClusterGraph(cluster),
        start_utc=start,
        reference_time_seconds=boston_static.makespan_seconds,
        reference_carbon_grams=boston_static.carbon_grams,
        reference_cost_usd=boston_static.cost_usd,
        quantum_seconds=600,
        max_elapsed_multiplier=2,
    )

    assert oracle.start_node_id == "boston"
    assert oracle.owner_path[0] == "boston"
    assert "france" in oracle.owner_path
    assert oracle.carbon_grams < boston_static.carbon_grams
