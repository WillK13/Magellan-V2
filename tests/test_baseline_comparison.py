from __future__ import annotations

from pathlib import Path

import pandas as pd

from magellan.carbon.store import CARBON_COLUMN, TIME_COLUMN, CarbonStore
from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.baseline_suite import REQUIRED_BASELINE_POLICIES, run_baseline_suite
from magellan.experiments.comparison import (
    ComparisonPolicy,
    ComparisonWorkload,
    best_at_dispatch_outcome,
    best_static_outcome,
    replay_causal_policy,
)
from magellan.graph.topology import ClusterGraph


def make_node(node_id: str, dataset: str, ip: str, longitude: float) -> NodeConfig:
    return NodeConfig(
        id=node_id,
        name=node_id.title(),
        vm_name=f"vm-{node_id}",
        zone="test-zone",
        internal_ip=ip,
        carbon_region=node_id,
        dataset_file=dataset,
        latitude=0.0,
        longitude=longitude,
        pue=1.0,
        compute_price_usd_per_hour=0.05,
        egress_price_usd_per_gb=0.02,
    )


def make_cluster() -> ClusterConfig:
    return ClusterConfig(
        epoch_seconds=600,
        default_bandwidth_mbps=1000,
        default_latency_ms=0,
        nodes=[
            make_node("boston", "boston.csv", "10.0.0.1", 0.0),
            make_node("france", "france.csv", "10.0.0.2", 10.0),
        ],
    )


def make_policy() -> ScoringPolicy:
    return ScoringPolicy.model_validate(
        {
            "horizon_seconds": 600,
            "weights": {"time": 0.05, "carbon": 0.9, "cost": 0.05},
            "pause": {
                "pause_seconds": 1,
                "idle_seconds": 600,
                "candidate_idle_seconds": [600],
                "resume_seconds": 1,
                "max_pause_window_seconds": 3600,
                "min_pause_gap_seconds": 3600,
            },
            "migration": {
                "min_migration_gap_seconds": 0,
                "required_improvement_fraction": 0,
                "network_energy_kwh_per_gb_base": 0,
                "network_energy_kwh_per_gb_km": 0,
            },
            "adaptive": {"enabled": False},
            "carbon_forecast": {
                "enabled": True,
                "provider": "persistence",
                "history_points": 4,
                "minimum_points": 1,
                "sample_interval_seconds": 600,
                "horizon_seconds": 3600,
                "forecast_sample_seconds": 300,
                "maximum_change_per_hour": 1000,
                "stale_after_seconds": 3600,
            },
            "clock": {
                "mode": "trace",
                "trace_start_utc": "2024-01-01T00:00:00Z",
                "trace_seconds_per_real_second": 1,
            },
        }
    )


def write_trace(path: Path, values: list[float]) -> None:
    timestamps = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=len(values),
        freq="1h",
    )
    pd.DataFrame({TIME_COLUMN: timestamps, CARBON_COLUMN: values}).to_csv(
        path,
        index=False,
    )


def store(tmp_path: Path) -> CarbonStore:
    write_trace(tmp_path / "boston.csv", [900.0] * 24)
    write_trace(tmp_path / "france.csv", [20.0] * 24)
    return CarbonStore(make_cluster(), tmp_path)


def workload() -> ComparisonWorkload:
    return ComparisonWorkload(
        name="test",
        duration_seconds=1800,
        power_kw=0.1,
        checkpoint_bytes=0,
        start_node_id="boston",
    )


def test_best_static_and_dispatch_choose_cleaner_node(tmp_path: Path) -> None:
    cluster = make_cluster()
    policy = make_policy()
    carbon = store(tmp_path)
    start = pd.Timestamp("2024-01-01T02:00:00Z")

    best_static = best_static_outcome(
        cluster=cluster,
        policy=policy,
        workload=workload(),
        carbon_store=carbon,
        start_utc=start,
    )
    dispatch = best_at_dispatch_outcome(
        cluster=cluster,
        policy=policy,
        workload=workload(),
        carbon_store=carbon,
        start_utc=start,
    )

    assert best_static.final_node_id == "france"
    assert dispatch.final_node_id == "france"
    assert best_static.carbon_grams < 5.0


def test_temporal_replay_cannot_migrate(tmp_path: Path) -> None:
    cluster = make_cluster()
    policy = make_policy()
    carbon = store(tmp_path)

    outcome = replay_causal_policy(
        label=ComparisonPolicy.TEMPORAL_ONLY,
        cluster=cluster,
        policy=policy,
        workload=workload(),
        carbon_store=carbon,
        graph=ClusterGraph(cluster),
        start_utc=pd.Timestamp("2024-01-01T02:00:00Z"),
    )

    assert outcome.completed
    assert outcome.migrations == 0
    assert outcome.final_node_id == "boston"


def test_magellan_causal_replay_can_migrate_to_cleaner_node(tmp_path: Path) -> None:
    cluster = make_cluster()
    policy = make_policy()
    carbon = store(tmp_path)

    outcome = replay_causal_policy(
        label=ComparisonPolicy.MAGELLAN_CAUSAL,
        cluster=cluster,
        policy=policy,
        workload=workload(),
        carbon_store=carbon,
        graph=ClusterGraph(cluster),
        start_utc=pd.Timestamp("2024-01-01T02:00:00Z"),
    )

    assert outcome.completed
    assert outcome.migrations >= 1
    assert "france" in outcome.owner_path


def test_full_suite_has_all_required_policies_and_oracle(tmp_path: Path) -> None:
    cluster = make_cluster()
    policy = make_policy()
    carbon = store(tmp_path)

    outcomes, metadata = run_baseline_suite(
        cluster=cluster,
        policy=policy,
        carbon_store=carbon,
        workload=workload(),
        start_utc="2024-01-01T02:00:00Z",
        oracle_quantum_seconds=600,
        oracle_max_elapsed_multiplier=2.5,
    )

    assert {item.policy for item in outcomes} == set(REQUIRED_BASELINE_POLICIES)
    oracle = next(item for item in outcomes if item.policy == "clairvoyant_oracle")
    assert oracle.start_node_id == "boston"
    assert oracle.completed
    assert metadata["policy_semantics"]["best_static"].startswith("Clairvoyant")
