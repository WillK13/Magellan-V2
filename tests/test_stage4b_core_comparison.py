from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.stage4b import (
    CORE_POLICIES,
    FrozenCalibrationGraph,
    Scenario,
    WorkloadCalibration,
    gaia_carbon_time_outcome,
    gaia_carbon_time_score,
    gaia_queue_parameters,
    load_node_slowdowns,
    load_stage4a1_edges,
    load_workload_calibrations,
    outcome_rows,
    replay_magellan_causal,
    summarize_policy_rows,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibration(class_id: str = "benchmark-json-medium", runtime: float = 120.0) -> WorkloadCalibration:
    return WorkloadCalibration(
        class_id=class_id,
        workload="json" if class_id.startswith("benchmark") else "dendro",
        variant="medium",
        canonical_runtime_seconds=runtime,
        power_kw=0.05,
        checkpoint_bytes=100_000_000,
        checkpoint_seconds=2.0,
        restore_seconds=3.0,
        migration_overhead_seconds=1.0,
    )


class StubCarbon:
    def average(self, node_id, start_utc, seconds):
        hour = pd.Timestamp(start_utc).hour
        return {0: 500.0, 1: 400.0, 2: 100.0, 3: 300.0}.get(hour, 500.0)

    def value_at(self, node_id, at_utc):
        return self.average(node_id, at_utc, 0.0)


def test_gaia_carbon_time_formula_and_wait_selection():
    assert gaia_carbon_time_score(
        immediate_carbon_grams=100.0,
        candidate_carbon_grams=60.0,
        wait_seconds=3600.0,
        queue_mean_runtime_seconds=3600.0,
    ) == pytest.approx(40.0 / 7200.0)
    cluster = load_cluster_config("config/cluster.gcp.json")
    item = calibration(runtime=3600.0)
    outcome = gaia_carbon_time_outcome(
        boston=cluster.get_node("boston"),
        calibration=item,
        node_slowdowns={"boston": 1.0},
        carbon_store=StubCarbon(),  # type: ignore[arg-type]
        arrival_utc=pd.Timestamp("2024-01-05T00:00:00Z"),
        runtime_scale=1.0,
        queue_parameters={"short": {"mean_runtime_seconds": 3600.0, "max_wait_seconds": 3 * 3600.0}},
        quantum_seconds=3600.0,
    )
    assert outcome.policy == "gaia_carbon_time"
    assert outcome.metadata["submission_wait_seconds"] == pytest.approx(7200.0)
    assert outcome.start_node_id == "boston" == outcome.final_node_id
    assert outcome.migrations == 0 and outcome.pauses == 0


def test_gaia_queue_parameters_use_queue_wide_means():
    values = {
        "short": calibration("dendro-r9-t1p0", runtime=80.0),
        "long1": calibration("benchmark-json-medium", runtime=130.0),
        "long2": calibration("llm-distilgpt2", runtime=150.0),
    }
    queues = gaia_queue_parameters(values, runtime_scale=60.0)
    assert queues["short"]["mean_runtime_seconds"] == pytest.approx(4800.0)
    assert queues["short"]["max_wait_seconds"] == pytest.approx(21600.0)
    assert queues["long"]["mean_runtime_seconds"] == pytest.approx((7800.0 + 9000.0) / 2.0)
    assert queues["long"]["max_wait_seconds"] == pytest.approx(86400.0)


def test_loads_frozen_workload_and_node_calibration(tmp_path: Path):
    a2, a3, a4 = tmp_path / "a2", tmp_path / "a3", tmp_path / "a4"
    write_csv(a2 / "migration_samples.csv", [
        {"workload": "json", "variant": "medium", "actual_checkpoint_bytes": 100, "actual_checkpoint_seconds": 2, "actual_restore_seconds": 3, "actual_migration_overhead_seconds": 1},
        {"workload": "dendro", "variant": "r9-t1", "actual_checkpoint_bytes": 200, "actual_checkpoint_seconds": 4, "actual_restore_seconds": 5, "actual_migration_overhead_seconds": 2},
        {"workload": "llm", "variant": "distilgpt2", "actual_checkpoint_bytes": 900, "actual_checkpoint_seconds": 10, "actual_restore_seconds": 11, "actual_migration_overhead_seconds": 3},
        {"workload": "llm", "variant": "distilgpt2", "actual_checkpoint_bytes": 1100, "actual_checkpoint_seconds": 12, "actual_restore_seconds": 13, "actual_migration_overhead_seconds": 5},
    ])
    write_csv(a3 / "profile_classes.csv", [
        {"class_id": "benchmark-json-medium", "workload": "json", "variant": "medium", "power_median_kw": 0.03},
        {"class_id": "dendro-r9-t1p0", "workload": "dendro", "variant": "r9-t1", "power_median_kw": 0.04},
        {"class_id": "llm-distilgpt2", "workload": "llm", "variant": "distilgpt2", "power_median_kw": 0.05},
    ])
    write_csv(a4 / "static_classes.csv", [
        {"class_id": "benchmark-json-medium", "runtime_seconds_median": 120},
        {"class_id": "dendro-r9-t1p0", "runtime_seconds_median": 80},
        {"class_id": "llm-distilgpt2", "runtime_seconds_median": 130},
    ])
    write_csv(a4 / "node_equivalence.csv", [
        {"node_id": "boston", "slowdown_vs_canonical": 1.0},
        {"node_id": "virginia", "slowdown_vs_canonical": 0.95},
    ])
    loaded = load_workload_calibrations(stage4a2_bundle=a2, stage4a3_bundle=a3, stage4a4_bundle=a4)
    assert loaded["llm-distilgpt2"].checkpoint_bytes == 1000
    assert loaded["llm-distilgpt2"].checkpoint_seconds == pytest.approx(11.0)
    assert loaded["llm-distilgpt2"].migration_overhead_seconds == pytest.approx(4.0)
    assert load_node_slowdowns(a4) == {"boston": 1.0, "virginia": 0.95}


def test_stage4a1_edges_follow_canonical_nested_network_bundle(tmp_path: Path):
    a1 = tmp_path / "a1"
    rows = [{
        "source_node_id": "boston",
        "destination_node_id": "virginia",
        "transfer_steady_bandwidth_mbps": "25",
    }]
    write_csv(a1 / "network" / "directed-mesh" / "edges.csv", rows)
    loaded = load_stage4a1_edges(a1, {"network_bundle": "network/directed-mesh"})
    assert loaded == rows


def test_frozen_graph_uses_stage4a1_affine_edge_model():
    cluster = load_cluster_config("config/cluster.gcp.json")
    graph = FrozenCalibrationGraph(
        cluster=cluster,
        workload=calibration(),
        edge_rows=[{
            "source_node_id": "boston",
            "destination_node_id": "virginia",
            "transfer_steady_bandwidth_mbps": "25",
            "transfer_fixed_seconds": "1.25",
            "measured_bandwidth_median_mbps": "20",
            "measured_rtt_median_ms": "200",
        }],
    )
    edge = graph.edge("boston", "virginia")
    assert edge.transfer_model_source == "measured_migration_transport_affine_ema"
    assert edge.transfer_fixed_seconds == pytest.approx(1.25)
    assert edge.transfer_steady_bandwidth_mbps == pytest.approx(25.0)
    assert edge.checkpoint_seconds == pytest.approx(2.0)
    assert edge.restore_seconds == pytest.approx(3.0)


def test_magellan_causal_replay_uses_realized_node_slowdown_progress():
    cluster = load_cluster_config("config/cluster.gcp.json")
    policy = load_policy_config("config/policy.prod.json")
    carbon = StubCarbon()
    rows = []
    for source in cluster.nodes:
        for destination in cluster.nodes:
            if source.id == destination.id:
                continue
            rows.append({
                "source_node_id": source.id,
                "destination_node_id": destination.id,
                "transfer_steady_bandwidth_mbps": "100",
                "transfer_fixed_seconds": "0.1",
                "measured_bandwidth_median_mbps": "100",
                "measured_rtt_median_ms": "10",
            })
    item = WorkloadCalibration(
        class_id="benchmark-json-medium",
        workload="json",
        variant="medium",
        canonical_runtime_seconds=1800.0,
        power_kw=0.03,
        checkpoint_bytes=1000,
        checkpoint_seconds=1.0,
        restore_seconds=1.0,
        migration_overhead_seconds=0.0,
    )
    graph = FrozenCalibrationGraph(cluster=cluster, workload=item, edge_rows=rows)
    slowdowns = {node.id: 1.0 for node in cluster.nodes}
    outcome = replay_magellan_causal(
        cluster=cluster,
        policy=policy,
        calibration=item,
        node_slowdowns=slowdowns,
        carbon_store=carbon,
        graph=graph,
        arrival_utc=pd.Timestamp("2024-06-05T00:00:00Z"),
        runtime_scale=1.0,
        max_decisions=100,
    )
    assert outcome.completed is True
    assert outcome.compute_seconds == pytest.approx(1800.0)
    assert outcome.start_node_id == "boston"
    assert outcome.decision_count >= 2


def test_outcome_rows_and_policy_summary_are_baseline_normalized():
    cluster = load_cluster_config("config/cluster.gcp.json")
    policy = load_policy_config("config/policy.prod.json")
    scenario = Scenario("s1", "benchmark-json-medium", pd.Timestamp("2024-01-05T00:00:00Z"))
    item = calibration()
    outcomes = []
    for index, name in enumerate(CORE_POLICIES, start=1):
        outcomes.append(
            __import__("magellan.experiments.comparison", fromlist=["PolicyOutcome"]).PolicyOutcome(
                policy=name,
                start_node_id="boston",
                final_node_id="boston",
                selected_initial_node_id="boston",
                makespan_seconds=100.0 * index,
                compute_seconds=100.0,
                paused_idle_seconds=0.0,
                pause_overhead_seconds=0.0,
                migration_seconds=0.0,
                carbon_grams=10.0 * index,
                cost_usd=1.0 * index,
                migrations=0,
                pauses=0,
                decision_count=0,
                owner_path=["boston"],
            )
        )
    rows = outcome_rows(scenario=scenario, calibration=item, outcomes=outcomes, policy=policy)
    assert rows[0]["time_ratio_vs_boston_static"] == pytest.approx(1.0)
    assert rows[-1]["carbon_ratio_vs_boston_static"] == pytest.approx(5.0)
    summary = summarize_policy_rows(rows)
    assert len(summary) == len(CORE_POLICIES)
    assert summary[0]["policy"] == "boston_static"
