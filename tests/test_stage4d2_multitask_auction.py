from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from magellan.bidding.models import AuctionStrategy
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.config.models import NodeResourceCapacity
from magellan.experiments.stage4b import WorkloadCalibration
from magellan.experiments.stage4d2 import (
    CANONICAL_TASK_MIX,
    LOWEST_SCORE_POLICY,
    LayoutTask,
    build_initial_layout,
    replay_capacity_policy,
)
from magellan.models.types import (
    ActionType,
    DecisionResult,
    ScoredAction,
    TaskResourceRequest,
)


def test_canonical_layout_is_measured_maximal_and_rotation_preserves_mix() -> None:
    node_ids = [
        "boston",
        "california",
        "south-australia",
        "nepal",
        "ethiopia",
        "france",
        "virginia",
    ]
    requests = {
        "benchmark-json-medium": TaskResourceRequest(cpu_cores=0.997, memory_mb=13),
        "dendro-r9-t1p0": TaskResourceRequest(cpu_cores=1.776, memory_mb=1379),
        "llm-distilgpt2": TaskResourceRequest(cpu_cores=0.763, memory_mb=1572),
    }
    signatures = {
        node_id: {(2, 0, 0), (1, 0, 1), (0, 1, 0), (0, 0, 2)}
        for node_id in node_ids
    }

    for rotation in range(4):
        layout = build_initial_layout(
            scenario_id=f"s{rotation}",
            node_ids=node_ids,
            requests=requests,
            rotation=rotation,
            maximal_signatures=signatures,
        )
        assert len(layout) == 11
        counts = {class_id: 0 for class_id in CANONICAL_TASK_MIX}
        for task in layout:
            counts[task.class_id] += 1
        assert counts == CANONICAL_TASK_MIX
        assert {task.initial_node_id for task in layout} == set(node_ids)


def _edge_rows(cluster) -> list[dict[str, str]]:
    rows = []
    for source in cluster.nodes:
        for destination in cluster.nodes:
            if source.id == destination.id:
                continue
            rows.append(
                {
                    "source_node_id": source.id,
                    "destination_node_id": destination.id,
                    "transfer_steady_bandwidth_mbps": "100",
                    "transfer_fixed_seconds": "1",
                    "measured_bandwidth_median_mbps": "100",
                    "measured_rtt_median_ms": "10",
                }
            )
    return rows


def test_capacity_replay_rejects_competing_bid_without_overcommit(monkeypatch) -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    policy = load_policy_config("config/policy.prod.json")
    capacities = {
        node.id: NodeResourceCapacity(cpu_cores=1.0, memory_mb=4096, gpu_count=0)
        for node in cluster.nodes
    }
    request = TaskResourceRequest(cpu_cores=1.0, memory_mb=128)
    layout = [
        LayoutTask("task-a", "benchmark-json-medium", "boston", request),
        LayoutTask("task-b", "benchmark-json-medium", "california", request),
    ]
    calibration = WorkloadCalibration(
        class_id="benchmark-json-medium",
        workload="benchmark",
        variant="json-medium",
        canonical_runtime_seconds=1800.0,
        power_kw=0.08,
        checkpoint_bytes=1,
        checkpoint_seconds=0.0,
        restore_seconds=0.0,
        migration_overhead_seconds=0.0,
    )

    def fake_evaluate_task(*, task, **kwargs):
        if task.current_node_id == "ethiopia":
            selected = ScoredAction(
                action=ActionType.CONTINUE,
                source_node_id="ethiopia",
                time_seconds=900,
                carbon_grams=1,
                cost_usd=1,
                normalized_time=0.1,
                normalized_carbon=0.1,
                normalized_cost=0.1,
                score=0.3,
            )
            return DecisionResult(selected=selected, ranked_actions=[selected], reason="continue")
        score = 0.1 if task.task_id == "task-a" else 0.2
        migrate = ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id=task.current_node_id,
            destination_node_id="ethiopia",
            time_seconds=901,
            carbon_grams=1,
            cost_usd=1,
            normalized_time=0.1,
            normalized_carbon=0.1,
            normalized_cost=0.1,
            score=score,
        )
        cont = ScoredAction(
            action=ActionType.CONTINUE,
            source_node_id=task.current_node_id,
            time_seconds=900,
            carbon_grams=2,
            cost_usd=1,
            normalized_time=0.1,
            normalized_carbon=0.2,
            normalized_cost=0.1,
            score=0.5,
        )
        return DecisionResult(
            selected=migrate,
            ranked_actions=[migrate, cont],
            reason="migrate",
        )

    monkeypatch.setattr("magellan.experiments.stage4d2.evaluate_task", fake_evaluate_task)
    monkeypatch.setattr(
        "magellan.experiments.stage4d2._realized_migration",
        lambda **kwargs: (0.0, 0.0, 0.0, {"transfer_model": "test"}),
    )
    monkeypatch.setattr(
        "magellan.experiments.stage4d2._compute_segment",
        lambda **kwargs: (0.0, 0.0),
    )

    task_rows, auction_rows, migration_rows, occupancy_rows = replay_capacity_policy(
        policy_label=LOWEST_SCORE_POLICY,
        auction_strategy=AuctionStrategy.LOWEST_SCORE,
        layout=layout,
        capacities=capacities,
        calibrations={"benchmark-json-medium": calibration},
        runtime_scales={"benchmark-json-medium": 1.0},
        node_slowdowns={node.id: 1.0 for node in cluster.nodes},
        cluster=cluster,
        policy=policy,
        carbon_store=object(),
        edge_rows=_edge_rows(cluster),
        arrival_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        scenario_id="test",
    )

    assert all(row["completed"] for row in task_rows)
    assert sum(int(row["bid_accepts"]) for row in task_rows) == 1
    assert sum(int(row["bid_rejections"]) for row in task_rows) >= 1
    assert len(migration_rows) == 1
    assert any(row["status"] == "rejected" for row in auction_rows)
    assert all(
        float(row["used_cpu_cores"]) <= float(row["capacity_cpu_cores"]) + 1e-9
        for row in occupancy_rows
    )


def test_replay_carbon_store_memoizes_identical_queries() -> None:
    from magellan.carbon.store import CarbonMetric
    from magellan.experiments.stage4d2 import ReplayCarbonStore

    cluster = load_cluster_config("config/cluster.gcp.json")
    policy = load_policy_config("config/policy.prod.json")
    store = ReplayCarbonStore(cluster, "datasets", carbon_metric=CarbonMetric.LIFECYCLE)
    at = pd.Timestamp("2024-04-20T12:00:00.123456789Z")

    first_average = store.average("ethiopia", at, 900.0)
    second_average = store.average("ethiopia", at, 900.0)
    assert second_average == first_average
    assert store.average_cache_hits == 1

    first_forecast = store.forecast(
        node_id="ethiopia",
        observed_at_utc=at,
        forecast_start_utc=at,
        duration_seconds=5400.0,
        policy=policy.carbon_forecast,
    )
    second_forecast = store.forecast(
        node_id="ethiopia",
        observed_at_utc=at,
        forecast_start_utc=at,
        duration_seconds=5400.0,
        policy=policy.carbon_forecast,
    )
    assert second_forecast is first_forecast
    assert store.forecast_cache_hits == 1
