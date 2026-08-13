from __future__ import annotations

from typing import Any

import pandas as pd

from magellan.carbon.store import CarbonStore, as_utc_timestamp
from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.comparison import (
    ComparisonPolicy,
    ComparisonWorkload,
    PolicyOutcome,
    best_at_dispatch_outcome,
    best_static_outcome,
    comparison_reference_scales,
    global_objective,
    replay_causal_policy,
    static_outcome,
)
from magellan.experiments.oracle import clairvoyant_oracle
from magellan.graph.topology import ClusterGraph


REQUIRED_BASELINE_POLICIES = (
    ComparisonPolicy.BOSTON_STATIC.value,
    ComparisonPolicy.FRANCE_STATIC.value,
    ComparisonPolicy.BEST_STATIC.value,
    ComparisonPolicy.BEST_AT_DISPATCH.value,
    ComparisonPolicy.TEMPORAL_ONLY.value,
    ComparisonPolicy.MAGELLAN_CAUSAL.value,
    "clairvoyant_oracle",
)


def run_baseline_suite(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    workload: ComparisonWorkload,
    start_utc: str | pd.Timestamp,
    oracle_quantum_seconds: float = 900.0,
    oracle_max_elapsed_multiplier: float = 3.0,
) -> tuple[list[PolicyOutcome], dict[str, Any]]:
    start = as_utc_timestamp(start_utc)
    graph = ClusterGraph(cluster)

    boston = static_outcome(
        label=ComparisonPolicy.BOSTON_STATIC.value,
        node=cluster.get_node("boston"),
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start,
    )
    france = static_outcome(
        label=ComparisonPolicy.FRANCE_STATIC.value,
        node=cluster.get_node("france"),
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start,
    )
    best_static = best_static_outcome(
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start,
    )
    best_dispatch = best_at_dispatch_outcome(
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon_store,
        start_utc=start,
    )
    temporal = replay_causal_policy(
        label=ComparisonPolicy.TEMPORAL_ONLY,
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon_store,
        graph=graph,
        start_utc=start,
    )
    magellan = replay_causal_policy(
        label=ComparisonPolicy.MAGELLAN_CAUSAL,
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon_store,
        graph=graph,
        start_utc=start,
    )

    pre_oracle = [boston, france, best_static, best_dispatch, temporal, magellan]
    scales = comparison_reference_scales(pre_oracle)
    oracle = clairvoyant_oracle(
        cluster=cluster,
        policy=policy,
        workload=workload,
        carbon_store=carbon_store,
        graph=graph,
        start_utc=start,
        reference_time_seconds=scales["time_seconds"],
        reference_carbon_grams=scales["carbon_grams"],
        reference_cost_usd=scales["cost_usd"],
        quantum_seconds=oracle_quantum_seconds,
        max_elapsed_multiplier=oracle_max_elapsed_multiplier,
    )
    outcomes = [*pre_oracle, oracle]
    objectives = {
        item.policy: global_objective(item, policy, scales)
        for item in outcomes
    }
    metadata = {
        "reference_scales": scales,
        "global_objective_values": objectives,
        "policy_semantics": {
            ComparisonPolicy.BOSTON_STATIC.value: (
                "Fixed Boston placement for entire workload"
            ),
            ComparisonPolicy.FRANCE_STATIC.value: (
                "Fixed France placement for entire workload"
            ),
            ComparisonPolicy.BEST_STATIC.value: (
                "Clairvoyant free initial placement; one node for entire workload"
            ),
            ComparisonPolicy.BEST_AT_DISPATCH.value: (
                "Causal submission-time free initial placement; then fixed"
            ),
            ComparisonPolicy.TEMPORAL_ONLY.value: (
                "Causal Magellan scoring with migration candidates disabled"
            ),
            ComparisonPolicy.MAGELLAN_CAUSAL.value: (
                "Causal v1.2 scoring replay using configured edge models; no live telemetry"
            ),
            "clairvoyant_oracle": (
                "Discretized full-trace oracle starting at the submission node"
            ),
        },
    }
    return outcomes, metadata
