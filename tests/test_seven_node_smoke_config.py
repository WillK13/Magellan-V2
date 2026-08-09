from pathlib import Path

from magellan.config.loader import load_cluster_config, load_policy_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_seven_node_smoke_config_is_fast_and_matches_production_nodes() -> None:
    production = load_cluster_config(REPO_ROOT / "config/cluster.gcp.json")
    smoke = load_cluster_config(REPO_ROOT / "config/cluster.gcp.smoke.json")

    assert smoke.epoch_seconds == 20
    assert [node.id for node in smoke.nodes] == [node.id for node in production.nodes]
    assert [str(node.internal_ip) for node in smoke.nodes] == [
        str(node.internal_ip) for node in production.nodes
    ]


def test_seven_node_smoke_policy_keeps_autonomous_adaptation_enabled() -> None:
    policy = load_policy_config(REPO_ROOT / "config/policy.gcp.smoke.json")

    assert policy.weights.time == 0.05
    assert policy.weights.carbon == 0.9
    assert policy.weights.cost == 0.05
    assert policy.migration.required_improvement_fraction == 0
    assert policy.migration.min_migration_gap_seconds == 90
    assert policy.adaptive.enabled is True
    assert policy.clock.mode == "trace"
    assert policy.clock.trace_seconds_per_real_second == 60
