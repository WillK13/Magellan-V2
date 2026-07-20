from pathlib import Path

from magellan.config.loader import load_cluster_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_CLUSTER_PATH = REPO_ROOT / "config" / "cluster.dev.json"


def test_dev_cluster_config_loads() -> None:
    config = load_cluster_config(DEV_CLUSTER_PATH)

    assert len(config.nodes) == 2
    assert config.api_port == 8040
    assert config.epoch_seconds == 30

    boston = config.get_node("boston")
    virginia = config.get_node("virginia")

    assert boston.zone == "us-east1-c"
    assert str(boston.internal_ip) == "10.142.0.2"

    assert virginia.zone == "northamerica-northeast1-c"
    assert str(virginia.internal_ip) == "10.162.0.2"


def test_production_configs_load() -> None:
    from magellan.config.loader import load_policy_config

    cluster = load_cluster_config(
        REPO_ROOT / "config" / "cluster.gcp.json"
    )
    policy = load_policy_config(
        REPO_ROOT / "config" / "policy.prod.json"
    )

    assert len(cluster.nodes) == 7
    assert cluster.reservation_ttl_seconds == 300
    assert policy.recovery.max_restart_attempts == 3
