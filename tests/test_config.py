from magellan.config.loader import load_cluster_config


def test_cluster_config_loads() -> None:
    config = load_cluster_config("config/cluster.json")

    assert len(config.nodes) == 7
    assert config.api_port == 8040
    assert config.epoch_seconds == 900
    assert config.get_node("france").zone == "europe-west9-b"
