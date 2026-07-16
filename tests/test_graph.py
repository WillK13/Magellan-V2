from magellan.config.loader import load_cluster_config
from magellan.graph.topology import ClusterGraph


def test_dev_graph_is_connected() -> None:
    cluster = load_cluster_config("config/cluster.dev.json")
    graph = ClusterGraph(cluster)

    edge = graph.edge("boston", "virginia")

    assert edge.distance_km > 0
    assert edge.bandwidth_mbps == 100
    assert edge.latency_ms == 50
