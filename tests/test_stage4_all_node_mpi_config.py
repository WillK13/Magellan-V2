from magellan.config.loader import load_cluster_config


def test_all_stage4_gcp_nodes_advertise_openmpi() -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    assert len(cluster.nodes) == 7
    for node in cluster.nodes:
        assert "mpirun" in node.capabilities.commands
        assert "mpiexec" in node.capabilities.commands
        assert "mpi" in node.capabilities.features
        assert node.capabilities.runtimes.get("openmpi") == "4.1.4"


def test_gcp_bootstrap_installs_dendro_runtime_libraries() -> None:
    text = open("scripts/bootstrap_gcp_node.sh", encoding="utf-8").read()
    assert "libgsl27" in text
    assert "libopenblas0-pthread" in text
