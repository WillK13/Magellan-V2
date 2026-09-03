from __future__ import annotations

from magellan.experiments.stage5a import (
    EXPECTED_STAGE5A_NODE_IDS,
    expected_directed_path_count,
    identical_hashes,
    stage5a_passes,
)


def _node_rows(sha: str) -> list[dict]:
    return [
        {
            "node_id": node_id,
            "repo_git_sha": sha,
            "daemon_git_sha": sha,
            "tracked_worktree_clean": True,
            "service_active": True,
            "health_ok": True,
            "capabilities_ready": True,
            "health_node_id": node_id,
            "cluster_sha256": "cluster",
            "policy_sha256": "policy",
        }
        for node_id in sorted(EXPECTED_STAGE5A_NODE_IDS)
    ]


def _dataset_rows() -> list[dict]:
    return [
        {
            "node_id": node_id,
            "dataset_file": dataset,
            "sha256": f"hash-{dataset}",
        }
        for node_id in sorted(EXPECTED_STAGE5A_NODE_IDS)
        for dataset in sorted(EXPECTED_STAGE5A_NODE_IDS)
    ]


def _mesh_rows() -> list[dict]:
    return [
        {
            "source": source,
            "destination": destination,
            "api_ok": True,
            "ssh_ok": True,
            "ok": True,
        }
        for source in sorted(EXPECTED_STAGE5A_NODE_IDS)
        for destination in sorted(EXPECTED_STAGE5A_NODE_IDS)
        if source != destination
    ]


def test_seven_nodes_have_42_directed_paths() -> None:
    assert expected_directed_path_count(7) == 42


def test_identical_hashes_groups_by_input() -> None:
    rows = [
        {"name": "a", "sha": "1"},
        {"name": "a", "sha": "1"},
        {"name": "b", "sha": "2"},
        {"name": "b", "sha": "3"},
    ]
    assert identical_hashes(rows, key_field="name", hash_field="sha") == {
        "a": True,
        "b": False,
    }


def test_stage5a_passes_exact_real_mesh() -> None:
    sha = "a" * 40
    assert stage5a_passes(
        node_rows=_node_rows(sha),
        dataset_rows=_dataset_rows(),
        mesh_rows=_mesh_rows(),
        expected_git_sha=sha,
    )


def test_stage5a_rejects_wrong_daemon_sha() -> None:
    sha = "a" * 40
    nodes = _node_rows(sha)
    nodes[0]["daemon_git_sha"] = "b" * 40
    assert not stage5a_passes(
        node_rows=nodes,
        dataset_rows=_dataset_rows(),
        mesh_rows=_mesh_rows(),
        expected_git_sha=sha,
    )


def test_stage5a_rejects_broken_directed_api_path() -> None:
    sha = "a" * 40
    mesh = _mesh_rows()
    mesh[0]["api_ok"] = False
    mesh[0]["ok"] = False
    assert not stage5a_passes(
        node_rows=_node_rows(sha),
        dataset_rows=_dataset_rows(),
        mesh_rows=mesh,
        expected_git_sha=sha,
    )


def test_stage5a_rejects_divergent_dataset_hash() -> None:
    sha = "a" * 40
    datasets = _dataset_rows()
    datasets[0]["sha256"] = "different"
    assert not stage5a_passes(
        node_rows=_node_rows(sha),
        dataset_rows=datasets,
        mesh_rows=_mesh_rows(),
        expected_git_sha=sha,
    )
