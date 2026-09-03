from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


EXPECTED_STAGE5A_NODE_IDS = {
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
}


def expected_directed_path_count(node_count: int) -> int:
    if node_count < 0:
        raise ValueError("node_count cannot be negative")
    return node_count * (node_count - 1)


def identical_hashes(
    rows: Iterable[dict[str, Any]],
    *,
    key_field: str,
    hash_field: str,
) -> dict[str, bool]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row[key_field])].add(str(row[hash_field]))
    return {key: len(values) == 1 for key, values in grouped.items()}


def stage5a_passes(
    *,
    node_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    mesh_rows: list[dict[str, Any]],
    expected_git_sha: str,
) -> bool:
    expected_nodes = EXPECTED_STAGE5A_NODE_IDS
    observed_nodes = {str(row["node_id"]) for row in node_rows}
    if observed_nodes != expected_nodes or len(node_rows) != len(expected_nodes):
        return False

    for row in node_rows:
        if str(row.get("repo_git_sha")) != expected_git_sha:
            return False
        if str(row.get("daemon_git_sha")) != expected_git_sha:
            return False
        if not bool(row.get("tracked_worktree_clean")):
            return False
        if not bool(row.get("service_active")):
            return False
        if not bool(row.get("health_ok")):
            return False
        if not bool(row.get("capabilities_ready")):
            return False
        if str(row.get("health_node_id")) != str(row.get("node_id")):
            return False

    if len({str(row["cluster_sha256"]) for row in node_rows}) != 1:
        return False
    if len({str(row["policy_sha256"]) for row in node_rows}) != 1:
        return False

    dataset_nodes = defaultdict(set)
    dataset_hashes = defaultdict(set)
    for row in dataset_rows:
        name = str(row["dataset_file"])
        dataset_nodes[name].add(str(row["node_id"]))
        dataset_hashes[name].add(str(row["sha256"]))
    if len(dataset_nodes) != len(expected_nodes):
        return False
    for name in dataset_nodes:
        if dataset_nodes[name] != expected_nodes:
            return False
        if len(dataset_hashes[name]) != 1:
            return False

    expected_paths = expected_directed_path_count(len(expected_nodes))
    if len(mesh_rows) != expected_paths:
        return False
    observed_pairs = {
        (str(row["source"]), str(row["destination"])) for row in mesh_rows
    }
    if len(observed_pairs) != expected_paths:
        return False
    for source, destination in observed_pairs:
        if source == destination:
            return False
        if source not in expected_nodes or destination not in expected_nodes:
            return False
    return all(
        bool(row.get("api_ok"))
        and bool(row.get("ssh_ok"))
        and bool(row.get("ok"))
        for row in mesh_rows
    )
