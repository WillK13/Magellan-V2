#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from magellan.config.loader import load_cluster_config


EXPECTED_STAGE5_CARBON_METRIC = "lifecycle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect local evidence for one Stage 5A Magellan daemon."
    )
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--service", default="magellan")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    result = subprocess.run(
        list(args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def request_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_systemd_environment(value: str) -> dict[str, str]:
    environment: dict[str, str] = {}
    for token in shlex.split(value):
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        environment[key] = item
    return environment


def parse_systemd_paths(value: str) -> list[str]:
    return [item for item in shlex.split(value) if item]


def expected_effective_environment(
    *,
    node_id: str,
    git_sha: str,
    git_branch: str,
    repository_root: Path,
    cluster_path: str,
    policy_path: str,
    datasets_path: str,
) -> dict[str, str]:
    state_root = repository_root / "runtime-state-gcp"
    return {
        "MAGELLAN_NODE_ID": node_id,
        "MAGELLAN_GIT_SHA": git_sha,
        "MAGELLAN_GIT_BRANCH": git_branch,
        "MAGELLAN_CONFIG": cluster_path,
        "MAGELLAN_POLICY": policy_path,
        "MAGELLAN_DATASETS": datasets_path,
        "MAGELLAN_CARBON_METRIC": EXPECTED_STAGE5_CARBON_METRIC,
        "MAGELLAN_STATE_ROOT": str(state_root),
        "MAGELLAN_REMOTE_STATE_ROOT": str(state_root),
        "MAGELLAN_REPOSITORY_ROOT": str(repository_root),
    }


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    node = cluster.get_node(args.node_id)

    repo_sha = command("git", "rev-parse", "HEAD")
    branch = command("git", "branch", "--show-current") or "DETACHED"
    dirty = command(
        "git", "status", "--porcelain", "--untracked-files=no"
    )
    service_active = (
        subprocess.run(
            ["systemctl", "is-active", "--quiet", args.service],
            check=False,
        ).returncode
        == 0
    )
    main_pid = command(
        "systemctl", "show", args.service, "--property=MainPID", "--value"
    )
    raw_environment = command(
        "systemctl", "show", args.service, "--property=Environment", "--value"
    )
    effective_environment = parse_systemd_environment(raw_environment)
    dropin_paths = parse_systemd_paths(
        command(
            "systemctl",
            "show",
            args.service,
            "--property=DropInPaths",
            "--value",
        )
    )

    repository_root = Path.cwd().resolve()
    expected_environment = expected_effective_environment(
        node_id=node.id,
        git_sha=args.expected_git_sha,
        git_branch=branch,
        repository_root=repository_root,
        cluster_path=args.cluster,
        policy_path=args.policy,
        datasets_path=args.datasets,
    )
    environment_mismatches = {
        key: {
            "expected": expected,
            "actual": effective_environment.get(key),
        }
        for key, expected in expected_environment.items()
        if effective_environment.get(key) != expected
    }

    state_root_value = effective_environment.get("MAGELLAN_STATE_ROOT", "")
    remote_state_root_value = effective_environment.get(
        "MAGELLAN_REMOTE_STATE_ROOT", ""
    )
    state_root = Path(state_root_value) if state_root_value else None
    remote_state_root = (
        Path(remote_state_root_value) if remote_state_root_value else None
    )
    state_root_exists = state_root is not None and state_root.is_dir()
    remote_state_root_exists = (
        remote_state_root is not None and remote_state_root.is_dir()
    )
    state_root_writable = (
        state_root is not None
        and state_root_exists
        and os.access(state_root, os.W_OK)
    )
    remote_state_root_writable = (
        remote_state_root is not None
        and remote_state_root_exists
        and os.access(remote_state_root, os.W_OK)
    )

    health = request_json(
        f"http://127.0.0.1:{cluster.api_port}/health",
        args.timeout,
    )
    capabilities = request_json(
        f"http://127.0.0.1:{cluster.api_port}/capabilities",
        args.timeout,
    )

    dataset_rows = []
    datasets_root = Path(args.datasets)
    for configured in cluster.nodes:
        path = datasets_root / configured.dataset_file
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset_rows.append(
            {
                "dataset_file": configured.dataset_file,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    payload = {
        "node_id": node.id,
        "node_name": node.name,
        "vm_name": node.vm_name,
        "zone": node.zone,
        "internal_ip": str(node.internal_ip),
        "machine_type": node.machine_type,
        "repo_git_sha": repo_sha,
        "repo_branch": branch,
        "expected_git_sha": args.expected_git_sha,
        "repo_sha_matches": repo_sha == args.expected_git_sha,
        "tracked_worktree_clean": not bool(dirty),
        "service_active": service_active,
        "service_main_pid": int(main_pid or 0),
        "cluster_sha256": sha256_file(Path(args.cluster)),
        "policy_sha256": sha256_file(Path(args.policy)),
        "python_version": sys.version.split()[0],
        "health": health,
        "health_ok": health.get("status") == "ok",
        "health_node_id": health.get("node_id"),
        "health_carbon_metric": health.get("carbon_metric"),
        "daemon_git_sha": health.get("deployment_git_sha"),
        "daemon_git_branch": health.get("deployment_git_branch"),
        "capabilities": capabilities,
        "capabilities_ready": bool(capabilities.get("ready")),
        "effective_environment": effective_environment,
        "expected_environment": expected_environment,
        "effective_environment_ok": not environment_mismatches,
        "effective_environment_mismatches": environment_mismatches,
        "systemd_dropin_paths": dropin_paths,
        "systemd_dropin_count": len(dropin_paths),
        "state_root_exists": state_root_exists,
        "state_root_writable": state_root_writable,
        "remote_state_root_exists": remote_state_root_exists,
        "remote_state_root_writable": remote_state_root_writable,
        "dataset_hashes": dataset_rows,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
