#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from magellan.config.loader import load_cluster_config


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
        "daemon_git_sha": health.get("deployment_git_sha"),
        "daemon_git_branch": health.get("deployment_git_branch"),
        "capabilities": capabilities,
        "capabilities_ready": bool(capabilities.get("ready")),
        "dataset_hashes": dataset_rows,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
