#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
    write_jsonl,
)
from magellan.experiments.stage5a import (
    EXPECTED_STAGE5A_NODE_IDS,
    expected_directed_path_count,
    stage5a_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy/validate one exact Magellan commit across all seven GCP nodes "
            "and freeze Stage 5A evidence. Run from Boston."
        )
    )
    parser.add_argument("--stage4e3-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--service", default="magellan")
    parser.add_argument("--connect-timeout", type=int, default=8)
    parser.add_argument(
        "--branch",
        help="Deployment branch. Defaults to the current local branch.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help=(
            "Synchronize every non-local node to the exact local origin branch SHA, "
            "then install/restart the Magellan systemd service on all seven nodes."
        ),
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def local_git(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def remote_cd(path: str) -> str:
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        return f'cd "$HOME"/{shlex.quote(path[2:])}'
    return f"cd {shlex.quote(path)}"


def source_shell(
    *,
    node_id: str,
    node_ip: str,
    local_node_id: str,
    ssh_user: str,
    timeout: int,
    command: str,
) -> list[str]:
    if node_id == local_node_id:
        return ["bash", "-lc", command]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        f"{ssh_user}@{node_ip}",
        command,
    ]


def deploy_remote_command(
    *,
    remote_repo: str,
    branch: str,
    target_sha: str,
    node_id: str,
    service: str,
) -> str:
    return f"""
set -euo pipefail
{remote_cd(remote_repo)}
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked working tree is dirty" >&2
  git status --short >&2
  exit 12
fi
git fetch origin
REMOTE_SHA="$(git rev-parse {shlex.quote('origin/' + branch)})"
if [[ "$REMOTE_SHA" != {shlex.quote(target_sha)} ]]; then
  echo "origin branch SHA mismatch: $REMOTE_SHA != {target_sha}" >&2
  exit 13
fi
git switch -C {shlex.quote(branch)} {shlex.quote('origin/' + branch)}
test "$(git rev-parse HEAD)" = {shlex.quote(target_sha)}
python -m compileall -q magellan scripts
scripts/install_magellan_systemd.sh {shlex.quote(node_id)} >/tmp/magellan-stage5a-systemd.log
sudo systemctl is-active --quiet {shlex.quote(service)}
curl -fsS --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:8040/health >/dev/null
echo STAGE5A_NODE_DEPLOYED node={shlex.quote(node_id)} sha=$(git rev-parse HEAD)
""".strip()


def local_restart_command(*, node_id: str, service: str) -> str:
    return f"""
set -euo pipefail
scripts/install_magellan_systemd.sh {shlex.quote(node_id)} >/tmp/magellan-stage5a-systemd.log
sudo systemctl is-active --quiet {shlex.quote(service)}
curl -fsS --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:8040/health >/dev/null
""".strip()


def probe_command(
    *,
    remote_repo: str,
    node_id: str,
    target_sha: str,
    cluster: str,
    policy: str,
    datasets: str,
    service: str,
) -> str:
    return " ".join(
        [
            remote_cd(remote_repo),
            "&&",
            ".venv/bin/python",
            "scripts/stage5a_node_probe.py",
            "--node-id",
            shlex.quote(node_id),
            "--expected-git-sha",
            shlex.quote(target_sha),
            "--cluster",
            shlex.quote(cluster),
            "--policy",
            shlex.quote(policy),
            "--datasets",
            shlex.quote(datasets),
            "--service",
            shlex.quote(service),
        ]
    )


def main() -> int:
    args = parse_args()
    source_e3 = Path(args.stage4e3_bundle)
    errors = validate_checksums(source_e3)
    if errors:
        raise RuntimeError("Stage 4E.3 checksum failure: " + "; ".join(errors))
    source_summary = json.loads(
        (source_e3 / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("passed") is not True:
        raise RuntimeError("Stage 4E.3 source bundle did not pass")

    cluster = load_cluster_config(args.cluster)
    cluster.get_node(args.local_node_id)
    node_ids = {node.id for node in cluster.nodes}
    if node_ids != EXPECTED_STAGE5A_NODE_IDS:
        raise RuntimeError(f"Unexpected Stage 5A node set: {sorted(node_ids)}")

    target_sha = local_git("rev-parse", "HEAD")
    local_branch = local_git("branch", "--show-current")
    branch = args.branch or local_branch
    if not branch:
        raise RuntimeError("Stage 5A requires a named local deployment branch")
    if local_branch != branch:
        raise RuntimeError(
            f"Local branch {local_branch!r} does not match deployment branch {branch!r}"
        )
    if local_git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Local tracked working tree must be clean before Stage 5A")

    # Require the exact commit to already be published before touching peers.
    run(["git", "fetch", "origin"])
    origin_sha = local_git("rev-parse", f"origin/{branch}")
    if origin_sha != target_sha:
        raise RuntimeError(
            f"Push the deployment branch first: local={target_sha} origin={origin_sha}"
        )

    print("== Stage 5A seven-node synchronized deployment ==")
    print(f"target_git_sha={target_sha}")
    print(f"branch={branch}")
    print(f"source_stage4e3={source_e3}")
    print(f"nodes={','.join(node.id for node in cluster.nodes)}")
    print(f"mode={'deploy+verify' if args.deploy else 'verify-only'}")

    if args.deploy:
        for node in cluster.nodes:
            print(f"[deploy] {node.id}", flush=True)
            if node.id == args.local_node_id:
                shell = source_shell(
                    node_id=node.id,
                    node_ip=str(node.internal_ip),
                    local_node_id=args.local_node_id,
                    ssh_user=args.ssh_user,
                    timeout=args.connect_timeout,
                    command=local_restart_command(
                        node_id=node.id,
                        service=args.service,
                    ),
                )
            else:
                shell = source_shell(
                    node_id=node.id,
                    node_ip=str(node.internal_ip),
                    local_node_id=args.local_node_id,
                    ssh_user=args.ssh_user,
                    timeout=args.connect_timeout,
                    command=deploy_remote_command(
                        remote_repo=args.remote_repo,
                        branch=branch,
                        target_sha=target_sha,
                        node_id=node.id,
                        service=args.service,
                    ),
                )
            result = run(shell, check=False, timeout=180)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Deployment failed for {node.id}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )

    node_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    probe_payloads: list[dict[str, Any]] = []
    for node in cluster.nodes:
        print(f"[probe] {node.id}", flush=True)
        shell = source_shell(
            node_id=node.id,
            node_ip=str(node.internal_ip),
            local_node_id=args.local_node_id,
            ssh_user=args.ssh_user,
            timeout=args.connect_timeout,
            command=probe_command(
                remote_repo=args.remote_repo,
                node_id=node.id,
                target_sha=target_sha,
                cluster=args.cluster,
                policy=args.policy,
                datasets=args.datasets,
                service=args.service,
            ),
        )
        result = run(shell, check=False, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Probe failed for {node.id}: "
                f"{(result.stderr or result.stdout).strip()}"
            )
        payload = json.loads(result.stdout)
        probe_payloads.append(payload)
        node_rows.append(
            {
                "node_id": payload["node_id"],
                "node_name": payload["node_name"],
                "vm_name": payload["vm_name"],
                "zone": payload["zone"],
                "internal_ip": payload["internal_ip"],
                "machine_type": payload["machine_type"],
                "repo_git_sha": payload["repo_git_sha"],
                "repo_branch": payload["repo_branch"],
                "daemon_git_sha": payload["daemon_git_sha"],
                "daemon_git_branch": payload["daemon_git_branch"],
                "tracked_worktree_clean": payload["tracked_worktree_clean"],
                "service_active": payload["service_active"],
                "service_main_pid": payload["service_main_pid"],
                "health_ok": payload["health_ok"],
                "health_node_id": payload["health_node_id"],
                "capabilities_ready": payload["capabilities_ready"],
                "cluster_sha256": payload["cluster_sha256"],
                "policy_sha256": payload["policy_sha256"],
                "python_version": payload["python_version"],
            }
        )
        for item in payload["dataset_hashes"]:
            dataset_rows.append(
                {
                    "node_id": node.id,
                    "dataset_file": item["dataset_file"],
                    "sha256": item["sha256"],
                    "bytes": item["bytes"],
                }
            )
        print(
            f"  sha={payload['repo_git_sha'][:12]} "
            f"daemon_sha={str(payload['daemon_git_sha'])[:12]} "
            f"service={'active' if payload['service_active'] else 'inactive'} "
            f"capabilities={'ready' if payload['capabilities_ready'] else 'drift'}",
            flush=True,
        )

    print("\n[mesh] validating 42 directed API + SSH/rsync paths", flush=True)
    mesh_result = run(
        [
            ".venv/bin/python",
            "scripts/validate_seven_node_mesh.py",
            "--cluster",
            args.cluster,
            "--local-node-id",
            args.local_node_id,
            "--ssh-user",
            args.ssh_user,
            "--connect-timeout",
            str(args.connect_timeout),
            "--json",
        ],
        check=False,
        timeout=180,
    )
    if mesh_result.returncode != 0:
        raise RuntimeError(
            "Seven-node mesh validation failed: "
            + (mesh_result.stderr or mesh_result.stdout).strip()
        )
    mesh_payload = json.loads(mesh_result.stdout)
    mesh_rows = list(mesh_payload["results"])

    passed = stage5a_passes(
        node_rows=node_rows,
        dataset_rows=dataset_rows,
        mesh_rows=mesh_rows,
        expected_git_sha=target_sha,
    )

    comparison_id = args.comparison_id or (
        f"stage5a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4e3_bundle": str(source_e3),
        "target_git_sha": target_sha,
        "deployment_branch": branch,
        "deployment_mode": "deploy+verify" if args.deploy else "verify-only",
        "node_count": len(node_rows),
        "expected_node_count": len(EXPECTED_STAGE5A_NODE_IDS),
        "directed_path_count": len(mesh_rows),
        "expected_directed_path_count": expected_directed_path_count(len(node_rows)),
        "api_paths_ok": sum(bool(row["api_ok"]) for row in mesh_rows),
        "ssh_paths_ok": sum(bool(row["ssh_ok"]) for row in mesh_rows),
        "exact_repo_sha_nodes": sum(
            row["repo_git_sha"] == target_sha for row in node_rows
        ),
        "exact_daemon_sha_nodes": sum(
            row["daemon_git_sha"] == target_sha for row in node_rows
        ),
        "active_service_nodes": sum(bool(row["service_active"]) for row in node_rows),
        "capabilities_ready_nodes": sum(
            bool(row["capabilities_ready"]) for row in node_rows
        ),
        "dataset_manifest_rows": len(dataset_rows),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5a_real_seven_node_deployment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "execution": (
                "Stage 5A runs against seven real GCP VMs. It is not a logical-node "
                "replay. Each node is independently probed over SSH and each directed "
                "source-to-destination API/SSH path is exercised from its source VM."
            ),
            "commit_identity": (
                "The systemd unit embeds MAGELLAN_GIT_SHA and MAGELLAN_GIT_BRANCH from "
                "the repository used to install it. /health exposes those values, so "
                "Stage 5A verifies both the on-disk repository SHA and the running "
                "daemon SHA against one exact target commit."
            ),
            "input_identity": (
                "Every node hashes cluster.gcp.json, policy.prod.json, and all seven "
                "carbon datasets. PASS requires identical hashes across nodes."
            ),
            "mesh": (
                "All 42 non-self directed pairs must pass the existing peer FastAPI "
                "health check and SSH/rsync transport check from the actual source node."
            ),
            "scope": (
                "Stage 5A establishes deployment identity and connectivity. It does not "
                "claim scheduling, contention, migration, or failure-recovery behavior; "
                "those are Stage 5B-5F."
            ),
        },
    }

    write_csv(root / "nodes.csv", node_rows, list(node_rows[0].keys()))
    write_csv(
        root / "dataset_hashes.csv",
        dataset_rows,
        ["node_id", "dataset_file", "sha256", "bytes"],
    )
    write_csv(
        root / "directed_mesh.csv",
        mesh_rows,
        ["source", "destination", "api_ok", "ssh_ok", "error", "ok"],
    )
    write_jsonl(root / "node_probes.jsonl", probe_payloads)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_5A_SEVEN_NODE_DEPLOYMENT_PASS" if passed else "STAGE_5A_SEVEN_NODE_DEPLOYMENT_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"git_sha: {target_sha}")
    print(f"nodes: {summary['exact_daemon_sha_nodes']}/{len(node_rows)} exact running SHA")
    print(
        f"directed_paths: {summary['api_paths_ok']}/{len(mesh_rows)} API, "
        f"{summary['ssh_paths_ok']}/{len(mesh_rows)} SSH/rsync"
    )
    print(f"capabilities_ready: {summary['capabilities_ready_nodes']}/{len(node_rows)}")
    print(f"dataset_manifest_rows: {len(dataset_rows)}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
