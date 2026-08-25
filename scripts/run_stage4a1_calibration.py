#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import sha256_file, write_checksums, write_csv, write_json
from magellan.experiments.stage4a1 import summarize_network_bundle, validate_stage4a1_node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 4A.1 final-hardware preflight and the complete directed WAN "
            "calibration/held-out validation campaign. Run from Boston or another "
            "cluster node with private-IP SSH access to every peer."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--expected-node-count", type=int, default=7)
    parser.add_argument("--expected-machine-type", default="e2-highmem-2")
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
    )
    parser.add_argument("--rtt-samples", type=int, default=15)
    parser.add_argument("--bandwidth-samples", type=int, default=3)
    parser.add_argument("--payload-bytes", type=int, default=8 * 1024 * 1024 + 123)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--calibration-id", default=None)
    return parser.parse_args()


def request_json(url: str, timeout: float = 10.0) -> Any:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run_at_source(
    *,
    local_node_id: str,
    source_node_id: str,
    source_ip: str,
    ssh_user: str,
    command: str,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    if source_node_id == local_node_id:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=8",
            f"{ssh_user}@{source_ip}",
            command,
        ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


REMOTE_HARDWARE_CODE = r'''
import json
import os
import pathlib
import platform
import subprocess
import urllib.request

repo = pathlib.Path.home() / "Magellan-V2"


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def metadata(path):
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/" + path,
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.read().decode().strip()

cpu_model = None
for raw in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
    if raw.lower().startswith("model name") and ":" in raw:
        cpu_model = raw.split(":", 1)[1].strip()
        break

memory_mb = None
for raw in pathlib.Path("/proc/meminfo").read_text().splitlines():
    if raw.startswith("MemTotal:"):
        memory_mb = int(raw.split()[1]) // 1024
        break

_, git_commit, _ = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
_, git_branch, _ = run(["git", "-C", str(repo), "branch", "--show-current"])
_, git_status, _ = run([
    "git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"
])
tracked_changes = [
    line for line in git_status.splitlines()
    if "magellan_v2.egg-info/" not in line
]
_, service_active, _ = run(["sudo", "-n", "systemctl", "is-active", "magellan"])
_, mpirun_output, _ = run(["mpirun", "--version"])

machine_type = metadata("instance/machine-type").rsplit("/", 1)[-1]
zone = metadata("instance/zone").rsplit("/", 1)[-1]
instance_name = metadata("instance/name")

print(json.dumps({
    "hostname": platform.node(),
    "instance_name": instance_name,
    "zone": zone,
    "machine_type": machine_type,
    "architecture": platform.machine(),
    "platform": platform.platform(),
    "cpu_model": cpu_model,
    "cpu_logical_count": os.cpu_count(),
    "memory_mb": memory_mb,
    "python_version": platform.python_version(),
    "mpirun_version": mpirun_output.splitlines()[0] if mpirun_output else None,
    "git_commit": git_commit,
    "git_branch": git_branch,
    "git_status_porcelain": tracked_changes,
    "service_active": service_active,
}, sort_keys=True))
'''


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    if len(cluster.nodes) != args.expected_node_count:
        raise RuntimeError(
            f"Configured node count={len(cluster.nodes)}; "
            f"expected {args.expected_node_count}"
        )
    if args.rtt_samples < 1 or args.bandwidth_samples < 1:
        raise ValueError("Sample counts must be positive")
    if args.payload_bytes <= 0:
        raise ValueError("--payload-bytes must be positive")

    local = cluster.get_node(args.local_node_id)
    local_commit = git_value("rev-parse", "HEAD")
    local_branch = git_value("branch", "--show-current")
    local_status = git_value("status", "--porcelain")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibration_id = (
        args.calibration_id
        or f"stage4a1-{timestamp}-{uuid4().hex[:8]}"
    )
    bundle = Path(args.measurements_root) / calibration_id
    if bundle.exists():
        raise FileExistsError(f"Calibration bundle already exists: {bundle}")
    bundle.mkdir(parents=True)

    print("== Stage 4A.1 final-hardware preflight ==")
    print(f"calibration_id={calibration_id}")
    print(f"local_commit={local_commit}")
    print(f"local_branch={local_branch}")

    preflight_rows: list[dict[str, Any]] = []
    hardware_by_node: dict[str, Any] = {}
    all_errors: list[str] = []

    remote_command = "python3 -c " + shlex.quote(REMOTE_HARDWARE_CODE)
    for node in cluster.nodes:
        base = f"http://{node.internal_ip}:{cluster.api_port}"
        health = request_json(f"{base}/health")
        capabilities = request_json(f"{base}/capabilities")
        auction = request_json(f"{base}/auction/status")
        result = run_at_source(
            local_node_id=local.id,
            source_node_id=node.id,
            source_ip=str(node.internal_ip),
            ssh_user=args.ssh_user,
            command=remote_command,
            timeout=max(30.0, args.timeout_seconds),
        )
        remote = json.loads(result.stdout.strip().splitlines()[-1])

        errors = validate_stage4a1_node(
            node,
            local_commit=local_commit,
            health=health,
            capabilities=capabilities,
            auction=auction,
            remote=remote,
            expected_machine_type=args.expected_machine_type,
            expected_carbon_metric=args.expected_carbon_metric,
            expected_state_token=args.expected_state_token,
        )
        all_errors.extend(errors)
        hardware_by_node[node.id] = {
            "configured": node.model_dump(mode="json"),
            "health": health,
            "capabilities": capabilities,
            "auction": auction,
            "observed_host": remote,
            "preflight_errors": errors,
            "preflight_passed": not errors,
        }
        preflight_rows.append(
            {
                "node_id": node.id,
                "vm_name": node.vm_name,
                "zone": node.zone,
                "machine_type": remote.get("machine_type"),
                "cpu_logical_count": remote.get("cpu_logical_count"),
                "memory_mb": remote.get("memory_mb"),
                "compute_price_usd_per_hour": node.compute_price_usd_per_hour,
                "carbon_region": node.carbon_region,
                "git_commit": remote.get("git_commit"),
                "service_active": remote.get("service_active"),
                "capabilities_ready": capabilities.get("ready"),
                "resource_busy_fraction": auction.get("resource_busy_fraction"),
                "preflight_passed": not errors,
            }
        )
        label = "PASS" if not errors else "FAIL"
        print(
            f"[{label}] {node.id:16} "
            f"machine={remote.get('machine_type')} "
            f"cpu={remote.get('cpu_logical_count')} "
            f"mem={remote.get('memory_mb')}MB "
            f"commit={str(remote.get('git_commit'))[:8]}"
        )

    write_json(bundle / "hardware.json", hardware_by_node)
    write_csv(
        bundle / "hardware.csv",
        preflight_rows,
        [
            "node_id",
            "vm_name",
            "zone",
            "machine_type",
            "cpu_logical_count",
            "memory_mb",
            "compute_price_usd_per_hour",
            "carbon_region",
            "git_commit",
            "service_active",
            "capabilities_ready",
            "resource_busy_fraction",
            "preflight_passed",
        ],
    )

    metadata = {
        "format_version": 1,
        "measurement_type": "stage4a1_hardware_network_calibration",
        "calibration_id": calibration_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": local_commit,
            "branch": local_branch,
            "status_porcelain": local_status,
        },
        "cluster": {
            "path": args.cluster,
            "sha256": sha256_file(args.cluster),
            "node_count": len(cluster.nodes),
            "node_ids": [node.id for node in cluster.nodes],
            "directed_edge_count": len(cluster.nodes) * (len(cluster.nodes) - 1),
        },
        "requirements": {
            "expected_node_count": args.expected_node_count,
            "expected_machine_type": args.expected_machine_type,
            "expected_carbon_metric": args.expected_carbon_metric,
            "expected_state_token": args.expected_state_token,
            "idle_cluster_required": True,
            "identical_git_commit_required": True,
        },
        "network_parameters": {
            "rtt_samples_per_edge": args.rtt_samples,
            "bandwidth_samples_per_edge": args.bandwidth_samples,
            "held_out_payload_bytes": args.payload_bytes,
            "timeout_seconds": args.timeout_seconds,
        },
        "hardware_preflight_passed": not all_errors,
        "hardware_preflight_errors": all_errors,
    }
    write_json(bundle / "metadata.json", metadata)

    if all_errors:
        write_checksums(bundle)
        print("\nSTAGE 4A.1 PREFLIGHT FAILED")
        for error in all_errors:
            print(f"- {error}")
        print(f"bundle: {bundle}")
        return 1

    print("\n== Complete 42-edge directed WAN campaign ==")
    network_root = bundle / "network"
    network_id = "directed-mesh"
    command = [
        "python",
        "scripts/measure_seven_node_network.py",
        "--cluster",
        args.cluster,
        "--local-node-id",
        args.local_node_id,
        "--ssh-user",
        args.ssh_user,
        "--rtt-samples",
        str(args.rtt_samples),
        "--bandwidth-samples",
        str(args.bandwidth_samples),
        "--payload-bytes",
        str(args.payload_bytes),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--measurements-root",
        str(network_root),
        "--measurement-id",
        network_id,
        "--expected-carbon-metric",
        args.expected_carbon_metric,
        "--expected-state-token",
        args.expected_state_token,
    ]
    subprocess.run(command, check=True)

    network_bundle = network_root / network_id
    subprocess.run(
        ["python", "scripts/validate_network_measurement.py", str(network_bundle)],
        check=True,
    )

    network_summary = summarize_network_bundle(network_bundle)
    summary = {
        "calibration_id": calibration_id,
        "hardware_preflight_passed": True,
        "node_count": len(cluster.nodes),
        "machine_type": args.expected_machine_type,
        "git_commit": local_commit,
        "network_bundle": str(network_bundle.relative_to(bundle)),
        "network": network_summary,
    }
    write_json(bundle / "summary.json", summary)
    write_checksums(bundle)

    print("\nSTAGE_4A1_CALIBRATION_PASS")
    print(f"bundle: {bundle}")
    print(f"nodes: {len(cluster.nodes)}")
    print(f"directed_edges: {network_summary['directed_edge_count']}")
    error = network_summary["absolute_transfer_prediction_error_percent"]
    if error is not None:
        print(
            "median_abs_transfer_prediction_error_pct: "
            f"{error['median']:.2f}"
        )
        print(
            "p95_abs_transfer_prediction_error_pct: "
            f"{error['p95']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
