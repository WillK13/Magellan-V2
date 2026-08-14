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
from magellan.experiments.measurement import (
    absolute_percent_error,
    directed_edge_pairs,
    predict_transfer_seconds,
    signed_percent_error,
    summarize_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the real directed WAN mesh described by the cluster config. "
            "Run from a cluster node that can SSH to every peer over private addresses."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--rtt-samples", type=int, default=10)
    parser.add_argument("--bandwidth-samples", type=int, default=2)
    parser.add_argument("--payload-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument(
        "--seed-telemetry",
        action="store_true",
        help=(
            "Record the measured RTT/transfer samples into each source daemon's "
            "edge telemetry after collecting the cold prediction."
        ),
    )
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
        help="Substring required in each node's telemetry state path.",
    )
    return parser.parse_args()


def request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_at_source(
    *,
    local_node_id: str,
    source_node_id: str,
    source_ip: str,
    ssh_user: str,
    command: str,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if source_node_id == local_node_id:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
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


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def python_command(code: str, *args: str) -> str:
    return " ".join(["python3", "-c", shlex.quote(code), *(shlex.quote(arg) for arg in args)])


def main() -> int:
    args = parse_args()
    if args.rtt_samples < 1 or args.bandwidth_samples < 1:
        raise ValueError("Sample counts must be positive")
    if args.payload_bytes <= 0:
        raise ValueError("--payload-bytes must be positive")

    cluster = load_cluster_config(args.cluster)
    local = cluster.get_node(args.local_node_id)
    node_by_id = {node.id: node for node in cluster.nodes}
    node_ids = [node.id for node in cluster.nodes]
    pairs = directed_edge_pairs(node_ids)
    if len(pairs) != len(node_ids) * (len(node_ids) - 1):
        raise RuntimeError("Directed mesh construction failed")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurement_id = args.measurement_id or f"network-{timestamp}-{uuid4().hex[:8]}"
    bundle = Path(args.measurements_root) / measurement_id
    if bundle.exists():
        raise FileExistsError(f"Measurement bundle already exists: {bundle}")
    (bundle / "raw").mkdir(parents=True)

    print("== Topology-driven WAN characterization ==")
    print(f"measurement_id={measurement_id}")
    print(f"directed_edges={len(pairs)}")
    print(
        f"rtt_samples={args.rtt_samples} bandwidth_samples={args.bandwidth_samples} "
        f"payload_bytes={args.payload_bytes}"
    )

    health: dict[str, dict[str, Any]] = {}
    print("== Verify seven peer APIs ==")
    for node in cluster.nodes:
        value = request_json(f"http://{node.internal_ip}:{cluster.api_port}/health")
        if value.get("node_id") != node.id:
            raise RuntimeError(f"Node identity mismatch for {node.id}: {value.get('node_id')}")
        if (
            args.expected_carbon_metric
            and value.get("carbon_metric") != args.expected_carbon_metric
        ):
            raise RuntimeError(
                f"{node.id} carbon metric={value.get('carbon_metric')} "
                f"expected={args.expected_carbon_metric}"
            )
        state_file = str(value.get("telemetry_state_file", ""))
        if args.expected_state_token and args.expected_state_token not in state_file:
            raise RuntimeError(
                f"{node.id} is not in isolated measurement state: {state_file}"
            )
        health[node.id] = value
        print(f"[OK] {node.id:16} {node.internal_ip} state={state_file}")

    payload_paths: dict[str, str] = {}
    create_code = """
import os, pathlib, sys
path = pathlib.Path(sys.argv[1])
size = int(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
if not path.is_file() or path.stat().st_size != size:
    tmp = path.with_suffix('.tmp')
    remaining = size
    with tmp.open('wb') as handle:
        while remaining:
            n = min(1024 * 1024, remaining)
            handle.write(os.urandom(n))
            remaining -= n
    os.replace(tmp, path)
print(path.stat().st_size)
"""
    print("== Prepare incompressible source payloads ==")
    for node in cluster.nodes:
        payload_path = f"/tmp/{measurement_id}-{node.id}.bin"
        result = run_at_source(
            local_node_id=local.id,
            source_node_id=node.id,
            source_ip=str(node.internal_ip),
            ssh_user=args.ssh_user,
            command=python_command(create_code, payload_path, str(args.payload_bytes)),
            timeout=max(60.0, args.timeout_seconds),
        )
        if int(result.stdout.strip().splitlines()[-1]) != args.payload_bytes:
            raise RuntimeError(f"Payload creation failed on {node.id}")
        payload_paths[node.id] = payload_path
        print(f"[payload] {node.id:16} {args.payload_bytes} bytes")

    rtt_rows: list[dict[str, Any]] = []
    bandwidth_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    rtt_code = """
import subprocess, sys
url = sys.argv[1]
count = int(sys.argv[2])
timeout = sys.argv[3]
for _ in range(count):
    value = subprocess.check_output([
        'curl', '-fsS', '-o', '/dev/null', '--connect-timeout', timeout,
        '--max-time', timeout, '-w', '%{time_total}', url,
    ], text=True).strip()
    print(float(value) * 1000.0)
"""
    bandwidth_code = """
import json, subprocess, sys, time
payload, user, destination_ip, measurement_id, source_id, destination_id, count = sys.argv[1:]
count = int(count)
target = f'{user}@{destination_ip}'
values = []
for sample in range(count):
    remote = f'/tmp/{measurement_id}-{source_id}-{destination_id}-{sample}.bin'
    started = time.perf_counter()
    subprocess.run([
        'rsync', '-az', '--delete',
        '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new',
        payload, f'{target}:{remote}',
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    duration = max(1e-9, time.perf_counter() - started)
    values.append(duration)
    subprocess.run([
        'ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
        target, f'rm -f {remote}',
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
print(json.dumps(values))
"""

    try:
        print(f"== Measure all {len(pairs)} directed edges ==")
        for index, (source_id, destination_id) in enumerate(pairs, start=1):
            source = node_by_id[source_id]
            destination = node_by_id[destination_id]
            source_api = f"http://{source.internal_ip}:{cluster.api_port}"
            telemetry = request_json(f"{source_api}/telemetry/edges/{destination_id}")

            rtt_result = run_at_source(
                local_node_id=local.id,
                source_node_id=source_id,
                source_ip=str(source.internal_ip),
                ssh_user=args.ssh_user,
                command=python_command(
                    rtt_code,
                    f"http://{destination.internal_ip}:{cluster.api_port}/health",
                    str(args.rtt_samples),
                    str(args.timeout_seconds),
                ),
                timeout=args.rtt_samples * args.timeout_seconds + 30.0,
            )
            rtt_values = [
                float(line)
                for line in rtt_result.stdout.splitlines()
                if line.strip()
            ]
            if len(rtt_values) != args.rtt_samples:
                raise RuntimeError(
                    f"RTT sample count mismatch for {source_id}->{destination_id}: "
                    f"{len(rtt_values)}"
                )
            for sample_index, value in enumerate(rtt_values, start=1):
                rtt_rows.append(
                    {
                        "source_node_id": source_id,
                        "destination_node_id": destination_id,
                        "sample": sample_index,
                        "rtt_ms": value,
                    }
                )

            transfer_result = run_at_source(
                local_node_id=local.id,
                source_node_id=source_id,
                source_ip=str(source.internal_ip),
                ssh_user=args.ssh_user,
                command=python_command(
                    bandwidth_code,
                    payload_paths[source_id],
                    args.ssh_user,
                    str(destination.internal_ip),
                    measurement_id,
                    source_id,
                    destination_id,
                    str(args.bandwidth_samples),
                ),
                timeout=args.bandwidth_samples * max(60.0, args.timeout_seconds) + 60.0,
            )
            transfer_values = json.loads(transfer_result.stdout.strip().splitlines()[-1])
            if len(transfer_values) != args.bandwidth_samples:
                raise RuntimeError(
                    f"Bandwidth sample count mismatch for {source_id}->{destination_id}"
                )

            effective_bandwidth = float(telemetry["effective_bandwidth_mbps"])
            effective_latency = float(telemetry["effective_latency_ms"])
            predicted_seconds = predict_transfer_seconds(
                size_bytes=args.payload_bytes,
                bandwidth_mbps=effective_bandwidth,
                latency_ms=effective_latency,
                bandwidth_is_end_to_end=(
                    telemetry.get("bandwidth_source")
                    == "measured_transfer_ema"
                ),
            )
            measured_bandwidths: list[float] = []
            for sample_index, duration in enumerate(transfer_values, start=1):
                duration = float(duration)
                bandwidth_mbps = args.payload_bytes * 8.0 / duration / 1_000_000.0
                measured_bandwidths.append(bandwidth_mbps)
                bandwidth_rows.append(
                    {
                        "source_node_id": source_id,
                        "destination_node_id": destination_id,
                        "sample": sample_index,
                        "payload_bytes": args.payload_bytes,
                        "duration_seconds": duration,
                        "bandwidth_mbps": bandwidth_mbps,
                        "predicted_duration_seconds": predicted_seconds,
                        "prediction_error_percent": signed_percent_error(
                            predicted_seconds, duration
                        ),
                        "absolute_prediction_error_percent": absolute_percent_error(
                            predicted_seconds, duration
                        ),
                    }
                )

            rtt_summary = summarize_samples(rtt_values)
            bandwidth_summary = summarize_samples(measured_bandwidths)
            duration_summary = summarize_samples(transfer_values)

            post_telemetry = telemetry
            post_predicted_seconds = predicted_seconds
            if args.seed_telemetry:
                for rtt_ms in rtt_values:
                    post_telemetry = request_json(
                        f"{source_api}/telemetry/edges/{destination_id}/sample",
                        method="POST",
                        payload={"latency_ms": rtt_ms},
                    )
                for duration in transfer_values:
                    post_telemetry = request_json(
                        f"{source_api}/telemetry/edges/{destination_id}/sample",
                        method="POST",
                        payload={
                            "transfer_bytes": args.payload_bytes,
                            "transfer_duration_seconds": float(duration),
                        },
                    )
                post_predicted_seconds = predict_transfer_seconds(
                    size_bytes=args.payload_bytes,
                    bandwidth_mbps=float(
                        post_telemetry["effective_bandwidth_mbps"]
                    ),
                    latency_ms=float(post_telemetry["effective_latency_ms"]),
                    bandwidth_is_end_to_end=(
                        post_telemetry.get("bandwidth_source")
                        == "measured_transfer_ema"
                    ),
                )
            edge_rows.append(
                {
                    "source_node_id": source_id,
                    "destination_node_id": destination_id,
                    "configured_latency_ms": telemetry["configured_latency_ms"],
                    "telemetry_latency_ms": effective_latency,
                    "telemetry_latency_source": telemetry["latency_source"],
                    "telemetry_latency_freshness": telemetry["latency_freshness"],
                    "measured_rtt_median_ms": rtt_summary.median,
                    "measured_rtt_p95_ms": rtt_summary.p95,
                    "rtt_cv": rtt_summary.coefficient_of_variation,
                    "configured_bandwidth_mbps": telemetry["configured_bandwidth_mbps"],
                    "telemetry_bandwidth_mbps": effective_bandwidth,
                    "telemetry_bandwidth_source": telemetry["bandwidth_source"],
                    "telemetry_bandwidth_freshness": telemetry["bandwidth_freshness"],
                    "measured_bandwidth_median_mbps": bandwidth_summary.median,
                    "measured_bandwidth_p95_mbps": bandwidth_summary.p95,
                    "bandwidth_cv": bandwidth_summary.coefficient_of_variation,
                    "measured_transfer_median_seconds": duration_summary.median,
                    "predicted_transfer_seconds": predicted_seconds,
                    "median_transfer_prediction_error_percent": signed_percent_error(
                        predicted_seconds, duration_summary.median
                    ),
                    "median_transfer_absolute_error_percent": absolute_percent_error(
                        predicted_seconds, duration_summary.median
                    ),
                    "post_seed_bandwidth_mbps": post_telemetry.get(
                        "effective_bandwidth_mbps"
                    ),
                    "post_seed_bandwidth_source": post_telemetry.get(
                        "bandwidth_source"
                    ),
                    "post_seed_latency_ms": post_telemetry.get(
                        "effective_latency_ms"
                    ),
                    "post_seed_latency_source": post_telemetry.get(
                        "latency_source"
                    ),
                    "post_seed_predicted_transfer_seconds": post_predicted_seconds,
                    "post_seed_transfer_prediction_error_percent": (
                        signed_percent_error(
                            post_predicted_seconds, duration_summary.median
                        )
                    ),
                    "post_seed_transfer_absolute_error_percent": (
                        absolute_percent_error(
                            post_predicted_seconds, duration_summary.median
                        )
                    ),
                }
            )
            line = (
                f"[{index:02d}/{len(pairs):02d}] {source_id:16} -> {destination_id:16} "
                f"rtt={rtt_summary.median:.1f}ms "
                f"bw={bandwidth_summary.median:.1f}Mbps "
                f"pred_err="
                f"{edge_rows[-1]['median_transfer_prediction_error_percent']:.1f}%"
            )
            if args.seed_telemetry:
                line += (
                    " post_seed_err="
                    f"{edge_rows[-1]['post_seed_transfer_prediction_error_percent']:.1f}%"
                )
            print(line)
    finally:
        cleanup_code = "import pathlib,sys; pathlib.Path(sys.argv[1]).unlink(missing_ok=True)"
        for node in cluster.nodes:
            try:
                run_at_source(
                    local_node_id=local.id,
                    source_node_id=node.id,
                    source_ip=str(node.internal_ip),
                    ssh_user=args.ssh_user,
                    command=python_command(cleanup_code, payload_paths[node.id]),
                    timeout=20.0,
                )
            except Exception:
                pass

    metadata = {
        "format_version": 1,
        "measurement_type": "directed_network",
        "measurement_id": measurement_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "status_porcelain": git_value("status", "--porcelain"),
        },
        "cluster": {
            "path": args.cluster,
            "sha256": sha256_file(args.cluster),
            "node_ids": node_ids,
            "node_count": len(node_ids),
            "directed_edge_count": len(pairs),
            "local_node_id": local.id,
            "api_port": cluster.api_port,
        },
        "parameters": {
            "rtt_samples": args.rtt_samples,
            "bandwidth_samples": args.bandwidth_samples,
            "payload_bytes": args.payload_bytes,
            "timeout_seconds": args.timeout_seconds,
            "rsync_mode": "-az --delete over SSH",
            "expected_carbon_metric": args.expected_carbon_metric,
            "expected_state_token": args.expected_state_token,
            "seed_telemetry": args.seed_telemetry,
        },
        "initial_health": health,
    }
    write_json(bundle / "metadata.json", metadata)
    write_csv(
        bundle / "rtt_samples.csv",
        rtt_rows,
        ["source_node_id", "destination_node_id", "sample", "rtt_ms"],
    )
    write_csv(
        bundle / "bandwidth_samples.csv",
        bandwidth_rows,
        [
            "source_node_id",
            "destination_node_id",
            "sample",
            "payload_bytes",
            "duration_seconds",
            "bandwidth_mbps",
            "predicted_duration_seconds",
            "prediction_error_percent",
            "absolute_prediction_error_percent",
        ],
    )
    write_csv(
        bundle / "edges.csv",
        edge_rows,
        list(edge_rows[0].keys()),
    )
    write_checksums(bundle)

    transfer_errors = [
        float(row["median_transfer_absolute_error_percent"])
        for row in edge_rows
        if row["median_transfer_absolute_error_percent"] is not None
    ]
    error_summary = summarize_samples(transfer_errors)
    post_seed_errors = [
        float(row["post_seed_transfer_absolute_error_percent"])
        for row in edge_rows
        if row["post_seed_transfer_absolute_error_percent"] is not None
    ]
    post_seed_error_summary = summarize_samples(post_seed_errors)
    print("\nDIRECTED NETWORK MEASUREMENT PASSED")
    print(f"bundle: {bundle}")
    print(f"directed_edges: {len(edge_rows)}")
    print(f"median_abs_transfer_prediction_error_pct: {error_summary.median:.2f}")
    print(f"p95_abs_transfer_prediction_error_pct: {error_summary.p95:.2f}")
    if args.seed_telemetry:
        print(
            "post_seed_median_abs_transfer_prediction_error_pct: "
            f"{post_seed_error_summary.median:.2f}"
        )
        print(
            "post_seed_p95_abs_transfer_prediction_error_pct: "
            f"{post_seed_error_summary.p95:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
