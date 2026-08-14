#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.request import Request, urlopen

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Force live topology-derived edge telemetry refresh on every node in "
            "a Magellan cluster before an experiment workload is launched."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Do not fail if a live probe leaves an edge on configured fallback.",
    )
    return parser.parse_args()


def post_json(url: str, timeout: float) -> Any:
    request = Request(url, data=b"", method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    node_count = len(cluster.nodes)
    expected_edges = node_count * max(0, node_count - 1)
    if node_count < 2:
        raise RuntimeError("Cluster edge refresh requires at least two nodes")

    print("== Live cluster edge telemetry refresh ==")
    print(f"nodes={node_count} directed_edges={expected_edges}")

    # Refresh source daemons in parallel. Each daemon serializes its own outgoing
    # bandwidth probes, avoiding source-NIC self-contention while keeping the
    # cluster-wide preflight short enough that early samples remain fresh.
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(node_count, 8)) as executor:
        futures = {
            executor.submit(
                post_json,
                f"http://{node.internal_ip}:{cluster.api_port}/telemetry/refresh",
                args.timeout_seconds,
            ): node
            for node in cluster.nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            results[node.id] = future.result()

    observed_edges = 0
    fallback_edges: list[str] = []
    for index, node in enumerate(cluster.nodes, start=1):
        result = results[node.id]
        if result.get("node_id") != node.id:
            raise RuntimeError(
                f"Node identity mismatch for {node.id}: {result.get('node_id')}"
            )
        edges = dict(result.get("edges") or {})
        if int(result.get("peer_count", -1)) != node_count - 1:
            raise RuntimeError(
                f"{node.id} reported peer_count={result.get('peer_count')}; "
                f"expected {node_count - 1}"
            )
        if len(edges) != node_count - 1:
            raise RuntimeError(
                f"{node.id} returned {len(edges)} edge views; expected {node_count - 1}"
            )

        fresh = 0
        for destination_id, view in edges.items():
            observed_edges += 1
            measured = (
                view.get("bandwidth_source") == "measured_transfer_ema"
                and view.get("latency_source") == "measured_http_rtt"
                and view.get("bandwidth_freshness") == "fresh"
                and view.get("latency_freshness") == "fresh"
            )
            if measured:
                fresh += 1
            else:
                fallback_edges.append(f"{node.id}->{destination_id}")

        print(
            f"[{index:02d}/{node_count:02d}] {node.id:16} "
            f"fresh_measured={fresh}/{node_count - 1}"
        )

    if observed_edges != expected_edges:
        raise RuntimeError(
            f"Observed {observed_edges} directed edges; expected {expected_edges}"
        )
    if fallback_edges and not args.allow_fallback:
        raise RuntimeError(
            "Live telemetry refresh left fallback edges: "
            + ", ".join(fallback_edges)
        )

    print("\nCLUSTER EDGE TELEMETRY REFRESH PASSED")
    print(f"nodes: {node_count}")
    print(f"directed_edges: {observed_edges}")
    print(f"fallback_edges: {len(fallback_edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
