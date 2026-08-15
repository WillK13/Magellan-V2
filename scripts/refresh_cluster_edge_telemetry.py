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
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help=(
            "Per-edge HTTP timeout. Each request refreshes exactly one directed "
            "edge, so this no longer needs to cover all peers on a source node."
        ),
    )
    parser.add_argument(
        "--source-workers",
        type=int,
        default=3,
        help=(
            "Maximum source daemons calibrated concurrently. Each source probes "
            "its outgoing edges serially; bounding cross-cluster concurrency avoids "
            "the preflight measuring probe-induced contention."
        ),
    )
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


def refresh_source_edges(
    source: Any,
    peers: list[Any],
    api_port: int,
    timeout: float,
) -> dict[str, Any]:
    """Refresh one source's directed edges with one bounded request per edge.

    The daemon already serializes outgoing rsync probes. Issuing an endpoint per
    edge prevents one HTTP request from needing to remain open for the entire
    source-node sweep, which can exceed a reasonable client timeout on WANs.
    """
    edges: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for destination in peers:
        url = (
            f"http://{source.internal_ip}:{api_port}/telemetry/edges/"
            f"{destination.id}/refresh"
        )
        try:
            edges[destination.id] = post_json(url, timeout)
        except Exception as exc:
            failures[destination.id] = f"{type(exc).__name__}: {exc}"
    return {
        "node_id": source.id,
        "peer_count": len(peers),
        "edges": edges,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    node_count = len(cluster.nodes)
    expected_edges = node_count * max(0, node_count - 1)
    if node_count < 2:
        raise RuntimeError("Cluster edge refresh requires at least two nodes")

    print("== Live cluster edge telemetry refresh ==")
    print(f"nodes={node_count} directed_edges={expected_edges}")

    if args.source_workers <= 0:
        raise ValueError("--source-workers must be positive")

    # Refresh a bounded number of source daemons concurrently. Within each source
    # we issue one request per directed edge. This keeps the client timeout scoped
    # to a single live calibration and avoids seven sources saturating each other's
    # destinations during the experiment preflight.
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=min(node_count, args.source_workers)
    ) as executor:
        futures = {}
        for node in cluster.nodes:
            peers = [peer for peer in cluster.nodes if peer.id != node.id]
            future = executor.submit(
                refresh_source_edges,
                node,
                peers,
                cluster.api_port,
                args.timeout_seconds,
            )
            futures[future] = node

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
        failures = dict(result.get("failures") or {})
        if failures:
            detail = ", ".join(
                f"{node.id}->{destination_id}: {message}"
                for destination_id, message in sorted(failures.items())
            )
            raise RuntimeError(f"Live edge refresh failed: {detail}")
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
                view.get("bandwidth_source") == "measured_migration_transport_ema"
                and view.get("transfer_model_source")
                == "measured_migration_transport_affine_ema"
                and view.get("latency_source") == "measured_http_rtt"
                and view.get("bandwidth_freshness") == "fresh"
                and view.get("transfer_model_freshness") == "fresh"
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
