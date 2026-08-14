#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a topology-driven directed-network measurement bundle"
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        errors.append("Missing metadata.json")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    edges_path = root / "edges.csv"
    rtt_path = root / "rtt_samples.csv"
    bandwidth_path = root / "bandwidth_samples.csv"
    for path in (edges_path, rtt_path, bandwidth_path):
        if not path.is_file():
            errors.append(f"Missing {path.name}")

    edges = read_csv(edges_path) if edges_path.is_file() else []
    rtt = read_csv(rtt_path) if rtt_path.is_file() else []
    bandwidth = read_csv(bandwidth_path) if bandwidth_path.is_file() else []
    node_ids = list(metadata.get("cluster", {}).get("node_ids", []))
    expected_edges = len(node_ids) * max(0, len(node_ids) - 1)
    if len(node_ids) < 2:
        errors.append(f"Expected at least 2 nodes, found {len(node_ids)}")
    metadata_edge_count = metadata.get("cluster", {}).get("directed_edge_count")
    if metadata_edge_count is not None and int(metadata_edge_count) != expected_edges:
        errors.append(
            f"Metadata directed_edge_count={metadata_edge_count}; expected {expected_edges}"
        )
    if len(edges) != expected_edges:
        errors.append(f"Expected {expected_edges} edge summaries, found {len(edges)}")

    rtt_samples = int(metadata.get("parameters", {}).get("rtt_samples", 0))
    bandwidth_samples = int(metadata.get("parameters", {}).get("bandwidth_samples", 0))
    if len(rtt) != expected_edges * rtt_samples:
        errors.append(
            f"Expected {expected_edges * rtt_samples} RTT rows, found {len(rtt)}"
        )
    if len(bandwidth) != expected_edges * bandwidth_samples:
        errors.append(
            f"Expected {expected_edges * bandwidth_samples} bandwidth rows, found {len(bandwidth)}"
        )

    pairs = {(row["source_node_id"], row["destination_node_id"]) for row in edges}
    if any(source == destination for source, destination in pairs):
        errors.append("Self-edge found in measurement")
    if len(pairs) != expected_edges:
        errors.append("Directed edge summaries are not unique/complete")

    for row in edges:
        for field in (
            "measured_rtt_median_ms",
            "measured_bandwidth_median_mbps",
            "measured_transfer_median_seconds",
            "predicted_transfer_seconds",
        ):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                errors.append(
                    f"Invalid {field} for {row.get('source_node_id')}->"
                    f"{row.get('destination_node_id')}"
                )
                continue
            if value <= 0:
                errors.append(
                    f"Non-positive {field} for {row['source_node_id']}->"
                    f"{row['destination_node_id']}"
                )

    if errors:
        print("NETWORK MEASUREMENT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("NETWORK MEASUREMENT BUNDLE PASSED")
    print(f"measurement_id: {metadata.get('measurement_id')}")
    print(f"nodes: {len(node_ids)}")
    print(f"directed_edges: {len(edges)}")
    print(f"rtt_samples: {len(rtt)}")
    print(f"bandwidth_samples: {len(bandwidth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
