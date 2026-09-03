#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5a import (
    EXPECTED_STAGE5A_NODE_IDS,
    expected_directed_path_count,
    stage5a_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 5A deployment bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = [
        "summary.json",
        "metadata.json",
        "nodes.csv",
        "dataset_hashes.csv",
        "directed_mesh.csv",
        "node_probes.jsonl",
        "checksums.sha256",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    hardened = int(metadata.get("format_version", 1)) >= 2
    nodes_raw = read_csv(root / "nodes.csv")
    datasets = read_csv(root / "dataset_hashes.csv")
    mesh_raw = read_csv(root / "directed_mesh.csv")
    target_sha = str(summary.get("target_git_sha") or "")

    nodes = []
    for row in nodes_raw:
        parsed = row | {
            "tracked_worktree_clean": truthy(row["tracked_worktree_clean"]),
            "service_active": truthy(row["service_active"]),
            "health_ok": truthy(row["health_ok"]),
            "capabilities_ready": truthy(row["capabilities_ready"]),
        }
        if hardened:
            parsed |= {
                "effective_environment_ok": truthy(row["effective_environment_ok"]),
                "systemd_dropin_count": int(row["systemd_dropin_count"]),
                "state_root_exists": truthy(row["state_root_exists"]),
                "state_root_writable": truthy(row["state_root_writable"]),
                "remote_state_root_exists": truthy(row["remote_state_root_exists"]),
                "remote_state_root_writable": truthy(row["remote_state_root_writable"]),
            }
        nodes.append(parsed)
    mesh = [
        row
        | {
            "api_ok": truthy(row["api_ok"]),
            "ssh_ok": truthy(row["ssh_ok"]),
            "ok": truthy(row["ok"]),
        }
        for row in mesh_raw
    ]

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if not target_sha or len(target_sha) != 40:
        errors.append("target_git_sha is missing or malformed")
    if {row["node_id"] for row in nodes} != EXPECTED_STAGE5A_NODE_IDS:
        errors.append("node identity set mismatch")
    if len(mesh) != expected_directed_path_count(len(EXPECTED_STAGE5A_NODE_IDS)):
        errors.append("directed mesh path count mismatch")
    if not stage5a_passes(
        node_rows=nodes,
        dataset_rows=datasets,
        mesh_rows=mesh,
        expected_git_sha=target_sha,
        require_effective_environment=hardened,
    ):
        errors.append("Stage 5A deployment invariants do not pass")

    source = Path(str(summary.get("source_stage4e3_bundle") or ""))
    if not source.is_dir():
        errors.append(f"source Stage 4E.3 bundle not found: {source}")
    elif validate_checksums(source):
        errors.append("source Stage 4E.3 checksum validation failed")

    dataset_hashes = defaultdict(set)
    for row in datasets:
        dataset_hashes[row["dataset_file"]].add(row["sha256"])
    divergent = [name for name, hashes in dataset_hashes.items() if len(hashes) != 1]
    if divergent:
        errors.append(f"dataset hashes diverge: {divergent}")

    if hardened:
        if sum(row["effective_environment_ok"] for row in nodes) != 7:
            errors.append("effective systemd environment is not exact on all nodes")
        if sum(row["systemd_dropin_count"] == 0 for row in nodes) != 7:
            errors.append("systemd drop-ins remain on one or more nodes")
        if sum(row.get("health_carbon_metric") == "lifecycle" for row in nodes) != 7:
            errors.append("lifecycle carbon metric is not active on all nodes")
        if sum(
            row["state_root_writable"] and row["remote_state_root_writable"]
            for row in nodes
        ) != 7:
            errors.append("state roots are not writable on all nodes")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5A_SEVEN_NODE_DEPLOYMENT_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {target_sha}")
    print(f"nodes: {len(nodes)}/7")
    print(f"running_exact_sha: {sum(row['daemon_git_sha'] == target_sha for row in nodes)}/7")
    print(f"capabilities_ready: {sum(row['capabilities_ready'] for row in nodes)}/7")
    if hardened:
        print(f"effective_environment: {sum(row['effective_environment_ok'] for row in nodes)}/7")
        print(f"zero_systemd_dropins: {sum(row['systemd_dropin_count'] == 0 for row in nodes)}/7")
        print(f"lifecycle_carbon_metric: {sum(row.get('health_carbon_metric') == 'lifecycle' for row in nodes)}/7")
        print(f"writable_state_roots: {sum(row['state_root_writable'] and row['remote_state_root_writable'] for row in nodes)}/7")
    print(f"directed_api_paths: {sum(row['api_ok'] for row in mesh)}/42")
    print(f"directed_ssh_paths: {sum(row['ssh_ok'] for row in mesh)}/42")
    print(f"dataset_manifest_rows: {len(datasets)}/49")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
