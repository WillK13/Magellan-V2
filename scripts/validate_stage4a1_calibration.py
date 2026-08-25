#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 4A.1 hardware/network calibration bundle"
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
    hardware_path = root / "hardware.json"
    hardware_csv_path = root / "hardware.csv"
    summary_path = root / "summary.json"
    for path in (metadata_path, hardware_path, hardware_csv_path, summary_path):
        if not path.is_file():
            errors.append(f"Missing {path.name}")

    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    hardware = (
        json.loads(hardware_path.read_text(encoding="utf-8"))
        if hardware_path.is_file()
        else {}
    )
    hardware_rows = read_csv(hardware_csv_path) if hardware_csv_path.is_file() else []
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )

    expected_nodes = int(metadata.get("requirements", {}).get("expected_node_count", 0))
    expected_machine = metadata.get("requirements", {}).get("expected_machine_type")
    if metadata.get("hardware_preflight_passed") is not True:
        errors.append("Hardware preflight did not pass")
    if len(hardware) != expected_nodes:
        errors.append(f"Expected {expected_nodes} hardware records, found {len(hardware)}")
    if len(hardware_rows) != expected_nodes:
        errors.append(
            f"Expected {expected_nodes} hardware CSV rows, found {len(hardware_rows)}"
        )

    commits: set[str] = set()
    for node_id, record in hardware.items():
        if record.get("preflight_passed") is not True:
            errors.append(f"Node preflight failed: {node_id}")
        observed = record.get("observed_host") or {}
        commits.add(str(observed.get("git_commit")))
        if expected_machine and observed.get("machine_type") != expected_machine:
            errors.append(
                f"{node_id}: machine_type={observed.get('machine_type')!r}; "
                f"expected {expected_machine!r}"
            )
    commits.discard("None")
    if len(commits) != 1:
        errors.append(f"Nodes do not share one commit: {sorted(commits)}")

    network_relative = summary.get("network_bundle")
    network_root = root / network_relative if network_relative else None
    if network_root is None or not network_root.is_dir():
        errors.append("Missing nested network bundle")
    else:
        result = subprocess.run(
            ["python", "scripts/validate_network_measurement.py", str(network_root)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            errors.append(
                "Nested network validation failed: "
                + (result.stdout + result.stderr).strip()
            )

    network_summary = summary.get("network") or {}
    expected_edges = expected_nodes * max(0, expected_nodes - 1)
    if int(network_summary.get("directed_edge_count", 0)) != expected_edges:
        errors.append(
            f"Expected {expected_edges} directed edges, found "
            f"{network_summary.get('directed_edge_count')}"
        )

    if errors:
        print("STAGE 4A.1 CALIBRATION BUNDLE FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("STAGE_4A1_CALIBRATION_BUNDLE_PASS")
    print(f"calibration_id: {metadata.get('calibration_id')}")
    print(f"nodes: {expected_nodes}")
    print(f"directed_edges: {expected_edges}")
    print(f"commit: {next(iter(commits)) if commits else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
