#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the complete carbon dataset directory to every GCP node."
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--project", default=None)
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    datasets = Path(args.datasets)
    csv_files = sorted(datasets.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {datasets}")

    validation = subprocess.run(
        [
            str(Path(__file__).with_name("validate_seven_node_deployment.py")),
            "--cluster",
            args.cluster,
            "--datasets",
            str(datasets),
        ],
        check=False,
    )
    if validation.returncode != 0:
        raise SystemExit("Dataset/config validation failed; refusing to sync")

    for node in cluster.nodes:
        mkdir_command = [
            "gcloud",
            "compute",
            "ssh",
            node.vm_name,
            "--zone",
            node.zone,
            "--command",
            f"mkdir -p {args.remote_repo}/datasets",
        ]
        if args.project:
            mkdir_command.extend(["--project", args.project])
        print(f"== {node.id}: create remote dataset directory ==", flush=True)
        subprocess.run(mkdir_command, check=True)

        scp_command = [
            "gcloud",
            "compute",
            "scp",
            *[str(path) for path in csv_files],
            f"{node.vm_name}:{args.remote_repo}/datasets/",
            "--zone",
            node.zone,
        ]
        if args.project:
            scp_command.extend(["--project", args.project])
        print(f"== {node.id}: copy {len(csv_files)} carbon datasets ==", flush=True)
        subprocess.run(scp_command, check=True)

    print("SEVEN-NODE DATASET SYNC PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
