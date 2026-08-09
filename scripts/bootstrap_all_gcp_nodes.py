#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull/install/test the deployment branch on every configured GCP node."
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--branch", default="seven-node-deployment")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--project", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--validate-datasets",
        action="store_true",
        help="Require carbon datasets to already exist on each remote node.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)

    for node in cluster.nodes:
        remote = " ".join(
            [
                f"cd {shlex.quote(args.remote_repo)}",
                "&&",
                f"MAGELLAN_BRANCH={shlex.quote(args.branch)}",
                f"MAGELLAN_BOOTSTRAP_RUN_TESTS={'0' if args.skip_tests else '1'}",
                f"MAGELLAN_BOOTSTRAP_VALIDATE_DATASETS={'1' if args.validate_datasets else '0'}",
                "scripts/bootstrap_gcp_node.sh",
                shlex.quote(node.id),
            ]
        )
        command = [
            "gcloud",
            "compute",
            "ssh",
            node.vm_name,
            "--zone",
            node.zone,
            "--command",
            remote,
        ]
        if args.project:
            command.extend(["--project", args.project])
        print(f"== bootstrap {node.id} ({node.vm_name}, {node.zone}) ==", flush=True)
        subprocess.run(command, check=True)

    print("SEVEN-NODE BOOTSTRAP PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
