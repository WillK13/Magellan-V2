#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install/restart the Magellan systemd service on every GCP node."
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--project", default=None)
    return parser.parse_args()

def remote_cd(path: str) -> str:
    """Render a remote cd command while preserving home expansion."""
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        relative = path[2:]
        return f'cd "$HOME"/{shlex.quote(relative)}'
    return f"cd {shlex.quote(path)}"


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)

    for node in cluster.nodes:
        remote = (
            f"{remote_cd(args.remote_repo)} && "
            f"scripts/install_magellan_systemd.sh {shlex.quote(node.id)}"
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
        print(f"== systemd {node.id} ==", flush=True)
        subprocess.run(command, check=True)

    print("SEVEN-NODE SYSTEMD INSTALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
