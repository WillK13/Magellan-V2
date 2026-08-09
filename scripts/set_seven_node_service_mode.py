#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Switch every GCP Magellan service between prod and smoke mode."
    )
    parser.add_argument("mode", choices=["prod", "smoke"])
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--project", default=None)
    parser.add_argument("--service", default="magellan")
    parser.add_argument(
        "--preserve-smoke-state",
        action="store_true",
        help="Do not clear runtime-state-gcp-smoke before entering smoke mode.",
    )
    return parser.parse_args()


def remote_cd(path: str) -> str:
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        return f'cd "$HOME"/{shlex.quote(path[2:])}'
    return f"cd {shlex.quote(path)}"


def smoke_command(remote_repo: str, service: str, reset_state: bool) -> str:
    reset = 'rm -rf "$REPO_ROOT/runtime-state-gcp-smoke"' if reset_state else ":"
    return f"""
set -euo pipefail
{remote_cd(remote_repo)}
REPO_ROOT="$(pwd)"
SERVICE={shlex.quote(service)}
sudo systemctl stop "$SERVICE"
{reset}
sudo mkdir -p "/etc/systemd/system/${{SERVICE}}.service.d"
cat <<EOF_MODE | sudo tee "/etc/systemd/system/${{SERVICE}}.service.d/20-magellan-mode.conf" >/dev/null
[Service]
Environment=MAGELLAN_CONFIG=config/cluster.gcp.smoke.json
Environment=MAGELLAN_POLICY=config/policy.gcp.smoke.json
Environment=MAGELLAN_STATE_ROOT=${{REPO_ROOT}}/runtime-state-gcp-smoke
Environment=MAGELLAN_REMOTE_STATE_ROOT=${{REPO_ROOT}}/runtime-state-gcp-smoke
EOF_MODE
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE"
sudo systemctl is-active --quiet "$SERVICE"
curl -fsS --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:8040/health >/dev/null
echo SMOKE_MODE_ACTIVE
""".strip()


def prod_command(remote_repo: str, service: str) -> str:
    return f"""
set -euo pipefail
{remote_cd(remote_repo)}
SERVICE={shlex.quote(service)}
sudo rm -f "/etc/systemd/system/${{SERVICE}}.service.d/20-magellan-mode.conf"
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE"
sudo systemctl is-active --quiet "$SERVICE"
curl -fsS --retry 20 --retry-delay 1 --retry-connrefused http://127.0.0.1:8040/health >/dev/null
echo PROD_MODE_ACTIVE
""".strip()


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)

    for node in cluster.nodes:
        print(f"== {args.mode} mode {node.id} ==", flush=True)
        if args.mode == "smoke":
            remote = smoke_command(
                args.remote_repo,
                args.service,
                reset_state=not args.preserve_smoke_state,
            )
        else:
            remote = prod_command(args.remote_repo, args.service)

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
        subprocess.run(command, check=True)

    print(f"SEVEN-NODE {args.mode.upper()} MODE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
