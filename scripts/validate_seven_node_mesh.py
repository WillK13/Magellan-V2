#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from magellan.config.loader import load_cluster_config
from magellan.config.models import NodeConfig


@dataclass
class PathResult:
    source: str
    destination: str
    api_ok: bool
    ssh_ok: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.api_ok and self.ssh_ok and self.error is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate all directed FastAPI and SSH/rsync paths in a Magellan cluster. "
            "Run this from one cluster node after passwordless SSH is configured."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--connect-timeout", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def _source_shell(
    source: NodeConfig,
    *,
    local_node_id: str,
    ssh_user: str,
    timeout: int,
    command: str,
) -> list[str]:
    if source.id == local_node_id:
        return ["bash", "-lc", command]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        f"{ssh_user}@{source.internal_ip}",
        command,
    ]


def _path_check(
    *,
    source: NodeConfig,
    destination: NodeConfig,
    api_port: int,
    local_node_id: str,
    ssh_user: str,
    timeout: int,
) -> PathResult:
    destination_ip = str(destination.internal_ip)
    python_check = (
        "import json,sys; value=json.load(sys.stdin); "
        f"assert value['node_id'] == {destination.id!r}, value"
    )
    api_command = (
        f"curl -fsS --connect-timeout {timeout} "
        f"http://{destination_ip}:{api_port}/health "
        f"| python3 -c {shlex.quote(python_check)}"
    )
    ssh_command = (
        "ssh -o BatchMode=yes "
        f"-o ConnectTimeout={timeout} "
        f"{shlex.quote(ssh_user + '@' + destination_ip)} "
        f"{shlex.quote('command -v rsync >/dev/null && command -v python3 >/dev/null')}"
    )

    try:
        api = subprocess.run(
            _source_shell(
                source,
                local_node_id=local_node_id,
                ssh_user=ssh_user,
                timeout=timeout,
                command=api_command,
            ),
            capture_output=True,
            text=True,
            timeout=max(10, timeout * 3),
        )
        ssh = subprocess.run(
            _source_shell(
                source,
                local_node_id=local_node_id,
                ssh_user=ssh_user,
                timeout=timeout,
                command=ssh_command,
            ),
            capture_output=True,
            text=True,
            timeout=max(10, timeout * 3),
        )
        errors = []
        if api.returncode != 0:
            errors.append(
                "api=" + (api.stderr or api.stdout or f"exit {api.returncode}").strip()
            )
        if ssh.returncode != 0:
            errors.append(
                "ssh=" + (ssh.stderr or ssh.stdout or f"exit {ssh.returncode}").strip()
            )
        return PathResult(
            source=source.id,
            destination=destination.id,
            api_ok=api.returncode == 0,
            ssh_ok=ssh.returncode == 0,
            error="; ".join(errors) or None,
        )
    except Exception as exc:
        return PathResult(
            source=source.id,
            destination=destination.id,
            api_ok=False,
            ssh_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    cluster.get_node(args.local_node_id)

    paths = [
        (source, destination)
        for source in cluster.nodes
        for destination in cluster.nodes
        if source.id != destination.id
    ]

    results: list[PathResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _path_check,
                source=source,
                destination=destination,
                api_port=cluster.api_port,
                local_node_id=args.local_node_id,
                ssh_user=args.ssh_user,
                timeout=args.connect_timeout,
            ): (source.id, destination.id)
            for source, destination in paths
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if not args.as_json:
                marker = "OK" if result.ok else "FAIL"
                print(
                    f"[{marker}] {result.source:16s} -> {result.destination:16s} "
                    f"api={'yes' if result.api_ok else 'no ':3s} "
                    f"ssh={'yes' if result.ssh_ok else 'no ':3s}"
                )
                if result.error:
                    print(f"       {result.error}")

    results.sort(key=lambda item: (item.source, item.destination))
    valid = all(item.ok for item in results)
    if args.as_json:
        print(
            json.dumps(
                {
                    "valid": valid,
                    "directed_path_count": len(results),
                    "results": [asdict(item) | {"ok": item.ok} for item in results],
                },
                indent=2,
            )
        )
    else:
        print(f"validated directed paths: {len(results)}")
        print("SEVEN-NODE MESH PASSED" if valid else "SEVEN-NODE MESH FAILED")

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
