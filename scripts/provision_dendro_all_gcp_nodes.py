#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import subprocess

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror the known-good Dendro build and parameter template from "
            "one experiment node to every configured GCP worker, then validate "
            "binary identity, dynamic-library resolution, and local MPI spawn."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--source-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--dendro-root", default="~/dgr-build")
    parser.add_argument(
        "--solver-relative-path", default="BSSN_GR/bssnSolver"
    )
    parser.add_argument(
        "--parameter-template", default="~/q1-magellan-magellan.toml"
    )
    return parser.parse_args()


def expand_home(value: str) -> Path:
    return Path(value).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=True,
        text=True,
        capture_output=capture,
    )


def remote_probe(
    *,
    user: str,
    ip: str,
    dendro_root: str,
    solver_relative_path: str,
    parameter_template: str,
) -> tuple[str, str, str]:
    solver = f"{dendro_root.rstrip('/')}/{solver_relative_path}"
    command = f"""
set -euo pipefail
solver={shlex.quote(solver)}
params={shlex.quote(parameter_template)}

test -x "$solver"
test -f "$params"

banner=$(mpirun --version | head -n 1)
case "$banner" in
  *4.1.4*) ;;
  *) echo "unexpected OpenMPI: $banner" >&2; exit 1 ;;
esac

missing=$(ldd "$solver" | grep 'not found' || true)
if [ -n "$missing" ]; then
  echo "$missing" >&2
  exit 1
fi

# Exercise two local MPI ranks without starting Dendro itself.
rank_count=$(mpirun -np 2 hostname | wc -l)
[ "$rank_count" -eq 2 ]

printf 'solver_sha256='
sha256sum "$solver" | awk '{{print $1}}'
printf 'parameter_sha256='
sha256sum "$params" | awk '{{print $1}}'
printf 'openmpi='
printf '%s\\n' "$banner"
"""
    result = run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8",
            f"{user}@{ip}",
            command,
        ],
        capture=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return (
        values.get("solver_sha256", ""),
        values.get("parameter_sha256", ""),
        values.get("openmpi", ""),
    )


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    source = cluster.get_node(args.source_node_id)

    dendro_root = expand_home(args.dendro_root)
    solver = dendro_root / args.solver_relative_path
    params = expand_home(args.parameter_template)
    if not solver.is_file():
        raise SystemExit(f"source solver does not exist: {solver}")
    if not solver.stat().st_mode & 0o111:
        raise SystemExit(f"source solver is not executable: {solver}")
    if not params.is_file():
        raise SystemExit(f"source parameter template does not exist: {params}")

    source_solver_sha = sha256(solver)
    source_parameter_sha = sha256(params)
    print(f"source_node={source.id}")
    print(f"source_solver={solver}")
    print(f"source_solver_sha256={source_solver_sha}")
    print(f"source_parameter_template={params}")
    print(f"source_parameter_sha256={source_parameter_sha}")

    # This script is intentionally run on the source node.  The Stage-4 WAN
    # harness already depends on passwordless internal SSH from Boston, making
    # the provisioning path deterministic and independent of public addresses.
    for node in cluster.nodes:
        print(f"\n== dendro provision {node.id} ==", flush=True)
        if node.id != source.id:
            remote_root = str(dendro_root)
            run([
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                f"{args.ssh_user}@{node.internal_ip}",
                f"mkdir -p {shlex.quote(remote_root)}",
            ])
            run([
                "rsync", "-az",
                f"{dendro_root}/",
                f"{args.ssh_user}@{node.internal_ip}:{remote_root}/",
            ])
            run([
                "rsync", "-az",
                str(params),
                f"{args.ssh_user}@{node.internal_ip}:{params}",
            ])

        solver_sha, parameter_sha, banner = remote_probe(
            user=args.ssh_user,
            ip=str(node.internal_ip),
            dendro_root=str(dendro_root),
            solver_relative_path=args.solver_relative_path,
            parameter_template=str(params),
        )
        if solver_sha != source_solver_sha:
            raise SystemExit(
                f"{node.id}: solver hash mismatch {solver_sha} != {source_solver_sha}"
            )
        if parameter_sha != source_parameter_sha:
            raise SystemExit(
                f"{node.id}: parameter hash mismatch "
                f"{parameter_sha} != {source_parameter_sha}"
            )
        print(
            f"[PASS] {node.id:16} openmpi={banner} "
            f"solver_sha256={solver_sha[:12]}"
        )

    print("\nSEVEN_NODE_DENDRO_PROVISION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
