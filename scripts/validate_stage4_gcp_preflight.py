#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magellan.config.models import ClusterConfig  # noqa: E402


EXPECTED_MACHINE_TYPE = "e2-highmem-2"
EXPECTED_MEMORY_MB = 16384
EXPECTED_PRICES = {
    "boston": 0.0904,
    "california": 0.1086,
    "south-australia": 0.1090,
    "nepal": 0.1086,
    "ethiopia": 0.0994,
    "france": 0.1049,
    "virginia": 0.0995,
}


def main() -> None:
    path = Path("config/cluster.gcp.json")
    cluster = ClusterConfig.model_validate_json(path.read_text(encoding="utf-8"))
    nodes = {node.id: node for node in cluster.nodes}

    if set(nodes) != set(EXPECTED_PRICES):
        raise SystemExit(
            f"unexpected GCP node set: {sorted(nodes)}; "
            f"expected {sorted(EXPECTED_PRICES)}"
        )

    for node_id, expected_price in EXPECTED_PRICES.items():
        node = nodes[node_id]
        if node.machine_type != EXPECTED_MACHINE_TYPE:
            raise SystemExit(
                f"{node_id}: machine_type={node.machine_type!r}; "
                f"expected {EXPECTED_MACHINE_TYPE!r}"
            )
        if node.resources.cpu_cores != 2:
            raise SystemExit(
                f"{node_id}: resources.cpu_cores={node.resources.cpu_cores}; expected 2"
            )
        if node.resources.memory_mb != EXPECTED_MEMORY_MB:
            raise SystemExit(
                f"{node_id}: resources.memory_mb={node.resources.memory_mb}; "
                f"expected {EXPECTED_MEMORY_MB}"
            )
        if node.capabilities.memory_mb != EXPECTED_MEMORY_MB:
            raise SystemExit(
                f"{node_id}: capabilities.memory_mb={node.capabilities.memory_mb}; "
                f"expected {EXPECTED_MEMORY_MB}"
            )
        if abs(node.compute_price_usd_per_hour - expected_price) > 1e-9:
            raise SystemExit(
                f"{node_id}: compute_price_usd_per_hour="
                f"{node.compute_price_usd_per_hour}; expected {expected_price}"
            )

    for node_id in ("boston", "virginia"):
        caps = nodes[node_id].capabilities
        if "mpirun" not in caps.commands:
            raise SystemExit(f"{node_id}: mpirun is not configured")
        if "mpi" not in caps.features:
            raise SystemExit(f"{node_id}: mpi feature is not configured")
        if caps.runtimes.get("openmpi") != "4.1.4":
            raise SystemExit(
                f"{node_id}: configured OpenMPI={caps.runtimes.get('openmpi')!r}; "
                "expected '4.1.4'"
            )

    print(json.dumps({
        "machine_type": EXPECTED_MACHINE_TYPE,
        "memory_mb": EXPECTED_MEMORY_MB,
        "prices_usd_per_hour": EXPECTED_PRICES,
    }, indent=2, sort_keys=True))
    print("STAGE_4_GCP_PREFLIGHT_CONFIG_PASS")


if __name__ == "__main__":
    main()
