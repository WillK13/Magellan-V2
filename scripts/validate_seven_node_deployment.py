#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.deployment.validation import validate_deployment


EXPECTED_NODE_IDS = {
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Magellan seven-node GCP config and carbon datasets."
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    policy = load_policy_config(args.policy)
    report = validate_deployment(
        cluster=cluster,
        policy=policy,
        datasets_directory=Path(args.datasets),
        expected_node_ids=EXPECTED_NODE_IDS,
    )

    payload = {
        "valid": report.valid,
        "node_count": len(report.node_ids),
        "node_ids": report.node_ids,
        "common_start_utc": report.common_start_utc,
        "common_end_utc": report.common_end_utc,
        "datasets": [asdict(item) for item in report.dataset_summaries],
        "warnings": report.warnings,
        "errors": report.errors,
    }

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("== Magellan seven-node deployment validation ==")
        print(f"nodes: {len(report.node_ids)} ({', '.join(report.node_ids)})")
        for item in report.dataset_summaries:
            print(
                f"[dataset] {item.node_id:16s} {item.dataset_file:26s} "
                f"rows={item.row_count} "
                f"range={item.start_utc}..{item.end_utc} "
                f"sha256={item.sha256[:12]}"
            )
        if report.common_start_utc is not None:
            print(
                "common carbon range: "
                f"{report.common_start_utc} .. {report.common_end_utc}"
            )
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
        print(
            "SEVEN-NODE DEPLOYMENT INPUTS PASSED"
            if report.valid
            else "SEVEN-NODE DEPLOYMENT INPUTS FAILED"
        )

    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
