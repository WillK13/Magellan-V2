#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5d import STAGE5D_RING, expected_hops, progress_is_monotonic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 5D migration-ring bundle.")
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = [
        "summary.json",
        "metadata.json",
        "hops.csv",
        "ownership_per_hop.csv",
        "final_ownership.csv",
        "migration_events.jsonl",
        "migration_journals.jsonl",
        "initial_state.json",
        "final_state.json",
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
    hops = read_csv(root / "hops.csv")
    final_ownership = read_csv(root / "final_ownership.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if list(summary.get("ring") or []) != list(STAGE5D_RING):
        errors.append("ring mismatch")
    if len(hops) != 7:
        errors.append(f"hop count {len(hops)} != 7")

    expected = expected_hops()
    migration_ids: set[str] = set()
    for index, row in enumerate(hops, start=1):
        if index > len(expected):
            break
        source, destination = expected[index - 1]
        if int(row["hop_index"]) != index:
            errors.append(f"hop index mismatch at row {index}")
        if row["source_node_id"] != source or row["destination_node_id"] != destination:
            errors.append(f"hop {index} route mismatch")
        if row["owner_before"] != source:
            errors.append(f"hop {index} owner_before mismatch")
        if int(row["generation_before"]) != index - 1:
            errors.append(f"hop {index} generation_before mismatch")
        if row["migrated"].lower() != "true":
            errors.append(f"hop {index} was not migrated")
        if row["owner_after"] != destination:
            errors.append(f"hop {index} owner_after mismatch")
        if int(row["generation_after"]) != index:
            errors.append(f"hop {index} generation_after mismatch")
        if row["destination_status_after"] != "running":
            errors.append(f"hop {index} destination is not running")
        if not row["destination_pid_after"]:
            errors.append(f"hop {index} destination PID missing")
        if row["source_record_role"] != "source" or row["source_record_status"] != "activated":
            errors.append(f"hop {index} source journal not activated")
        if row["destination_record_role"] != "destination" or row["destination_record_status"] != "activated":
            errors.append(f"hop {index} destination journal not activated")
        if row["bid_status"] not in {"accepted", "consumed"}:
            errors.append(f"hop {index} bid status is {row['bid_status']}")
        if row["ownership_converged"].lower() != "true":
            errors.append(f"hop {index} ownership did not converge")
        if float(row["total_downtime_seconds"]) <= 0:
            errors.append(f"hop {index} downtime is not positive")
        migration_id = row["migration_id"]
        if not migration_id or migration_id in migration_ids:
            errors.append(f"hop {index} migration id missing/duplicate")
        migration_ids.add(migration_id)

    if not progress_is_monotonic(hops):
        errors.append("progress regressed across ring")
    if summary.get("final_owner_node_id") != "boston":
        errors.append("final owner is not Boston")
    if int(summary.get("final_generation", -1)) != 7:
        errors.append("final generation is not 7")
    if summary.get("ownership_converged_final") is not True:
        errors.append("final ownership did not converge")
    if len({row["source_node_id"] for row in hops}) != 7:
        errors.append("not every node acted as source")
    if len({row["destination_node_id"] for row in hops}) != 7:
        errors.append("not every node acted as destination")

    final_values = {
        (row["owner_node_id"], row["generation"])
        for row in final_ownership
    }
    if final_values != {("boston", "7")}:
        errors.append(f"final ownership rows disagree: {sorted(final_values)}")
    if len(final_ownership) != 7:
        errors.append(f"final ownership coverage {len(final_ownership)} != 7")

    s5a = Path(str(summary.get("source_stage5a_bundle") or ""))
    s5c = Path(str(summary.get("source_stage5c_bundle") or ""))
    for path, label in ((s5a, "Stage 5A"), (s5c, "Stage 5C")):
        if not path.is_dir():
            errors.append(f"{label} source bundle missing: {path}")
        elif validate_checksums(path):
            errors.append(f"{label} source checksum validation failed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5D_MIGRATION_RING_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print("hops: 7/7")
    print("source_nodes: 7/7")
    print("destination_nodes: 7/7")
    print("source_journals_activated: 7/7")
    print("destination_journals_activated: 7/7")
    print("progress_monotonic: True")
    print("final_owner: boston")
    print("final_generation: 7")
    print("ownership_converged: True")
    print(f"total_downtime_seconds: {float(summary.get('total_downtime_seconds') or 0):.3f}")
    print(f"max_hop_downtime_seconds: {float(summary.get('max_hop_downtime_seconds') or 0):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
