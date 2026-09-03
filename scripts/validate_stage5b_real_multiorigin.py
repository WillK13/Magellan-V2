#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5b import STAGE5B_SOURCE_IDS, stage5b_passes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage 5B bundle.")
    parser.add_argument("bundle")
    args = parser.parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    required = [
        "summary.json", "metadata.json", "sources.csv", "decisions.csv", "bids.csv",
        "migrations.csv", "ownership.csv", "final_tasks.csv", "events.jsonl",
        "node_evidence.jsonl", "checksums.sha256",
    ]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"Missing {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    sources = read_csv(root / "sources.csv")
    decisions = read_csv(root / "decisions.csv")
    bids = read_csv(root / "bids.csv")
    migrations = read_csv(root / "migrations.csv")
    ownership = read_csv(root / "ownership.csv")

    source_stage5a = Path(str(summary.get("source_stage5a_bundle") or ""))
    if not source_stage5a.is_dir() or validate_checksums(source_stage5a):
        errors.append("source Stage 5A bundle missing or checksum-invalid")
    else:
        s5a = json.loads((source_stage5a / "summary.json").read_text(encoding="utf-8"))
        if s5a.get("passed") is not True:
            errors.append("source Stage 5A did not pass")
        if str(s5a.get("target_git_sha")) != str(summary.get("git_sha")):
            errors.append("Stage 5A / Stage 5B git SHA mismatch")

    ownership_ok = bool(summary.get("ownership_converged"))
    reconstructed = stage5b_passes(
        source_rows=sources,
        decision_rows=decisions,
        bid_rows=bids,
        migration_rows=migrations,
        ownership_ok=ownership_ok,
        expected_git_sha=str(summary.get("git_sha")),
    )
    if summary.get("passed") is not True or not reconstructed:
        errors.append("Stage 5B pass invariants failed")

    if len(ownership) != len(STAGE5B_SOURCE_IDS) * 7:
        errors.append(f"ownership rows {len(ownership)} != {len(STAGE5B_SOURCE_IDS) * 7}")
    if int(summary.get("task_count") or 0) != len(STAGE5B_SOURCE_IDS):
        errors.append("task_count mismatch")
    if int(summary.get("scheduler_decision_count") or 0) != len(decisions):
        errors.append("scheduler decision count mismatch")
    if int(summary.get("bid_count") or 0) != len(bids):
        errors.append("bid count mismatch")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5B_REAL_MULTIORIGIN_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print(f"sources: {summary.get('trigger_success_count')}/{len(STAGE5B_SOURCE_IDS)}")
    print(f"scheduler_decisions: {len(decisions)}")
    print(f"bid_source_nodes: {summary.get('bid_source_node_count')}")
    print(f"bid_destination_nodes: {summary.get('bid_destination_node_count')}")
    print(f"bids: {len(bids)}")
    print(f"successful_migrations: {summary.get('successful_migration_count')}")
    print("ownership_converged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
