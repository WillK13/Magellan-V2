#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5c import (
    is_resource_contention_rejection,
    is_successful_bid_status,
)
from magellan.experiments.stage5e3 import (
    BENCHMARK_CLASS_ID,
    STAGE5E3_DESTINATION_ID,
    STAGE5E3_SOURCE_IDS,
    stage5e3_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 5E.3 real-process contention bundle."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)

    required = [
        "summary.json",
        "metadata.json",
        "sources.csv",
        "decisions.csv",
        "bids.csv",
        "migrations.csv",
        "ownership.csv",
        "final_tasks.csv",
        "pre_live_witness.csv",
        "post_live_witness.csv",
        "resource_ledger.csv",
        "cleanup.csv",
        "auction_before.json",
        "auction_after.json",
        "events.jsonl",
        "node_evidence.jsonl",
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
    sources = read_csv(root / "sources.csv")
    decisions = read_csv(root / "decisions.csv")
    bids = read_csv(root / "bids.csv")
    migrations = read_csv(root / "migrations.csv")
    final_tasks = read_csv(root / "final_tasks.csv")
    pre_witness = read_csv(root / "pre_live_witness.csv")
    post_witness = read_csv(root / "post_live_witness.csv")
    resource_rows = read_csv(root / "resource_ledger.csv")
    cleanup = read_csv(root / "cleanup.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if summary.get("destination_node_id") != STAGE5E3_DESTINATION_ID:
        errors.append("destination mismatch")
    if list(summary.get("source_node_ids") or []) != list(STAGE5E3_SOURCE_IDS):
        errors.append("source node list mismatch")
    if summary.get("benchmark_class_id") != BENCHMARK_CLASS_ID:
        errors.append("benchmark class mismatch")

    for key, label in (
        ("source_stage5a_bundle", "Stage 5A"),
        ("source_stage5c_bundle", "Stage 5C"),
        ("source_stage5e2_bundle", "Stage 5E.2"),
        ("source_stage4d1_bundle", "Stage 4D.1"),
    ):
        path = Path(str(summary.get(key) or ""))
        if not path.is_dir():
            errors.append(f"{label} source bundle missing: {path}")
        elif validate_checksums(path):
            errors.append(f"{label} source checksum validation failed")

    resident = [row for row in final_tasks if row.get("role") == "resident"]
    if len(resident) != 1:
        errors.append("resident final-task coverage mismatch")
        resident_task_id = ""
    else:
        resident_task_id = resident[0]["task_id"]

    request_cpu = float(summary.get("benchmark_cpu_cores") or 0.0)
    request_memory = int(float(summary.get("benchmark_memory_mb") or 0))
    capacity_cpu = float(summary.get("destination_cpu_capacity") or 0.0)

    if resident_task_id and not stage5e3_passes(
        source_rows=sources,
        decision_rows=decisions,
        bid_rows=bids,
        migration_rows=migrations,
        final_rows=final_tasks,
        ownership_ok=summary.get("ownership_converged") is True,
        resident_task_id=resident_task_id,
        benchmark_cpu_cores=request_cpu,
        benchmark_memory_mb=request_memory,
        capacity_cpu_cores=capacity_cpu,
        expected_git_sha=str(summary.get("git_sha") or ""),
        pre_witness_rows=pre_witness,
        post_witness_rows=post_witness,
        resource_rows=resource_rows,
    ):
        errors.append("Stage 5E.3 pass invariants failed")

    challenge_bids = [
        row for row in bids
        if row.get("destination_node_id") == STAGE5E3_DESTINATION_ID
    ]
    successful = [
        row for row in challenge_bids
        if is_successful_bid_status(str(row.get("status")))
    ]
    rejected = [row for row in challenge_bids if row.get("status") == "rejected"]
    contention = [row for row in rejected if is_resource_contention_rejection(row)]
    completed = [row for row in migrations if row.get("status") == "completed"]
    failed = [row for row in migrations if row.get("status") == "failed"]

    if len(pre_witness) != 5 or sum(truthy(row.get("live")) for row in pre_witness) != 5:
        errors.append("pre-auction physical witness is not 5/5 live")
    if len(post_witness) != 5 or sum(truthy(row.get("live")) for row in post_witness) != 5:
        errors.append("post-auction physical witness is not 5/5 live")
    if sum(
        truthy(row.get("live")) and row.get("node_id") == STAGE5E3_DESTINATION_ID
        for row in post_witness
    ) != 2:
        errors.append("post-auction Ethiopia physical witness is not 2/2")
    if len(resource_rows) != 7:
        errors.append(f"resource ledger coverage {len(resource_rows)} != 7")
    if sum(truthy(row.get("reservation_matches_expected")) for row in resource_rows) != 7:
        errors.append("resource ledger reservation match is not 7/7")
    if sum(truthy(row.get("capacity_respected")) for row in resource_rows) != 7:
        errors.append("resource capacity respected is not 7/7")
    if len(cleanup) != 5 or sum(truthy(row.get("cleanup_ok")) for row in cleanup) != 5:
        errors.append("cleanup is not 5/5")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5E3_REAL_PROCESS_CONTENTION_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print(f"destination: {STAGE5E3_DESTINATION_ID}")
    print(f"sources: {len(sources)}/{len(STAGE5E3_SOURCE_IDS)}")
    print(f"scheduler_decisions: {len(decisions)}")
    print(
        f"bids: {len(challenge_bids)} successful={len(successful)} "
        f"rejected={len(rejected)} resource_contention={len(contention)}"
    )
    print(f"successful_migrations: {len(completed)}")
    print(f"failed_migrations: {len(failed)}")
    print("pre_live_witness: 5/5")
    print("post_live_witness: 5/5")
    print("destination_live_after: 2/2")
    print("resource_matches: 7/7")
    print("capacity_respected: 7/7")
    print("cleanup: 5/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
