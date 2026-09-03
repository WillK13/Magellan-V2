#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from magellan.experiments.bundle import validate_checksums
from magellan.experiments.stage5c import (
    STAGE5C_DESTINATION_ID,
    STAGE5C_SOURCE_IDS,
    is_resource_contention_rejection,
    is_successful_bid_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage 5C real contention bundle."
    )
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
        "sources.csv",
        "decisions.csv",
        "bids.csv",
        "migrations.csv",
        "ownership.csv",
        "final_tasks.csv",
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

    summary = json.loads(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    sources = read_csv(root / "sources.csv")
    decisions = read_csv(root / "decisions.csv")
    bids = read_csv(root / "bids.csv")
    migrations = read_csv(root / "migrations.csv")
    final_tasks = read_csv(root / "final_tasks.csv")

    if summary.get("passed") is not True:
        errors.append("summary passed is not true")
    if summary.get("destination_node_id") != STAGE5C_DESTINATION_ID:
        errors.append("destination mismatch")
    if list(summary.get("source_node_ids") or []) != list(
        STAGE5C_SOURCE_IDS
    ):
        errors.append("source node list mismatch")

    s5a = Path(str(summary.get("source_stage5a_bundle") or ""))
    s5b = Path(str(summary.get("source_stage5b_bundle") or ""))
    d41 = Path(str(summary.get("source_stage4d1_bundle") or ""))
    for path, label in (
        (s5a, "Stage 5A"),
        (s5b, "Stage 5B"),
        (d41, "Stage 4D.1"),
    ):
        if not path.is_dir():
            errors.append(f"{label} source bundle missing: {path}")
        elif validate_checksums(path):
            errors.append(f"{label} source checksum validation failed")

    if len(sources) != len(STAGE5C_SOURCE_IDS):
        errors.append(
            f"source coverage {len(sources)} != {len(STAGE5C_SOURCE_IDS)}"
        )
    if sum(row.get("trigger_ok", "").lower() == "true" for row in sources) != len(
        STAGE5C_SOURCE_IDS
    ):
        errors.append("not all source triggers succeeded")

    if len(decisions) != len(STAGE5C_SOURCE_IDS):
        errors.append(
            f"decision coverage {len(decisions)} != {len(STAGE5C_SOURCE_IDS)}"
        )
    for row in decisions:
        if row["selected_action"] != "migrate":
            errors.append(
                f"{row['node_id']} did not select migrate"
            )
        if (
            row["selected_destination_node_id"]
            != STAGE5C_DESTINATION_ID
        ):
            errors.append(
                f"{row['node_id']} did not select Ethiopia"
            )

    successful = [
        row for row in bids
        if is_successful_bid_status(row["status"])
    ]
    rejected = [
        row for row in bids
        if row["status"] == "rejected"
    ]
    contention = [
        row for row in rejected
        if is_resource_contention_rejection(row)
    ]
    if len(bids) != len(STAGE5C_SOURCE_IDS):
        errors.append(
            f"bid coverage {len(bids)} != {len(STAGE5C_SOURCE_IDS)}"
        )
    if len(successful) != 1:
        errors.append(f"successful bid outcomes {len(successful)} != 1")
    if len(rejected) != len(STAGE5C_SOURCE_IDS) - 1:
        errors.append(
            f"rejected bids {len(rejected)} != {len(STAGE5C_SOURCE_IDS)-1}"
        )
    if len(contention) != len(rejected):
        errors.append(
            "one or more rejected bids were not resource-contention rejections"
        )
    if {
        row["source_node_id"] for row in bids
    } != set(STAGE5C_SOURCE_IDS):
        errors.append("bid source-node coverage mismatch")
    if any(
        row["destination_node_id"] != STAGE5C_DESTINATION_ID
        for row in bids
    ):
        errors.append("a challenge bid was not stored at Ethiopia")

    completed = [
        row for row in migrations
        if row["status"] == "completed"
    ]
    failed = [
        row for row in migrations
        if row["status"] == "failed"
    ]
    if len(completed) != 1:
        errors.append(
            f"completed migrations {len(completed)} != 1"
        )
    if failed:
        errors.append(f"failed migrations observed: {len(failed)}")

    if summary.get("ownership_converged") is not True:
        errors.append("ownership did not converge")

    resident = [
        row for row in final_tasks
        if row["role"] == "resident"
    ]
    challengers = [
        row for row in final_tasks
        if row["role"] == "challenger"
    ]
    if len(resident) != 1:
        errors.append("resident final-task coverage mismatch")
    elif (
        resident[0]["final_owner_node_id"]
        != STAGE5C_DESTINATION_ID
    ):
        errors.append("resident left Ethiopia")
    if len(challengers) != len(STAGE5C_SOURCE_IDS):
        errors.append("challenger final-task coverage mismatch")
    if sum(
        row["final_owner_node_id"] == STAGE5C_DESTINATION_ID
        for row in challengers
    ) != 1:
        errors.append(
            "exactly one challenger did not finish owned by Ethiopia"
        )

    request_cpu = float(summary.get("benchmark_cpu_cores") or 0)
    capacity_cpu = float(
        summary.get("destination_cpu_capacity") or 0
    )
    if 2 * request_cpu > capacity_cpu + 1e-9:
        errors.append("resident + one benchmark request does not fit")
    if 3 * request_cpu <= capacity_cpu + 1e-9:
        errors.append(
            "resident + two benchmark requests unexpectedly fit"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    print("STAGE_5C_REAL_CONTENTION_BUNDLE_PASS")
    print(f"comparison_id: {summary.get('comparison_id')}")
    print(f"git_sha: {summary.get('git_sha')}")
    print(f"destination: {STAGE5C_DESTINATION_ID}")
    print(
        f"sources: {len(sources)}/{len(STAGE5C_SOURCE_IDS)}"
    )
    print(f"scheduler_decisions: {len(decisions)}")
    print(
        f"bids: {len(bids)} successful={len(successful)} "
        f"rejected={len(rejected)}"
    )
    print(
        f"resource_contention_rejections: "
        f"{len(contention)}/{len(rejected)}"
    )
    print(f"successful_migrations: {len(completed)}")
    print("failed_migrations: 0")
    print("ownership_converged: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
