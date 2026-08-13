#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from magellan.experiments.bundle import validate_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Magellan experiment bundle and its internal evidence."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors: list[str] = []

    required = [
        "manifest.json",
        "summary.json",
        "decisions.csv",
        "decision_candidates.csv",
        "migrations.csv",
        "ownership.csv",
        "task_results.csv",
        "checksums.sha256",
        "raw/events.jsonl",
        "raw/observations.jsonl",
    ]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"Missing required bundle file: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    errors.extend(validate_checksums(root))
    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "summary.json")
    decisions = csv_rows(root / "decisions.csv")
    candidates = csv_rows(root / "decision_candidates.csv")
    migrations = csv_rows(root / "migrations.csv")
    ownership = csv_rows(root / "ownership.csv")
    results = csv_rows(root / "task_results.csv")

    node_ids = manifest.get("cluster", {}).get("node_ids", [])
    if len(node_ids) != 7 or len(set(node_ids)) != 7:
        errors.append(f"Expected seven unique node IDs; found {node_ids}")
    for node_id in node_ids:
        if not (root / "raw" / f"{node_id}.json").is_file():
            errors.append(f"Missing raw evidence for node: {node_id}")

    run_id = manifest.get("run_id")
    if not run_id or summary.get("run_id") != run_id:
        errors.append("Manifest and summary run IDs do not match")
    if len(results) != 1 or results[0].get("task_id") != run_id:
        errors.append("task_results.csv must contain exactly the experiment run")
    if summary.get("status") != "completed":
        errors.append(f"Experiment status is not completed: {summary.get('status')}")
    if summary.get("last_error"):
        errors.append(f"Completed experiment has last_error: {summary.get('last_error')}")
    if not decisions:
        errors.append("No scheduler decisions were recorded")
    if len(candidates) < len(decisions):
        errors.append("There are fewer candidate records than scheduler decisions")

    decision_sequences: dict[str, list[int]] = {}
    for row in decisions:
        node_id = row.get("node_id", "")
        try:
            sequence = int(row.get("sequence") or 0)
        except ValueError:
            errors.append(f"Invalid decision sequence: {row.get('sequence')}")
            continue
        decision_sequences.setdefault(node_id, []).append(sequence)
    for node_id, sequences in decision_sequences.items():
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            errors.append(f"Decision sequences are not strictly increasing for {node_id}")

    generations: list[int] = []
    for row in ownership:
        raw = row.get("generation")
        if raw in {None, ""}:
            continue
        try:
            generations.append(int(raw))
        except ValueError:
            errors.append(f"Invalid ownership generation: {raw}")
    if generations and generations != sorted(generations):
        errors.append("Ownership generations move backwards")

    successful_migrations = [row for row in migrations if row.get("status") == "completed"]
    expected_migrations = int(summary.get("successful_migration_count", 0))
    if len(successful_migrations) != expected_migrations:
        errors.append("Migration CSV count does not match summary")
    if manifest.get("requirements", {}).get("require_migration") and not successful_migrations:
        errors.append("Manifest requires migration but no successful migration was recorded")

    if errors:
        print("EXPERIMENT RUN INVALID")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("EXPERIMENT RUN PASSED")
    print(f"experiment_id: {manifest.get('experiment_id')}")
    print(f"run_id: {run_id}")
    print(f"nodes: {len(node_ids)}")
    print(f"decisions: {len(decisions)}")
    print(f"candidates: {len(candidates)}")
    print(f"migrations: {len(successful_migrations)}")
    print(f"final_owner: {summary.get('final_owner_node_id')}")
    print(f"final_generation: {summary.get('final_generation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
