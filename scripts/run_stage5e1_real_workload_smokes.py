#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage5e1 import (
    STAGE5E1_CASES,
    SmokeCase,
    stage5e1_passes,
    summarize_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5E.1 real benchmark/LLM/Dendro migration smoke tests "
            "on the hardened seven-node deployment."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--stage4a3-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--ssh-user", default=os.getenv("MAGELLAN_SSH_USER", "WILL"))
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--profile-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    return parser.parse_args()


def load_passed_bundle(path: Path, label: str) -> dict[str, Any]:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum failure: " + "; ".join(errors))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(f"{label} bundle did not pass")
    return summary


def local_git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def request_json(url: str) -> Any:
    with urlopen(url, timeout=10.0) as response:
        return json.load(response)


def active_tasks(api: str) -> list[tuple[str, str, str]]:
    rows = []
    for item in request_json(f"{api}/tasks").get("tasks", []):
        state = item.get("state", {})
        if state.get("status") in {"running", "paused", "migrating", "recovering"}:
            rows.append(
                (
                    str(state.get("task_id")),
                    str(state.get("owner_node_id")),
                    str(state.get("status")),
                )
            )
    return rows


def hardened_preflight(cluster, expected_sha: str) -> None:
    for node in cluster.nodes:
        api = f"http://{node.internal_ip}:{cluster.api_port}"
        health = request_json(f"{api}/health")
        if health.get("deployment_git_sha") != expected_sha:
            raise RuntimeError(
                f"{node.id} daemon SHA drifted: {health.get('deployment_git_sha')} != {expected_sha}"
            )
        if health.get("carbon_metric") != "lifecycle":
            raise RuntimeError(f"{node.id} carbon metric is not lifecycle")
        telemetry_state = str(health.get("telemetry_state_file") or "")
        if "runtime-state-gcp" not in telemetry_state or "runtime-state-gcp-measurement" in telemetry_state:
            raise RuntimeError(
                f"{node.id} is not using hardened runtime-state-gcp: {telemetry_state}"
            )
        active = active_tasks(api)
        if active:
            raise RuntimeError(f"{node.id} has pre-existing active tasks: {active}")


def case_command(
    case: SmokeCase,
    *,
    case_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    common = [
        "--cluster", args.cluster,
        "--local-node-id", "boston",
        "--ssh-user", args.ssh_user,
        "--timeout-seconds", str(args.timeout_seconds),
        "--measurements-root", str(case_root),
        "--measurement-id", case.case_id,
        "--expected-carbon-metric", "lifecycle",
        "--expected-state-token", "runtime-state-gcp",
    ]

    if case.class_id == "benchmark-json-medium":
        return [
            "python", "scripts/measure_stage4a2_workload.py",
            *common,
            "--source", case.source_node_id,
            "--destination", case.destination_node_id,
            "--workload", "benchmark",
            "--benchmark", "json",
            "--size", "medium",
            "--benchmark-iterations", "1000000",
            "--seed", "42",
            "--profile-seconds", str(args.profile_seconds),
            "--sample-interval-seconds", str(args.sample_interval_seconds),
            "--minimum-progress", "2",
        ]

    if case.class_id == "dendro-r9-t1p0":
        return [
            "python", "scripts/measure_stage4a2_workload.py",
            *common,
            "--source", case.source_node_id,
            "--destination", case.destination_node_id,
            "--workload", "dendro",
            "--resolution", "9",
            "--time-end", "1.0",
            "--profile-seconds", str(args.profile_seconds),
            "--sample-interval-seconds", str(args.sample_interval_seconds),
            "--minimum-progress", "1",
        ]

    if case.class_id == "llm-distilgpt2":
        return [
            "python", "scripts/measure_llm_migration.py",
            *common,
            "--state-root", "runtime-state-gcp",
            "--path", f"{case.source_node_id},{case.destination_node_id}",
            "--model", "experiment-assets/models/distilgpt2",
            "--migrate-after-step", "2",
            "--steps-between-migrations", "1",
            "--post-path-steps", "1",
            "--checkpoint-every", "1",
            "--sleep-per-step", "2",
            "--torch-threads", "2",
            "--profile-seconds", str(args.profile_seconds),
            "--profile-sample-interval-seconds", str(args.sample_interval_seconds),
        ]

    raise ValueError(case.class_id)


def main() -> int:
    args = parse_args()
    stage5a_path = Path(args.stage5a_bundle)
    stage4a3_path = Path(args.stage4a3_bundle)
    stage5a_summary = load_passed_bundle(stage5a_path, "Stage 5A")
    # Stage 4A.3 uses its own summary format but still has passed + checksums.
    load_passed_bundle(stage4a3_path, "Stage 4A.3")

    target_sha = str(stage5a_summary.get("target_git_sha") or stage5a_summary.get("git_sha") or "")
    if not target_sha:
        raise RuntimeError("Stage 5A bundle does not expose target Git SHA")
    if local_git_sha() != target_sha:
        raise RuntimeError(
            f"Local Git SHA {local_git_sha()} does not match hardened Stage 5A SHA {target_sha}"
        )

    cluster = load_cluster_config(args.cluster)
    hardened_preflight(cluster, target_sha)

    comparison_id = args.comparison_id or (
        f"stage5e1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    case_root = root / "cases"
    if root.exists():
        raise FileExistsError(root)
    case_root.mkdir(parents=True)

    print("== Stage 5E.1 real heterogeneous workload migration smokes ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage5a={stage5a_path}")
    print(f"source_stage4a3={stage4a3_path}")
    print(f"git_sha={target_sha}")
    print("cases=" + ",".join(case.case_id for case in STAGE5E1_CASES))
    print("state_root=runtime-state-gcp carbon_metric=lifecycle")

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(STAGE5E1_CASES, start=1):
        print(
            f"\n[case {index}/{len(STAGE5E1_CASES)}] {case.case_id} "
            f"{case.source_node_id} -> {case.destination_node_id}",
            flush=True,
        )
        hardened_preflight(cluster, target_sha)
        command = case_command(case, case_root=case_root, args=args)
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

        child = case_root / case.case_id
        child_summary = json.loads((child / "summary.json").read_text(encoding="utf-8"))
        child_errors = validate_checksums(child)
        if child_errors:
            raise RuntimeError(
                f"{case.case_id} child checksum failure: " + "; ".join(child_errors)
            )
        row = summarize_case(case, child_summary)
        row["child_bundle"] = str(child)
        rows.append(row)
        if not row["child_passed"] or not row["resume_validation_passed"]:
            raise RuntimeError(f"{case.case_id} smoke failed: {row}")
        print(
            f"  PASS migration_count={row['migration_count']} "
            f"checkpoint_bytes={row['checkpoint_bytes']} "
            f"downtime_seconds={row['downtime_seconds']}",
            flush=True,
        )
        hardened_preflight(cluster, target_sha)

    passed = stage5e1_passes(rows)
    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage5a_bundle": str(stage5a_path),
        "source_stage4a3_bundle": str(stage4a3_path),
        "git_sha": target_sha,
        "case_count": len(rows),
        "passed_case_count": sum(1 for row in rows if row["child_passed"] and row["resume_validation_passed"]),
        "cases": rows,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5e1_real_heterogeneous_workload_migration_smokes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Bridge the frozen Stage 4A workload classes to the hardened Stage 5 "
                "seven-daemon deployment by running one real migration for the measured "
                "benchmark, LLM, and Dendro workload forms."
            ),
            "benchmark": (
                "Real checkpointable magellan.workloads.benchmark JSON/medium workload; "
                "Boston -> California."
            ),
            "llm": (
                "Real Hugging Face distilgpt2 CPU training with model weights, tokenizer, "
                "AdamW optimizer state, PyTorch RNG state, and exact-step resume validation; "
                "California -> France."
            ),
            "dendro": (
                "Real upstream Dendro-GR BSSN r9/t1.0, two MPI ranks, native checkpoint "
                "discovery/restore; Virginia -> Nepal."
            ),
            "migration_path": (
                "Destinations are operator-directed for smoke coverage, while compatibility, "
                "destination bid/reservation, checkpoint, rsync, restore/activation, ownership, "
                "and telemetry use the production Magellan migration path."
            ),
            "scope": (
                "Stage 5E.1 validates workload-specific real migration correctness. Mixed "
                "placement, physical contention, and multi-task policy behavior are deferred "
                "to Stage 5E.2-5E.4."
            ),
        },
    }
    write_csv(root / "cases.csv", rows, list(rows[0].keys()))
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_5E1_REAL_WORKLOAD_SMOKES_PASS" if passed else "STAGE_5E1_REAL_WORKLOAD_SMOKES_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"cases: {sum(1 for row in rows if row['child_passed'] and row['resume_validation_passed'])}/{len(rows)}")
    for row in rows:
        print(
            f"  {row['case_id']:22s} {row['source_node_id']:16s} -> "
            f"{row['destination_node_id']:16s} migration={row['migration_count']} "
            f"resume={row['resume_validation_passed']}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
