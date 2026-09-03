#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
    write_jsonl,
)
from magellan.experiments.stage5b import (
    STAGE5B_SOURCE_IDS,
    active_task_ids,
    ownership_converged,
    stage5b_passes,
)


TRACE_TIME_UTC = "2024-08-20T12:00:00Z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5B real multi-origin decentralized scheduling on the seven-node cluster."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--trace-time-utc", default=TRACE_TIME_UTC)
    parser.add_argument("--convergence-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=15.0)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def try_json(url: str, timeout: float = 8.0) -> Any | None:
    try:
        return request_json(url, timeout=timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def base_url(node: Any, port: int) -> str:
    return f"http://{node.internal_ip}:{port}"


def local_git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def event_query(api: str, after_sequence: int, task_id: str) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "after_sequence": after_sequence,
            "task_id": task_id,
            "limit": 100000,
        }
    )
    return list(request_json(f"{api}/experiment/events?{query}").get("events", []))


def task_state(api: str, task_id: str) -> dict[str, Any] | None:
    payload = try_json(f"{api}/tasks")
    if not isinstance(payload, dict):
        return None
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == task_id:
            return state
    return None


def definition_payload(comparison_id: str, node_ids: list[str]) -> dict[str, Any]:
    return {
        "definition_id": f"stage5b-counter-{comparison_id}",
        "profile": {
            "workload_type": "stage5b-counter",
            "power_kw": 0.6,
            "checkpoint_bytes": 1024,
            "data_bytes": 0,
            "prestaged_node_ids": node_ids,
            "estimated_remaining_seconds": 86400,
            "accumulated_cost_usd": 0,
            "cost_cap_usd": 100.0,
            "priority": 20,
            "deadline_at_utc": None,
            "resource_request": {
                "cpu_cores": 0.1,
                "memory_mb": 64,
                "gpu_count": 0,
                "accelerator_type": None,
            },
            "compatibility": {
                "architectures": ["x86_64"],
                "operating_systems": ["linux"],
                "minimum_cpu_cores": 1,
                "minimum_memory_mb": 256,
                "required_commands": ["python3"],
                "required_runtimes": {"python": ">=3.11,<3.12"},
                "required_features": ["python-module", "application-checkpoint"],
                "checkpoint_architecture_independent": True,
            },
        },
        "runtime": {
            "module": "magellan.workloads.counter",
            "arguments": [
                "--checkpoint-file", "{checkpoint_file}",
                "--interval-seconds", "0.5",
                "--max-value", "1000000",
                "--progress-file", "{progress_file}",
                "--completion-file", "{completion_file}",
                "--output-dir", "{output_directory}",
            ],
            "environment": {},
            "working_directory": ".",
            "checkpoint_relative_path": "checkpoint/counter.json",
            "progress_relative_path": "runtime/progress.json",
            "completion_relative_path": "runtime/completion.json",
            "output_relative_directory": "output",
            "stop_timeout_seconds": 10,
        },
        "artifacts": [],
    }


def wait_definition(cluster, definition_id: str, revision: int, digest: str, timeout: float) -> None:
    pending = {node.id for node in cluster.nodes}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            api = base_url(node, cluster.api_port)
            value = try_json(f"{api}/task-definitions/{definition_id}?revision={revision}")
            if isinstance(value, dict) and value.get("digest") == digest:
                pending.remove(node.id)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Definition did not converge to {sorted(pending)}")


def wait_runs(cluster, run_ids: list[str], timeout: float) -> None:
    pending = {(node.id, run_id) for node in cluster.nodes for run_id in run_ids}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            api = base_url(node, cluster.api_port)
            for run_id in run_ids:
                key = (node.id, run_id)
                if key not in pending:
                    continue
                value = try_json(f"{api}/task-runs/{run_id}")
                if isinstance(value, dict) and value.get("run", {}).get("run_id") == run_id:
                    pending.remove(key)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Task runs did not converge; remaining={sorted(pending)[:12]}")


def wait_ownership(cluster, run_ids: list[str], timeout: float):
    deadline = time.monotonic() + timeout
    last_snapshots: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        last_snapshots = {
            node.id: request_json(f"{base_url(node, cluster.api_port)}/ownership/snapshot")
            for node in cluster.nodes
        }
        ok, rows = ownership_converged(last_snapshots, run_ids)
        if ok:
            return True, rows, last_snapshots
        time.sleep(2)
    ok, rows = ownership_converged(last_snapshots, run_ids)
    return ok, rows, last_snapshots


def main() -> int:
    args = parse_args()
    source = Path(args.stage5a_bundle)
    errors = validate_checksums(source)
    if errors:
        raise RuntimeError("Stage 5A checksum failure: " + "; ".join(errors))
    stage5a = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if stage5a.get("passed") is not True:
        raise RuntimeError("Stage 5A source bundle did not pass")

    cluster = load_cluster_config(args.cluster)
    node_by_id = {node.id: node for node in cluster.nodes}
    if not set(STAGE5B_SOURCE_IDS).issubset(node_by_id):
        raise RuntimeError("Stage 5B source nodes are not present in cluster")

    target_sha = str(stage5a["target_git_sha"])
    if local_git_sha() != target_sha:
        raise RuntimeError(
            "Stage 5B must run from the exact SHA frozen by its Stage 5A bundle: "
            f"local={local_git_sha()} stage5a={target_sha}"
        )

    comparison_id = args.comparison_id or (
        f"stage5b-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 5B real multi-origin decentralized scheduling ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage5a={source}")
    print(f"git_sha={target_sha}")
    print(f"sources={','.join(STAGE5B_SOURCE_IDS)}")
    print(f"controlled_trace_time={args.trace_time_utc}")
    print("scheduler_mode=operator_only; explicit evaluate trigger uses production scoring/bidding/migration")

    health_by_node: dict[str, dict[str, Any]] = {}
    event_baselines: dict[str, int] = {}
    bid_baselines: dict[str, set[str]] = {}
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health")
        if health.get("deployment_git_sha") != target_sha:
            raise RuntimeError(f"{node.id} daemon SHA drifted from Stage 5A")
        active = active_task_ids(request_json(f"{api}/tasks"))
        if active:
            raise RuntimeError(
                f"Stage 5B requires no pre-existing active tasks; {node.id} has {active}"
            )
        health_by_node[node.id] = health
        event_baselines[node.id] = int(
            request_json(f"{api}/experiment/events/status").get("last_sequence", 0)
        )
        bid_baselines[node.id] = {
            str(item["bid_id"]) for item in request_json(f"{api}/bids")
        }
        print(f"[preflight] {node.id:16s} sha={target_sha[:12]} active_tasks=0")

    definition = definition_payload(comparison_id, [node.id for node in cluster.nodes])
    registration_api = base_url(node_by_id[STAGE5B_SOURCE_IDS[0]], cluster.api_port)
    created = request_json(
        f"{registration_api}/task-definitions",
        method="POST",
        payload=definition,
    )
    wait_definition(
        cluster,
        str(created["definition_id"]),
        int(created["revision"]),
        str(created["digest"]),
        args.convergence_timeout_seconds,
    )
    print(f"[catalog] definition converged: {created['definition_id']}@{created['revision']}")

    source_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    run_origin: dict[str, str] = {}
    for source_id in STAGE5B_SOURCE_IDS:
        node = node_by_id[source_id]
        api = base_url(node, cluster.api_port)
        run_view = request_json(
            f"{api}/task-runs",
            method="POST",
            payload={
                "definition_id": created["definition_id"],
                "revision": created["revision"],
                "initial_owner_node_id": source_id,
                "idempotency_key": f"{comparison_id}:{source_id}",
                "auto_start": True,
                "labels": {
                    "purpose": "stage5b-real-multiorigin",
                    "comparison_id": comparison_id,
                    "scheduler_mode": "operator_only",
                    "origin_node_id": source_id,
                },
            },
        )
        run_id = str(run_view["run"]["run_id"])
        run_ids.append(run_id)
        run_origin[run_id] = source_id
        source_rows.append(
            {
                "source_node_id": source_id,
                "task_id": run_id,
                "daemon_git_sha": health_by_node[source_id]["deployment_git_sha"],
                "trigger_ok": False,
                "trigger_error": "",
                "trigger_return_owner": "",
                "trigger_return_status": "",
            }
        )
        print(f"[submit] {source_id:16s} task={run_id}")

    wait_runs(cluster, run_ids, args.convergence_timeout_seconds)
    print("[catalog] all four task runs converged to all seven nodes")

    print(f"[checkpoint] allowing {args.checkpoint_wait_seconds:g}s for application checkpoints")
    time.sleep(args.checkpoint_wait_seconds)
    for run_id in run_ids:
        source_id = run_origin[run_id]
        state = task_state(base_url(node_by_id[source_id], cluster.api_port), run_id)
        if not state or state.get("status") != "running":
            raise RuntimeError(f"{source_id}/{run_id} is not running before evaluation: {state}")
        print(
            f"  {source_id:16s} progress={state.get('progress_completed_units')} "
            f"remaining={state.get('estimated_remaining_seconds')}"
        )

    print("\n[evaluate] triggering four source daemons concurrently", flush=True)
    by_source_row = {row["source_node_id"]: row for row in source_rows}

    def trigger(source_id: str, run_id: str):
        node = node_by_id[source_id]
        api = base_url(node, cluster.api_port)
        query = urlencode({"trace_time_utc": args.trace_time_utc})
        return request_json(
            f"{api}/tasks/{run_id}/evaluate?{query}",
            method="POST",
            timeout=240.0,
        )

    with ThreadPoolExecutor(max_workers=len(run_ids)) as pool:
        futures = {
            pool.submit(trigger, source_id, run_id): (source_id, run_id)
            for run_id, source_id in run_origin.items()
        }
        for future in as_completed(futures):
            source_id, run_id = futures[future]
            row = by_source_row[source_id]
            try:
                value = future.result()
                row["trigger_ok"] = True
                state = value.get("state", {})
                row["trigger_return_owner"] = state.get("owner_node_id", "")
                row["trigger_return_status"] = state.get("status", "")
                print(
                    f"  [done] {source_id:16s} task={run_id} "
                    f"owner={row['trigger_return_owner']} status={row['trigger_return_status']}"
                )
            except Exception as exc:
                row["trigger_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [fail] {source_id:16s} task={run_id} {row['trigger_error']}")

    time.sleep(args.settle_seconds)

    decision_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []
    bid_rows: list[dict[str, Any]] = []
    raw_node_evidence: list[dict[str, Any]] = []

    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        node_events: list[dict[str, Any]] = []
        for run_id in run_ids:
            events = event_query(api, event_baselines[node.id], run_id)
            node_events.extend(events)
            for event in events:
                event_rows.append(event)
                if event.get("event_type") == "scheduler_decision":
                    selected = event.get("payload", {}).get("decision", {}).get("selected", {})
                    decision_rows.append(
                        {
                            "node_id": node.id,
                            "task_id": run_id,
                            "origin_node_id": run_origin[run_id],
                            "selected_action": selected.get("action"),
                            "selected_destination_node_id": selected.get("destination_node_id") or "",
                            "selected_score": selected.get("score"),
                            "trace_time_utc": event.get("trace_time_utc"),
                            "sequence": event.get("sequence"),
                        }
                    )
                elif event.get("event_type") in {"migration_completed", "migration_failed"}:
                    payload = event.get("payload", {})
                    migration_rows.append(
                        {
                            "node_id": node.id,
                            "task_id": run_id,
                            "status": "completed" if event.get("event_type") == "migration_completed" else "failed",
                            "source_node_id": payload.get("source_node_id", ""),
                            "destination_node_id": payload.get("destination_node_id", ""),
                            "migration_id": payload.get("migration_id", ""),
                            "bid_id": payload.get("bid_id", ""),
                            "total_downtime_seconds": payload.get("total_downtime_seconds", ""),
                            "error": payload.get("error", ""),
                        }
                    )

        new_bids = [
            item
            for item in request_json(f"{api}/bids")
            if str(item.get("bid_id")) not in bid_baselines[node.id]
            and str(item.get("task_id")) in run_origin
        ]
        for item in new_bids:
            bid_rows.append(
                {
                    "reporting_node_id": node.id,
                    "bid_id": item.get("bid_id"),
                    "task_id": item.get("task_id"),
                    "source_node_id": item.get("source_node_id"),
                    "destination_node_id": item.get("destination_node_id"),
                    "status": item.get("status"),
                    "candidate_score": item.get("candidate", {}).get("score"),
                    "decision_reason": item.get("decision_reason"),
                    "credit_before": item.get("credit_before"),
                    "credit_after": item.get("credit_after"),
                }
            )
        raw_node_evidence.append(
            {
                "node_id": node.id,
                "health": request_json(f"{api}/health"),
                "events": node_events,
                "new_bids": new_bids,
                "ownership_snapshot": request_json(f"{api}/ownership/snapshot"),
            }
        )
        print(
            f"[collect] {node.id:16s} decisions={sum(r['node_id']==node.id for r in decision_rows)} "
            f"new_bids={len(new_bids)}"
        )

    ownership_ok, ownership_rows, snapshots = wait_ownership(
        cluster, run_ids, args.convergence_timeout_seconds
    )

    final_rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        updates = []
        for node_id, snapshot in snapshots.items():
            for update in snapshot.get("updates", []):
                if update.get("task_id") == run_id:
                    updates.append(update)
        newest = max(updates, key=lambda item: int(item.get("generation", 0)))
        final_rows.append(
            {
                "task_id": run_id,
                "origin_node_id": run_origin[run_id],
                "final_owner_node_id": newest.get("owner_node_id"),
                "generation": newest.get("generation"),
                "status": newest.get("status"),
                "last_migration_id": newest.get("last_migration_id") or "",
            }
        )

    passed = stage5b_passes(
        source_rows=source_rows,
        decision_rows=decision_rows,
        bid_rows=bid_rows,
        migration_rows=migration_rows,
        ownership_ok=ownership_ok,
        expected_git_sha=target_sha,
    )

    bid_sources = sorted({str(row["source_node_id"]) for row in bid_rows})
    bid_destinations = sorted({str(row["destination_node_id"]) for row in bid_rows})
    accepted = [row for row in bid_rows if row["status"] == "accepted"]
    rejected = [row for row in bid_rows if row["status"] == "rejected"]
    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage5a_bundle": str(source),
        "git_sha": target_sha,
        "controlled_trace_time_utc": args.trace_time_utc,
        "source_node_ids": list(STAGE5B_SOURCE_IDS),
        "task_count": len(run_ids),
        "trigger_success_count": sum(bool(row["trigger_ok"]) for row in source_rows),
        "scheduler_decision_count": len(decision_rows),
        "decision_source_node_count": len({str(row["node_id"]) for row in decision_rows}),
        "bid_count": len(bid_rows),
        "bid_source_node_ids": bid_sources,
        "bid_source_node_count": len(bid_sources),
        "bid_destination_node_ids": bid_destinations,
        "bid_destination_node_count": len(bid_destinations),
        "accepted_bid_count": len(accepted),
        "rejected_bid_count": len(rejected),
        "successful_migration_count": sum(row["status"] == "completed" for row in migration_rows),
        "failed_migration_count": sum(row["status"] == "failed" for row in migration_rows),
        "ownership_converged": ownership_ok,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5b_real_multiorigin_decentralized_scheduling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "execution": (
                "Four real source VMs each originate and execute one checkpointable task. "
                "Boston only coordinates submission timing and sends an evaluate-now HTTP trigger; "
                "the production SchedulerService on each owning source performs scoring, destination "
                "selection, peer bid submission, and accepted migration independently."
            ),
            "background_scheduler": (
                "Runs are labeled scheduler_mode=operator_only to prevent the 900-second background "
                "loop from racing the controlled trigger. The explicit evaluation endpoint calls the "
                "same production _evaluate_task_locked path and does not choose or force an action."
            ),
            "trace": (
                "All four source daemons evaluate against the same controlled summer trace timestamp "
                "to remove service-start clock skew while retaining production carbon scoring."
            ),
            "capacity": (
                "Stage 5B uses lightweight 0.1-core counter tasks to avoid intentionally creating "
                "resource contention. Measured-capacity contention is isolated in Stage 5C."
            ),
            "pass_condition": (
                "PASS requires a production scheduler decision on the correct origin daemon for every "
                "task, successful triggers on all four sources, real bid records from at least two "
                "distinct source daemons stored by their destination daemons, no failed migrations, "
                "and converged ownership for every task across all seven nodes. It does not require "
                "a particular destination, migration count, or carbon-favorable outcome."
            ),
        },
    }

    write_csv(root / "sources.csv", source_rows, list(source_rows[0].keys()))
    write_csv(
        root / "decisions.csv",
        decision_rows,
        list(decision_rows[0].keys()) if decision_rows else ["node_id"],
    )
    write_csv(
        root / "bids.csv",
        bid_rows,
        list(bid_rows[0].keys()) if bid_rows else ["reporting_node_id"],
    )
    write_csv(
        root / "migrations.csv",
        migration_rows,
        list(migration_rows[0].keys()) if migration_rows else ["node_id"],
    )
    write_csv(root / "ownership.csv", ownership_rows, list(ownership_rows[0].keys()))
    write_csv(root / "final_tasks.csv", final_rows, list(final_rows[0].keys()))
    write_jsonl(root / "events.jsonl", event_rows)
    write_jsonl(root / "node_evidence.jsonl", raw_node_evidence)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    print("\n[cleanup] stopping tasks at their converged owners", flush=True)
    for row in final_rows:
        owner = str(row["final_owner_node_id"])
        run_id = str(row["task_id"])
        try:
            request_json(
                f"{base_url(node_by_id[owner], cluster.api_port)}/tasks/{run_id}/stop",
                method="POST",
                timeout=60.0,
            )
            print(f"  stopped {run_id} on {owner}")
        except Exception as exc:
            print(f"  cleanup warning {run_id}/{owner}: {type(exc).__name__}: {exc}")

    marker = "STAGE_5B_REAL_MULTIORIGIN_PASS" if passed else "STAGE_5B_REAL_MULTIORIGIN_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"sources: {summary['trigger_success_count']}/{len(STAGE5B_SOURCE_IDS)} evaluated")
    print(f"scheduler_decisions: {len(decision_rows)}")
    print(f"bid_sources: {len(bid_sources)} ({','.join(bid_sources)})")
    print(f"bid_destinations: {len(bid_destinations)} ({','.join(bid_destinations)})")
    print(f"bids: {len(bid_rows)} accepted={len(accepted)} rejected={len(rejected)}")
    print(
        f"migrations: successful={summary['successful_migration_count']} "
        f"failed={summary['failed_migration_count']}"
    )
    print(f"ownership_converged: {ownership_ok}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
