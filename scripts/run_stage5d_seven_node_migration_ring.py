#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
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
from magellan.experiments.stage5b import active_task_ids, ownership_converged
from magellan.experiments.stage5d import STAGE5D_RING, expected_hops, stage5d_passes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5D: one real checkpointable task traverses a controlled "
            "seven-node migration ring and returns to Boston."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--stage5c-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=5.0)
    parser.add_argument("--post-hop-settle-seconds", type=float, default=1.0)
    parser.add_argument("--convergence-timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
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


def require_bundle(path: Path, label: str) -> dict[str, Any]:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum failure: " + "; ".join(errors))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(f"{label} source bundle did not pass")
    return summary


def definition_payload(comparison_id: str, node_ids: list[str]) -> dict[str, Any]:
    return {
        "definition_id": f"stage5d-ring-counter-{comparison_id}",
        "profile": {
            "workload_type": "stage5d-ring-counter",
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


def wait_definition(
    cluster,
    definition_id: str,
    revision: int,
    digest: str,
    timeout: float,
) -> None:
    pending = {node.id for node in cluster.nodes}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            value = try_json(
                f"{base_url(node, cluster.api_port)}/task-definitions/"
                f"{definition_id}?revision={revision}"
            )
            if isinstance(value, dict) and value.get("digest") == digest:
                pending.remove(node.id)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Definition did not converge to {sorted(pending)}")


def wait_run(cluster, run_id: str, timeout: float) -> None:
    pending = {node.id for node in cluster.nodes}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            value = try_json(f"{base_url(node, cluster.api_port)}/task-runs/{run_id}")
            if isinstance(value, dict) and value.get("run", {}).get("run_id") == run_id:
                pending.remove(node.id)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Task run did not converge to {sorted(pending)}")


def task_state(api: str, task_id: str) -> dict[str, Any] | None:
    payload = try_json(f"{api}/tasks")
    if not isinstance(payload, dict):
        return None
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == task_id:
            return state
    return None


def wait_running_progress(
    api: str,
    task_id: str,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = task_state(api, task_id)
        if (
            isinstance(last, dict)
            and last.get("status") == "running"
            and last.get("pid")
            and last.get("progress_completed_units") is not None
        ):
            return last
        time.sleep(0.5)
    raise RuntimeError(f"Task did not expose running progress: {last}")


def wait_ownership_one(cluster, task_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    last_snapshots: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        last_snapshots = {
            node.id: request_json(
                f"{base_url(node, cluster.api_port)}/ownership/snapshot"
            )
            for node in cluster.nodes
        }
        ok, rows = ownership_converged(last_snapshots, [task_id])
        if ok:
            return True, rows, last_snapshots
        time.sleep(1)
    ok, rows = ownership_converged(last_snapshots, [task_id])
    return ok, rows, last_snapshots


def newest_update(snapshots: dict[str, dict[str, Any]], task_id: str) -> dict[str, Any]:
    updates = [
        update
        for snapshot in snapshots.values()
        for update in snapshot.get("updates", [])
        if update.get("task_id") == task_id
    ]
    if not updates:
        raise RuntimeError(f"No ownership update for {task_id}")
    return max(updates, key=lambda item: int(item.get("generation", 0)))


def experiment_events(api: str, task_id: str) -> list[dict[str, Any]]:
    return list(
        request_json(
            f"{api}/experiment/events?after_sequence=0&task_id={task_id}&limit=100000"
        ).get("events", [])
    )


def migration_event(
    api: str,
    task_id: str,
    migration_id: str,
) -> dict[str, Any] | None:
    for event in reversed(experiment_events(api, task_id)):
        if (
            event.get("event_type") == "migration_completed"
            and event.get("payload", {}).get("migration_id") == migration_id
        ):
            return event
    return None


def counter_artifacts(
    node: Any,
    task_id: str,
    *,
    ssh_user: str,
) -> dict[str, Any]:
    """Read durable counter artifacts from the node's effective state root."""

    quoted_task = shlex.quote(task_id)
    script = f"""set -euo pipefail
ROOT=\"$(sudo systemctl show magellan --property=Environment --value \\
  | tr ' ' '\\n' \\
  | sed -n 's/^MAGELLAN_STATE_ROOT=//p' \\
  | tail -1)\"
test -n \"$ROOT\"
TASK_DIR=\"$ROOT/tasks/{quoted_task}\"
python3 - \"$ROOT\" \"$TASK_DIR\" <<'PY_ARTIFACT'
import json
import pathlib
import re
import sys

root = sys.argv[1]
task_dir = pathlib.Path(sys.argv[2])
checkpoint_path = task_dir / "checkpoint" / "counter.json"
progress_path = task_dir / "runtime" / "progress.json"
log_path = task_dir / "logs" / "process.log"

checkpoint = json.loads(checkpoint_path.read_text())
progress = json.loads(progress_path.read_text())
log_text = log_path.read_text()
resumed = [int(v) for v in re.findall(r"\\[counter\\] resumed value=(\\d+)", log_text)]
stopped = [int(v) for v in re.findall(r"\\[counter\\] stopped value=(\\d+)", log_text)]

print(json.dumps({
    "state_root": root,
    "checkpoint_value": checkpoint.get("value"),
    "checkpoint_node_id": checkpoint.get("node_id"),
    "checkpoint_updated_at_unix": checkpoint.get("updated_at_unix"),
    "progress_value": progress.get("completed_units"),
    "progress_node_id": progress.get("node_id"),
    "progress_updated_at_utc": progress.get("updated_at_utc"),
    "last_resumed_value": resumed[-1] if resumed else None,
    "last_stopped_value": stopped[-1] if stopped else None,
}))
PY_ARTIFACT
"""

    if str(node.id) == "boston":
        result = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10",
                f"{ssh_user}@{node.internal_ip}",
                "bash", "-s",
            ],
            input=script,
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    ssh_user = os.getenv("MAGELLAN_SSH_USER", "WILL")
    s5a_path = Path(args.stage5a_bundle)
    s5c_path = Path(args.stage5c_bundle)
    s5a = require_bundle(s5a_path, "Stage 5A")
    require_bundle(s5c_path, "Stage 5C")

    cluster = load_cluster_config(args.cluster)
    node_by_id = {node.id: node for node in cluster.nodes}
    ring_nodes = set(STAGE5D_RING[:-1])
    if ring_nodes != set(node_by_id):
        raise RuntimeError(
            f"Stage 5D ring must cover exactly the cluster nodes: "
            f"ring={sorted(ring_nodes)} cluster={sorted(node_by_id)}"
        )

    target_sha = str(s5a["target_git_sha"])
    if local_git_sha() != target_sha:
        raise RuntimeError(
            "Stage 5D must run from the exact SHA frozen by its current Stage 5A bundle: "
            f"local={local_git_sha()} stage5a={target_sha}"
        )

    comparison_id = args.comparison_id or (
        f"stage5d-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 5D real seven-node migration ring ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage5a={s5a_path}")
    print(f"source_stage5c={s5c_path}")
    print(f"git_sha={target_sha}")
    print("ring=" + " -> ".join(STAGE5D_RING))
    print(
        "mode=operator-directed destinations; every hop still uses production "
        "compatibility, checkpoint validation, destination bidding/arbitration, "
        "rsync transfer, activation, and ownership broadcast"
    )

    health_by_node: dict[str, dict[str, Any]] = {}
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health")
        if health.get("deployment_git_sha") != target_sha:
            raise RuntimeError(f"{node.id} daemon SHA drifted from Stage 5A")
        active = active_task_ids(request_json(f"{api}/tasks"))
        if active:
            raise RuntimeError(
                f"Stage 5D requires no pre-existing active tasks; {node.id} has {active}"
            )
        health_by_node[node.id] = health
        print(f"[preflight] {node.id:16s} sha={target_sha[:12]} active_tasks=0")

    boston_api = base_url(node_by_id["boston"], cluster.api_port)
    definition = definition_payload(comparison_id, list(node_by_id))
    created = request_json(
        f"{boston_api}/task-definitions",
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

    view = request_json(
        f"{boston_api}/task-runs",
        method="POST",
        payload={
            "definition_id": str(created["definition_id"]),
            "revision": int(created["revision"]),
            "initial_owner_node_id": "boston",
            "idempotency_key": f"{comparison_id}:ring-task",
            "auto_start": True,
            "labels": {
                "purpose": "stage5d-seven-node-migration-ring",
                "comparison_id": comparison_id,
                "scheduler_mode": "operator_only",
                "origin_node_id": "boston",
            },
        },
    )
    task_id = str(view["run"]["run_id"])
    wait_run(cluster, task_id, args.convergence_timeout_seconds)
    print(f"[submit] task={task_id} owner=boston")
    print(f"[checkpoint] allowing {args.checkpoint_wait_seconds:g}s for initial progress")
    time.sleep(args.checkpoint_wait_seconds)
    initial_state = wait_running_progress(
        boston_api,
        task_id,
        args.convergence_timeout_seconds,
    )
    initial_artifacts = counter_artifacts(
        node_by_id["boston"], task_id, ssh_user=ssh_user
    )
    print(
        f"[initial] generation={initial_state.get('generation')} "
        f"registry_progress={initial_state.get('progress_completed_units')} "
        f"checkpoint={initial_artifacts.get('checkpoint_value')} "
        f"pid={initial_state.get('pid')}"
    )

    hop_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_journals: list[dict[str, Any]] = []
    ownership_rows_all: list[dict[str, Any]] = []

    for hop_index, (source_id, destination_id) in enumerate(expected_hops(), start=1):
        source_api = base_url(node_by_id[source_id], cluster.api_port)
        destination_api = base_url(node_by_id[destination_id], cluster.api_port)
        before = wait_running_progress(
            source_api,
            task_id,
            args.convergence_timeout_seconds,
        )
        if before.get("owner_node_id") != source_id:
            raise RuntimeError(
                f"Hop {hop_index} owner mismatch before migration: {before}"
            )
        if int(before.get("generation", -1)) != hop_index - 1:
            raise RuntimeError(
                f"Hop {hop_index} generation mismatch before migration: {before}"
            )

        print(
            f"\n[hop {hop_index}/7] {source_id} -> {destination_id} "
            f"generation={before.get('generation')} "
            f"progress={before.get('progress_completed_units')}",
            flush=True,
        )
        started = time.monotonic()
        response = request_json(
            f"{source_api}/tasks/{task_id}/migrate/{destination_id}",
            method="POST",
            timeout=360.0,
        )
        request_wall = time.monotonic() - started
        if response.get("migrated") is not True:
            raise RuntimeError(f"Hop {hop_index} migration was not accepted: {response}")

        response_state = response.get("state") or {}
        migration_id = str(response_state.get("last_migration_id") or "")
        bid = response.get("bid") or {}
        if not migration_id:
            raise RuntimeError(f"Hop {hop_index} missing migration id: {response}")

        source_artifacts = counter_artifacts(
            node_by_id[source_id], task_id, ssh_user=ssh_user
        )
        time.sleep(args.post_hop_settle_seconds)
        after = wait_running_progress(
            destination_api,
            task_id,
            args.convergence_timeout_seconds,
        )
        destination_artifacts = counter_artifacts(
            node_by_id[destination_id], task_id, ssh_user=ssh_user
        )
        ownership_ok, ownership_rows, snapshots = wait_ownership_one(
            cluster,
            task_id,
            args.convergence_timeout_seconds,
        )
        ownership_rows_all.extend(
            {**row, "hop_index": hop_index}
            for row in ownership_rows
        )
        newest = newest_update(snapshots, task_id)

        source_record = request_json(f"{source_api}/migrations/{migration_id}")
        destination_record = request_json(
            f"{destination_api}/migrations/{migration_id}"
        )
        event = migration_event(source_api, task_id, migration_id)
        if event is None:
            raise RuntimeError(
                f"Hop {hop_index} missing migration_completed event {migration_id}"
            )
        payload = event.get("payload", {})
        all_events.append(event)
        all_journals.extend(
            [
                {"reporting_node_id": source_id, **source_record},
                {"reporting_node_id": destination_id, **destination_record},
            ]
        )

        hop_row = {
            "hop_index": hop_index,
            "source_node_id": source_id,
            "destination_node_id": destination_id,
            "source_daemon_git_sha": health_by_node[source_id]["deployment_git_sha"],
            "destination_daemon_git_sha": health_by_node[destination_id]["deployment_git_sha"],
            "owner_before": before.get("owner_node_id"),
            "generation_before": before.get("generation"),
            "source_status_before": before.get("status"),
            "source_pid_before": before.get("pid"),
            "registry_progress_before": before.get("progress_completed_units"),
            "source_state_root": source_artifacts.get("state_root"),
            "source_checkpoint_value": source_artifacts.get("checkpoint_value"),
            "source_progress_value": source_artifacts.get("progress_value"),
            "source_stopped_value": source_artifacts.get("last_stopped_value"),
            "progress_before": source_artifacts.get("checkpoint_value"),
            "migrated": bool(response.get("migrated")),
            "already_migrated": bool(response.get("already_migrated", False)),
            "bid_id": bid.get("bid_id", ""),
            "bid_status": bid.get("status", ""),
            "migration_id": migration_id,
            "request_wall_seconds": request_wall,
            "total_downtime_seconds": payload.get("total_downtime_seconds"),
            "checkpoint_seconds": payload.get("checkpoint_seconds"),
            "transfer_seconds": payload.get("transfer_seconds"),
            "restore_seconds": payload.get("restore_seconds"),
            "activation_seconds": payload.get("activation_seconds"),
            "checkpoint_transfer_bytes": payload.get("checkpoint_transfer_bytes"),
            "owner_after": after.get("owner_node_id"),
            "generation_after": after.get("generation"),
            "destination_status_after": after.get("status"),
            "destination_pid_after": after.get("pid"),
            "registry_progress_after": after.get("progress_completed_units"),
            "destination_state_root": destination_artifacts.get("state_root"),
            "destination_resume_value": destination_artifacts.get("last_resumed_value"),
            "destination_checkpoint_value": destination_artifacts.get("checkpoint_value"),
            "destination_progress_value": destination_artifacts.get("progress_value"),
            "progress_after": destination_artifacts.get("last_resumed_value"),
            "source_record_role": source_record.get("role"),
            "source_record_status": source_record.get("status"),
            "destination_record_role": destination_record.get("role"),
            "destination_record_status": destination_record.get("status"),
            "ownership_converged": ownership_ok,
            "converged_owner": newest.get("owner_node_id"),
            "converged_generation": newest.get("generation"),
        }
        hop_rows.append(hop_row)
        print(
            f"  complete migration={migration_id[:12]} "
            f"generation={after.get('generation')} "
            f"checkpoint={source_artifacts.get('checkpoint_value')} "
            f"resume={destination_artifacts.get('last_resumed_value')} "
            f"registry={before.get('progress_completed_units')}->"
            f"{after.get('progress_completed_units')} "
            f"downtime={float(payload.get('total_downtime_seconds') or 0):.3f}s "
            f"ownership_converged={ownership_ok}",
            flush=True,
        )

    final_ok, final_ownership_rows, final_snapshots = wait_ownership_one(
        cluster,
        task_id,
        args.convergence_timeout_seconds,
    )
    final = newest_update(final_snapshots, task_id)
    final_state = wait_running_progress(
        boston_api,
        task_id,
        args.convergence_timeout_seconds,
    )
    final_artifacts = counter_artifacts(
        node_by_id["boston"], task_id, ssh_user=ssh_user
    )

    passed = stage5d_passes(
        hop_rows=hop_rows,
        final_owner_node_id=str(final.get("owner_node_id")),
        final_generation=int(final.get("generation", -1)),
        ownership_converged_final=final_ok,
        expected_git_sha=target_sha,
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage5a_bundle": str(s5a_path),
        "source_stage5c_bundle": str(s5c_path),
        "git_sha": target_sha,
        "task_id": task_id,
        "ring": list(STAGE5D_RING),
        "hop_count": len(hop_rows),
        "successful_hop_count": sum(bool(row["migrated"]) for row in hop_rows),
        "unique_source_nodes": len({row["source_node_id"] for row in hop_rows}),
        "unique_destination_nodes": len({row["destination_node_id"] for row in hop_rows}),
        "final_owner_node_id": final.get("owner_node_id"),
        "final_generation": final.get("generation"),
        "final_status": final_state.get("status"),
        "final_pid": final_state.get("pid"),
        "initial_registry_progress_completed_units": initial_state.get("progress_completed_units"),
        "final_registry_progress_completed_units": final_state.get("progress_completed_units"),
        "initial_progress_completed_units": initial_artifacts.get("checkpoint_value"),
        "final_progress_completed_units": final_artifacts.get("checkpoint_value"),
        "ownership_converged_final": final_ok,
        "total_downtime_seconds": sum(
            float(row["total_downtime_seconds"] or 0) for row in hop_rows
        ),
        "max_hop_downtime_seconds": max(
            float(row["total_downtime_seconds"] or 0) for row in hop_rows
        ),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5d_real_seven_node_migration_ring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "question": (
                "Can Magellan's real checkpoint-transfer-restore and ownership mechanism "
                "move one stateful task through every peer as both source and destination?"
            ),
            "route": " -> ".join(STAGE5D_RING),
            "control": (
                "The experiment operator chooses the next destination. Stage 5D does not "
                "test carbon-policy destination choice; Stages 5B/5C already test autonomous "
                "production decisions."
            ),
            "real_mechanism": (
                "Every hop uses the normal operator migration endpoint, which still performs "
                "compatibility checking, checkpoint validation, real destination bidding and "
                "ResourceLedger admission, rsync checkpoint transfer, remote activation, "
                "migration journaling, and ownership broadcast."
            ),
            "pass_condition": (
                "Seven successful hops; every cluster node appears once as source and once as "
                "destination; generation advances exactly 0 through 7; source and destination "
                "migration journals are activated for every hop; destination process is running; "
                "accounted progress never regresses; ownership converges after every hop; final "
                "owner is Boston at generation 7."
            ),
        },
    }

    write_csv(root / "hops.csv", hop_rows, list(hop_rows[0].keys()))
    write_csv(
        root / "ownership_per_hop.csv",
        ownership_rows_all,
        list(ownership_rows_all[0].keys()),
    )
    write_csv(
        root / "final_ownership.csv",
        final_ownership_rows,
        list(final_ownership_rows[0].keys()),
    )
    write_jsonl(root / "migration_events.jsonl", all_events)
    write_jsonl(root / "migration_journals.jsonl", all_journals)
    write_json(root / "initial_state.json", initial_state)
    write_json(root / "final_state.json", final_state)
    write_json(root / "initial_artifacts.json", initial_artifacts)
    write_json(root / "final_artifacts.json", final_artifacts)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    print("\n[cleanup] stopping ring task on final owner", flush=True)
    try:
        request_json(
            f"{boston_api}/tasks/{task_id}/stop",
            method="POST",
            timeout=60.0,
        )
        print(f"  stopped {task_id} on boston")
    except Exception as exc:
        print(f"  cleanup warning: {type(exc).__name__}: {exc}")

    marker = "STAGE_5D_MIGRATION_RING_PASS" if passed else "STAGE_5D_MIGRATION_RING_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"task: {task_id}")
    print(f"hops: {summary['successful_hop_count']}/7")
    print(
        f"source_nodes: {summary['unique_source_nodes']}/7 "
        f"destination_nodes: {summary['unique_destination_nodes']}/7"
    )
    print(
        f"final_owner: {summary['final_owner_node_id']} "
        f"generation={summary['final_generation']}"
    )
    print(
        f"progress: {summary['initial_progress_completed_units']} -> "
        f"{summary['final_progress_completed_units']}"
    )
    print(
        f"downtime: total={summary['total_downtime_seconds']:.3f}s "
        f"max_hop={summary['max_hop_downtime_seconds']:.3f}s"
    )
    print(f"ownership_converged_final: {final_ok}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
