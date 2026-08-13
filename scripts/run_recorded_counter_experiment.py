#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.bundle import sha256_file, write_checksums, write_json
from magellan.experiments.collector import materialize_bundle_tables


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one checkpointable seven-node counter experiment and write a complete "
            "reproducibility bundle. Run this from a cluster node that can reach all private APIs."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.smoke.json")
    parser.add_argument("--policy", default="config/policy.gcp.smoke.json")
    parser.add_argument(
        "--definition",
        default="config/submissions/gcp-seven-node-smoke-definition.json",
    )
    parser.add_argument("--initial-node-id", default="boston")
    parser.add_argument("--runs-root", default="experiments/runs")
    parser.add_argument("--timeout-seconds", type=float, default=360.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--require-migration", action="store_true")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    return parser.parse_args()


def request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 8.0,
) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def try_json(url: str) -> Any | None:
    try:
        return request_json(url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def base_url(node: Any, port: int) -> str:
    return f"http://{node.internal_ip}:{port}"


def task_state(api: str, run_id: str) -> dict[str, Any] | None:
    payload = try_json(f"{api}/tasks")
    if not isinstance(payload, dict):
        return None
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == run_id:
            return state
    return None


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def remote_cd(path: str) -> str:
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        return f'cd "$HOME"/{shlex.quote(path[2:])}'
    return f"cd {shlex.quote(path)}"


def cluster_git_snapshot(
    nodes: list[Any],
    *,
    local_node_id: str,
    ssh_user: str,
    remote_repo: str,
) -> dict[str, dict[str, str]]:
    local_commit = git_value("rev-parse", "HEAD")
    snapshots: dict[str, dict[str, str]] = {}
    for node in nodes:
        if node.id == local_node_id:
            snapshot = {
                "commit": local_commit or "",
                "branch": git_value("branch", "--show-current") or "",
                "status_porcelain": git_value("status", "--porcelain") or "",
            }
        else:
            command = (
                f"{remote_cd(remote_repo)} && "
                "printf 'COMMIT=%s\\n' \"$(git rev-parse HEAD)\" && "
                "printf 'BRANCH=%s\\n' \"$(git branch --show-current)\" && "
                "printf 'DIRTY=%s\\n' \"$(git status --porcelain | wc -l)\""
            )
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    f"{ssh_user}@{node.internal_ip}",
                    command,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            values = dict(
                line.split("=", 1)
                for line in result.stdout.splitlines()
                if "=" in line
            )
            snapshot = {
                "commit": values.get("COMMIT", ""),
                "branch": values.get("BRANCH", ""),
                "dirty_file_count": values.get("DIRTY", ""),
            }
        snapshots[node.id] = snapshot
        if local_commit and snapshot.get("commit") != local_commit:
            raise RuntimeError(
                f"Git commit mismatch: {node.id}={snapshot.get('commit')} "
                f"local={local_commit}"
            )
    return snapshots


def file_record(path_value: str | Path) -> dict[str, Any]:
    path = Path(path_value)
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def query_events(api: str, after_sequence: int, run_id: str) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "after_sequence": after_sequence,
            "task_id": run_id,
            "limit": 100000,
        }
    )
    payload = request_json(f"{api}/experiment/events?{query}")
    return list(payload.get("events", []))


def filtered_task_items(values: Any, run_id: str) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [item for item in values if item.get("task_id") == run_id]


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    policy_config = load_policy_config(args.policy)
    initial = cluster.get_node(args.initial_node_id)
    initial_api = base_url(initial, cluster.api_port)
    definition_path = Path(args.definition)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = args.experiment_id or f"counter-{timestamp}-{uuid4().hex[:8]}"
    bundle_dir = Path(args.runs_root) / experiment_id
    if bundle_dir.exists():
        raise FileExistsError(f"Experiment bundle already exists: {bundle_dir}")
    (bundle_dir / "raw").mkdir(parents=True)

    started_at_utc = utc_now_iso()
    event_baselines: dict[str, int] = {}
    initial_health: dict[str, dict[str, Any]] = {}
    carbon_accounting: dict[str, str] | None = None

    print("== Verify identical Git commit across seven nodes ==")
    cluster_git = cluster_git_snapshot(
        cluster.nodes,
        local_node_id=initial.id,
        ssh_user=args.ssh_user,
        remote_repo=args.remote_repo,
    )
    for node_id, snapshot in cluster_git.items():
        print(f"[git] {node_id:16} {snapshot.get('commit', '')[:12]}")

    print("== Verify experiment event streams ==")
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health")
        event_status = request_json(f"{api}/experiment/events/status")
        active_policy = request_json(f"{api}/policy")
        if health.get("node_id") != node.id or event_status.get("node_id") != node.id:
            raise RuntimeError(f"Node identity mismatch for {node.id}")
        if int(health.get("epoch_seconds", -1)) != cluster.epoch_seconds:
            raise RuntimeError(
                f"{node.id} epoch mismatch: active={health.get('epoch_seconds')} "
                f"expected={cluster.epoch_seconds}"
            )
        expected_weights = policy_config.weights.model_dump()
        if active_policy.get("baseline_weights") != expected_weights:
            raise RuntimeError(
                f"{node.id} policy mismatch: active={active_policy.get('baseline_weights')} "
                f"expected={expected_weights}"
            )
        node_carbon = {
            "metric": str(event_status.get("carbon_metric", "")),
            "column": str(event_status.get("carbon_column", "")),
        }
        if not node_carbon["metric"] or not node_carbon["column"]:
            raise RuntimeError(f"{node.id} does not report its carbon metric")
        if health.get("carbon_metric") != node_carbon["metric"]:
            raise RuntimeError(f"{node.id} health/event carbon metric mismatch")
        if carbon_accounting is None:
            carbon_accounting = node_carbon
        elif node_carbon != carbon_accounting:
            raise RuntimeError(
                f"Carbon accounting mismatch: {node.id}={node_carbon} "
                f"expected={carbon_accounting}"
            )
        event_baselines[node.id] = int(event_status.get("last_sequence", 0))
        initial_health[node.id] = health
        print(
            f"[OK] {node.id:16} event_sequence={event_baselines[node.id]} "
            f"carbon={node_carbon['metric']}"
        )

    created = request_json(
        f"{initial_api}/task-definitions",
        method="POST",
        payload=definition,
    )
    definition_id = created["definition_id"]
    revision = created["revision"]
    print(f"definition={definition_id}@{revision}")

    print("== Wait for definition anti-entropy convergence ==")
    convergence_deadline = time.monotonic() + min(90.0, args.timeout_seconds)
    pending = {node.id for node in cluster.nodes}
    while pending and time.monotonic() < convergence_deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            api = base_url(node, cluster.api_port)
            value = try_json(f"{api}/task-definitions/{definition_id}?revision={revision}")
            if isinstance(value, dict) and value.get("digest") == created.get("digest"):
                pending.remove(node.id)
                print(f"[catalog] {node.id:16} converged")
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"Definition did not converge to: {sorted(pending)}")

    run_request = {
        "definition_id": definition_id,
        "revision": revision,
        "initial_owner_node_id": initial.id,
        "idempotency_key": f"experiment-{experiment_id}",
        "auto_start": True,
        "labels": {
            "purpose": "experiment-infrastructure-stage1",
            "experiment_id": experiment_id,
        },
    }
    run_view = request_json(
        f"{initial_api}/task-runs",
        method="POST",
        payload=run_request,
    )
    run_id = run_view["run"]["run_id"]
    print(f"run_id={run_id} initial_owner={initial.id}")

    observations: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    deadline = time.monotonic() + args.timeout_seconds
    last_print: tuple[Any, ...] | None = None

    print("== Observe task ==")
    while time.monotonic() < deadline:
        states: list[tuple[str, dict[str, Any]]] = []
        observed_at = utc_now_iso()
        for node in cluster.nodes:
            api = base_url(node, cluster.api_port)
            state = task_state(api, run_id)
            telemetry = try_json(f"{api}/telemetry/tasks/{run_id}")
            policy_state = try_json(f"{api}/policy/tasks/{run_id}")
            if state is not None:
                states.append((node.id, state))
            observations.append(
                {
                    "observed_at_utc": observed_at,
                    "node_id": node.id,
                    "state": state,
                    "telemetry": telemetry,
                    "policy": policy_state,
                }
            )

        if states:
            status_rank = {
                "failed": 5,
                "completed": 5,
                "running": 4,
                "paused": 3,
                "migrating": 3,
                "stopped": 2,
                "remote": 1,
            }
            _, newest = max(
                states,
                key=lambda item: (
                    int(item[1].get("generation", 0)),
                    status_rank.get(str(item[1].get("status")), 0),
                    item[0] == item[1].get("owner_node_id"),
                ),
            )
            line = (
                newest.get("owner_node_id"),
                newest.get("generation"),
                newest.get("status"),
                newest.get("progress_completed_units"),
            )
            if line != last_print:
                print(
                    f"[state] owner={line[0]} generation={line[1]} "
                    f"status={line[2]} progress={line[3]}"
                )
                last_print = line
            if newest.get("status") in {"completed", "failed"}:
                final_state = newest
                break
        time.sleep(args.poll_seconds)

    if final_state is None:
        raise RuntimeError(f"Task did not reach a terminal state within {args.timeout_seconds:g}s")

    finished_at_utc = utc_now_iso()
    node_evidence: dict[str, dict[str, Any]] = {}
    print("== Collect per-node evidence ==")
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        evidence = {
            "node_id": node.id,
            "health": request_json(f"{api}/health"),
            "event_start_sequence": event_baselines[node.id],
            "event_end_status": request_json(f"{api}/experiment/events/status"),
            "events": query_events(api, event_baselines[node.id], run_id),
            "bids": filtered_task_items(request_json(f"{api}/bids"), run_id),
            "migrations": filtered_task_items(request_json(f"{api}/migrations"), run_id),
            "ownership_snapshot": request_json(f"{api}/ownership/snapshot"),
            "task_state": task_state(api, run_id),
            "task_run": try_json(f"{api}/task-runs/{run_id}"),
            "task_telemetry": try_json(f"{api}/telemetry/tasks/{run_id}"),
            "edge_telemetry": request_json(f"{api}/telemetry/edges"),
            "policy_state": try_json(f"{api}/policy/tasks/{run_id}"),
        }
        node_evidence[node.id] = evidence
        print(
            f"[collect] {node.id:16} events={len(evidence['events'])} "
            f"bids={len(evidence['bids'])} migrations={len(evidence['migrations'])}"
        )

    dataset_records = {
        node.id: file_record(Path("datasets") / node.dataset_file)
        for node in cluster.nodes
    }
    manifest = {
        "format_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "created_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "exact_tag": git_value("describe", "--tags", "--exact-match", "HEAD"),
            "status_porcelain": git_value("status", "--porcelain"),
            "nodes": cluster_git,
        },
        "inputs": {
            "cluster": file_record(args.cluster),
            "policy": file_record(args.policy),
            "definition": file_record(args.definition),
            "datasets": dataset_records,
        },
        "carbon_accounting": carbon_accounting,
        "cluster": {
            "node_ids": [node.id for node in cluster.nodes],
            "api_port": cluster.api_port,
            "epoch_seconds": cluster.epoch_seconds,
            "initial_node_id": initial.id,
            "initial_health": initial_health,
        },
        "submission": {
            "definition_id": definition_id,
            "revision": revision,
            "definition_digest": created.get("digest"),
            "run_request": run_request,
        },
        "event_start_sequences": event_baselines,
        "requirements": {"require_migration": args.require_migration},
    }
    write_json(bundle_dir / "manifest.json", manifest)

    summary = materialize_bundle_tables(
        bundle_dir,
        node_evidence=node_evidence,
        observations=observations,
        run_id=run_id,
        final_state=final_state,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
    )
    write_checksums(bundle_dir)

    if final_state.get("status") != "completed":
        raise RuntimeError(f"Recorded workload failed: {final_state.get('last_error')}")
    if summary["decision_count"] < 1:
        raise RuntimeError("No structured scheduler decisions were recorded")
    if args.require_migration and summary["successful_migration_count"] < 1:
        raise RuntimeError("Experiment required a migration, but none completed")

    print("\nRECORDED COUNTER EXPERIMENT PASSED")
    print(f"experiment_id: {experiment_id}")
    print(f"run_id: {run_id}")
    print(f"bundle: {bundle_dir}")
    print(f"decisions: {summary['decision_count']}")
    print(f"migrations: {summary['successful_migration_count']}")
    print(f"owners: {' -> '.join(summary['owners_observed'])}")
    print(f"carbon_g: {summary['final_accounting'].get('accumulated_carbon_grams')}")
    print(f"cost_usd: {summary['final_accounting'].get('accumulated_cost_usd')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
