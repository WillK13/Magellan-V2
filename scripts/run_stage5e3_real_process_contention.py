#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
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
from magellan.experiments.stage4d2 import read_resource_model
from magellan.experiments.stage5c import (
    active_task_ids,
    is_resource_contention_rejection,
    is_successful_bid_status,
    ownership_converged,
)
from magellan.experiments.stage5e3 import (
    BENCHMARK_CLASS_ID,
    STAGE5E3_DESTINATION_ID,
    STAGE5E3_SOURCE_IDS,
    stage5e3_passes,
)
from magellan.experiments.workload_population import benchmark_definition
from magellan.submission.models import TaskDefinitionSubmission


TRACE_TIME_UTC = "2024-08-20T12:00:00Z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5E.3 real destination-side measured-resource contention "
            "on the seven-node cluster."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--stage5c-bundle", required=True)
    parser.add_argument("--stage5e2-bundle", required=True)
    parser.add_argument("--stage4d1-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--trace-time-utc", default=TRACE_TIME_UTC)
    parser.add_argument("--convergence-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--checkpoint-wait-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    parser.add_argument("--benchmark-iterations", type=int, default=1_000_000)
    parser.add_argument("--ssh-user", default=os.getenv("MAGELLAN_SSH_USER", "WILL"))
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--witness-timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 240.0,
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


def event_query(
    api: str,
    after_sequence: int,
    task_id: str,
) -> list[dict[str, Any]]:
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


def ps_snapshot(
    node: Any,
    *,
    local_node_id: str,
    ssh_user: str,
    timeout: float = 15.0,
) -> str:
    command = ["ps", "-eo", "sid=,pid=,stat=,rss=,pcpu="]
    if node.id == local_node_id:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
    else:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=5",
                f"{ssh_user}@{node.internal_ip}",
                "ps -eo sid=,pid=,stat=,rss=,pcpu=",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"direct process witness failed on {node.id}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def parse_ps_sessions(raw: str) -> dict[int, dict[str, Any]]:
    sessions: dict[int, dict[str, Any]] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            sid = int(fields[0])
            pid = int(fields[1])
            rss_kb = float(fields[3])
            cpu_percent = float(fields[4])
        except (TypeError, ValueError):
            continue
        state = fields[2].strip().upper()
        record = sessions.setdefault(
            sid,
            {
                "process_count": 0,
                "process_state": None,
                "memory_rss_mb": 0.0,
                "cpu_utilization_percent": 0.0,
            },
        )
        record["process_count"] += 1
        record["memory_rss_mb"] += max(0.0, rss_kb) / 1024.0
        record["cpu_utilization_percent"] += max(0.0, cpu_percent)
        if pid == sid:
            record["process_state"] = state
    return sessions


def capture_live_witness(
    *,
    targets: list[dict[str, Any]],
    cluster: Any,
    local_node_id: str,
    ssh_user: str,
) -> list[dict[str, Any]]:
    node_by_id = {node.id: node for node in cluster.nodes}
    state_by_task: dict[str, dict[str, Any]] = {}
    for target in targets:
        state = task_state(target["api"], target["task_id"]) or {}
        state_by_task[str(target["task_id"])] = state

    node_ids = sorted({str(target["node_id"]) for target in targets})
    ps_by_node: dict[str, dict[int, dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(node_ids))) as pool:
        futures = {
            pool.submit(
                ps_snapshot,
                node_by_id[node_id],
                local_node_id=local_node_id,
                ssh_user=ssh_user,
            ): node_id
            for node_id in node_ids
        }
        for future, node_id in futures.items():
            ps_by_node[node_id] = parse_ps_sessions(future.result())

    sampled_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for target in targets:
        state = state_by_task[str(target["task_id"])]
        pid = int(state.get("pid") or 0)
        session = ps_by_node.get(str(target["node_id"]), {}).get(pid, {}) if pid else {}
        process_count = int(session.get("process_count") or 0)
        process_state = str(session.get("process_state") or "").upper()
        rss_mb = float(session.get("memory_rss_mb") or 0.0)
        live = (
            str(state.get("status") or "") == "running"
            and pid > 0
            and process_count >= 1
            and process_state[:1] not in {"Z", "X"}
            and rss_mb > 0.0
        )
        rows.append(
            {
                "sampled_at_utc": sampled_at,
                "phase": target["phase"],
                "task_id": target["task_id"],
                "role": target["role"],
                "origin_node_id": target["origin_node_id"],
                "node_id": target["node_id"],
                "class_id": BENCHMARK_CLASS_ID,
                "status": state.get("status"),
                "progress_completed_units": state.get("progress_completed_units"),
                "pid": pid or None,
                "process_count": process_count,
                "process_state": process_state or None,
                "cpu_utilization_percent": session.get("cpu_utilization_percent"),
                "memory_rss_mb": rss_mb,
                "live": live,
            }
        )
    return rows


def wait_live_witness(
    *,
    targets: list[dict[str, Any]],
    cluster: Any,
    local_node_id: str,
    ssh_user: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = capture_live_witness(
            targets=targets,
            cluster=cluster,
            local_node_id=local_node_id,
            ssh_user=ssh_user,
        )
        last_rows = rows
        if len(rows) == len(targets) and all(bool(row.get("live")) for row in rows):
            return rows
        time.sleep(0.5)
    missing = [
        f"{row.get('node_id')}:{row.get('task_id')} status={row.get('status')} "
        f"pid={row.get('pid')} proc={row.get('process_count')} "
        f"state={row.get('process_state')} rss={row.get('memory_rss_mb')}"
        for row in last_rows if not bool(row.get("live"))
    ]
    raise TimeoutError(
        "Stage 5E.3 did not produce the required all-live physical witness: "
        + "; ".join(missing)
    )


def collect_resource_rows(
    *,
    cluster: Any,
    final_rows: list[dict[str, Any]],
    benchmark_cpu_cores: float,
    benchmark_memory_mb: int,
) -> list[dict[str, Any]]:
    owner_counts: dict[str, int] = {node.id: 0 for node in cluster.nodes}
    for row in final_rows:
        owner = str(row.get("final_owner_node_id"))
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
    rows: list[dict[str, Any]] = []
    for node in cluster.nodes:
        health = request_json(f"{base_url(node, cluster.api_port)}/health")
        count = owner_counts.get(node.id, 0)
        expected_cpu = count * benchmark_cpu_cores
        expected_memory = count * benchmark_memory_mb
        actual_cpu = float(health.get("reserved_cpu_cores") or 0.0)
        actual_memory = int(float(health.get("reserved_memory_mb") or 0))
        busy = float(health.get("resource_busy_fraction") or 0.0)
        rows.append(
            {
                "node_id": node.id,
                "expected_owned_task_count": count,
                "actual_owned_task_count": int(health.get("owned_task_count") or 0),
                "expected_reserved_cpu_cores": expected_cpu,
                "reserved_cpu_cores": actual_cpu,
                "expected_reserved_memory_mb": expected_memory,
                "reserved_memory_mb": actual_memory,
                "resource_busy_fraction": busy,
                "available_cpu_cores": health.get("available_cpu_cores"),
                "reservation_matches_expected": (
                    abs(actual_cpu - expected_cpu) <= 1e-6
                    and actual_memory == expected_memory
                ),
                "capacity_respected": busy <= 1.0 + 1e-9,
            }
        )
    return rows


def definition_payload(
    comparison_id: str,
    node_ids: list[str],
    resource_request: dict[str, Any],
    *,
    iterations: int,
) -> dict[str, Any]:
    payload = benchmark_definition(
        definition_id=f"stage5e3-real-benchmark-{comparison_id}",
        benchmark="json",
        size="medium",
        seed=53,
        iterations=iterations,
        node_ids=node_ids,
    )
    payload["profile"]["resource_request"] = resource_request
    payload["profile"]["estimated_remaining_seconds"] = 86400
    payload["profile"]["cost_cap_usd"] = 100.0
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


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
                if (
                    isinstance(value, dict)
                    and value.get("run", {}).get("run_id") == run_id
                ):
                    pending.remove(key)
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(
            f"Task runs did not converge; remaining={sorted(pending)[:12]}"
        )


def wait_ownership(cluster, run_ids: list[str], timeout: float):
    deadline = time.monotonic() + timeout
    last_snapshots: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        last_snapshots = {
            node.id: request_json(
                f"{base_url(node, cluster.api_port)}/ownership/snapshot"
            )
            for node in cluster.nodes
        }
        ok, rows = ownership_converged(last_snapshots, run_ids)
        if ok:
            return True, rows, last_snapshots
        time.sleep(2)
    ok, rows = ownership_converged(last_snapshots, run_ids)
    return ok, rows, last_snapshots


def submit_run(
    *,
    api: str,
    definition_id: str,
    revision: int,
    owner_node_id: str,
    comparison_id: str,
    role: str,
) -> str:
    view = request_json(
        f"{api}/task-runs",
        method="POST",
        payload={
            "definition_id": definition_id,
            "revision": revision,
            "initial_owner_node_id": owner_node_id,
            "idempotency_key": f"{comparison_id}:{role}:{owner_node_id}",
            "auto_start": True,
            "labels": {
                "purpose": "stage5e3-real-contention",
                "comparison_id": comparison_id,
                "scheduler_mode": "operator_only",
                "origin_node_id": owner_node_id,
                "stage5e3_role": role,
            },
        },
    )
    return str(view["run"]["run_id"])


def main() -> int:
    args = parse_args()
    s5a_path = Path(args.stage5a_bundle)
    s5c_path = Path(args.stage5c_bundle)
    s5e2_path = Path(args.stage5e2_bundle)
    d41_path = Path(args.stage4d1_bundle)

    s5a = require_bundle(s5a_path, "Stage 5A")
    s5c = require_bundle(s5c_path, "Stage 5C")
    s5e2 = require_bundle(s5e2_path, "Stage 5E.2")
    require_bundle(d41_path, "Stage 4D.1")

    cluster = load_cluster_config(args.cluster)
    node_by_id = {node.id: node for node in cluster.nodes}
    if not set(STAGE5E3_SOURCE_IDS).issubset(node_by_id):
        raise RuntimeError("Stage 5E.3 source nodes are not present in cluster")
    destination = node_by_id[STAGE5E3_DESTINATION_ID]

    target_sha = str(s5a["target_git_sha"])
    if local_git_sha() != target_sha:
        raise RuntimeError(
            "Stage 5E.3 must run from the exact SHA frozen by its current Stage 5A bundle: "
            f"local={local_git_sha()} stage5a={target_sha}"
        )
    print(
        "[provenance] Stage 5E.3 builds on frozen Stage 5C contention semantics "
        "and frozen Stage 5E.2 physical workload evidence"
    )

    capacities, requests = read_resource_model(d41_path)
    benchmark_request = requests[BENCHMARK_CLASS_ID]
    production_capacity = destination.resources
    if production_capacity.cpu_cores is None:
        raise RuntimeError("Stage 5E.3 requires explicit destination CPU capacity")
    cpu_capacity = float(production_capacity.cpu_cores)
    request_cpu = float(benchmark_request.cpu_cores)
    if request_cpu + request_cpu > cpu_capacity + 1e-9:
        raise RuntimeError(
            "Frozen benchmark request no longer permits resident + one challenger"
        )
    if request_cpu * 3 <= cpu_capacity + 1e-9:
        raise RuntimeError(
            "Frozen benchmark request does not create one-additional-fit contention"
        )

    comparison_id = args.comparison_id or (
        f"stage5e3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 5E.3 real destination-side measured-resource contention ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage5a={s5a_path}")
    print(f"source_stage5c={s5c_path}")
    print(f"source_stage5e2={s5e2_path}")
    print(f"source_stage4d1={d41_path}")
    print(f"git_sha={target_sha}")
    print(f"destination={STAGE5E3_DESTINATION_ID}")
    print(f"sources={','.join(STAGE5E3_SOURCE_IDS)}")
    print(f"controlled_trace_time={args.trace_time_utc}")
    print(
        "resource_model="
        f"{BENCHMARK_CLASS_ID} cpu={benchmark_request.cpu_cores:.9f} "
        f"memory={benchmark_request.memory_mb}MB; "
        f"destination_cpu={cpu_capacity:.3f}; "
        "resident + exactly one challenger fits"
    )

    event_baselines: dict[str, int] = {}
    bid_baselines: dict[str, set[str]] = {}
    health_by_node: dict[str, dict[str, Any]] = {}
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health")
        if health.get("deployment_git_sha") != target_sha:
            raise RuntimeError(f"{node.id} daemon SHA drifted from Stage 5A")
        active = active_task_ids(request_json(f"{api}/tasks"))
        if active:
            raise RuntimeError(
                f"Stage 5E.3 requires no pre-existing active tasks; {node.id} has {active}"
            )
        health_by_node[node.id] = health
        event_baselines[node.id] = int(
            request_json(f"{api}/experiment/events/status").get(
                "last_sequence", 0
            )
        )
        bid_baselines[node.id] = {
            str(item["bid_id"])
            for item in request_json(f"{api}/bids")
        }
        print(
            f"[preflight] {node.id:16s} "
            f"sha={target_sha[:12]} active_tasks=0"
        )

    definition = definition_payload(
        comparison_id,
        [node.id for node in cluster.nodes],
        benchmark_request.model_dump(mode="json"),
        iterations=args.benchmark_iterations,
    )
    registration_api = base_url(
        node_by_id[STAGE5E3_SOURCE_IDS[0]], cluster.api_port
    )
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
    print(
        f"[catalog] definition converged: "
        f"{created['definition_id']}@{created['revision']}"
    )

    destination_api = base_url(destination, cluster.api_port)
    resident_id = submit_run(
        api=destination_api,
        definition_id=str(created["definition_id"]),
        revision=int(created["revision"]),
        owner_node_id=STAGE5E3_DESTINATION_ID,
        comparison_id=comparison_id,
        role="resident",
    )
    wait_runs(cluster, [resident_id], args.convergence_timeout_seconds)
    time.sleep(args.checkpoint_wait_seconds)
    resident_state = task_state(destination_api, resident_id)
    if not resident_state or resident_state.get("status") != "running":
        raise RuntimeError(f"Resident task is not running: {resident_state}")
    auction_before = request_json(f"{destination_api}/auction/status")
    print(
        f"[resident] task={resident_id} "
        f"reserved_cpu={auction_before.get('reserved_cpu_cores')} "
        f"available_cpu={auction_before.get('available_cpu_cores')}"
    )

    source_rows: list[dict[str, Any]] = []
    run_ids: list[str] = []
    run_origin: dict[str, str] = {}
    for source_id in STAGE5E3_SOURCE_IDS:
        api = base_url(node_by_id[source_id], cluster.api_port)
        run_id = submit_run(
            api=api,
            definition_id=str(created["definition_id"]),
            revision=int(created["revision"]),
            owner_node_id=source_id,
            comparison_id=comparison_id,
            role="challenger",
        )
        run_ids.append(run_id)
        run_origin[run_id] = source_id
        source_rows.append(
            {
                "source_node_id": source_id,
                "task_id": run_id,
                "daemon_git_sha": health_by_node[source_id][
                    "deployment_git_sha"
                ],
                "trigger_ok": False,
                "trigger_error": "",
                "trigger_return_owner": "",
                "trigger_return_status": "",
            }
        )
        print(f"[submit] {source_id:16s} challenger={run_id}")

    all_ids = [resident_id, *run_ids]
    wait_runs(cluster, all_ids, args.convergence_timeout_seconds)
    print("[catalog] resident + four real benchmark challengers converged to all seven nodes")
    print(
        f"[checkpoint] allowing {args.checkpoint_wait_seconds:g}s "
        "for challenger checkpoints"
    )
    time.sleep(args.checkpoint_wait_seconds)

    for run_id in run_ids:
        source_id = run_origin[run_id]
        state = task_state(
            base_url(node_by_id[source_id], cluster.api_port),
            run_id,
        )
        if not state or state.get("status") != "running":
            raise RuntimeError(
                f"{source_id}/{run_id} is not running before evaluation: {state}"
            )

    pre_targets = [
        {
            "phase": "pre_auction",
            "task_id": resident_id,
            "role": "resident",
            "origin_node_id": STAGE5E3_DESTINATION_ID,
            "node_id": STAGE5E3_DESTINATION_ID,
            "api": destination_api,
        },
        *(
            {
                "phase": "pre_auction",
                "task_id": run_id,
                "role": "challenger",
                "origin_node_id": run_origin[run_id],
                "node_id": run_origin[run_id],
                "api": base_url(node_by_id[run_origin[run_id]], cluster.api_port),
            }
            for run_id in run_ids
        ),
    ]
    print("[witness-pre] requiring all five real benchmark processes live")
    pre_witness_rows = wait_live_witness(
        targets=pre_targets,
        cluster=cluster,
        local_node_id=args.local_node_id,
        ssh_user=args.ssh_user,
        timeout_seconds=args.witness_timeout_seconds,
    )
    print("[witness-pre] 5/5 real benchmark processes live before arbitration")

    print(
        "\n[evaluate] triggering four source daemons concurrently; "
        "all must independently select Ethiopia",
        flush=True,
    )
    by_source_row = {
        row["source_node_id"]: row for row in source_rows
    }

    def trigger(source_id: str, run_id: str):
        api = base_url(node_by_id[source_id], cluster.api_port)
        query = urlencode({"trace_time_utc": args.trace_time_utc})
        return request_json(
            f"{api}/tasks/{run_id}/evaluate?{query}",
            method="POST",
            timeout=300.0,
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
                row["trigger_return_owner"] = state.get(
                    "owner_node_id", ""
                )
                row["trigger_return_status"] = state.get("status", "")
                print(
                    f"  [done] {source_id:16s} task={run_id} "
                    f"owner={row['trigger_return_owner']} "
                    f"status={row['trigger_return_status']}"
                )
            except Exception as exc:
                row["trigger_error"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"  [fail] {source_id:16s} task={run_id} "
                    f"{row['trigger_error']}"
                )

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
            events = event_query(
                api,
                event_baselines[node.id],
                run_id,
            )
            node_events.extend(events)
            for event in events:
                event_rows.append(event)
                if event.get("event_type") == "scheduler_decision":
                    selected = (
                        event.get("payload", {})
                        .get("decision", {})
                        .get("selected", {})
                    )
                    decision_rows.append(
                        {
                            "node_id": node.id,
                            "task_id": run_id,
                            "origin_node_id": run_origin[run_id],
                            "selected_action": selected.get("action"),
                            "selected_destination_node_id": (
                                selected.get("destination_node_id") or ""
                            ),
                            "selected_score": selected.get("score"),
                            "trace_time_utc": event.get("trace_time_utc"),
                            "sequence": event.get("sequence"),
                        }
                    )
                elif event.get("event_type") in {
                    "migration_completed",
                    "migration_failed",
                }:
                    payload = event.get("payload", {})
                    migration_rows.append(
                        {
                            "node_id": node.id,
                            "task_id": run_id,
                            "status": (
                                "completed"
                                if event.get("event_type")
                                == "migration_completed"
                                else "failed"
                            ),
                            "source_node_id": payload.get(
                                "source_node_id", ""
                            ),
                            "destination_node_id": payload.get(
                                "destination_node_id", ""
                            ),
                            "migration_id": payload.get(
                                "migration_id", ""
                            ),
                            "bid_id": payload.get("bid_id", ""),
                            "total_downtime_seconds": payload.get(
                                "total_downtime_seconds", ""
                            ),
                            "error": payload.get("error", ""),
                        }
                    )

        new_bids = [
            item
            for item in request_json(f"{api}/bids")
            if str(item.get("bid_id"))
            not in bid_baselines[node.id]
            and str(item.get("task_id")) in run_origin
        ]
        for item in new_bids:
            metrics = item.get("auction_metrics") or {}
            bid_rows.append(
                {
                    "reporting_node_id": node.id,
                    "bid_id": item.get("bid_id"),
                    "task_id": item.get("task_id"),
                    "source_node_id": item.get("source_node_id"),
                    "destination_node_id": item.get(
                        "destination_node_id"
                    ),
                    "status": item.get("status"),
                    "candidate_score": (
                        item.get("candidate", {}).get("score")
                    ),
                    "decision_reason": item.get("decision_reason"),
                    "auction_strategy": item.get("auction_strategy"),
                    "auction_rank": item.get("auction_rank"),
                    "resource_fit": item.get("resource_fit"),
                    "credit_before": item.get("auction_credit_before"),
                    "credit_after": item.get("auction_credit_after"),
                    "requested_cpu_cores": metrics.get(
                        "requested_cpu_cores"
                    ),
                    "requested_memory_mb": metrics.get(
                        "requested_memory_mb"
                    ),
                }
            )
        raw_node_evidence.append(
            {
                "node_id": node.id,
                "health": request_json(f"{api}/health"),
                "events": node_events,
                "new_bids": new_bids,
                "ownership_snapshot": request_json(
                    f"{api}/ownership/snapshot"
                ),
            }
        )
        print(
            f"[collect] {node.id:16s} "
            f"decisions={sum(r['node_id']==node.id for r in decision_rows)} "
            f"new_bids={len(new_bids)}"
        )

    ownership_ok, ownership_rows, snapshots = wait_ownership(
        cluster,
        all_ids,
        args.convergence_timeout_seconds,
    )

    final_rows: list[dict[str, Any]] = []
    for run_id in all_ids:
        updates = []
        for snapshot in snapshots.values():
            for update in snapshot.get("updates", []):
                if update.get("task_id") == run_id:
                    updates.append(update)
        newest = max(
            updates,
            key=lambda item: int(item.get("generation", 0)),
        )
        final_rows.append(
            {
                "task_id": run_id,
                "role": (
                    "resident" if run_id == resident_id else "challenger"
                ),
                "origin_node_id": (
                    STAGE5E3_DESTINATION_ID
                    if run_id == resident_id
                    else run_origin[run_id]
                ),
                "final_owner_node_id": newest.get("owner_node_id"),
                "generation": newest.get("generation"),
                "status": newest.get("status"),
                "last_migration_id": (
                    newest.get("last_migration_id") or ""
                ),
            }
        )

    final_by_task = {str(row["task_id"]): row for row in final_rows}
    post_targets = []
    for run_id in all_ids:
        final_owner = str(final_by_task[run_id]["final_owner_node_id"])
        post_targets.append(
            {
                "phase": "post_auction",
                "task_id": run_id,
                "role": "resident" if run_id == resident_id else "challenger",
                "origin_node_id": (
                    STAGE5E3_DESTINATION_ID if run_id == resident_id else run_origin[run_id]
                ),
                "node_id": final_owner,
                "api": base_url(node_by_id[final_owner], cluster.api_port),
            }
        )
    print("[witness-post] requiring all five real benchmark processes live after arbitration")
    post_witness_rows = wait_live_witness(
        targets=post_targets,
        cluster=cluster,
        local_node_id=args.local_node_id,
        ssh_user=args.ssh_user,
        timeout_seconds=args.witness_timeout_seconds,
    )
    destination_live = sum(
        str(row.get("node_id")) == STAGE5E3_DESTINATION_ID
        and bool(row.get("live"))
        for row in post_witness_rows
    )
    print(
        f"[witness-post] 5/5 live after arbitration; "
        f"destination_live={destination_live}/2"
    )

    resource_rows = collect_resource_rows(
        cluster=cluster,
        final_rows=final_rows,
        benchmark_cpu_cores=request_cpu,
        benchmark_memory_mb=int(benchmark_request.memory_mb),
    )
    destination_resource = next(
        row for row in resource_rows if row["node_id"] == STAGE5E3_DESTINATION_ID
    )
    print(
        "[resources] Ethiopia "
        f"owned={destination_resource['actual_owned_task_count']} "
        f"reserved_cpu={destination_resource['reserved_cpu_cores']:.6f} "
        f"available_cpu={destination_resource['available_cpu_cores']} "
        f"match={destination_resource['reservation_matches_expected']}"
    )

    auction_after = request_json(f"{destination_api}/auction/status")
    challenge_bids = [
        row for row in bid_rows
        if str(row.get("task_id")) in run_origin
        and str(row.get("destination_node_id"))
        == STAGE5E3_DESTINATION_ID
    ]
    successful_bids = [
        row for row in challenge_bids
        if is_successful_bid_status(str(row.get("status")))
    ]
    rejected_bids = [
        row for row in challenge_bids
        if str(row.get("status")) == "rejected"
    ]
    contention_rejections = [
        row for row in rejected_bids
        if is_resource_contention_rejection(row)
    ]

    passed = stage5e3_passes(
        source_rows=source_rows,
        decision_rows=decision_rows,
        bid_rows=bid_rows,
        migration_rows=migration_rows,
        final_rows=final_rows,
        ownership_ok=ownership_ok,
        resident_task_id=resident_id,
        benchmark_cpu_cores=request_cpu,
        benchmark_memory_mb=int(benchmark_request.memory_mb),
        capacity_cpu_cores=cpu_capacity,
        expected_git_sha=target_sha,
        pre_witness_rows=pre_witness_rows,
        post_witness_rows=post_witness_rows,
        resource_rows=resource_rows,
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage5a_bundle": str(s5a_path),
        "source_stage5c_bundle": str(s5c_path),
        "source_stage5e2_bundle": str(s5e2_path),
        "source_stage4d1_bundle": str(d41_path),
        "git_sha": target_sha,
        "controlled_trace_time_utc": args.trace_time_utc,
        "destination_node_id": STAGE5E3_DESTINATION_ID,
        "source_node_ids": list(STAGE5E3_SOURCE_IDS),
        "resident_task_id": resident_id,
        "benchmark_class_id": BENCHMARK_CLASS_ID,
        "benchmark_cpu_cores": request_cpu,
        "benchmark_memory_mb": int(
            benchmark_request.memory_mb
        ),
        "destination_cpu_capacity": cpu_capacity,
        "trigger_success_count": sum(
            bool(row["trigger_ok"]) for row in source_rows
        ),
        "scheduler_decision_count": len(decision_rows),
        "challenge_bid_count": len(challenge_bids),
        "successful_bid_outcome_count": len(successful_bids),
        "rejected_bid_count": len(rejected_bids),
        "resource_contention_rejection_count": len(
            contention_rejections
        ),
        "successful_migration_count": sum(
            row["status"] == "completed"
            for row in migration_rows
        ),
        "failed_migration_count": sum(
            row["status"] == "failed"
            for row in migration_rows
        ),
        "ownership_converged": ownership_ok,
        "pre_live_witness_count": sum(bool(row.get("live")) for row in pre_witness_rows),
        "post_live_witness_count": sum(bool(row.get("live")) for row in post_witness_rows),
        "post_destination_live_count": destination_live,
        "resource_match_node_count": sum(
            bool(row.get("reservation_matches_expected")) for row in resource_rows
        ),
        "capacity_respected_node_count": sum(
            bool(row.get("capacity_respected")) for row in resource_rows
        ),
        "destination_final_owned_task_count": destination_resource.get(
            "actual_owned_task_count"
        ),
        "destination_final_reserved_cpu": destination_resource.get(
            "reserved_cpu_cores"
        ),
        "destination_final_available_cpu": destination_resource.get(
            "available_cpu_cores"
        ),
        "auction_before_reserved_cpu": auction_before.get(
            "reserved_cpu_cores"
        ),
        "auction_before_available_cpu": auction_before.get(
            "available_cpu_cores"
        ),
        "auction_after_reserved_cpu": auction_after.get(
            "reserved_cpu_cores"
        ),
        "auction_after_available_cpu": auction_after.get(
            "available_cpu_cores"
        ),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": (
            "stage5e3_real_destination_measured_resource_contention"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "execution": (
                "One real benchmark-json-medium process is kept resident on Ethiopia. "
                "Four source VMs each run the same real checkpointable benchmark and "
                "independently evaluate migration; production scoring must select Ethiopia. "
                "All bids are handled by Ethiopia's actual BidArbiter."
            ),
            "resource_grounding": (
                "The resident and all four challengers are the actual "
                "benchmark-json-medium workload used in Stage 4A/5E.2, while admission "
                "uses the exact frozen Stage 4D.1 p95 resource request. Direct process "
                "witnesses are captured before and after arbitration."
            ),
            "capacity_shape": (
                "Ethiopia has two configured CPU cores. One frozen benchmark "
                "request remains resident, leaving enough declared capacity "
                "for exactly one additional benchmark request but not two."
            ),
            "auction_window": (
                "All four source evaluations are triggered concurrently. "
                "The production destination arbiter's configured bid window "
                "collects competing bids before applying lowest_score ranking "
                "and ResourceLedger admission."
            ),
            "pass_condition": (
                "PASS requires five real benchmark processes live before arbitration; "
                "four correct source-daemon decisions to Ethiopia; four destination bids; "
                "exactly one successful migration and three resource-contention rejections; "
                "five real processes live afterward with exactly two on Ethiopia; exact "
                "ResourceLedger ownership/reservation agreement; zero failed migrations; "
                "and converged ownership across all seven nodes."
            ),
            "scope": (
                "Stage 5E.3 is the physical analogue of Stage 5C: contention is created "
                "by real benchmark processes rather than counters that only declare measured "
                "resource demand. It validates admission/rejection and physical co-location; "
                "broader heterogeneous policy comparison belongs to Stage 5E.4."
            ),
        },
    }

    write_csv(
        root / "sources.csv",
        source_rows,
        list(source_rows[0].keys()),
    )
    write_csv(
        root / "decisions.csv",
        decision_rows,
        list(decision_rows[0].keys())
        if decision_rows else ["node_id"],
    )
    write_csv(
        root / "bids.csv",
        bid_rows,
        list(bid_rows[0].keys())
        if bid_rows else ["reporting_node_id"],
    )
    write_csv(
        root / "migrations.csv",
        migration_rows,
        list(migration_rows[0].keys())
        if migration_rows else ["node_id"],
    )
    write_csv(
        root / "ownership.csv",
        ownership_rows,
        list(ownership_rows[0].keys()),
    )
    write_csv(
        root / "final_tasks.csv",
        final_rows,
        list(final_rows[0].keys()),
    )
    write_csv(
        root / "pre_live_witness.csv",
        pre_witness_rows,
        list(pre_witness_rows[0].keys()),
    )
    write_csv(
        root / "post_live_witness.csv",
        post_witness_rows,
        list(post_witness_rows[0].keys()),
    )
    write_csv(
        root / "resource_ledger.csv",
        resource_rows,
        list(resource_rows[0].keys()),
    )
    write_json(root / "auction_before.json", auction_before)
    write_json(root / "auction_after.json", auction_after)
    write_jsonl(root / "events.jsonl", event_rows)
    write_jsonl(root / "node_evidence.jsonl", raw_node_evidence)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    print("\n[cleanup] stopping resident/challenger tasks", flush=True)
    cleanup_rows: list[dict[str, Any]] = []
    for row in final_rows:
        owner = str(row["final_owner_node_id"])
        run_id = str(row["task_id"])
        cleanup_ok = False
        cleanup_status = "error"
        cleanup_error = ""
        try:
            value = request_json(
                f"{base_url(node_by_id[owner], cluster.api_port)}/"
                f"tasks/{run_id}/stop",
                method="POST",
                timeout=60.0,
            )
            state = value.get("state", value) if isinstance(value, dict) else {}
            cleanup_status = str(state.get("status") or "stopped")
            cleanup_ok = cleanup_status in {"stopped", "completed", "failed"}
            print(f"  stopped {run_id} on {owner} status={cleanup_status}")
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
            print(f"  cleanup warning {run_id}/{owner}: {cleanup_error}")
        cleanup_rows.append(
            {
                "task_id": run_id,
                "owner_node_id": owner,
                "cleanup_ok": cleanup_ok,
                "cleanup_status": cleanup_status,
                "cleanup_error": cleanup_error,
            }
        )

    cleanup_ok_count = sum(bool(row["cleanup_ok"]) for row in cleanup_rows)
    passed = bool(passed and cleanup_ok_count == len(final_rows))
    summary["cleanup_ok_count"] = cleanup_ok_count
    summary["cleanup_expected_count"] = len(final_rows)
    summary["passed"] = passed
    write_csv(
        root / "cleanup.csv",
        cleanup_rows,
        list(cleanup_rows[0].keys()),
    )
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = (
        "STAGE_5E3_REAL_PROCESS_CONTENTION_PASS"
        if passed else
        "STAGE_5E3_REAL_PROCESS_CONTENTION_FAIL"
    )
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(
        f"destination: {STAGE5E3_DESTINATION_ID} "
        f"resident_cpu={request_cpu:.6f}/{cpu_capacity:.1f}"
    )
    print(
        f"sources: {summary['trigger_success_count']}/"
        f"{len(STAGE5E3_SOURCE_IDS)} evaluated"
    )
    print(f"scheduler_decisions: {len(decision_rows)}")
    print(
        f"challenge_bids: {len(challenge_bids)} "
        f"successful={len(successful_bids)} "
        f"rejected={len(rejected_bids)} "
        f"resource_contention={len(contention_rejections)}"
    )
    print(
        f"migrations: successful={summary['successful_migration_count']} "
        f"failed={summary['failed_migration_count']}"
    )
    print(f"ownership_converged: {ownership_ok}")
    print(f"pre_live_witness: {sum(bool(r.get('live')) for r in pre_witness_rows)}/5")
    print(f"post_live_witness: {sum(bool(r.get('live')) for r in post_witness_rows)}/5")
    print(f"destination_live_after: {destination_live}/2")
    print(
        f"resource_matches: {sum(bool(r.get('reservation_matches_expected')) for r in resource_rows)}/"
        f"{len(resource_rows)}"
    )
    print(f"cleanup: {cleanup_ok_count}/{len(final_rows)}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
