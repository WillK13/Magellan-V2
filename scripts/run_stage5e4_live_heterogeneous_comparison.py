#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
from magellan.experiments.stage5c import active_task_ids, ownership_converged
from magellan.experiments.stage5e2 import (
    BENCHMARK_CLASS_ID,
    DENDRO_CLASS_ID,
    LLM_CLASS_ID,
    physical_definition,
)
from magellan.experiments.stage5e4 import (
    DECISION_CLASSES,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_DECISION_SOURCE_COUNT,
    EXPECTED_LOAD_ID,
    EXPECTED_SEASON,
    EXPECTED_SOURCE_SCENARIO_ID,
    EXPECTED_TASK_COUNT,
    MAGELLAN_POLICY,
    MIN_NODE_SAMPLE_COVERAGE_FRACTION,
    MIN_SAMPLE_COVERAGE_FRACTION,
    POLICIES,
    STATIC_POLICY,
    evaluation_source_groups,
    layout_fingerprint,
    read_stage4d3_u75_layout,
    stage5e4_passes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5E.4 fixed-duration live heterogeneous static-vs-Magellan "
            "comparison using the frozen Stage 4D.3 winter/u75 layout."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--stage5e3-bundle", required=True)
    parser.add_argument("--stage5e2-bundle", required=True)
    parser.add_argument("--stage5e1-bundle", required=True)
    parser.add_argument("--stage4d3-bundle", required=True)
    parser.add_argument("--stage4d1-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--expected-policy", default="config/policy.stage5e4.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--trial-seconds", type=float, default=240.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=15.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--witness-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--convergence-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--benchmark-iterations", type=int, default=1_000_000)
    parser.add_argument("--llm-model", default="experiment-assets/models/distilgpt2")
    parser.add_argument(
        "--dendro-definition",
        default="config/submissions/dendro-bssn-template.json",
    )
    parser.add_argument(
        "--dendro-solver",
        default="/home/WILL/dgr-build/BSSN_GR/bssnSolver",
    )
    parser.add_argument(
        "--dendro-parameter-template",
        default="/home/WILL/q1-magellan-magellan.toml",
    )
    parser.add_argument("--ssh-user", default=os.getenv("MAGELLAN_SSH_USER", "WILL"))
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--remote-repo", default="/home/WILL/Magellan-V2")
    parser.add_argument("--service", default="magellan")
    parser.add_argument("--trace-anchor-max-delta-seconds", type=float, default=300.0)
    return parser.parse_args()


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def try_json(url: str, timeout: float = 10.0) -> Any | None:
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


def require_stage5a_policy(bundle: Path, expected_policy: str) -> None:
    rows_path = bundle / "nodes.csv"
    with rows_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7:
        raise RuntimeError(f"Stage 5A policy provenance expected 7 nodes, found {len(rows)}")
    policies = {str(row.get("effective_policy") or "") for row in rows}
    if policies != {expected_policy}:
        raise RuntimeError(
            "Stage 5E.4 requires Stage 5A deployment with "
            f"{expected_policy}; observed effective policies={sorted(policies)}"
        )


def task_state(api: str, task_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
    value = try_json(f"{api}/task-runs/{task_id}", timeout=timeout)
    if not isinstance(value, dict):
        return None
    state = value.get("state")
    return state if isinstance(state, dict) else None


def event_query(api: str, after_sequence: int, task_id: str) -> list[dict[str, Any]]:
    query = urlencode({"after_sequence": after_sequence, "task_id": task_id})
    value = request_json(f"{api}/experiment/events?{query}", timeout=30.0)
    if isinstance(value, dict):
        return list(value.get("events") or [])
    return list(value or [])


def restart_command(node: Any, *, local_node_id: str, ssh_user: str, service: str) -> None:
    command = f"sudo systemctl restart {shlex.quote(service)}"
    if node.id == local_node_id:
        args = ["bash", "-lc", command]
    else:
        args = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=6", f"{ssh_user}@{node.internal_ip}", command,
        ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"service restart failed on {node.id}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def restart_cluster(
    cluster: Any,
    *,
    expected_sha: str,
    local_node_id: str,
    ssh_user: str,
    service: str,
) -> None:
    print("[reset] restarting all seven daemons to reset the trace clock", flush=True)
    with ThreadPoolExecutor(max_workers=len(cluster.nodes)) as pool:
        futures = {
            pool.submit(
                restart_command,
                node,
                local_node_id=local_node_id,
                ssh_user=ssh_user,
                service=service,
            ): node.id
            for node in cluster.nodes
        }
        for future in as_completed(futures):
            future.result()
    deadline = time.monotonic() + 90.0
    pending = {node.id for node in cluster.nodes}
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            health = try_json(f"{base_url(node, cluster.api_port)}/health", timeout=4.0)
            if (
                isinstance(health, dict)
                and health.get("deployment_git_sha") == expected_sha
                and health.get("auction_strategy") == "lowest_score"
            ):
                pending.remove(node.id)
        if pending:
            time.sleep(1.0)
    if pending:
        raise RuntimeError(f"daemons did not recover after restart: {sorted(pending)}")


def hardened_preflight(cluster: Any, expected_sha: str) -> dict[str, dict[str, Any]]:
    health_by_node: dict[str, dict[str, Any]] = {}
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health", timeout=15.0)
        errors: list[str] = []
        if health.get("deployment_git_sha") != expected_sha:
            errors.append(f"sha={health.get('deployment_git_sha')}")
        if health.get("carbon_metric") != "lifecycle":
            errors.append(f"carbon={health.get('carbon_metric')}")
        if health.get("auction_strategy") != "lowest_score":
            errors.append(f"auction={health.get('auction_strategy')}")
        if float(health.get("runtime_reconcile_seconds") or 0.0) > 5.0 + 1e-9:
            errors.append(f"runtime_reconcile={health.get('runtime_reconcile_seconds')}")
        state_file = str(health.get("telemetry_state_file") or "")
        if "runtime-state-gcp" not in state_file or "runtime-state-gcp-measurement" in state_file:
            errors.append(f"state={state_file}")
        active = active_task_ids(request_json(f"{api}/tasks"))
        if active:
            errors.append(f"active={active}")
        if float(health.get("resource_busy_fraction") or 0.0) > 1e-9:
            errors.append(f"busy={health.get('resource_busy_fraction')}")
        if errors:
            raise RuntimeError(f"{node.id} preflight failed: " + "; ".join(errors))
        health_by_node[node.id] = health
        print(
            f"[preflight] {node.id:16s} sha={expected_sha[:12]} "
            "active=0 busy=0 auction=lowest_score",
            flush=True,
        )
    # Stage 5E.4 is deliberately aligned to the frozen Jan-5 winter trace.
    forecast = request_json(
        f"{base_url(cluster.get_node('boston'), cluster.api_port)}/carbon/forecast/boston"
        "?horizon_seconds=0",
        timeout=15.0,
    )
    generated = str(forecast.get("generated_at_utc") or "")
    if not generated.startswith("2024-01-05T"):
        raise RuntimeError(
            "Stage 5E.4 requires the deployed Stage 5E.4 trace policy anchored on 2024-01-05; "
            f"Boston clock reported {generated!r}"
        )
    return health_by_node


def remote_check(node: Any, *, ssh_user: str, command: str, timeout: float = 60.0) -> None:
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=6", f"{ssh_user}@{node.internal_ip}", command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"asset preflight failed on {node.id}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def workload_asset_preflight(cluster: Any, layout: list[dict[str, str]], args: argparse.Namespace) -> None:
    node_by_id = {node.id: node for node in cluster.nodes}
    llm_nodes = sorted({row["initial_node_id"] for row in layout if row["class_id"] == LLM_CLASS_ID})
    dendro_nodes = sorted({row["initial_node_id"] for row in layout if row["class_id"] == DENDRO_CLASS_ID})
    model_path = f"{args.remote_repo}/{args.llm_model}"
    llm_cmd = (
        f"test -d {shlex.quote(model_path)} && "
        f"{shlex.quote(args.remote_repo)}/.venv/bin/python -c "
        + shlex.quote("import torch, transformers; print(torch.__version__, transformers.__version__)")
    )
    for node_id in llm_nodes:
        remote_check(node_by_id[node_id], ssh_user=args.ssh_user, command=llm_cmd)
        print(f"[asset] {node_id:16s} LLM ready")
    dendro_cmd = (
        f"test -x {shlex.quote(args.dendro_solver)} && "
        f"test -f {shlex.quote(args.dendro_parameter_template)} && "
        "command -v mpirun >/dev/null"
    )
    for node_id in dendro_nodes:
        remote_check(node_by_id[node_id], ssh_user=args.ssh_user, command=dendro_cmd)
        print(f"[asset] {node_id:16s} Dendro ready")


def read_checkpoint_bytes(stage5e1_path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with (stage5e1_path / "cases.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            class_id = str(row.get("class_id") or "")
            raw = row.get("checkpoint_bytes")
            if class_id and raw not in (None, ""):
                values[class_id] = max(values.get(class_id, 0), int(float(raw)))
    missing = set(EXPECTED_CLASS_COUNTS) - set(values)
    if missing:
        raise RuntimeError(f"Stage 5E.1 checkpoint evidence missing {sorted(missing)}")
    return values


def available_bytes(
    node: Any,
    *,
    path: str,
    local_node_id: str,
    ssh_user: str,
) -> int:
    if node.id == local_node_id:
        stats = os.statvfs(path)
        return int(stats.f_bavail * stats.f_frsize)
    code = "import os,sys; s=os.statvfs(sys.argv[1]); print(int(s.f_bavail*s.f_frsize))"
    command = f"python3 -c {shlex.quote(code)} {shlex.quote(path)}"
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            f"{ssh_user}@{node.internal_ip}", command,
        ],
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"disk preflight failed on {node.id}: {result.stderr.strip()}")
    return int(result.stdout.strip())


def disk_preflight(
    cluster: Any,
    *,
    layout: list[dict[str, str]],
    checkpoint_bytes: dict[str, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    planned: dict[str, int] = defaultdict(int)
    for row in layout:
        planned[row["initial_node_id"]] += checkpoint_bytes[row["class_id"]]
    largest = max(checkpoint_bytes.values())
    rows: list[dict[str, Any]] = []
    for node in cluster.nodes:
        free = available_bytes(
            node,
            path=args.remote_repo,
            local_node_id=args.local_node_id,
            ssh_user=args.ssh_user,
        )
        # One GiB operational reserve + current local checkpoints + one largest
        # possible incoming checkpoint. This is sufficient for the u75 trial and
        # catches the ENOSPC failure that Stage 5E.2 exposed.
        required = 1024**3 + int(planned.get(node.id, 0)) + largest
        row = {
            "node_id": node.id,
            "planned_checkpoint_bytes": int(planned.get(node.id, 0)),
            "largest_incoming_checkpoint_bytes": largest,
            "required_free_bytes": required,
            "available_free_bytes": free,
            "sufficient": free >= required,
        }
        rows.append(row)
        print(
            f"[disk] {node.id:16s} free={free/1024**3:.2f}GiB "
            f"required={required/1024**3:.2f}GiB ok={row['sufficient']}"
        )
    bad = [row for row in rows if not row["sufficient"]]
    if bad:
        raise RuntimeError("Stage 5E.4 disk headroom preflight failed")
    return rows


def wait_definition(cluster: Any, definition_id: str, revision: int, digest: str, timeout: float) -> None:
    pending = {node.id for node in cluster.nodes}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            if node.id not in pending:
                continue
            value = try_json(
                f"{base_url(node, cluster.api_port)}/task-definitions/{definition_id}?revision={revision}",
                timeout=5.0,
            )
            if isinstance(value, dict) and value.get("digest") == digest:
                pending.remove(node.id)
        if pending:
            time.sleep(0.5)
    if pending:
        raise RuntimeError(f"definition did not converge: {sorted(pending)}")


def wait_runs(cluster: Any, run_ids: list[str], timeout: float) -> None:
    pending = {(node.id, run_id) for node in cluster.nodes for run_id in run_ids}
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for node in cluster.nodes:
            api = base_url(node, cluster.api_port)
            for run_id in run_ids:
                key = (node.id, run_id)
                if key not in pending:
                    continue
                value = try_json(f"{api}/task-runs/{run_id}", timeout=4.0)
                if isinstance(value, dict) and value.get("run", {}).get("run_id") == run_id:
                    pending.remove(key)
        if pending:
            time.sleep(0.5)
    if pending:
        raise RuntimeError(f"task runs did not converge: {sorted(pending)[:12]}")


def telemetry_record_is_live(telemetry: dict[str, Any]) -> bool:
    try:
        count = int(telemetry.get("process_count") or 0)
        rss = float(telemetry.get("memory_rss_mb") or 0.0)
    except (TypeError, ValueError):
        return False
    state = str(telemetry.get("process_state") or "").upper()
    return count >= 1 and state[:1] not in {"Z", "X"} and rss > 0.0


def wait_base_ready(rows: list[dict[str, Any]], timeout: float) -> None:
    minimum = {BENCHMARK_CLASS_ID: 2.0, LLM_CLASS_ID: 1.0}
    deadline = time.monotonic() + timeout
    last: list[str] = []
    while time.monotonic() < deadline:
        pending: list[str] = []
        for row in rows:
            state = task_state(row["api"], row["task_id"])
            if not state:
                pending.append(f"{row['task_id']}:missing")
                continue
            status = str(state.get("status"))
            if status in {"failed", "completed", "stopped"}:
                raise RuntimeError(f"base workload terminated before readiness: {state}")
            telemetry = try_json(f"{row['api']}/telemetry/tasks/{row['task_id']}", timeout=5.0) or {}
            progress = float(state.get("progress_completed_units") or 0.0)
            if status != "running" or progress < minimum[row["class_id"]] or not telemetry_record_is_live(telemetry):
                pending.append(
                    f"{row['node_id']}:{row['class_id']} status={status} progress={progress} "
                    f"proc={telemetry.get('process_count')} rss={telemetry.get('memory_rss_mb')}"
                )
        if not pending:
            return
        last = pending
        time.sleep(1.0)
    raise TimeoutError("base workloads not ready: " + "; ".join(last[:12]))


def ps_snapshot(node: Any, *, ssh_user: str, local_node_id: str) -> str:
    command = "ps -eo sid=,pid=,stat=,rss=,pcpu="
    if node.id == local_node_id:
        args = ["bash", "-lc", command]
    else:
        args = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            f"{ssh_user}@{node.internal_ip}", command,
        ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=15.0)
    if result.returncode != 0:
        raise RuntimeError(f"ps witness failed on {node.id}")
    return result.stdout


def parse_ps_sessions(raw: str) -> dict[int, dict[str, Any]]:
    sessions: dict[int, dict[str, Any]] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            sid, pid = int(fields[0]), int(fields[1])
            rss, cpu = float(fields[3]), float(fields[4])
        except ValueError:
            continue
        record = sessions.setdefault(
            sid,
            {"process_count": 0, "process_state": None, "memory_rss_mb": 0.0, "cpu_utilization_percent": 0.0},
        )
        record["process_count"] += 1
        record["memory_rss_mb"] += max(0.0, rss) / 1024.0
        record["cpu_utilization_percent"] += max(0.0, cpu)
        if pid == sid:
            record["process_state"] = fields[2].upper()
    return sessions


def capture_live_witness(
    rows: list[dict[str, Any]],
    *,
    cluster: Any,
    ssh_user: str,
    local_node_id: str,
    phase: str,
    policy: str,
) -> list[dict[str, Any]]:
    """Capture one near-simultaneous direct process/accounting witness.

    The exact Dendro r9/t1p0 calibration job is intentionally short.  Stage 5E.2
    already established that it can be witnessed reliably when registry-state and
    process-table reads are issued concurrently.  Do the same here: serial reads
    across nine tasks plus seven nodes can consume the workload's entire lifetime
    and create a false negative before the comparison even starts.
    """
    node_by_id = {node.id: node for node in cluster.nodes}
    node_ids = sorted({str(row["node_id"]) for row in rows})
    state_by_task: dict[str, dict[str, Any]] = {}
    ps_by_node: dict[str, dict[int, dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max(1, len(rows) + len(node_ids))) as pool:
        state_futures = {
            pool.submit(task_state, row["api"], row["task_id"], 15.0): str(row["task_id"])
            for row in rows
        }
        ps_futures = {
            pool.submit(
                ps_snapshot,
                node_by_id[node_id],
                ssh_user=ssh_user,
                local_node_id=local_node_id,
            ): node_id
            for node_id in node_ids
        }
        for future, task_id in state_futures.items():
            state_by_task[task_id] = future.result() or {}
        for future, node_id in ps_futures.items():
            ps_by_node[node_id] = parse_ps_sessions(future.result())

    sampled = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    for row in rows:
        state = state_by_task[str(row["task_id"])]
        pid = int(state.get("pid") or 0)
        session = ps_by_node.get(str(row["node_id"]), {}).get(pid, {}) if pid else {}
        proc_state = str(session.get("process_state") or "").upper()
        rss = float(session.get("memory_rss_mb") or 0.0)
        live = (
            state.get("status") == "running" and pid > 0
            and int(session.get("process_count") or 0) >= 1
            and proc_state[:1] not in {"Z", "X"} and rss > 0
        )
        output.append(
            {
                "policy": policy,
                "phase": phase,
                "sampled_at_utc": sampled,
                "task_id": row["task_id"],
                "class_id": row["class_id"],
                "initial_node_id": row["initial_node_id"],
                "node_id": row["node_id"],
                "status": state.get("status"),
                "progress_completed_units": state.get("progress_completed_units"),
                "pid": pid or None,
                "process_count": int(session.get("process_count") or 0),
                "process_state": proc_state or None,
                "cpu_utilization_percent": session.get("cpu_utilization_percent"),
                "memory_rss_mb": rss,
                "accumulated_runtime_seconds": float(state.get("accumulated_runtime_seconds") or 0.0),
                "accumulated_migration_seconds": float(state.get("accumulated_migration_seconds") or 0.0),
                "accumulated_carbon_grams": float(state.get("accumulated_carbon_grams") or 0.0),
                "accumulated_compute_carbon_grams": float(state.get("accumulated_compute_carbon_grams") or 0.0),
                "accumulated_transfer_carbon_grams": float(state.get("accumulated_transfer_carbon_grams") or 0.0),
                "accumulated_cost_usd": float(state.get("accumulated_cost_usd") or 0.0),
                "accumulated_compute_cost_usd": float(state.get("accumulated_compute_cost_usd") or 0.0),
                "accumulated_transfer_cost_usd": float(state.get("accumulated_transfer_cost_usd") or 0.0),
                "live": live,
            }
        )
    return output


def wait_all_live(rows: list[dict[str, Any]], *, cluster: Any, args: argparse.Namespace, policy: str) -> list[dict[str, Any]]:
    deadline = time.monotonic() + args.witness_timeout_seconds
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = capture_live_witness(
            rows,
            cluster=cluster,
            ssh_user=args.ssh_user,
            local_node_id=args.local_node_id,
            phase="pre_trial",
            policy=policy,
        )
        if len(last) == len(rows) and all(bool(row["live"]) for row in last):
            return last
        time.sleep(0.25)
    detail = "; ".join(
        f"{row['node_id']}:{row['class_id']} status={row['status']} state={row['process_state']} rss={row['memory_rss_mb']}"
        for row in last if not row["live"]
    )
    raise TimeoutError("all-live witness failed: " + detail)


def trace_anchor(cluster: Any) -> str:
    boston = cluster.get_node("boston")
    forecast = request_json(
        f"{base_url(boston, cluster.api_port)}/carbon/forecast/boston?horizon_seconds=0",
        timeout=15.0,
    )
    return str(forecast["generated_at_utc"])


SAMPLE_REQUEST_TIMEOUT_SECONDS = 5.0
SAMPLE_REQUEST_ATTEMPTS = 2


def sample_request_json(url: str) -> tuple[Any | None, str]:
    errors: list[str] = []
    for attempt in range(1, SAMPLE_REQUEST_ATTEMPTS + 1):
        try:
            return request_json(url, timeout=SAMPLE_REQUEST_TIMEOUT_SECONDS), ""
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"attempt{attempt}:{type(exc).__name__}:{exc}")
    return None, " | ".join(errors)


def collect_cluster_sample(
    *,
    cluster: Any,
    policy: str,
    sample_index: int,
    elapsed_seconds: float,
    run_ids: set[str],
) -> list[dict[str, Any]]:
    sampled = datetime.now(timezone.utc).isoformat()

    def sample_node(node: Any) -> dict[str, Any]:
        api = base_url(node, cluster.api_port)
        health, health_error = sample_request_json(f"{api}/health")
        telemetry, telemetry_error = sample_request_json(f"{api}/telemetry/tasks")
        sample_complete = isinstance(health, dict) and isinstance(telemetry, list)
        task_records = (
            [
                item
                for item in telemetry
                if str(item.get("task_id")) in run_ids
                and str(item.get("node_id")) == node.id
            ]
            if isinstance(telemetry, list)
            else []
        )
        busy_fraction = health.get("resource_busy_fraction") if isinstance(health, dict) else None
        error_parts = []
        if health_error:
            error_parts.append(f"health={health_error}")
        if telemetry_error:
            error_parts.append(f"telemetry={telemetry_error}")
        return {
            "policy": policy,
            "sample_index": sample_index,
            "sampled_at_utc": sampled,
            "elapsed_seconds": elapsed_seconds,
            "node_id": node.id,
            "sample_complete": sample_complete,
            "sample_error": "; ".join(error_parts),
            "owned_task_count": health.get("owned_task_count") if isinstance(health, dict) else None,
            "reserved_cpu_cores": health.get("reserved_cpu_cores") if isinstance(health, dict) else None,
            "reserved_memory_mb": health.get("reserved_memory_mb") if isinstance(health, dict) else None,
            "resource_busy_fraction": busy_fraction,
            "available_cpu_cores": health.get("available_cpu_cores") if isinstance(health, dict) else None,
            "task_telemetry_count": len(task_records),
            "task_cpu_percent_sum": sum(float(item.get("cpu_utilization_percent") or 0.0) for item in task_records),
            "task_rss_mb_sum": sum(float(item.get("memory_rss_mb") or 0.0) for item in task_records),
            "capacity_respected": (
                float(busy_fraction or 0.0) <= 1.0 + 1e-9
                if sample_complete
                else None
            ),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(cluster.nodes))) as pool:
        future_by_node = {pool.submit(sample_node, node): node for node in cluster.nodes}
        for future in as_completed(future_by_node):
            node = future_by_node[future]
            row = future.result()
            rows.append(row)
            if not row["sample_complete"]:
                print(
                    f"[sample-warning] {policy} sample={sample_index} node={node.id} "
                    f"{row['sample_error']}"
                )
    return sorted(rows, key=lambda row: str(row["node_id"]))


def wait_ownership(cluster: Any, run_ids: list[str], timeout: float) -> tuple[bool, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    snapshots: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        snapshots = {
            node.id: request_json(f"{base_url(node, cluster.api_port)}/ownership/snapshot", timeout=10.0)
            for node in cluster.nodes
        }
        ok, rows = ownership_converged(snapshots, run_ids)
        if ok:
            return ok, rows, snapshots
        time.sleep(1.0)
    ok, rows = ownership_converged(snapshots, run_ids)
    return ok, rows, snapshots


def newest_owner(snapshots: dict[str, dict[str, Any]], task_id: str) -> str:
    updates: list[dict[str, Any]] = []
    for snapshot in snapshots.values():
        updates.extend(update for update in snapshot.get("updates", []) if update.get("task_id") == task_id)
    if not updates:
        raise RuntimeError(f"no ownership update for {task_id}")
    newest = max(updates, key=lambda item: int(item.get("generation", 0)))
    return str(newest["owner_node_id"])


def submit_run(
    *, api: str, definition: dict[str, Any], owner_node_id: str, policy: str, comparison_id: str, slot: int
) -> str:
    view = request_json(
        f"{api}/task-runs",
        method="POST",
        payload={
            "definition_id": definition["definition_id"],
            "revision": definition["revision"],
            "initial_owner_node_id": owner_node_id,
            "idempotency_key": f"{comparison_id}:{policy}:{slot:02d}",
            "auto_start": True,
            "labels": {
                "purpose": "stage5e4-live-heterogeneous-comparison",
                "comparison_id": comparison_id,
                "stage5e4_policy": policy,
                "scheduler_mode": "operator_only",
                "initial_node_id": owner_node_id,
            },
        },
        timeout=120.0,
    )
    return str(view["run"]["run_id"])


def collect_trial_events(
    *,
    cluster: Any,
    run_ids: set[str],
    event_baselines: dict[str, int],
    bid_baselines: dict[str, set[str]],
    policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    bids: list[dict[str, Any]] = []
    migrations: list[dict[str, Any]] = []
    seen_events: set[tuple[str, int]] = set()
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        for task_id in run_ids:
            for event in event_query(api, event_baselines[node.id], task_id):
                key = (node.id, int(event.get("sequence") or 0))
                if key in seen_events:
                    continue
                seen_events.add(key)
                event_type = str(event.get("event_type") or "")
                payload = event.get("payload") or {}
                if event_type == "scheduler_decision":
                    selected = (payload.get("decision") or {}).get("selected") or {}
                    decisions.append(
                        {
                            "policy": policy,
                            "node_id": node.id,
                            "task_id": task_id,
                            "sequence": event.get("sequence"),
                            "trace_time_utc": event.get("trace_time_utc"),
                            "selected_action": selected.get("action"),
                            "selected_destination_node_id": selected.get("destination_node_id") or "",
                            "selected_score": selected.get("score"),
                        }
                    )
                elif event_type in {"migration_completed", "migration_failed"}:
                    migrations.append(
                        {
                            "policy": policy,
                            "reporting_node_id": node.id,
                            "task_id": task_id,
                            "status": "completed" if event_type == "migration_completed" else "failed",
                            "source_node_id": payload.get("source_node_id") or "",
                            "destination_node_id": payload.get("destination_node_id") or "",
                            "migration_id": payload.get("migration_id") or "",
                            "bid_id": payload.get("bid_id") or "",
                            "total_downtime_seconds": payload.get("total_downtime_seconds") or "",
                            "error": payload.get("error") or "",
                        }
                    )
        for item in request_json(f"{api}/bids", timeout=20.0):
            if str(item.get("bid_id")) in bid_baselines[node.id] or str(item.get("task_id")) not in run_ids:
                continue
            metrics = item.get("auction_metrics") or {}
            bids.append(
                {
                    "policy": policy,
                    "reporting_node_id": node.id,
                    "bid_id": item.get("bid_id"),
                    "task_id": item.get("task_id"),
                    "source_node_id": item.get("source_node_id"),
                    "destination_node_id": item.get("destination_node_id"),
                    "status": item.get("status"),
                    "decision_reason": item.get("decision_reason"),
                    "auction_strategy": item.get("auction_strategy"),
                    "auction_rank": item.get("auction_rank"),
                    "resource_fit": item.get("resource_fit"),
                    "requested_cpu_cores": metrics.get("requested_cpu_cores"),
                    "requested_memory_mb": metrics.get("requested_memory_mb"),
                }
            )
    return decisions, bids, migrations


def run_source_evaluation_epoch(
    *,
    source_rows: list[dict[str, Any]],
    anchor: str,
    request_timeout_seconds: float,
    witness_completed_monotonic: float,
) -> list[dict[str, Any]]:
    """Evaluate one source daemon's frozen cohort in production epoch order.

    Production ``SchedulerService.run_epoch()`` snapshots that daemon's running
    task IDs and awaits each evaluation sequentially. Different daemons run their
    own epochs independently, so the caller executes one of these loops per
    source daemon concurrently.
    """
    results: list[dict[str, Any]] = []
    ordered_rows = sorted(source_rows, key=lambda row: str(row.get("task_id") or ""))

    for source_sequence_index, row in enumerate(ordered_rows):
        request_started = time.monotonic()
        query = urlencode({"trace_time_utc": anchor})
        try:
            value = request_json(
                f"{row['api']}/tasks/{row['task_id']}/evaluate?{query}",
                method="POST",
                timeout=request_timeout_seconds,
            )
            state = value.get("state", {}) if isinstance(value, dict) else {}
            results.append(
                {
                    "policy": MAGELLAN_POLICY,
                    "task_id": row["task_id"],
                    "class_id": row["class_id"],
                    "initial_node_id": row["initial_node_id"],
                    "source_sequence_index": source_sequence_index,
                    "request_started_seconds_after_witness": (
                        request_started - witness_completed_monotonic
                    ),
                    "request_completed_seconds_after_witness": (
                        time.monotonic() - witness_completed_monotonic
                    ),
                    "trigger_ok": True,
                    "trigger_error": "",
                    "returned_owner_node_id": state.get("owner_node_id") or "",
                    "returned_status": state.get("status") or "",
                }
            )
        except Exception as exc:
            # Match run_epoch's per-task failure isolation: one failed evaluation
            # must not prevent later tasks on the same source daemon from being
            # evaluated. The trial still fails after the fixed window if any
            # trigger failed.
            results.append(
                {
                    "policy": MAGELLAN_POLICY,
                    "task_id": row["task_id"],
                    "class_id": row["class_id"],
                    "initial_node_id": row["initial_node_id"],
                    "source_sequence_index": source_sequence_index,
                    "request_started_seconds_after_witness": (
                        request_started - witness_completed_monotonic
                    ),
                    "request_completed_seconds_after_witness": (
                        time.monotonic() - witness_completed_monotonic
                    ),
                    "trigger_ok": False,
                    "trigger_error": f"{type(exc).__name__}: {exc}",
                    "returned_owner_node_id": "",
                    "returned_status": "",
                }
            )

    return results


def run_trial(
    *,
    policy: str,
    comparison_id: str,
    cluster: Any,
    layout: list[dict[str, str]],
    requests: dict[str, Any],
    target_sha: str,
    args: argparse.Namespace,
    root: Path,
    dendro_template: dict[str, Any],
) -> dict[str, Any]:
    node_by_id = {node.id: node for node in cluster.nodes}
    restart_cluster(
        cluster,
        expected_sha=target_sha,
        local_node_id=args.local_node_id,
        ssh_user=args.ssh_user,
        service=args.service,
    )
    hardened_preflight(cluster, target_sha)

    event_baselines: dict[str, int] = {}
    bid_baselines: dict[str, set[str]] = {}
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        event_baselines[node.id] = int(request_json(f"{api}/experiment/events/status").get("last_sequence", 0))
        bid_baselines[node.id] = {str(item["bid_id"]) for item in request_json(f"{api}/bids")}

    prepared: list[dict[str, Any]] = []
    node_ids = [node.id for node in cluster.nodes]
    print(f"\n== trial {policy} ==")
    print("[prepare] registering the frozen 9-task winter/u75 real-workload population")
    for slot, source_row in enumerate(layout):
        class_id = source_row["class_id"]
        owner_id = source_row["initial_node_id"]
        definition_id = f"{comparison_id}-{policy}-{slot:02d}-{class_id.replace('-', '_')}"
        definition_payload = physical_definition(
            class_id=class_id,
            definition_id=definition_id,
            request=requests[class_id],
            node_ids=node_ids,
            seed=700 + slot,
            benchmark_iterations=args.benchmark_iterations,
            llm_model=args.llm_model,
            dendro_template=dendro_template,
            dendro_solver=args.dendro_solver,
            dendro_parameter_template=args.dendro_parameter_template,
        )
        api = base_url(node_by_id[owner_id], cluster.api_port)
        created = request_json(f"{api}/task-definitions", method="POST", payload=definition_payload, timeout=120.0)
        wait_definition(
            cluster,
            str(created["definition_id"]),
            int(created["revision"]),
            str(created["digest"]),
            args.convergence_timeout_seconds,
        )
        prepared.append(
            {
                "slot": slot,
                "source_task_id": source_row["task_id"],
                "class_id": class_id,
                "initial_node_id": owner_id,
                "node_id": owner_id,
                "api": api,
                "definition_id": str(created["definition_id"]),
                "revision": int(created["revision"]),
            }
        )

    launched: list[dict[str, Any]] = []
    cleanup_rows: list[dict[str, Any]] = []
    error: str | None = None
    try:
        base_prepared = [row for row in prepared if row["class_id"] != DENDRO_CLASS_ID]
        dendro_prepared = [row for row in prepared if row["class_id"] == DENDRO_CLASS_ID]

        def launch(row: dict[str, Any]) -> dict[str, Any]:
            run_id = submit_run(
                api=row["api"],
                definition={"definition_id": row["definition_id"], "revision": row["revision"]},
                owner_node_id=row["initial_node_id"],
                policy=policy,
                comparison_id=comparison_id,
                slot=int(row["slot"]),
            )
            return {**row, "task_id": run_id}

        print("[launch-base] starting 6 benchmark/LLM tasks concurrently")
        with ThreadPoolExecutor(max_workers=len(base_prepared)) as pool:
            for future in as_completed([pool.submit(launch, row) for row in base_prepared]):
                value = future.result()
                launched.append(value)
                print(f"  started {value['task_id']} {value['initial_node_id']:16s} {value['class_id']}")
        wait_runs(cluster, [row["task_id"] for row in launched], args.convergence_timeout_seconds)
        print("[steady-base] waiting for all 6 benchmark/LLM tasks to be genuinely live with progress")
        wait_base_ready(launched, args.ready_timeout_seconds)

        # Prepare the controlled trace instant before launching the brief Dendro
        # jobs.  After the all-live witness there must be no serial state-snapshot
        # phase before Magellan's evaluation requests are issued.
        anchor = trace_anchor(cluster)

        print("[launch-dendro] starting 3 exact dendro-r9-t1p0 workloads concurrently")
        with ThreadPoolExecutor(max_workers=len(dendro_prepared)) as pool:
            for future in as_completed([pool.submit(launch, row) for row in dendro_prepared]):
                value = future.result()
                launched.append(value)
                print(f"  started {value['task_id']} {value['initial_node_id']:16s} {value['class_id']}")

        # Do not wait for seven-node registry convergence here: the exact Dendro
        # calibration workload is shorter than that distributed bookkeeping path.
        # The direct witness is the physical epoch boundary; ownership convergence
        # is checked normally at the end of the trial.
        witness = wait_all_live(launched, cluster=cluster, args=args, policy=policy)
        witness_completed_monotonic = time.monotonic()
        print(f"[witness] {sum(bool(row['live']) for row in witness)}/{EXPECTED_TASK_COUNT} real tasks simultaneously live")

        initial_progress = {
            row["task_id"]: row.get("progress_completed_units") for row in witness
        }
        # The direct witness already contains the authoritative state snapshot. Use
        # those exact values as the fixed-window accounting baseline instead of
        # issuing nine more serial API reads that could outlive Dendro r9/t1p0.
        baseline_accounting = {
            str(row["task_id"]): {
                "runtime_seconds": float(row.get("accumulated_runtime_seconds") or 0.0),
                "migration_seconds": float(row.get("accumulated_migration_seconds") or 0.0),
                "carbon_grams": float(row.get("accumulated_carbon_grams") or 0.0),
                "compute_carbon_grams": float(row.get("accumulated_compute_carbon_grams") or 0.0),
                "transfer_carbon_grams": float(row.get("accumulated_transfer_carbon_grams") or 0.0),
                "cost_usd": float(row.get("accumulated_cost_usd") or 0.0),
                "compute_cost_usd": float(row.get("accumulated_compute_cost_usd") or 0.0),
                "transfer_cost_usd": float(row.get("accumulated_transfer_cost_usd") or 0.0),
            }
            for row in witness
        }
        if len(baseline_accounting) != EXPECTED_TASK_COUNT:
            raise RuntimeError("direct witness did not provide accounting baselines for all nine tasks")

        print(f"[trial] trace_anchor={anchor} duration={args.trial_seconds:.0f}s")
        start_monotonic = time.monotonic()
        run_ids = {row["task_id"] for row in launched}

        evaluation_pool: ThreadPoolExecutor | None = None
        evaluation_futures: dict[Future[list[dict[str, Any]]], str] = {}
        evaluation_results: list[dict[str, Any]] = []
        evaluation_trigger_delay_seconds = 0.0
        evaluation_source_daemon_count = 0
        if policy == MAGELLAN_POLICY:
            cohort = [row for row in launched if row["class_id"] in DECISION_CLASSES]
            source_groups = evaluation_source_groups(cohort)
            evaluation_source_daemon_count = len(source_groups)
            if evaluation_source_daemon_count != EXPECTED_DECISION_SOURCE_COUNT:
                raise RuntimeError(
                    "Stage 5E.4 decision cohort source geometry drifted: "
                    f"{evaluation_source_daemon_count} != {EXPECTED_DECISION_SOURCE_COUNT}"
                )
            print(
                f"[evaluate] triggering {len(cohort)} benchmark/LLM production evaluations "
                f"across {evaluation_source_daemon_count} source daemons immediately after "
                "the 9/9 physical witness; tasks are sequential within each source"
            )
            evaluation_pool = ThreadPoolExecutor(max_workers=evaluation_source_daemon_count)
            for source_node_id, source_rows in source_groups:
                future = evaluation_pool.submit(
                    run_source_evaluation_epoch,
                    source_rows=source_rows,
                    anchor=anchor,
                    request_timeout_seconds=args.request_timeout_seconds,
                    witness_completed_monotonic=witness_completed_monotonic,
                )
                evaluation_futures[future] = source_node_id
            evaluation_trigger_delay_seconds = time.monotonic() - witness_completed_monotonic
            print(
                f"[evaluate] source_epoch_submission_delay_after_witness="
                f"{evaluation_trigger_delay_seconds:.3f}s"
            )

        samples: list[dict[str, Any]] = []
        sample_index = 0
        next_sample = 0.0
        while True:
            elapsed = time.monotonic() - start_monotonic
            if elapsed + 1e-9 >= next_sample:
                samples.extend(
                    collect_cluster_sample(
                        cluster=cluster,
                        policy=policy,
                        sample_index=sample_index,
                        elapsed_seconds=elapsed,
                        run_ids=run_ids,
                    )
                )
                sample_index += 1
                next_sample += args.sample_interval_seconds
            if elapsed >= args.trial_seconds:
                break
            time.sleep(min(0.5, max(0.05, next_sample - elapsed)))

        actual_trial_seconds = time.monotonic() - start_monotonic
        if evaluation_futures:
            unfinished = [future for future in evaluation_futures if not future.done()]
            if unfinished:
                raise RuntimeError(
                    f"{len(unfinished)} source scheduler epochs exceeded the fixed "
                    f"{args.trial_seconds:.0f}s trial window"
                )
            for future, source_node_id in evaluation_futures.items():
                try:
                    evaluation_results.extend(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"source scheduler epoch failed on {source_node_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            evaluation_results.sort(
                key=lambda row: (
                    str(row["initial_node_id"]),
                    int(row["source_sequence_index"]),
                )
            )
            if any(not row["trigger_ok"] for row in evaluation_results):
                raise RuntimeError(f"scheduler evaluation failure: {evaluation_results}")
        if evaluation_pool is not None:
            evaluation_pool.shutdown(wait=True)

        ownership_ok, ownership_rows, snapshots = wait_ownership(
            cluster,
            list(run_ids),
            args.convergence_timeout_seconds,
        )
        if not ownership_ok:
            raise RuntimeError("ownership did not converge at end of trial")

        final_rows: list[dict[str, Any]] = []
        for row in launched:
            owner = newest_owner(snapshots, row["task_id"])
            state = task_state(base_url(node_by_id[owner], cluster.api_port), row["task_id"], timeout=20.0)
            if not state:
                raise RuntimeError(f"missing final state for {row['task_id']} on {owner}")
            if state.get("status") == "failed":
                raise RuntimeError(f"workload failed during {policy}: {state}")
            final_rows.append(
                {
                    "policy": policy,
                    "task_id": row["task_id"],
                    "source_task_id": row["source_task_id"],
                    "class_id": row["class_id"],
                    "initial_node_id": row["initial_node_id"],
                    "final_owner_node_id": owner,
                    "generation": state.get("generation"),
                    "status": state.get("status"),
                    "initial_progress_completed_units": initial_progress.get(row["task_id"]),
                    "final_progress_completed_units": state.get("progress_completed_units"),
                    "progress_total_units": state.get("progress_total_units"),
                    "baseline_runtime_seconds": baseline_accounting[row["task_id"]]["runtime_seconds"],
                    "final_runtime_seconds": state.get("accumulated_runtime_seconds"),
                    "measurement_runtime_seconds": max(0.0, float(state.get("accumulated_runtime_seconds") or 0.0) - baseline_accounting[row["task_id"]]["runtime_seconds"]),
                    "baseline_migration_seconds": baseline_accounting[row["task_id"]]["migration_seconds"],
                    "final_migration_seconds": state.get("accumulated_migration_seconds"),
                    "measurement_migration_seconds": max(0.0, float(state.get("accumulated_migration_seconds") or 0.0) - baseline_accounting[row["task_id"]]["migration_seconds"]),
                    "baseline_compute_carbon_grams": baseline_accounting[row["task_id"]]["compute_carbon_grams"],
                    "baseline_transfer_carbon_grams": baseline_accounting[row["task_id"]]["transfer_carbon_grams"],
                    "baseline_carbon_grams": baseline_accounting[row["task_id"]]["carbon_grams"],
                    "final_compute_carbon_grams": state.get("accumulated_compute_carbon_grams"),
                    "final_transfer_carbon_grams": state.get("accumulated_transfer_carbon_grams"),
                    "final_carbon_grams": state.get("accumulated_carbon_grams"),
                    "measurement_compute_carbon_grams": max(0.0, float(state.get("accumulated_compute_carbon_grams") or 0.0) - baseline_accounting[row["task_id"]]["compute_carbon_grams"]),
                    "measurement_transfer_carbon_grams": max(0.0, float(state.get("accumulated_transfer_carbon_grams") or 0.0) - baseline_accounting[row["task_id"]]["transfer_carbon_grams"]),
                    "measurement_carbon_grams": max(0.0, float(state.get("accumulated_carbon_grams") or 0.0) - baseline_accounting[row["task_id"]]["carbon_grams"]),
                    "baseline_compute_cost_usd": baseline_accounting[row["task_id"]]["compute_cost_usd"],
                    "baseline_transfer_cost_usd": baseline_accounting[row["task_id"]]["transfer_cost_usd"],
                    "baseline_cost_usd": baseline_accounting[row["task_id"]]["cost_usd"],
                    "final_compute_cost_usd": state.get("accumulated_compute_cost_usd"),
                    "final_transfer_cost_usd": state.get("accumulated_transfer_cost_usd"),
                    "final_cost_usd": state.get("accumulated_cost_usd"),
                    "measurement_compute_cost_usd": max(0.0, float(state.get("accumulated_compute_cost_usd") or 0.0) - baseline_accounting[row["task_id"]]["compute_cost_usd"]),
                    "measurement_transfer_cost_usd": max(0.0, float(state.get("accumulated_transfer_cost_usd") or 0.0) - baseline_accounting[row["task_id"]]["transfer_cost_usd"]),
                    "measurement_cost_usd": max(0.0, float(state.get("accumulated_cost_usd") or 0.0) - baseline_accounting[row["task_id"]]["cost_usd"]),
                    "last_migration_id": state.get("last_migration_id") or "",
                }
            )

        decisions, bids, migrations = collect_trial_events(
            cluster=cluster,
            run_ids=run_ids,
            event_baselines=event_baselines,
            bid_baselines=bid_baselines,
            policy=policy,
        )
        successful_migrations = [row for row in migrations if row["status"] == "completed"]
        failed_migrations = [row for row in migrations if row["status"] == "failed"]
        complete_samples = [row for row in samples if bool(row.get("sample_complete"))]
        capacity_violations = [row for row in complete_samples if not bool(row["capacity_respected"])]
        sample_coverage_fraction = len(complete_samples) / len(samples) if samples else 0.0
        node_sample_counts = Counter(str(row["node_id"]) for row in samples)
        node_complete_counts = Counter(str(row["node_id"]) for row in complete_samples)
        node_coverage = {
            node.id: (
                node_complete_counts[node.id] / node_sample_counts[node.id]
                if node_sample_counts[node.id]
                else 0.0
            )
            for node in cluster.nodes
        }
        min_node_sample_coverage_fraction = min(node_coverage.values()) if node_coverage else 0.0
        if sample_coverage_fraction + 1e-12 < MIN_SAMPLE_COVERAGE_FRACTION:
            raise RuntimeError(
                f"resource telemetry coverage too low: {sample_coverage_fraction:.3f} "
                f"< {MIN_SAMPLE_COVERAGE_FRACTION:.3f}"
            )
        if min_node_sample_coverage_fraction + 1e-12 < MIN_NODE_SAMPLE_COVERAGE_FRACTION:
            raise RuntimeError(
                f"per-node resource telemetry coverage too low: "
                f"{min_node_sample_coverage_fraction:.3f} "
                f"< {MIN_NODE_SAMPLE_COVERAGE_FRACTION:.3f}; {node_coverage}"
            )

        total_carbon = sum(float(row.get("measurement_carbon_grams") or 0.0) for row in final_rows)
        total_cost = sum(float(row.get("measurement_cost_usd") or 0.0) for row in final_rows)
        total_runtime = sum(float(row.get("measurement_runtime_seconds") or 0.0) for row in final_rows)
        completed_count = sum(str(row.get("status")) == "completed" for row in final_rows)
        avg_busy = (
            sum(float(row.get("resource_busy_fraction") or 0.0) for row in complete_samples) / len(complete_samples)
            if complete_samples else 0.0
        )
        complete_round_equivalents = len(complete_samples) / len(cluster.nodes) if cluster.nodes else 0.0
        avg_cpu_percent = (
            sum(float(row.get("task_cpu_percent_sum") or 0.0) for row in complete_samples) / complete_round_equivalents
            if complete_round_equivalents else 0.0
        )

        trial = {
            "policy": policy,
            "task_count": len(launched),
            "class_counts": dict(Counter(row["class_id"] for row in launched)),
            "layout_fingerprint": [list(item) for item in layout_fingerprint(layout)],
            "configured_trial_seconds": args.trial_seconds,
            "actual_trial_seconds": actual_trial_seconds,
            "trace_anchor_utc": anchor,
            "trace_date_utc": anchor[:10],
            "pre_live_witness_count": sum(bool(row["live"]) for row in witness),
            "evaluation_source_daemon_count": evaluation_source_daemon_count,
            "evaluation_trigger_delay_seconds_after_witness": evaluation_trigger_delay_seconds,
            "scheduler_decision_count": len(decisions),
            "bid_count": len(bids),
            "accepted_or_consumed_bid_count": sum(str(row.get("status")) in {"accepted", "consumed"} for row in bids),
            "rejected_bid_count": sum(str(row.get("status")) == "rejected" for row in bids),
            "successful_migration_count": len(successful_migrations),
            "failed_migration_count": len(failed_migrations),
            "migration_downtime_seconds_total": sum(float(row.get("total_downtime_seconds") or 0.0) for row in successful_migrations),
            "ownership_converged": ownership_ok,
            "completed_task_count": completed_count,
            "total_runtime_seconds": total_runtime,
            "total_carbon_grams": total_carbon,
            "total_cost_usd": total_cost,
            "average_cluster_resource_busy_fraction": avg_busy,
            "mean_observed_cluster_task_cpu_percent": avg_cpu_percent,
            "capacity_violation_sample_count": len(capacity_violations),
            "sample_round_count": int(len(samples) / len(cluster.nodes)) if cluster.nodes else 0,
            "complete_sample_count": len(complete_samples),
            "total_sample_count": len(samples),
            "sample_coverage_fraction": sample_coverage_fraction,
            "min_node_sample_coverage_fraction": min_node_sample_coverage_fraction,
            "sample_error_count": len(samples) - len(complete_samples),
            "cleanup_ok_count": 0,
        }

        trial_root = root / policy
        trial_root.mkdir(parents=True, exist_ok=True)
        write_csv(trial_root / "initial_layout.csv", layout, list(layout[0].keys()))
        write_csv(trial_root / "pre_live_witness.csv", witness, list(witness[0].keys()))
        write_csv(trial_root / "resource_samples.csv", samples, list(samples[0].keys()))
        write_csv(trial_root / "evaluation_triggers.csv", evaluation_results, list(evaluation_results[0].keys()) if evaluation_results else ["policy", "task_id"])
        write_csv(trial_root / "decisions.csv", decisions, list(decisions[0].keys()) if decisions else ["policy", "task_id"])
        write_csv(trial_root / "bids.csv", bids, list(bids[0].keys()) if bids else ["policy", "bid_id"])
        write_csv(trial_root / "migrations.csv", migrations, list(migrations[0].keys()) if migrations else ["policy", "task_id"])
        write_csv(trial_root / "ownership.csv", ownership_rows, list(ownership_rows[0].keys()))
        write_csv(trial_root / "final_tasks.csv", final_rows, list(final_rows[0].keys()))

        # Cleanup after all evidence is durable. Terminal Dendro tasks need no stop call.
        for row in final_rows:
            task_id = str(row["task_id"])
            owner = str(row["final_owner_node_id"])
            status = str(row.get("status") or "")
            ok = False
            cleanup_status = status
            cleanup_error = ""
            try:
                if status in {"completed", "failed", "stopped"}:
                    ok = True
                else:
                    value = request_json(
                        f"{base_url(node_by_id[owner], cluster.api_port)}/tasks/{task_id}/stop",
                        method="POST",
                        timeout=900.0,
                    )
                    state = value.get("state", value) if isinstance(value, dict) else {}
                    cleanup_status = str(state.get("status") or "stopped")
                    ok = cleanup_status in {"stopped", "completed", "failed"}
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
            cleanup_rows.append(
                {
                    "policy": policy,
                    "task_id": task_id,
                    "owner_node_id": owner,
                    "cleanup_ok": ok,
                    "cleanup_status": cleanup_status,
                    "cleanup_error": cleanup_error,
                }
            )
        trial["cleanup_ok_count"] = sum(bool(row["cleanup_ok"]) for row in cleanup_rows)
        write_csv(trial_root / "cleanup.csv", cleanup_rows, list(cleanup_rows[0].keys()))
        write_json(trial_root / "summary.json", trial)
        write_checksums(trial_root)

        print(
            f"[{policy}] carbon={total_carbon:.6f}g cost=${total_cost:.6f} "
            f"decisions={len(decisions)} bids={len(bids)} migrations={len(successful_migrations)} "
            f"capacity_violations={len(capacity_violations)} cleanup={trial['cleanup_ok_count']}/{EXPECTED_TASK_COUNT}"
        )
        return trial
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Best-effort cleanup from every known current owner/local state.
        print(f"[trial-error] {policy}: {error}")
        for row in launched:
            task_id = row["task_id"]
            cleaned = False
            for node in cluster.nodes:
                state = task_state(base_url(node, cluster.api_port), task_id, timeout=3.0)
                if not state or state.get("owner_node_id") != node.id:
                    continue
                if state.get("status") in {"completed", "failed", "stopped"}:
                    cleaned = True
                    break
                try:
                    request_json(
                        f"{base_url(node, cluster.api_port)}/tasks/{task_id}/stop",
                        method="POST",
                        timeout=120.0,
                    )
                    cleaned = True
                    break
                except Exception:
                    pass
            cleanup_rows.append({"policy": policy, "task_id": task_id, "owner_node_id": "", "cleanup_ok": cleaned, "cleanup_status": "error", "cleanup_error": error})
        trial_root = root / policy
        trial_root.mkdir(parents=True, exist_ok=True)
        if cleanup_rows:
            write_csv(trial_root / "cleanup.csv", cleanup_rows, list(cleanup_rows[0].keys()))
        write_json(trial_root / "summary.json", {"policy": policy, "passed": False, "error": error})
        write_checksums(trial_root)
        raise


def main() -> int:
    args = parse_args()
    if args.trial_seconds < 60:
        raise ValueError("Stage 5E.4 trial_seconds must be at least 60")
    if args.sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    s5a_path = Path(args.stage5a_bundle)
    s5e3_path = Path(args.stage5e3_bundle)
    s5e2_path = Path(args.stage5e2_bundle)
    s5e1_path = Path(args.stage5e1_bundle)
    d43_path = Path(args.stage4d3_bundle)
    d41_path = Path(args.stage4d1_bundle)

    s5a = require_bundle(s5a_path, "Stage 5A")
    require_stage5a_policy(s5a_path, args.expected_policy)
    require_bundle(s5e3_path, "Stage 5E.3")
    require_bundle(s5e2_path, "Stage 5E.2")
    require_bundle(s5e1_path, "Stage 5E.1")
    require_bundle(d43_path, "Stage 4D.3")
    require_bundle(d41_path, "Stage 4D.1")

    target_sha = str(s5a.get("target_git_sha") or s5a.get("git_sha") or "")
    if local_git_sha() != target_sha:
        raise RuntimeError(
            f"local SHA {local_git_sha()} != Stage 5A SHA {target_sha}"
        )

    cluster = load_cluster_config(args.cluster)
    case, layout = read_stage4d3_u75_layout(d43_path)
    capacities, requests = read_resource_model(d41_path)
    cluster_cpu = sum(float(capacity.cpu_cores or 0.0) for capacity in capacities.values())
    planned_cpu = sum(float(requests[row["class_id"]].cpu_cores) for row in layout)
    achieved = planned_cpu / cluster_cpu
    frozen_achieved = float(case["achieved_initial_cpu_fraction"])
    if abs(achieved - frozen_achieved) > 1e-6:
        raise RuntimeError(f"live u75 CPU fraction {achieved} != frozen Stage 4D.3 {frozen_achieved}")

    comparison_id = args.comparison_id or (
        f"stage5e4-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 5E.4 live heterogeneous scheduling comparison ==")
    print(f"comparison_id={comparison_id}")
    print(f"git_sha={target_sha}")
    print(f"source_stage4d3={d43_path}")
    print(f"source_stage4d1={d41_path}")
    print(f"source_stage5e2={s5e2_path}")
    print(f"source_stage5e3={s5e3_path}")
    print(
        f"frozen_case={EXPECTED_SEASON}/{EXPECTED_LOAD_ID} "
        f"source_scenario={EXPECTED_SOURCE_SCENARIO_ID} tasks={len(layout)} "
        f"mix=3/3/3 planned_cpu={planned_cpu:.6f}/{cluster_cpu:.1f} ({achieved*100:.2f}%)"
    )
    print(f"policies={','.join(POLICIES)} fixed_trial_seconds={args.trial_seconds:.0f}")
    print("decision_cohort=3 benchmark + 3 LLM; Dendro remains real physical background load")

    # Validate assets/disk once before either policy; each trial restarts the daemons.
    hardened_preflight(cluster, target_sha)
    workload_asset_preflight(cluster, layout, args)
    checkpoint_bytes = read_checkpoint_bytes(s5e1_path)
    disk_rows = disk_preflight(
        cluster,
        layout=layout,
        checkpoint_bytes=checkpoint_bytes,
        args=args,
    )
    dendro_template = json.loads(Path(args.dendro_definition).read_text(encoding="utf-8"))

    trial_summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        trial_summaries.append(
            run_trial(
                policy=policy,
                comparison_id=comparison_id,
                cluster=cluster,
                layout=layout,
                requests=requests,
                target_sha=target_sha,
                args=args,
                root=root,
                dendro_template=dendro_template,
            )
        )

    by_policy = {row["policy"]: row for row in trial_summaries}
    static = by_policy[STATIC_POLICY]
    magellan = by_policy[MAGELLAN_POLICY]
    static_carbon = float(static["total_carbon_grams"])
    magellan_carbon = float(magellan["total_carbon_grams"])
    static_cost = float(static["total_cost_usd"])
    magellan_cost = float(magellan["total_cost_usd"])
    anchor_delta = abs(
        (datetime.fromisoformat(str(magellan["trace_anchor_utc"]).replace("Z", "+00:00"))
         - datetime.fromisoformat(str(static["trace_anchor_utc"]).replace("Z", "+00:00"))).total_seconds()
    )
    if anchor_delta > args.trace_anchor_max_delta_seconds:
        raise RuntimeError(
            f"policy trace anchors differ by {anchor_delta:.1f}s, exceeding "
            f"{args.trace_anchor_max_delta_seconds:.1f}s"
        )

    passed = stage5e4_passes(
        trial_summaries=trial_summaries,
        expected_layout_fingerprint=layout_fingerprint(layout),
    )
    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "git_sha": target_sha,
        "source_stage5a_bundle": str(s5a_path),
        "source_stage5e3_bundle": str(s5e3_path),
        "source_stage5e2_bundle": str(s5e2_path),
        "source_stage5e1_bundle": str(s5e1_path),
        "source_stage4d3_bundle": str(d43_path),
        "source_stage4d1_bundle": str(d41_path),
        "deployed_policy_path": args.expected_policy,
        "season": EXPECTED_SEASON,
        "load_id": EXPECTED_LOAD_ID,
        "source_stage4d2_scenario_id": EXPECTED_SOURCE_SCENARIO_ID,
        "task_count": EXPECTED_TASK_COUNT,
        "class_counts": EXPECTED_CLASS_COUNTS,
        "planned_cpu_cores": planned_cpu,
        "cluster_cpu_cores": cluster_cpu,
        "planned_cpu_fraction": achieved,
        "trial_seconds": args.trial_seconds,
        "trace_anchor_delta_seconds": anchor_delta,
        "policies": trial_summaries,
        "static_total_carbon_grams": static_carbon,
        "magellan_total_carbon_grams": magellan_carbon,
        "carbon_ratio_vs_static": magellan_carbon / static_carbon if static_carbon else None,
        "carbon_savings_percent_vs_static": 100.0 * (1.0 - magellan_carbon / static_carbon) if static_carbon else None,
        "static_total_cost_usd": static_cost,
        "magellan_total_cost_usd": magellan_cost,
        "cost_ratio_vs_static": magellan_cost / static_cost if static_cost else None,
        "successful_migrations": int(magellan["successful_migration_count"]),
        "failed_migrations": int(magellan["failed_migration_count"]),
        "capacity_violation_sample_count": int(static["capacity_violation_sample_count"]) + int(magellan["capacity_violation_sample_count"]),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5e4_live_heterogeneous_static_vs_magellan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "source_case": (
                "Uses the exact frozen Stage 4D.3 winter/u75 nine-task layout: three "
                "benchmark-json-medium, three llm-distilgpt2, and three dendro-r9-t1p0 tasks."
            ),
            "trace": (
                "Stage 5A explicitly installs config/policy.stage5e4.json, a copy of the "
                "production policy with only trace_start_utc anchored to the frozen winter "
                "arrival date 2024-01-05. Production policy.prod.json remains unchanged. All "
                "seven daemons are restarted before each trial so the trace clock resets."
            ),
            "policies": (
                "static_initial_layout runs the real workload population without scheduler "
                "evaluation. magellan_lowest_score triggers one synchronized production-style "
                "epoch across the four source daemons owning the six benchmark/LLM tasks: source "
                "daemons execute concurrently while each daemon evaluates its task IDs sequentially, "
                "matching SchedulerService.run_epoch(). The three exact Dendro tasks remain real "
                "physical background load. Both trials use the same fixed "
                "wall-clock measurement duration."
            ),
            "physical_evidence": (
                "Each trial requires a direct process-session witness with all nine real "
                "workloads simultaneously alive before the measurement window. ResourceLedger "
                "and live task telemetry are sampled throughout the fixed-duration window."
            ),
            "metrics": (
                "Primary live-system metrics are lifecycle carbon and cost deltas accumulated "
                "by authoritative task states during the fixed measurement window (excluding "
                "launch/warm-up), scheduler decisions, bids, migrations/downtime, "
                "completion/progress, observed CPU/RSS, resource occupancy, and capacity violations."
            ),
            "scope": (
                "This is a controlled synchronized live scheduler-epoch comparison, not the "
                "100-task scale study (Stage 4E) and not a fairness-policy comparison (Stage 4D.4)."
            ),
        },
    }
    write_csv(root / "frozen_initial_layout.csv", layout, list(layout[0].keys()))
    write_csv(root / "disk_preflight.csv", disk_rows, list(disk_rows[0].keys()))
    write_csv(root / "policy_summary.csv", trial_summaries, list(trial_summaries[0].keys()))
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_5E4_LIVE_HETEROGENEOUS_COMPARISON_PASS" if passed else "STAGE_5E4_LIVE_HETEROGENEOUS_COMPARISON_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"layout: winter/u75 tasks=9 mix=3 benchmark + 3 llm + 3 dendro cpu={achieved*100:.2f}%")
    print(f"static: carbon={static_carbon:.6f}g cost=${static_cost:.6f} migrations=0")
    print(
        f"magellan: carbon={magellan_carbon:.6f}g cost=${magellan_cost:.6f} "
        f"decisions={magellan['scheduler_decision_count']} bids={magellan['bid_count']} "
        f"migrations={magellan['successful_migration_count']}"
    )
    if static_carbon:
        print(
            f"carbon_ratio={summary['carbon_ratio_vs_static']:.4f} "
            f"savings={summary['carbon_savings_percent_vs_static']:.2f}%"
        )
    print(f"capacity_violations={summary['capacity_violation_sample_count']}")
    print(f"trace_anchor_delta={anchor_delta:.2f}s")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
