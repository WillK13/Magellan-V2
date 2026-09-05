#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage4d2 import (
    maximal_packing_signatures,
    read_resource_model,
)
from magellan.experiments.stage5e2 import (
    BENCHMARK_CLASS_ID,
    DENDRO_CLASS_ID,
    EXPECTED_CLASS_COUNTS,
    LLM_CLASS_ID,
    STAGE5E2_LAYOUT,
    physical_definition,
    physical_task_specs,
    stage5e2_passes,
    validate_physical_layout,
)


TASK_FIELDS = [
    "task_index",
    "task_id",
    "definition_id",
    "node_id",
    "class_id",
    "declared_cpu_cores",
    "declared_memory_mb",
    "declared_gpu_count",
    "launched",
    "steady_running",
    "initial_progress",
    "progress_min",
    "progress_max",
    "progress_advanced",
    "telemetry_sample_count",
    "cpu_sample_count",
    "mean_cpu_percent",
    "p95_cpu_percent",
    "rss_sample_count",
    "mean_memory_rss_mb",
    "max_memory_rss_mb",
    "min_process_count",
    "max_process_count",
    "max_checkpoint_bytes",
    "cleanup_ok",
    "cleanup_status",
]

NODE_FIELDS = [
    "node_id",
    "packing_signature",
    "task_count",
    "benchmark_count",
    "dendro_count",
    "llm_count",
    "expected_reserved_cpu_cores",
    "expected_reserved_memory_mb",
    "expected_reserved_gpu_count",
    "effective_cpu_cores",
    "effective_memory_mb",
    "effective_gpu_count",
    "cpu_fraction",
    "memory_fraction",
    "is_frozen_maximal_packing",
    "all_tasks_running",
    "reservation_matches_expected",
    "capacity_respected",
    "health_sample_count",
    "max_reserved_cpu_cores",
    "max_reserved_memory_mb",
    "max_resource_busy_fraction",
    "min_available_cpu_cores",
    "actual_cpu_sample_count",
    "mean_actual_task_cpu_percent",
    "max_actual_task_cpu_percent",
    "actual_rss_sample_count",
    "mean_actual_task_rss_mb",
    "max_actual_task_rss_mb",
]

TASK_SAMPLE_FIELDS = [
    "sample_index",
    "sampled_at_utc",
    "task_index",
    "task_id",
    "node_id",
    "class_id",
    "status",
    "progress_completed_units",
    "pid",
    "process_count",
    "process_state",
    "cpu_utilization_percent",
    "memory_rss_mb",
    "checkpoint_bytes",
    "telemetry_last_sample_at_utc",
    "telemetry_sample_count",
    "telemetry_freshness",
    "telemetry_age_seconds",
]

NODE_SAMPLE_FIELDS = [
    "sample_index",
    "sampled_at_utc",
    "node_id",
    "owned_task_count",
    "reserved_cpu_cores",
    "reserved_memory_mb",
    "reserved_gpu_count",
    "resource_busy_fraction",
    "available_cpu_cores",
    "available_memory_mb",
    "actual_task_cpu_percent",
    "actual_task_rss_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 5E.2: 11 simultaneous real benchmark/LLM/Dendro tasks "
            "in the exact measured maximal packings frozen by Stage 4D.1."
        )
    )
    parser.add_argument("--stage5a-bundle", required=True)
    parser.add_argument("--stage5e1-bundle", required=True)
    parser.add_argument("--stage4d1-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--ssh-user", default=os.getenv("MAGELLAN_SSH_USER", "WILL"))
    parser.add_argument("--remote-repo", default="/home/WILL/Magellan-V2")
    parser.add_argument("--profile-seconds", type=float, default=30.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=900.0)
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


def base_url(node: Any, port: int) -> str:
    return f"http://{node.internal_ip}:{port}"


def local_git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_passed_bundle(path: Path, label: str) -> dict[str, Any]:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum failure: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(f"{label} bundle did not pass")
    return summary


def task_state(api: str, run_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
    try:
        value = request_json(f"{api}/task-runs/{run_id}", timeout=timeout)
    except Exception:
        return None
    state = value.get("state") if isinstance(value, dict) else None
    return state if isinstance(state, dict) else None


def active_tasks(api: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for item in request_json(f"{api}/tasks", timeout=15.0).get("tasks", []):
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


def hardened_preflight(cluster: Any, expected_sha: str) -> None:
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health", timeout=15.0)
        errors: list[str] = []
        if health.get("deployment_git_sha") != expected_sha:
            errors.append(f"sha={health.get('deployment_git_sha')}")
        if health.get("carbon_metric") != "lifecycle":
            errors.append(f"carbon={health.get('carbon_metric')}")
        state_file = str(health.get("telemetry_state_file") or "")
        if "runtime-state-gcp" not in state_file or "runtime-state-gcp-measurement" in state_file:
            errors.append(f"state={state_file}")
        active = active_tasks(api)
        if active:
            errors.append(f"active_tasks={active}")
        if float(health.get("resource_busy_fraction") or 0.0) > 1e-9:
            errors.append(f"busy={health.get('resource_busy_fraction')}")
        if errors:
            raise RuntimeError(f"{node.id} hardened preflight failed: " + "; ".join(errors))
        print(
            f"[preflight] {node.id:16s} sha={expected_sha[:12]} "
            "active_tasks=0 busy=0"
        )


def remote_check(node: Any, *, ssh_user: str, command: str, timeout: float = 60.0) -> None:
    result = subprocess.run(
        [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{ssh_user}@{node.internal_ip}",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"asset/dependency preflight failed on {node.id}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def workload_asset_preflight(cluster: Any, args: argparse.Namespace) -> None:
    node_by_id = {node.id: node for node in cluster.nodes}
    llm_nodes = sorted(
        node_id
        for node_id, classes in STAGE5E2_LAYOUT.items()
        if LLM_CLASS_ID in classes
    )
    dendro_nodes = sorted(
        node_id
        for node_id, classes in STAGE5E2_LAYOUT.items()
        if DENDRO_CLASS_ID in classes
    )
    llm_model = f"{args.remote_repo}/{args.llm_model}"
    llm_command = (
        "set -e; "
        f"test -d {shlex.quote(llm_model)}; "
        f"{shlex.quote(args.remote_repo + '/.venv/bin/python')} -c "
        + shlex.quote("import torch, transformers")
    )
    for node_id in llm_nodes:
        remote_check(node_by_id[node_id], ssh_user=args.ssh_user, command=llm_command)
        print(f"[asset] {node_id:16s} LLM model+dependencies ready")

    dendro_command = (
        "set -e; command -v mpirun >/dev/null; "
        f"test -x {shlex.quote(args.dendro_solver)}; "
        f"test -f {shlex.quote(args.dendro_parameter_template)}"
    )
    for node_id in dendro_nodes:
        remote_check(node_by_id[node_id], ssh_user=args.ssh_user, command=dendro_command)
        print(f"[asset] {node_id:16s} Dendro solver+MPI ready")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            output.append(parsed)
    return output


def wait_steady(
    *,
    launched: list[dict[str, Any]],
    cluster: Any,
    timeout_seconds: float,
) -> dict[str, float]:
    minimum_progress = {
        BENCHMARK_CLASS_ID: 2.0,
        LLM_CLASS_ID: 1.0,
        DENDRO_CLASS_ID: 1.0,
    }
    deadline = time.monotonic() + timeout_seconds
    initial_progress: dict[str, float] = {}
    last_not_ready: list[str] = []
    while time.monotonic() < deadline:
        not_ready: list[str] = []
        for row in launched:
            state = task_state(row["api"], row["task_id"])
            if state is None:
                not_ready.append(f"{row['task_id']}:missing")
                continue
            status = str(state.get("status"))
            if status in {"failed", "completed", "stopped"}:
                raise RuntimeError(
                    f"{row['class_id']} task {row['task_id']} terminated before steady state: {state}"
                )
            try:
                telemetry = request_json(
                    f"{row['api']}/telemetry/tasks/{row['task_id']}", timeout=15.0
                )
            except Exception:
                telemetry = {}
            progress = float(state.get("progress_completed_units") or 0.0)
            if (
                status != "running"
                or progress < minimum_progress[row["class_id"]]
                or int(telemetry.get("process_count") or 0) < 1
            ):
                not_ready.append(
                    f"{row['node_id']}:{row['class_id']} status={status} "
                    f"progress={progress} proc={telemetry.get('process_count')}"
                )
                continue
            initial_progress[row["task_id"]] = progress
        if not not_ready and len(initial_progress) == len(launched):
            return initial_progress
        last_not_ready = not_ready
        time.sleep(2.0)
    raise TimeoutError(
        "11-task physical packing did not reach simultaneous steady state; "
        + "; ".join(last_not_ready[:12])
    )


def fetch_task_sample(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    state = task_state(row["api"], row["task_id"], timeout=20.0)
    if state is None:
        raise RuntimeError(f"Task disappeared during profile: {row['task_id']}")
    telemetry = request_json(
        f"{row['api']}/telemetry/tasks/{row['task_id']}", timeout=20.0
    )
    return {
        "sample_index": sample_index,
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_index": row["task_index"],
        "task_id": row["task_id"],
        "node_id": row["node_id"],
        "class_id": row["class_id"],
        "status": state.get("status"),
        "progress_completed_units": state.get("progress_completed_units"),
        "pid": state.get("pid"),
        "process_count": telemetry.get("process_count"),
        "process_state": telemetry.get("process_state"),
        "cpu_utilization_percent": telemetry.get("cpu_utilization_percent"),
        "memory_rss_mb": telemetry.get("memory_rss_mb"),
        "checkpoint_bytes": telemetry.get("checkpoint_bytes"),
        "telemetry_last_sample_at_utc": telemetry.get("last_sample_at_utc"),
        "telemetry_sample_count": telemetry.get("sample_count"),
        "telemetry_freshness": telemetry.get("freshness"),
        "telemetry_age_seconds": telemetry.get("age_seconds"),
    }


def profile_physical_packing(
    *,
    launched: list[dict[str, Any]],
    cluster: Any,
    profile_seconds: float,
    sample_interval_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_by_id = {node.id: node for node in cluster.nodes}
    task_samples: list[dict[str, Any]] = []
    node_samples: list[dict[str, Any]] = []
    sample_index = 0
    started = time.monotonic()
    while True:
        with ThreadPoolExecutor(max_workers=24) as pool:
            task_futures = {
                pool.submit(fetch_task_sample, row, sample_index): row
                for row in launched
            }
            health_futures = {
                pool.submit(
                    request_json,
                    f"{base_url(node, cluster.api_port)}/health",
                    timeout=20.0,
                ): node.id
                for node in cluster.nodes
            }
            current_task_samples = [future.result() for future in task_futures]
            health_by_node = {
                node_id: future.result()
                for future, node_id in health_futures.items()
            }

        task_samples.extend(current_task_samples)
        for row in current_task_samples:
            if row["status"] != "running":
                raise RuntimeError(
                    f"Task left RUNNING during physical profile: {row['task_id']} status={row['status']}"
                )

        per_node_cpu: dict[str, float] = defaultdict(float)
        per_node_rss: dict[str, float] = defaultdict(float)
        for row in current_task_samples:
            if row["cpu_utilization_percent"] is not None:
                per_node_cpu[row["node_id"]] += float(row["cpu_utilization_percent"])
            if row["memory_rss_mb"] is not None:
                per_node_rss[row["node_id"]] += float(row["memory_rss_mb"])

        now = datetime.now(timezone.utc).isoformat()
        for node_id in STAGE5E2_LAYOUT:
            health = health_by_node[node_id]
            node_samples.append(
                {
                    "sample_index": sample_index,
                    "sampled_at_utc": now,
                    "node_id": node_id,
                    "owned_task_count": health.get("owned_task_count"),
                    "reserved_cpu_cores": health.get("reserved_cpu_cores"),
                    "reserved_memory_mb": health.get("reserved_memory_mb"),
                    "reserved_gpu_count": health.get("reserved_gpu_count"),
                    "resource_busy_fraction": health.get("resource_busy_fraction"),
                    "available_cpu_cores": health.get("available_cpu_cores"),
                    "available_memory_mb": health.get("available_memory_mb"),
                    "actual_task_cpu_percent": per_node_cpu.get(node_id),
                    "actual_task_rss_mb": per_node_rss.get(node_id),
                }
            )

        total_cpu = sum(per_node_cpu.values())
        total_rss = sum(per_node_rss.values())
        print(
            f"[sample {sample_index:02d}] 11/11 running "
            f"task_cpu={total_cpu:.1f}% task_rss={total_rss:.1f}MB"
        )
        sample_index += 1
        if time.monotonic() - started >= profile_seconds and sample_index >= 2:
            break
        time.sleep(sample_interval_seconds)
    return task_samples, node_samples


def summarize_tasks(
    *,
    launched: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    initial_progress: dict[str, float],
    cleanup: dict[str, tuple[bool, str]],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_task[str(sample["task_id"])].append(sample)
    rows: list[dict[str, Any]] = []
    for launched_row in sorted(launched, key=lambda row: int(row["task_index"])):
        run_id = str(launched_row["task_id"])
        task_samples = by_task.get(run_id, [])
        cpu = numeric([row.get("cpu_utilization_percent") for row in task_samples])
        rss = numeric([row.get("memory_rss_mb") for row in task_samples])
        progress = numeric([row.get("progress_completed_units") for row in task_samples])
        process_count = numeric([row.get("process_count") for row in task_samples])
        checkpoint = numeric([row.get("checkpoint_bytes") for row in task_samples])
        cleanup_ok, cleanup_status = cleanup.get(run_id, (False, "missing"))
        rows.append(
            {
                "task_index": launched_row["task_index"],
                "task_id": run_id,
                "definition_id": launched_row["definition_id"],
                "node_id": launched_row["node_id"],
                "class_id": launched_row["class_id"],
                "declared_cpu_cores": launched_row["request"].cpu_cores,
                "declared_memory_mb": launched_row["request"].memory_mb,
                "declared_gpu_count": launched_row["request"].gpu_count,
                "launched": True,
                "steady_running": bool(task_samples) and all(
                    row.get("status") == "running" for row in task_samples
                ),
                "initial_progress": initial_progress.get(run_id),
                "progress_min": min(progress) if progress else initial_progress.get(run_id, 0.0),
                "progress_max": max(progress) if progress else initial_progress.get(run_id, 0.0),
                "progress_advanced": bool(progress)
                and max(progress) > float(initial_progress.get(run_id, progress[0])),
                "telemetry_sample_count": len(task_samples),
                "cpu_sample_count": len(cpu),
                "mean_cpu_percent": statistics.fmean(cpu) if cpu else None,
                "p95_cpu_percent": percentile(cpu, 0.95),
                "rss_sample_count": len(rss),
                "mean_memory_rss_mb": statistics.fmean(rss) if rss else None,
                "max_memory_rss_mb": max(rss) if rss else None,
                "min_process_count": min(process_count) if process_count else None,
                "max_process_count": max(process_count) if process_count else None,
                "max_checkpoint_bytes": int(max(checkpoint)) if checkpoint else 0,
                "cleanup_ok": cleanup_ok,
                "cleanup_status": cleanup_status,
            }
        )
    return rows


def summarize_nodes(
    *,
    plan_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    node_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in node_samples:
        samples_by_node[str(sample["node_id"])].append(sample)
    for row in task_rows:
        tasks_by_node[str(row["node_id"])].append(row)

    output: list[dict[str, Any]] = []
    for plan in plan_rows:
        node_id = str(plan["node_id"])
        samples = samples_by_node.get(node_id, [])
        expected_cpu = float(plan["expected_reserved_cpu_cores"])
        expected_memory = int(plan["expected_reserved_memory_mb"])
        reserved_cpu = numeric([row.get("reserved_cpu_cores") for row in samples])
        reserved_memory = numeric([row.get("reserved_memory_mb") for row in samples])
        busy = numeric([row.get("resource_busy_fraction") for row in samples])
        available_cpu = numeric([row.get("available_cpu_cores") for row in samples])
        actual_cpu = numeric([row.get("actual_task_cpu_percent") for row in samples])
        actual_rss = numeric([row.get("actual_task_rss_mb") for row in samples])
        owned = numeric([row.get("owned_task_count") for row in samples])
        reservation_matches = bool(samples) and all(
            abs(value - expected_cpu) <= 1e-6 for value in reserved_cpu
        ) and all(abs(value - expected_memory) <= 1e-6 for value in reserved_memory)
        capacity_cpu = float(plan["effective_cpu_cores"])
        capacity_memory = float(plan["effective_memory_mb"])
        capacity_respected = (
            bool(samples)
            and all(value <= capacity_cpu + 1e-9 for value in reserved_cpu)
            and all(value <= capacity_memory + 1e-9 for value in reserved_memory)
            and all(value <= 1.0 + 1e-9 for value in busy)
            and all(int(value) == int(plan["task_count"]) for value in owned)
        )
        row = dict(plan)
        row.update(
            {
                "all_tasks_running": bool(tasks_by_node[node_id])
                and all(bool(item["steady_running"]) for item in tasks_by_node[node_id]),
                "reservation_matches_expected": reservation_matches,
                "capacity_respected": capacity_respected,
                "health_sample_count": len(samples),
                "max_reserved_cpu_cores": max(reserved_cpu) if reserved_cpu else None,
                "max_reserved_memory_mb": max(reserved_memory) if reserved_memory else None,
                "max_resource_busy_fraction": max(busy) if busy else None,
                "min_available_cpu_cores": min(available_cpu) if available_cpu else None,
                "actual_cpu_sample_count": len(actual_cpu),
                "mean_actual_task_cpu_percent": statistics.fmean(actual_cpu) if actual_cpu else None,
                "max_actual_task_cpu_percent": max(actual_cpu) if actual_cpu else None,
                "actual_rss_sample_count": len(actual_rss),
                "mean_actual_task_rss_mb": statistics.fmean(actual_rss) if actual_rss else None,
                "max_actual_task_rss_mb": max(actual_rss) if actual_rss else None,
            }
        )
        output.append(row)
    return output


def cleanup_runs(
    launched: list[dict[str, Any]],
    stop_timeout_seconds: float,
) -> dict[str, tuple[bool, str]]:
    def stop_one(row: dict[str, Any]) -> tuple[str, bool, str]:
        run_id = str(row["task_id"])
        try:
            state = task_state(row["api"], run_id, timeout=20.0)
            if state is None:
                return run_id, False, "state-missing"
            status = str(state.get("status") or "")
            if status in {"stopped", "completed"}:
                return run_id, True, status
            request_json(
                f"{row['api']}/tasks/{run_id}/stop",
                method="POST",
                timeout=stop_timeout_seconds,
            )
            final = task_state(row["api"], run_id, timeout=20.0) or {}
            final_status = str(final.get("status") or "unknown")
            return run_id, final_status in {"stopped", "completed"}, final_status
        except Exception as exc:
            return run_id, False, f"{type(exc).__name__}: {exc}"

    result: dict[str, tuple[bool, str]] = {}
    if not launched:
        return result
    with ThreadPoolExecutor(max_workers=len(launched)) as pool:
        futures = [pool.submit(stop_one, row) for row in launched]
        for future in as_completed(futures):
            run_id, ok, status = future.result()
            result[run_id] = (ok, status)
    return result


def main() -> int:
    args = parse_args()
    if args.profile_seconds <= 0 or args.sample_interval_seconds <= 0:
        raise ValueError("profile/sample intervals must be positive")
    if args.benchmark_iterations < 1:
        raise ValueError("benchmark iterations must be positive")

    stage5a_path = Path(args.stage5a_bundle)
    stage5e1_path = Path(args.stage5e1_bundle)
    stage4d1_path = Path(args.stage4d1_bundle)
    s5a = require_passed_bundle(stage5a_path, "Stage 5A")
    s5e1 = require_passed_bundle(stage5e1_path, "Stage 5E.1")
    require_passed_bundle(stage4d1_path, "Stage 4D.1")

    target_sha = str(s5a.get("target_git_sha") or "")
    if not target_sha or local_git_sha() != target_sha:
        raise RuntimeError(
            "Stage 5E.2 must run from the exact SHA frozen by its Stage 5A bundle: "
            f"local={local_git_sha()} stage5a={target_sha}"
        )
    if int(s5e1.get("passed_case_count") or 0) != 3:
        raise RuntimeError("Stage 5E.1 does not contain 3/3 passing workload smokes")

    cluster = load_cluster_config(args.cluster)
    node_by_id = {node.id: node for node in cluster.nodes}
    if set(node_by_id) != set(STAGE5E2_LAYOUT):
        raise RuntimeError("Stage 5E.2 requires the canonical seven-node cluster")

    capacities, requests = read_resource_model(stage4d1_path)
    signatures = maximal_packing_signatures(stage4d1_path)
    plan_rows = validate_physical_layout(
        capacities=capacities,
        requests=requests,
        maximal_signatures=signatures,
    )
    cluster_cpu = sum(float(capacities[node_id].cpu_cores or 0.0) for node_id in STAGE5E2_LAYOUT)
    planned_cpu = sum(float(row["expected_reserved_cpu_cores"]) for row in plan_rows)

    comparison_id = args.comparison_id or (
        f"stage5e2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 5E.2 physical heterogeneous maximal packing ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage5a={stage5a_path}")
    print(f"source_stage5e1={stage5e1_path}")
    print(f"source_stage4d1={stage4d1_path}")
    print(f"git_sha={target_sha}")
    print("mix=4 benchmark-json-medium + 4 llm-distilgpt2 + 3 dendro-r9-t1p0")
    print(
        f"planned_measured_cpu={planned_cpu:.6f}/{cluster_cpu:.1f} "
        f"({planned_cpu / cluster_cpu * 100.0:.2f}%)"
    )
    for row in plan_rows:
        print(
            f"[packing] {row['node_id']:16s} sig={row['packing_signature']:5s} "
            f"tasks={row['task_count']} cpu={float(row['expected_reserved_cpu_cores']):.6f} "
            f"maximal={row['is_frozen_maximal_packing']}"
        )

    launched: list[dict[str, Any]] = []
    task_samples: list[dict[str, Any]] = []
    node_samples: list[dict[str, Any]] = []
    initial_progress: dict[str, float] = {}
    cleanup: dict[str, tuple[bool, str]] = {}
    error: str | None = None

    try:
        hardened_preflight(cluster, target_sha)
        workload_asset_preflight(cluster, args)

        dendro_template = json.loads(Path(args.dendro_definition).read_text(encoding="utf-8"))
        node_ids = [node.id for node in cluster.nodes]
        prepared: list[dict[str, Any]] = []

        print("\n[prepare] registering 11 real workload definitions")
        for spec in physical_task_specs():
            class_slug = spec.class_id.replace("-", "_")
            definition_id = f"{comparison_id}-{spec.task_index:02d}-{class_slug}"
            definition = physical_definition(
                class_id=spec.class_id,
                definition_id=definition_id,
                request=requests[spec.class_id],
                node_ids=node_ids,
                seed=42 + spec.task_index,
                benchmark_iterations=args.benchmark_iterations,
                llm_model=args.llm_model,
                dendro_template=dendro_template,
                dendro_solver=args.dendro_solver,
                dendro_parameter_template=args.dendro_parameter_template,
            )
            owner = node_by_id[spec.node_id]
            api = base_url(owner, cluster.api_port)
            created = request_json(
                f"{api}/task-definitions",
                method="POST",
                payload=definition,
                timeout=args.request_timeout_seconds,
            )
            prepared.append(
                {
                    "task_index": spec.task_index,
                    "node_id": spec.node_id,
                    "class_id": spec.class_id,
                    "definition_id": str(created["definition_id"]),
                    "revision": int(created["revision"]),
                    "api": api,
                    "request": requests[spec.class_id],
                }
            )

        def launch_one(row: dict[str, Any]) -> dict[str, Any]:
            view = request_json(
                f"{row['api']}/task-runs",
                method="POST",
                payload={
                    "definition_id": row["definition_id"],
                    "revision": row["revision"],
                    "initial_owner_node_id": row["node_id"],
                    "idempotency_key": f"{comparison_id}:physical:{row['task_index']:02d}",
                    "auto_start": True,
                    "labels": {
                        "purpose": "stage5e2-physical-heterogeneous-packing",
                        "comparison_id": comparison_id,
                        "scheduler_mode": "operator_only",
                        "class_id": row["class_id"],
                        "initial_node_id": row["node_id"],
                    },
                },
                timeout=args.request_timeout_seconds,
            )
            return {**row, "task_id": str(view["run"]["run_id"])}

        print("[launch] starting all 11 workload processes concurrently")
        with ThreadPoolExecutor(max_workers=11) as pool:
            futures = [pool.submit(launch_one, row) for row in prepared]
            for future in as_completed(futures):
                row = future.result()
                launched.append(row)
                print(
                    f"  started {row['task_id']} {row['node_id']:16s} {row['class_id']}"
                )
        launched.sort(key=lambda row: int(row["task_index"]))
        if len(launched) != 11:
            raise RuntimeError(f"Only {len(launched)}/11 tasks launched")

        print("[steady] waiting for all 11 real workloads to run and make progress")
        initial_progress = wait_steady(
            launched=launched,
            cluster=cluster,
            timeout_seconds=args.ready_timeout_seconds,
        )
        print("[steady] 11/11 simultaneously running with observed progress")

        # Check the actual Magellan ledger before collecting physical telemetry.
        for plan in plan_rows:
            health = request_json(
                f"{base_url(node_by_id[str(plan['node_id'])], cluster.api_port)}/health",
                timeout=20.0,
            )
            expected_cpu = float(plan["expected_reserved_cpu_cores"])
            if abs(float(health.get("reserved_cpu_cores") or 0.0) - expected_cpu) > 1e-6:
                raise RuntimeError(
                    f"{plan['node_id']} reservation does not match frozen packing: "
                    f"actual={health.get('reserved_cpu_cores')} expected={expected_cpu}"
                )
            if float(health.get("resource_busy_fraction") or 0.0) > 1.0 + 1e-9:
                raise RuntimeError(f"{plan['node_id']} resource ledger oversubscribed")

        print(
            f"[profile] collecting {args.profile_seconds:.0f}s of concurrent task CPU/RSS telemetry"
        )
        task_samples, node_samples = profile_physical_packing(
            launched=launched,
            cluster=cluster,
            profile_seconds=args.profile_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {error}")
    finally:
        if launched:
            print("[cleanup] stopping physical packing tasks")
            cleanup = cleanup_runs(launched, args.stop_timeout_seconds)
            clean_count = sum(ok for ok, _ in cleanup.values())
            print(f"  stopped/completed {clean_count}/{len(launched)}")

    task_rows = summarize_tasks(
        launched=launched,
        samples=task_samples,
        initial_progress=initial_progress,
        cleanup=cleanup,
    )
    node_rows = summarize_nodes(
        plan_rows=plan_rows,
        task_rows=task_rows,
        node_samples=node_samples,
    )
    passed = error is None and stage5e2_passes(task_rows, node_rows)

    class_counts = defaultdict(int)
    for row in task_rows:
        class_counts[str(row["class_id"])] += 1
    cpu_values = numeric([row.get("actual_task_cpu_percent") for row in node_samples])
    # node_samples contains one row per node per sample, so this is per-node sample data.
    sample_totals: dict[int, float] = defaultdict(float)
    rss_totals: dict[int, float] = defaultdict(float)
    for row in node_samples:
        if row.get("actual_task_cpu_percent") is not None:
            sample_totals[int(row["sample_index"])] += float(row["actual_task_cpu_percent"])
        if row.get("actual_task_rss_mb") is not None:
            rss_totals[int(row["sample_index"])] += float(row["actual_task_rss_mb"])

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage5a_bundle": str(stage5a_path),
        "source_stage5e1_bundle": str(stage5e1_path),
        "source_stage4d1_bundle": str(stage4d1_path),
        "git_sha": target_sha,
        "task_count": len(task_rows),
        "expected_task_count": 11,
        "node_count": len(node_rows),
        "expected_node_count": 7,
        "class_counts": dict(class_counts),
        "expected_class_counts": EXPECTED_CLASS_COUNTS,
        "planned_reserved_cpu_cores": planned_cpu,
        "effective_cluster_cpu_cores": cluster_cpu,
        "planned_cpu_fraction": planned_cpu / cluster_cpu,
        "frozen_maximal_packing_nodes": sum(
            bool(row["is_frozen_maximal_packing"]) for row in node_rows
        ),
        "reservation_match_nodes": sum(
            bool(row["reservation_matches_expected"]) for row in node_rows
        ),
        "capacity_respected_nodes": sum(
            bool(row["capacity_respected"]) for row in node_rows
        ),
        "steady_running_tasks": sum(bool(row["steady_running"]) for row in task_rows),
        "cleanup_ok_tasks": sum(bool(row["cleanup_ok"]) for row in task_rows),
        "profile_sample_rounds": len(sample_totals),
        "mean_cluster_task_cpu_percent": (
            statistics.fmean(sample_totals.values()) if sample_totals else None
        ),
        "max_cluster_task_cpu_percent": max(sample_totals.values()) if sample_totals else None,
        "mean_cluster_task_cpu_fraction_of_14_cores": (
            statistics.fmean(sample_totals.values()) / (cluster_cpu * 100.0)
            if sample_totals else None
        ),
        "max_cluster_task_rss_mb": max(rss_totals.values()) if rss_totals else None,
        "error": error,
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage5e2_real_physical_heterogeneous_maximal_packing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Physically realize the exact 11-task, 88.36%-measured-CPU umax mix "
                "used by Stage 4D.2/4D.3 with actual benchmark, DistilGPT-2, and "
                "Dendro-GR processes on the hardened seven-node cluster."
            ),
            "layout": STAGE5E2_LAYOUT,
            "resource_requests": (
                "Each real task's Magellan resource_request is replaced only at the "
                "declaration layer with the exact Stage 4D.1 p95 request derived from "
                "Stage 4A.3. Runtime/checkpoint/compatibility behavior remains the real "
                "workload implementation validated by Stage 5E.1."
            ),
            "placement": (
                "Placement is operator-fixed and scheduler_mode=operator_only because "
                "Stage 5E.2 tests physical co-location and the resource model, not carbon "
                "policy choice. Each node is assigned a frozen maximal packing."
            ),
            "telemetry": (
                "Magellan /telemetry/tasks provides per-process-group CPU utilization, RSS, "
                "process count, checkpoint bytes, and progress while /health provides the "
                "declared reservation ledger. Samples are collected while all 11 tasks are "
                "simultaneously RUNNING."
            ),
            "pass_criteria": (
                "11/11 real tasks launch and remain running during the profile; every task "
                "has CPU and RSS samples; all seven node reservation ledgers exactly match "
                "the frozen Stage 4D.1 packing and remain <= capacity; all tasks clean up."
            ),
            "scope": (
                "Stage 5E.2 validates physical heterogeneous co-location. Actual-workload "
                "destination contention and policy comparisons remain Stage 5E.3/5E.4."
            ),
        },
    }

    write_csv(root / "planned_layout.csv", plan_rows, list(plan_rows[0].keys()))
    if task_samples:
        write_csv(root / "task_profile_samples.csv", task_samples, TASK_SAMPLE_FIELDS)
    else:
        write_csv(root / "task_profile_samples.csv", [], TASK_SAMPLE_FIELDS)
    if node_samples:
        write_csv(root / "node_profile_samples.csv", node_samples, NODE_SAMPLE_FIELDS)
    else:
        write_csv(root / "node_profile_samples.csv", [], NODE_SAMPLE_FIELDS)
    write_csv(root / "tasks.csv", task_rows, TASK_FIELDS)
    write_csv(root / "nodes.csv", node_rows, NODE_FIELDS)
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = (
        "STAGE_5E2_PHYSICAL_HETEROGENEOUS_PACKING_PASS"
        if passed
        else "STAGE_5E2_PHYSICAL_HETEROGENEOUS_PACKING_FAIL"
    )
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"tasks: {summary['steady_running_tasks']}/11 simultaneous real workloads")
    print(
        "mix: "
        f"benchmark={class_counts[BENCHMARK_CLASS_ID]} "
        f"llm={class_counts[LLM_CLASS_ID]} dendro={class_counts[DENDRO_CLASS_ID]}"
    )
    print(
        f"maximal_packings: {summary['frozen_maximal_packing_nodes']}/7 "
        f"reservation_matches: {summary['reservation_match_nodes']}/7 "
        f"capacity_respected: {summary['capacity_respected_nodes']}/7"
    )
    print(
        f"planned_cpu: {planned_cpu:.3f}/{cluster_cpu:.1f} "
        f"({planned_cpu / cluster_cpu * 100.0:.2f}%)"
    )
    if summary["mean_cluster_task_cpu_percent"] is not None:
        print(
            f"observed_task_cpu: mean={summary['mean_cluster_task_cpu_percent']:.1f}% "
            f"max={summary['max_cluster_task_cpu_percent']:.1f}% across cluster"
        )
    print(f"cleanup: {summary['cleanup_ok_tasks']}/11")
    if error:
        print(f"error: {error}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
