#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.llm_validation import (
    build_resume_validation,
    checkpoint_event_at_or_after,
    last_checkpoint_event,
)
from magellan.experiments.measurement import absolute_percent_error, signed_percent_error
from magellan.experiments.stage4a2 import fresh_run_idempotency_key, summarize_profile_samples


MIB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Hugging Face causal-LM training process, migrate its genuine "
            "model/optimizer/RNG checkpoint, and verify exact resume continuity."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--state-root", default="runtime-state-gcp-measurement")
    parser.add_argument(
        "--path",
        default="boston,virginia",
        help=(
            "Comma-separated migration path. Two nodes performs one migration; "
            "for example boston,virginia,boston,virginia also captures a warm "
            "second Boston->Virginia migration without creating another task."
        ),
    )
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--migrate-after-step", type=int, default=1)
    parser.add_argument("--steps-between-migrations", type=int, default=1)
    parser.add_argument("--post-path-steps", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--sleep-per-step", type=float, default=3.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=2.5,
        help="Minimum free disk required on every node in the validation path.",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--profile-seconds",
        type=float,
        default=0.0,
        help="Optional pre-migration telemetry profile window per hop.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Collect the initial-node workload profile and stop without migrating.",
    )
    parser.add_argument(
        "--profile-sample-interval-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
    )
    parser.add_argument(
        "--allow-fallback-edge",
        action="store_true",
        help="Permit a migration even if the cached WAN transfer model is fallback-only.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check node health, optional LLM dependencies, and disk without starting a task.",
    )
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


def try_json(url: str, timeout: float = 8.0) -> Any | None:
    try:
        return request_json(url, timeout=timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def base_url(node: Any, port: int) -> str:
    return f"http://{node.internal_ip}:{port}"


def remote_cd(path: str) -> str:
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        return f'cd "$HOME"/{shlex.quote(path[2:])}'
    return f"cd {shlex.quote(path)}"


def run_on_node(
    *,
    local_node_id: str,
    node: Any,
    ssh_user: str,
    command: str,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if node.id == local_node_id:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{ssh_user}@{node.internal_ip}",
            command,
        ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def available_bytes(
    *,
    local_node_id: str,
    node: Any,
    ssh_user: str,
    remote_repo: str,
) -> int:
    result = run_on_node(
        local_node_id=local_node_id,
        node=node,
        ssh_user=ssh_user,
        command=(
            f"{remote_cd(remote_repo)} && "
            "df -Pk . | tail -1 | awk '{print $4}'"
        ),
        timeout=30.0,
    )
    return int(result.stdout.strip().splitlines()[-1]) * 1024


def llm_dependency_versions(
    *,
    local_node_id: str,
    node: Any,
    ssh_user: str,
    remote_repo: str,
) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m, json, torch, transformers; "
        "print(json.dumps({'torch': torch.__version__, "
        "'transformers': transformers.__version__, "
        "'safetensors': m.version('safetensors')}))"
    )
    command = (
        f"{remote_cd(remote_repo)} && source .venv/bin/activate && "
        f"python -c {shlex.quote(code)}"
    )
    try:
        result = run_on_node(
            local_node_id=local_node_id,
            node=node,
            ssh_user=ssh_user,
            command=command,
            timeout=60.0,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "missing LLM dependencies").strip()
        raise RuntimeError(
            f"LLM dependencies are not ready on {node.id}: {detail}. "
            "Activate ~/Magellan-V2/.venv and install `torch` plus `-e '.[llm]'`."
        ) from exc
    return json.loads(result.stdout.strip().splitlines()[-1])


def task_state(api: str, run_id: str) -> dict[str, Any] | None:
    payload = try_json(f"{api}/tasks")
    if not isinstance(payload, dict):
        return None
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == run_id:
            return state
    return None


def query_events(api: str, after_sequence: int, run_id: str) -> list[dict[str, Any]]:
    query = urlencode(
        {"after_sequence": after_sequence, "task_id": run_id, "limit": 10000}
    )
    payload = request_json(f"{api}/experiment/events?{query}")
    return list(payload.get("events", []))


def remote_task_path(
    *,
    remote_repo: str,
    ssh_user: str,
    state_root: str,
    run_id: str,
    relative: str,
) -> str:
    if remote_repo == "~":
        root = f"/home/{ssh_user}"
    elif remote_repo.startswith("~/"):
        root = f"/home/{ssh_user}/{remote_repo[2:]}"
    else:
        root = remote_repo
    return f"{root}/{state_root}/tasks/{run_id}/{relative}"


def read_remote_text(
    *,
    local_node_id: str,
    node: Any,
    ssh_user: str,
    path: str,
) -> str | None:
    command = f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || true"
    result = run_on_node(
        local_node_id=local_node_id,
        node=node,
        ssh_user=ssh_user,
        command=command,
        timeout=30.0,
    )
    value = result.stdout.strip()
    return value or None


def read_remote_json(**kwargs: Any) -> dict[str, Any] | None:
    value = read_remote_text(**kwargs)
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_remote_jsonl(**kwargs: Any) -> list[dict[str, Any]]:
    value = read_remote_text(**kwargs)
    if value is None:
        return []
    events: list[dict[str, Any]] = []
    for raw in value.splitlines():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def task_file(
    *,
    remote_repo: str,
    ssh_user: str,
    state_root: str,
    run_id: str,
    relative: str,
) -> str:
    return remote_task_path(
        remote_repo=remote_repo,
        ssh_user=ssh_user,
        state_root=state_root,
        run_id=run_id,
        relative=relative,
    )


def read_progress(
    *,
    args: argparse.Namespace,
    node: Any,
    run_id: str,
) -> dict[str, Any] | None:
    return read_remote_json(
        local_node_id=args.local_node_id,
        node=node,
        ssh_user=args.ssh_user,
        path=task_file(
            remote_repo=args.remote_repo,
            ssh_user=args.ssh_user,
            state_root=args.state_root,
            run_id=run_id,
            relative="runtime/progress.json",
        ),
    )


def read_ready(
    *,
    args: argparse.Namespace,
    node: Any,
    run_id: str,
) -> dict[str, Any] | None:
    return read_remote_json(
        local_node_id=args.local_node_id,
        node=node,
        ssh_user=args.ssh_user,
        path=task_file(
            remote_repo=args.remote_repo,
            ssh_user=args.ssh_user,
            state_root=args.state_root,
            run_id=run_id,
            relative="runtime/ready.json",
        ),
    )


def read_checkpoint_metrics(
    *,
    args: argparse.Namespace,
    node: Any,
    run_id: str,
) -> list[dict[str, Any]]:
    return read_remote_jsonl(
        local_node_id=args.local_node_id,
        node=node,
        ssh_user=args.ssh_user,
        path=task_file(
            remote_repo=args.remote_repo,
            ssh_user=args.ssh_user,
            state_root=args.state_root,
            run_id=run_id,
            relative="runtime/checkpoint-metrics.jsonl",
        ),
    )


def wait_definition(
    api: str,
    definition_id: str,
    revision: int,
    digest: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = try_json(f"{api}/task-definitions/{definition_id}?revision={revision}")
        if isinstance(value, dict) and value.get("digest") == digest:
            return
        time.sleep(1)
    raise TimeoutError(f"Definition {definition_id}@{revision} did not converge to {api}")


def wait_checkpointed_progress(
    *,
    args: argparse.Namespace,
    node: Any,
    api: str,
    run_id: str,
    minimum_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        if state is not None and state.get("status") in {"failed", "completed"}:
            raise RuntimeError(f"LLM task terminated before migration: {state}")
        progress = read_progress(args=args, node=node, run_id=run_id)
        metrics = read_checkpoint_metrics(args=args, node=node, run_id=run_id)
        checkpoint = checkpoint_event_at_or_after(
            metrics,
            minimum_steps=minimum_steps,
            reasons={"periodic"},
        )
        telemetry = try_json(f"{api}/telemetry/tasks/{run_id}")
        if (
            state is not None
            and state.get("status") == "running"
            and progress is not None
            and int(progress.get("completed_units", -1)) >= minimum_steps
            and checkpoint is not None
            and isinstance(telemetry, dict)
            and int(telemetry.get("checkpoint_bytes") or 0) > 0
        ):
            return progress, checkpoint, telemetry
        time.sleep(args.poll_seconds)
    raise TimeoutError(
        f"Task {run_id} did not reach checkpointed training step {minimum_steps}"
    )


def collect_profile_samples(
    *,
    args: argparse.Namespace,
    node: Any,
    api: str,
    run_id: str,
    hop: int,
) -> list[dict[str, Any]]:
    if args.profile_seconds <= 0:
        return []
    samples: list[dict[str, Any]] = []
    last_telemetry_sample_at: str | None = None
    deadline = time.monotonic() + args.profile_seconds
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        if state is None or state.get("status") != "running":
            raise RuntimeError(f"LLM left RUNNING during profile window: {state}")
        telemetry = request_json(f"{api}/telemetry/tasks/{run_id}")
        telemetry_sample_at = telemetry.get("last_sample_at_utc")
        if telemetry_sample_at == last_telemetry_sample_at:
            time.sleep(min(args.profile_sample_interval_seconds, 1.0))
            continue
        last_telemetry_sample_at = telemetry_sample_at
        progress = read_progress(args=args, node=node, run_id=run_id) or {}
        samples.append(
            {
                "hop": hop,
                "telemetry_last_sample_at_utc": telemetry_sample_at,
                "telemetry_sample_count": telemetry.get("sample_count"),
                "node_id": node.id,
                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "progress_completed_units": progress.get("completed_units"),
                "progress_total_units": progress.get("total_units"),
                "process_count": telemetry.get("process_count"),
                "process_state": telemetry.get("process_state"),
                "cpu_utilization_percent": telemetry.get("cpu_utilization_percent"),
                "memory_rss_mb": telemetry.get("memory_rss_mb"),
                "checkpoint_bytes": telemetry.get("checkpoint_bytes"),
                "measured_power_kw": telemetry.get("measured_power_kw"),
                "power_source": telemetry.get("power_source"),
                "power_confidence": telemetry.get("power_confidence"),
                "progress_rate_units_per_second": telemetry.get("progress_rate_units_per_second"),
                "estimated_remaining_seconds": telemetry.get("estimated_remaining_seconds"),
                "telemetry_freshness": telemetry.get("freshness"),
                "telemetry_age_seconds": telemetry.get("age_seconds"),
            }
        )
        time.sleep(args.profile_sample_interval_seconds)
    return samples


def wait_resumed_progress(
    *,
    args: argparse.Namespace,
    node: Any,
    api: str,
    run_id: str,
    source_completed_steps: int,
    additional_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    minimum = source_completed_steps + additional_steps
    deadline = time.monotonic() + args.timeout_seconds
    last_ready: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        if state is not None and state.get("status") in {"failed", "completed"}:
            raise RuntimeError(f"LLM task terminated after migration: {state}")
        ready = read_ready(args=args, node=node, run_id=run_id)
        if ready is not None:
            last_ready = ready
        progress = read_progress(args=args, node=node, run_id=run_id)
        if (
            state is not None
            and state.get("status") == "running"
            and last_ready is not None
            and progress is not None
            and int(progress.get("completed_units", -1)) >= minimum
        ):
            return last_ready, progress
        time.sleep(args.poll_seconds)
    raise TimeoutError(
        f"Task {run_id} did not continue {additional_steps} step(s) after resume"
    )


def build_definition(
    *,
    definition_id: str,
    initial_node_id: str,
    node_ids: list[str],
    model: str,
    checkpoint_every: int,
    sleep_per_step: float,
    torch_threads: int,
) -> dict[str, Any]:
    training_text = (
        "Magellan migrates long-running stateful machine-learning workloads across "
        "geographically distributed computing regions while preserving optimizer state."
    )
    return {
        "definition_id": definition_id,
        "profile": {
            "workload_type": "causal-lm-training-validation",
            "power_kw": 0.08,
            "checkpoint_bytes": 0,
            "data_bytes": 0,
            "prestaged_node_ids": node_ids,
            "estimated_remaining_seconds": 86400,
            "accumulated_cost_usd": 0,
            "cost_cap_usd": 10.0,
            "priority": 50,
            "deadline_at_utc": None,
            "resource_request": {
                "cpu_cores": 2,
                "memory_mb": 3072,
                "gpu_count": 0,
                "accelerator_type": None,
            },
            "compatibility": {
                "architectures": ["x86_64"],
                "operating_systems": ["linux"],
                "minimum_cpu_cores": 2,
                "minimum_memory_mb": 3072,
                "required_commands": ["python3"],
                "required_runtimes": {"python": ">=3.11,<3.12"},
                "required_features": ["python-module", "application-checkpoint"],
                "checkpoint_architecture_independent": True,
            },
        },
        "runtime": {
            "module": "magellan.workloads.llm_train",
            "arguments": [
                "--checkpoint-dir",
                "{checkpoint_directory}",
                "--ready-file",
                "{readiness_file}",
                "--progress-file",
                "{progress_file}",
                "--checkpoint-metrics-file",
                "{task_directory}/runtime/checkpoint-metrics.jsonl",
                "--model",
                model,
                "--max-steps",
                "1000000",
                "--sleep-per-step",
                str(sleep_per_step),
                "--checkpoint-every",
                str(checkpoint_every),
                "--learning-rate",
                "0.00005",
                "--device",
                "cpu",
                "--torch-threads",
                str(torch_threads),
                "--text",
                training_text,
                "--completion-file",
                "{completion_file}",
                "--output-dir",
                "{output_directory}",
            ],
            "environment": {"TOKENIZERS_PARALLELISM": "false"},
            "working_directory": ".",
            "checkpoint_relative_path": "checkpoint/complete.json",
            "checkpoint_manifest_relative_path": "complete.json",
            "readiness_relative_path": "runtime/ready.json",
            "readiness_timeout_seconds": 1200,
            "progress_relative_path": "runtime/progress.json",
            "completion_relative_path": "runtime/completion.json",
            "output_relative_directory": "output",
            "stop_timeout_seconds": 600,
            "minimum_process_count": 1,
        },
        "artifacts": [],
    }


def parse_path(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("--path requires at least two node IDs")
    for left, right in zip(values, values[1:]):
        if left == right:
            raise ValueError("Consecutive path nodes must differ")
    return values


def main() -> int:
    args = parse_args()
    if args.migrate_after_step < 1:
        raise ValueError("--migrate-after-step must be positive")
    if args.steps_between_migrations < 1:
        raise ValueError("--steps-between-migrations must be positive")
    if args.post_path_steps < 1:
        raise ValueError("--post-path-steps must be positive")
    if args.checkpoint_every < 1:
        raise ValueError("LLM migration validation requires --checkpoint-every >= 1")
    if args.minimum_free_gib <= 0:
        raise ValueError("--minimum-free-gib must be positive")
    if args.profile_seconds < 0 or args.profile_sample_interval_seconds <= 0:
        raise ValueError("LLM profile intervals must be non-negative/positive")

    path = parse_path(args.path)
    cluster = load_cluster_config(args.cluster)
    local = cluster.get_node(args.local_node_id)
    node_by_id = {node.id: node for node in cluster.nodes}
    for node_id in path:
        cluster.get_node(node_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurement_id = args.measurement_id or f"llm-{timestamp}-{uuid4().hex[:8]}"
    bundle = Path(args.measurements_root) / measurement_id
    if bundle.exists():
        raise FileExistsError(f"Measurement bundle already exists: {bundle}")

    unique_path_nodes = list(dict.fromkeys(path))
    health: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, dict[str, Any]] = {}
    disk_before: dict[str, int] = {}

    print("== Stage 3B.1 real LLM migration validation ==")
    print(f"measurement_id={measurement_id}")
    print(f"model={args.model}")
    print(f"path={' -> '.join(path)}")

    for node_id in unique_path_nodes:
        node = node_by_id[node_id]
        api = base_url(node, cluster.api_port)
        value = request_json(f"{api}/health")
        if value.get("node_id") != node_id:
            raise RuntimeError(f"Node identity mismatch on {node_id}")
        if (
            args.expected_carbon_metric
            and value.get("carbon_metric") != args.expected_carbon_metric
        ):
            raise RuntimeError(
                f"{node_id} carbon metric={value.get('carbon_metric')} "
                f"expected={args.expected_carbon_metric}"
            )
        state_file = str(value.get("telemetry_state_file", ""))
        if args.expected_state_token and args.expected_state_token not in state_file:
            raise RuntimeError(
                f"{node_id} is not in isolated measurement state: {state_file}"
            )
        health[node_id] = value
        dependencies[node_id] = llm_dependency_versions(
            local_node_id=local.id,
            node=node,
            ssh_user=args.ssh_user,
            remote_repo=args.remote_repo,
        )
        disk_before[node_id] = available_bytes(
            local_node_id=local.id,
            node=node,
            ssh_user=args.ssh_user,
            remote_repo=args.remote_repo,
        )
        minimum_free = int(args.minimum_free_gib * 1024**3)
        if disk_before[node_id] < minimum_free:
            raise RuntimeError(
                f"Insufficient preflight disk on {node_id}: "
                f"free={disk_before[node_id]} required={minimum_free}"
            )
        print(
            f"[OK] {node_id:16} free={disk_before[node_id] / (1024**3):.2f}GiB "
            f"torch={dependencies[node_id]['torch']} "
            f"transformers={dependencies[node_id]['transformers']}"
        )

    if args.preflight_only:
        print("LLM EXPERIMENT PREFLIGHT PASSED")
        return 0

    initial_id = path[0]
    initial = node_by_id[initial_id]
    initial_api = base_url(initial, cluster.api_port)
    definition_id = f"llm-migration-{measurement_id[-8:]}"
    definition = build_definition(
        definition_id=definition_id,
        initial_node_id=initial_id,
        node_ids=unique_path_nodes,
        model=args.model,
        checkpoint_every=args.checkpoint_every,
        sleep_per_step=args.sleep_per_step,
        torch_threads=args.torch_threads,
    )
    created = request_json(
        f"{initial_api}/task-definitions",
        method="POST",
        payload=definition,
        timeout=30.0,
    )
    for node_id in unique_path_nodes:
        wait_definition(
            base_url(node_by_id[node_id], cluster.api_port),
            created["definition_id"],
            int(created["revision"]),
            created["digest"],
            min(180.0, args.timeout_seconds),
        )
    print(f"[definition] {created['definition_id']}@{created['revision']} converged")

    run_request = {
        "definition_id": created["definition_id"],
        "revision": created["revision"],
        "initial_owner_node_id": initial_id,
        "idempotency_key": fresh_run_idempotency_key(measurement_id),
        "auto_start": True,
        "labels": {
            "purpose": "real-llm-migration-validation",
            "scheduler_mode": "operator_only",
            "measurement_id": measurement_id,
            "model": args.model,
        },
    }
    run_view = request_json(
        f"{initial_api}/task-runs",
        method="POST",
        payload=run_request,
        timeout=args.timeout_seconds,
    )
    run_id = run_view["run"]["run_id"]
    print(f"[run] {run_id} owner={initial_id}")

    (bundle / "raw").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    profile_samples: list[dict[str, Any]] = []
    current_minimum_step = args.migrate_after_step

    for hop_index, (source_id, destination_id) in enumerate(
        zip(path, path[1:]), start=1
    ):
        source = node_by_id[source_id]
        destination = node_by_id[destination_id]
        source_api = base_url(source, cluster.api_port)
        destination_api = base_url(destination, cluster.api_port)

        progress_before, periodic_checkpoint, task_telemetry = wait_checkpointed_progress(
            args=args,
            node=source,
            api=source_api,
            run_id=run_id,
            minimum_steps=current_minimum_step,
        )
        checkpoint_bytes = int(task_telemetry.get("checkpoint_bytes") or 0)
        if checkpoint_bytes <= 0:
            raise RuntimeError("LLM task telemetry did not expose checkpoint size")

        profile_samples.extend(
            collect_profile_samples(
                args=args,
                node=source,
                api=source_api,
                run_id=run_id,
                hop=hop_index,
            )
        )

        if args.profile_only:
            try:
                request_json(
                    f"{source_api}/tasks/{run_id}/stop",
                    method="POST",
                    timeout=args.timeout_seconds,
                )
            except Exception as exc:
                print(f"[WARN] profile-only task stop failed: {type(exc).__name__}: {exc}")
            if profile_samples:
                write_csv(
                    bundle / "profile_samples.csv",
                    profile_samples,
                    list(profile_samples[0].keys()),
                )
            profile_summary = summarize_profile_samples(profile_samples)
            summary = {
                "measurement_id": measurement_id,
                "model": args.model,
                "workload": "llm",
                "variant": args.model,
                "source_node_id": source_id,
                "run_id": run_id,
                "profile": profile_summary,
                "profile_only": True,
                "passed": bool(profile_samples),
            }
            metadata = {
                "format_version": 1,
                "measurement_type": "stage4a3_workload_profile",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "measurement_id": measurement_id,
                "model": args.model,
                "source_node_id": source_id,
                "profile_seconds": args.profile_seconds,
                "profile_sample_interval_seconds": args.profile_sample_interval_seconds,
                "health": health,
                "dependencies": dependencies,
                "disk_before": disk_before,
                "definition": definition,
                "created_definition": created,
                "run": run_view,
                "periodic_checkpoint": periodic_checkpoint,
                "methodology": {
                    "profile_mode": (
                        "Profile a scheduler-isolated real LLM training process on the initial "
                        "final-hardware node, retain application checkpoint telemetry, then stop "
                        "without migration."
                    )
                },
            }
            write_json(bundle / "metadata.json", metadata)
            write_json(bundle / "summary.json", summary)
            write_checksums(bundle)
            if not summary["passed"]:
                raise RuntimeError("LLM profile-only run produced no telemetry samples")
            print("STAGE_4A3_PROFILE_MEASUREMENT_PASS")
            print(f"bundle: {bundle}")
            print(f"run_id: {run_id}")
            print(f"samples: {len(profile_samples)}")
            return 0

        required_free = checkpoint_bytes * 2 + 512 * MIB
        for node in (source, destination):
            free = available_bytes(
                local_node_id=local.id,
                node=node,
                ssh_user=args.ssh_user,
                remote_repo=args.remote_repo,
            )
            if free < required_free:
                raise RuntimeError(
                    f"Insufficient disk on {node.id}: free={free} "
                    f"required={required_free} checkpoint={checkpoint_bytes}"
                )

        edge = request_json(f"{source_api}/telemetry/edges/{destination_id}")
        if (
            not args.allow_fallback_edge
            and edge.get("transfer_model_source") == "configured_fallback"
        ):
            raise RuntimeError(
                f"Cached WAN model for {source_id}->{destination_id} is fallback-only. "
                "Run scripts/refresh_cluster_edge_telemetry.py before the LLM experiment."
            )

        calibration_before = request_json(f"{source_api}/telemetry/calibration")
        edge_calibration = next(
            (
                item
                for item in calibration_before
                if item.get("source_node_id") == source_id
                and item.get("destination_node_id") == destination_id
            ),
            None,
        )
        event_status = request_json(f"{source_api}/experiment/events/status")
        event_start = int(event_status.get("last_sequence", 0))

        print(
            f"[hop {hop_index}] {source_id}->{destination_id} "
            f"step={progress_before['completed_units']} "
            f"checkpoint={checkpoint_bytes / MIB:.1f}MiB"
        )
        migration_response = request_json(
            f"{source_api}/tasks/{run_id}/migrate/{destination_id}",
            method="POST",
            timeout=args.timeout_seconds,
        )
        if not migration_response.get("migrated"):
            raise RuntimeError(f"Migration did not complete: {migration_response}")

        source_metrics = read_checkpoint_metrics(args=args, node=source, run_id=run_id)
        shutdown_checkpoint = last_checkpoint_event(source_metrics, reason="shutdown")
        if shutdown_checkpoint is None:
            raise RuntimeError("Source LLM did not record its stop-induced checkpoint")

        ready, progress_after = wait_resumed_progress(
            args=args,
            node=destination,
            api=destination_api,
            run_id=run_id,
            source_completed_steps=int(shutdown_checkpoint["completed_steps"]),
            additional_steps=(
                args.post_path_steps
                if hop_index == len(path) - 1
                else args.steps_between_migrations
            ),
        )
        resume_validation = build_resume_validation(
            source_checkpoint=shutdown_checkpoint,
            destination_ready=ready,
            destination_progress=progress_after,
        )
        if not resume_validation.passed:
            raise RuntimeError(
                "LLM checkpoint resume validation failed: "
                f"{resume_validation.as_dict()}"
            )

        events = query_events(source_api, event_start, run_id)
        completed_events = [
            item for item in events if item.get("event_type") == "migration_completed"
        ]
        if not completed_events:
            raise RuntimeError(f"No migration_completed event for hop {hop_index}")
        migration_event = completed_events[-1]
        actual = migration_event["payload"]
        bid = migration_response.get("bid") or {}
        candidate = bid.get("candidate") or {}
        predicted = candidate.get("details") or {}

        predicted_checkpoint = float(predicted.get("checkpoint_seconds", 0.0))
        predicted_transfer = float(predicted.get("transfer_seconds", 0.0))
        predicted_restore = float(predicted.get("restore_seconds", 0.0))
        predicted_overhead = float(predicted.get("migration_overhead_seconds", 0.0))
        predicted_downtime = float(
            predicted.get(
                "predicted_downtime_seconds",
                predicted_checkpoint
                + predicted_transfer
                + predicted_restore
                + predicted_overhead,
            )
        )
        actual_checkpoint = float(actual["checkpoint_seconds"])
        actual_transfer = float(actual["transfer_seconds"])
        actual_restore = float(actual["restore_seconds"])
        actual_overhead = float(actual.get("migration_overhead_seconds", 0.0))
        actual_downtime = float(actual["total_downtime_seconds"])

        row = {
            "measurement_id": measurement_id,
            "run_id": run_id,
            "hop": hop_index,
            "model": args.model,
            "source_node_id": source_id,
            "destination_node_id": destination_id,
            "source_progress_before_request": progress_before.get("completed_units"),
            "source_periodic_checkpoint_step": periodic_checkpoint.get("completed_steps"),
            "source_shutdown_checkpoint_step": shutdown_checkpoint.get("completed_steps"),
            "source_shutdown_checkpoint_id": shutdown_checkpoint.get("checkpoint_id"),
            "source_app_checkpoint_seconds": shutdown_checkpoint.get("duration_seconds"),
            "source_app_checkpoint_bytes": shutdown_checkpoint.get("size_bytes"),
            "telemetry_checkpoint_bytes_before": checkpoint_bytes,
            "actual_checkpoint_bytes": actual.get("checkpoint_bytes"),
            "destination_resumed_checkpoint_id": ready.get("resumed_checkpoint_id"),
            "destination_ready_steps": ready.get("completed_steps"),
            "destination_progress_after_resume": progress_after.get("completed_units"),
            "destination_optimizer_state_loaded": ready.get("optimizer_state_loaded"),
            "checkpoint_id_matches": resume_validation.checkpoint_id_matches,
            "resumed_at_same_step": resume_validation.resumed_at_same_step,
            "progress_continued": resume_validation.progress_continued,
            "resume_validation_passed": resume_validation.passed,
            "candidate_calibration_source": predicted.get("calibration_source"),
            "candidate_transfer_model": predicted.get("transfer_model"),
            "candidate_transfer_model_source": predicted.get("transfer_model_source"),
            "cached_transfer_model_source": edge.get("transfer_model_source"),
            "cached_transfer_model_age_seconds": edge.get("transfer_model_age_seconds"),
            "calibration_sample_count_before": (
                edge_calibration.get("sample_count") if edge_calibration else 0
            ),
            "predicted_checkpoint_seconds": predicted_checkpoint,
            "actual_checkpoint_seconds": actual_checkpoint,
            "checkpoint_error_percent": signed_percent_error(
                predicted_checkpoint, actual_checkpoint
            ),
            "predicted_transfer_seconds": predicted_transfer,
            "actual_transfer_seconds": actual_transfer,
            "transfer_error_percent": signed_percent_error(
                predicted_transfer, actual_transfer
            ),
            "predicted_restore_seconds": predicted_restore,
            "actual_restore_seconds": actual_restore,
            "restore_error_percent": signed_percent_error(
                predicted_restore, actual_restore
            ),
            "predicted_migration_overhead_seconds": predicted_overhead,
            "actual_migration_overhead_seconds": actual_overhead,
            "predicted_downtime_seconds": predicted_downtime,
            "actual_downtime_seconds": actual_downtime,
            "downtime_error_percent": signed_percent_error(
                predicted_downtime, actual_downtime
            ),
        }
        rows.append(row)
        write_json(
            bundle / "raw" / f"hop-{hop_index:02d}.json",
            {
                "progress_before": progress_before,
                "periodic_checkpoint": periodic_checkpoint,
                "source_shutdown_checkpoint": shutdown_checkpoint,
                "edge_telemetry": edge,
                "calibration_before": edge_calibration,
                "migration_response": migration_response,
                "migration_event": migration_event,
                "destination_ready": ready,
                "destination_progress": progress_after,
                "resume_validation": resume_validation.as_dict(),
            },
        )
        print(
            f"[verified] checkpoint_id_match=yes optimizer=yes "
            f"progress={shutdown_checkpoint['completed_steps']}"
            f"->{progress_after['completed_units']} "
            f"downtime={actual_downtime:.3f}s"
        )
        current_minimum_step = (
            int(progress_after["completed_units"]) + args.steps_between_migrations
        )

    final_node_id = path[-1]
    final_node = node_by_id[final_node_id]
    final_api = base_url(final_node, cluster.api_port)
    try:
        request_json(
            f"{final_api}/tasks/{run_id}/stop",
            method="POST",
            timeout=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"[WARN] final task stop failed: {type(exc).__name__}: {exc}")

    fieldnames = list(rows[0]) if rows else []
    write_csv(bundle / "llm_migrations.csv", rows, fieldnames)
    if profile_samples:
        write_csv(
            bundle / "profile_samples.csv",
            profile_samples,
            list(profile_samples[0].keys()),
        )

    calibrated = [
        row
        for row in rows
        if row.get("candidate_calibration_source") != "configured_fallback"
    ]
    transfer_apes = [
        value
        for row in calibrated
        if (
            value := absolute_percent_error(
                float(row["predicted_transfer_seconds"]),
                float(row["actual_transfer_seconds"]),
            )
        )
        is not None
    ]
    downtime_apes = [
        value
        for row in calibrated
        if (
            value := absolute_percent_error(
                float(row["predicted_downtime_seconds"]),
                float(row["actual_downtime_seconds"]),
            )
        )
        is not None
    ]
    summary = {
        "measurement_id": measurement_id,
        "model": args.model,
        "path": path,
        "run_id": run_id,
        "migration_count": len(rows),
        "resume_validations_passed": sum(
            bool(row["resume_validation_passed"]) for row in rows
        ),
        "optimizer_resume_validations_passed": sum(
            bool(row["destination_optimizer_state_loaded"]) for row in rows
        ),
        "calibrated_migration_count": len(calibrated),
        "checkpoint_bytes_median": (
            median(float(row["actual_checkpoint_bytes"]) for row in rows)
            if rows
            else None
        ),
        "application_checkpoint_seconds_median": (
            median(float(row["source_app_checkpoint_seconds"]) for row in rows)
            if rows
            else None
        ),
        "calibrated_transfer_ape_median_pct": (
            median(transfer_apes) if transfer_apes else None
        ),
        "calibrated_downtime_ape_median_pct": (
            median(downtime_apes) if downtime_apes else None
        ),
        "profile": summarize_profile_samples(profile_samples),
        "passed": bool(rows)
        and all(bool(row["resume_validation_passed"]) for row in rows),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "real_llm_migration_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_id": measurement_id,
        "model": args.model,
        "path": path,
        "checkpoint_every": args.checkpoint_every,
        "migrate_after_step": args.migrate_after_step,
        "steps_between_migrations": args.steps_between_migrations,
        "post_path_steps": args.post_path_steps,
        "profile_seconds": args.profile_seconds,
        "profile_sample_interval_seconds": args.profile_sample_interval_seconds,
        "carbon_metric": args.expected_carbon_metric,
        "state_root": args.state_root,
        "health": health,
        "dependencies": dependencies,
        "disk_free_bytes_before": disk_before,
        "definition": definition,
        "created_definition": created,
        "run": run_view,
        "methodology": {
            "checkpoint_contents": [
                "Hugging Face model weights/config",
                "tokenizer",
                "AdamW optimizer state",
                "PyTorch RNG state",
                "completed training step",
            ],
            "correctness_checks": [
                "source stop checkpoint ID equals destination resumed checkpoint ID",
                "destination loads optimizer state",
                "destination resumes at exactly the source checkpoint step",
                "training progress advances after restore",
            ],
            "scheduler_role": (
                "Operator endpoint forces the validation path, but the migration uses the "
                "same candidate scoring, bid/reservation, checkpoint, rsync, activation, "
                "restore, ownership, and telemetry path as Magellan migration."
            ),
        },
    }
    write_json(bundle / "metadata.json", metadata)
    write_json(bundle / "summary.json", summary)
    write_checksums(bundle)

    if not summary["passed"]:
        raise RuntimeError("Real LLM migration validation did not pass")

    print("\nREAL LLM MIGRATION VALIDATION PASSED")
    print(f"bundle: {bundle}")
    print(f"run_id: {run_id}")
    print(f"migrations: {len(rows)}")
    print(f"checkpoint_bytes_median: {summary['checkpoint_bytes_median']}")
    print(
        "application_checkpoint_seconds_median: "
        f"{summary['application_checkpoint_seconds_median']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
