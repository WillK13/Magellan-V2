#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.stage4a2 import fresh_run_idempotency_key
from magellan.experiments.stage4a4 import llm_training_definition
from magellan.experiments.workload_population import benchmark_definition, dendro_definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure one scheduler-isolated Stage 4A.4 static completion run.")
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--node", required=True)
    parser.add_argument("--workload", choices=["benchmark", "dendro", "llm"], required=True)
    parser.add_argument("--benchmark", choices=["nbody", "json", "matmul"], default=None)
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--benchmark-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dendro-definition", default="config/submissions/dendro-bssn-template.json")
    parser.add_argument("--dendro-solver", default="/home/WILL/dgr-build/BSSN_GR/bssnSolver")
    parser.add_argument("--dendro-parameter-template", default="/home/WILL/q1-magellan-magellan.toml")
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--model", default="experiment-assets/models/distilgpt2")
    parser.add_argument("--llm-max-steps", type=int, default=None)
    parser.add_argument("--llm-checkpoint-every", type=int, default=1)
    parser.add_argument("--llm-sleep-per-step", type=float, default=2.0)
    parser.add_argument("--llm-torch-threads", type=int, default=2)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", required=True)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--scope", choices=["canonical", "equivalence"], required=True)
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument("--expected-state-token", default="runtime-state-gcp-measurement")
    return parser.parse_args()


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def base_url(node: Any, port: int) -> str:
    return f"http://{node.internal_ip}:{port}"


def task_state(api: str, run_id: str) -> dict[str, Any] | None:
    payload = request_json(f"{api}/tasks")
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == run_id:
            return state
    return None


def idle_preflight(api: str, node_id: str, expected_carbon_metric: str, expected_state_token: str) -> dict[str, Any]:
    health = request_json(f"{api}/health")
    auction = request_json(f"{api}/auction/status")
    capabilities = request_json(f"{api}/capabilities")
    errors: list[str] = []
    if health.get("node_id") != node_id:
        errors.append("node identity mismatch")
    if health.get("carbon_metric") != expected_carbon_metric:
        errors.append(f"carbon metric={health.get('carbon_metric')}")
    if expected_state_token not in str(health.get("telemetry_state_file", "")):
        errors.append("not using isolated measurement state")
    if capabilities.get("ready") is not True or capabilities.get("drift"):
        errors.append(f"capability drift={capabilities.get('drift')}")
    for field in ("owned_task_count", "pending_bid_count", "active_reservation_count", "paused_task_count"):
        if int(health.get(field, 0) or 0) != 0:
            errors.append(f"{field}={health.get(field)}")
    if float(auction.get("resource_busy_fraction") or 0) > 1e-9:
        errors.append(f"resource_busy_fraction={auction.get('resource_busy_fraction')}")
    if errors:
        raise RuntimeError(f"{node_id} is not idle/ready: " + "; ".join(errors))
    return {"health": health, "auction": auction, "capabilities": capabilities}


def wait_definition(api: str, definition_id: str, revision: int, digest: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = request_json(f"{api}/task-definitions/{definition_id}?revision={revision}")
        except Exception:
            value = None
        if isinstance(value, dict) and value.get("digest") == digest:
            return
        time.sleep(1)
    raise TimeoutError(f"Definition {definition_id}@{revision} did not converge")


def build_definition(args: argparse.Namespace, definition_id: str) -> dict[str, Any]:
    if args.workload == "benchmark":
        if args.benchmark is None or args.benchmark_iterations is None:
            raise ValueError("benchmark workload requires --benchmark and --benchmark-iterations")
        return benchmark_definition(
            definition_id=definition_id,
            benchmark=args.benchmark,
            size=args.size,
            seed=args.seed,
            iterations=args.benchmark_iterations,
            node_ids=[args.node],
        )
    if args.workload == "dendro":
        template = json.loads(Path(args.dendro_definition).read_text(encoding="utf-8"))
        return dendro_definition(
            definition_id=definition_id,
            template=template,
            solver_path=args.dendro_solver,
            parameter_template_path=args.dendro_parameter_template,
            resolution=args.resolution,
            time_end=args.time_end,
            eligible_nodes=[args.node],
        )
    if args.llm_max_steps is None:
        raise ValueError("llm workload requires --llm-max-steps")
    return llm_training_definition(
        definition_id=definition_id,
        model=args.model,
        node_ids=[args.node],
        max_steps=args.llm_max_steps,
        checkpoint_every=args.llm_checkpoint_every,
        sleep_per_step=args.llm_sleep_per_step,
        torch_threads=args.llm_torch_threads,
    )


def iso_delta_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end.replace("Z", "+00:00")) - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()


def main() -> int:
    args = parse_args()
    if args.sample_interval_seconds <= 0 or args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("sample/poll/timeout values must be positive")
    cluster = load_cluster_config(args.cluster)
    node = cluster.get_node(args.node)
    api = base_url(node, cluster.api_port)
    preflight = idle_preflight(api, args.node, args.expected_carbon_metric, args.expected_state_token)

    bundle = Path(args.measurements_root) / args.measurement_id
    if bundle.exists():
        raise FileExistsError(f"Measurement bundle already exists: {bundle}")
    bundle.mkdir(parents=True)

    definition_id = f"{args.measurement_id}-definition"
    definition = build_definition(args, definition_id)
    created = request_json(f"{api}/task-definitions", method="POST", payload=definition, timeout=30)
    wait_definition(api, created["definition_id"], int(created["revision"]), created["digest"], min(180.0, args.timeout_seconds))
    run_request = {
        "definition_id": created["definition_id"],
        "revision": created["revision"],
        "initial_owner_node_id": args.node,
        "idempotency_key": fresh_run_idempotency_key(args.measurement_id),
        "auto_start": True,
        "labels": {
            "purpose": "stage4a4-static-completion",
            "scheduler_mode": "operator_only",
            "measurement_id": args.measurement_id,
            "class_id": args.class_id,
            "scope": args.scope,
            "trial": str(args.trial),
        },
    }
    run_view = request_json(f"{api}/task-runs", method="POST", payload=run_request, timeout=args.timeout_seconds)
    run_id = run_view["run"]["run_id"]
    print(f"[run] {run_id} class={args.class_id} node={args.node} scope={args.scope} trial={args.trial:02d}")

    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.timeout_seconds
    telemetry_samples: list[dict[str, Any]] = []
    last_sample_at: str | None = None
    final_state: dict[str, Any] | None = None
    try:
        while time.monotonic() < deadline:
            state = task_state(api, run_id)
            if state is None:
                time.sleep(args.poll_seconds)
                continue
            if state.get("status") == "failed":
                raise RuntimeError(f"Static task failed: {state}")
            if state.get("status") == "completed":
                final_state = state
                break
            if state.get("status") != "running":
                raise RuntimeError(f"Static task left RUNNING without completion: {state}")
            try:
                telemetry = request_json(f"{api}/telemetry/tasks/{run_id}")
            except Exception:
                telemetry = None
            if isinstance(telemetry, dict):
                sampled_at = telemetry.get("last_sample_at_utc")
                if sampled_at and sampled_at != last_sample_at:
                    last_sample_at = sampled_at
                    telemetry_samples.append({
                        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "telemetry_last_sample_at_utc": sampled_at,
                        "status": state.get("status"),
                        "progress_completed_units": state.get("progress_completed_units"),
                        "progress_total_units": state.get("progress_total_units"),
                        "process_count": telemetry.get("process_count"),
                        "cpu_utilization_percent": telemetry.get("cpu_utilization_percent"),
                        "memory_rss_mb": telemetry.get("memory_rss_mb"),
                        "checkpoint_bytes": telemetry.get("checkpoint_bytes"),
                        "measured_power_kw": telemetry.get("measured_power_kw"),
                        "progress_rate_units_per_second": telemetry.get("progress_rate_units_per_second"),
                    })
            time.sleep(args.poll_seconds)
        if final_state is None:
            raise TimeoutError(f"Static task {run_id} did not complete before timeout")

        wall_seconds = iso_delta_seconds(final_state.get("started_at_utc"), final_state.get("completed_at_utc"))
        if wall_seconds is None:
            wall_seconds = time.monotonic() - started_monotonic
        summary = {
            "measurement_id": args.measurement_id,
            "class_id": args.class_id,
            "scope": args.scope,
            "trial": args.trial,
            "node_id": args.node,
            "run_id": run_id,
            "workload": args.workload,
            "variant": args.size if args.workload == "benchmark" else (f"r{args.resolution}-t{args.time_end:g}" if args.workload == "dendro" else args.model),
            "status": final_state.get("status"),
            "generation": final_state.get("generation"),
            "owner_node_id": final_state.get("owner_node_id"),
            "wall_seconds": wall_seconds,
            "telemetry_sample_count": len(telemetry_samples),
            "progress_completed_units": final_state.get("progress_completed_units"),
            "progress_total_units": final_state.get("progress_total_units"),
            "accumulated_runtime_seconds": final_state.get("accumulated_runtime_seconds"),
            "accumulated_paused_seconds": final_state.get("accumulated_paused_seconds"),
            "last_migration_id": final_state.get("last_migration_id"),
            "accumulated_migration_seconds": final_state.get("accumulated_migration_seconds"),
            "accumulated_compute_cost_usd": final_state.get("accumulated_compute_cost_usd"),
            "accumulated_transfer_cost_usd": final_state.get("accumulated_transfer_cost_usd"),
            "accumulated_cost_usd": final_state.get("accumulated_cost_usd"),
            "accumulated_compute_carbon_grams": final_state.get("accumulated_compute_carbon_grams"),
            "accumulated_transfer_carbon_grams": final_state.get("accumulated_transfer_carbon_grams"),
            "accumulated_carbon_grams": final_state.get("accumulated_carbon_grams"),
            "passed": bool(
                final_state.get("status") == "completed"
                and final_state.get("owner_node_id") == args.node
                and int(final_state.get("generation") or 0) == 0
                and final_state.get("last_migration_id") is None
                and float(final_state.get("accumulated_paused_seconds") or 0.0) <= 1e-9
                and float(final_state.get("accumulated_migration_seconds") or 0.0) <= 1e-9
                and float(final_state.get("accumulated_transfer_cost_usd") or 0.0) <= 1e-12
                and float(final_state.get("accumulated_transfer_carbon_grams") or 0.0) <= 1e-12
            ),
        }
        metadata = {
            "format_version": 1,
            "measurement_type": "stage4a4_static_completion",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preflight": preflight,
            "definition": definition,
            "created_definition": created,
            "run": run_view,
            "final_state": final_state,
            "methodology": {
                "scheduler_mode": "operator_only",
                "migration_policy": "No pause or migration is permitted; task runs to natural completion on its initial node.",
            },
        }
        if telemetry_samples:
            write_csv(bundle / "telemetry_samples.csv", telemetry_samples, list(telemetry_samples[0].keys()))
        write_json(bundle / "summary.json", summary)
        write_json(bundle / "metadata.json", metadata)
        write_checksums(bundle)
        if summary["passed"] is not True:
            raise RuntimeError(f"Static completion invariant failed: {summary}")
        print("STAGE_4A4_STATIC_MEASUREMENT_PASS")
        print(f"bundle: {bundle}")
        print(f"run_id: {run_id}")
        print(f"wall_seconds: {wall_seconds:.3f}")
        print(f"runtime_seconds: {float(summary['accumulated_runtime_seconds'] or 0):.3f}")
        print(f"samples: {len(telemetry_samples)}")
        return 0
    finally:
        try:
            state = task_state(api, run_id)
            if state is not None and state.get("status") in {"running", "paused", "recovering", "migrating"}:
                request_json(f"{api}/tasks/{run_id}/stop", method="POST", timeout=min(120.0, args.timeout_seconds))
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
