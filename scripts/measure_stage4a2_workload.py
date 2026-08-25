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
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import write_checksums, write_csv, write_json
from magellan.experiments.measurement import signed_percent_error
from magellan.experiments.stage4a2 import summarize_profile_samples
from magellan.experiments.workload_population import benchmark_definition, dendro_definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one real Stage-4 workload on final hardware, force one "
            "normal Magellan migration, and record measured resource/checkpoint/"
            "restore behavior. Supports checkpointable benchmark and Dendro workloads."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--workload", choices=["benchmark", "dendro"], required=True)
    parser.add_argument("--benchmark", choices=["nbody", "json", "matmul"], default=None)
    parser.add_argument("--size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--benchmark-iterations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--profile-seconds", type=float, default=20.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-progress", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
    )
    return parser.parse_args()


def request_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 15.0,
) -> Any:
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
    raise TimeoutError(f"Definition {definition_id}@{revision} did not converge to {api}")


def query_events(api: str, after_sequence: int, run_id: str) -> list[dict[str, Any]]:
    query = urlencode({"after_sequence": after_sequence, "task_id": run_id, "limit": 10000})
    value = request_json(f"{api}/experiment/events?{query}")
    return list(value.get("events", []))


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


def profile_task(
    *,
    api: str,
    run_id: str,
    profile_seconds: float,
    sample_interval_seconds: float,
    minimum_progress: float,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    ready_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        if state is not None and state.get("status") in {"failed", "completed"}:
            raise RuntimeError(f"Task terminated before profiling: {state}")
        try:
            telemetry = request_json(f"{api}/telemetry/tasks/{run_id}")
        except Exception:
            telemetry = None
        progress = float((state or {}).get("progress_completed_units") or 0.0)
        checkpoint_bytes = int((telemetry or {}).get("checkpoint_bytes") or 0)
        if (
            state is not None
            and state.get("status") == "running"
            and progress >= minimum_progress
            and checkpoint_bytes > 0
        ):
            ready_state = state
            break
        time.sleep(2)
    if ready_state is None:
        raise TimeoutError(f"Task {run_id} did not reach checkpointed progress")

    samples: list[dict[str, Any]] = []
    last_telemetry_sample_at: str | None = None
    end = time.monotonic() + profile_seconds
    while time.monotonic() < end:
        state = task_state(api, run_id)
        if state is None or state.get("status") != "running":
            raise RuntimeError(f"Task left RUNNING during profile window: {state}")
        telemetry = request_json(f"{api}/telemetry/tasks/{run_id}")
        telemetry_sample_at = telemetry.get("last_sample_at_utc")
        if telemetry_sample_at == last_telemetry_sample_at:
            time.sleep(min(sample_interval_seconds, 1.0))
            continue
        last_telemetry_sample_at = telemetry_sample_at
        samples.append(
            {
                "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
                "telemetry_last_sample_at_utc": telemetry_sample_at,
                "telemetry_sample_count": telemetry.get("sample_count"),
                "progress_completed_units": state.get("progress_completed_units"),
                "progress_total_units": state.get("progress_total_units"),
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
        time.sleep(sample_interval_seconds)
    final_state = task_state(api, run_id)
    if final_state is None:
        raise RuntimeError("Task disappeared after profile window")
    return samples, final_state


def wait_resumed(
    *, api: str, run_id: str, minimum_progress: float, timeout_seconds: float, poll_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        if state is not None:
            last = state
            if state.get("status") == "failed":
                raise RuntimeError(f"Migrated task failed: {state}")
            progress = float(state.get("progress_completed_units") or 0.0)
            if (
                state.get("status") == "running"
                and state.get("resumed_from_checkpoint") is True
                and progress >= minimum_progress
            ):
                return state
        time.sleep(poll_seconds)
    raise TimeoutError(f"Task {run_id} did not resume from checkpoint: {last}")


def build_definition(args: argparse.Namespace, definition_id: str, eligible_nodes: list[str]) -> dict[str, Any]:
    if args.workload == "benchmark":
        if args.benchmark is None:
            raise ValueError("--benchmark is required for benchmark workload")
        return benchmark_definition(
            definition_id=definition_id,
            benchmark=args.benchmark,
            size=args.size,
            seed=args.seed,
            iterations=args.benchmark_iterations,
            node_ids=eligible_nodes,
        )
    template = json.loads(Path(args.dendro_definition).read_text(encoding="utf-8"))
    return dendro_definition(
        definition_id=definition_id,
        template=template,
        solver_path=args.dendro_solver,
        parameter_template_path=args.dendro_parameter_template,
        resolution=args.resolution,
        time_end=args.time_end,
        eligible_nodes=eligible_nodes,
    )


def main() -> int:
    args = parse_args()
    if args.source == args.destination:
        raise ValueError("source and destination must differ")
    if args.profile_seconds <= 0 or args.sample_interval_seconds <= 0:
        raise ValueError("profile/sample intervals must be positive")

    cluster = load_cluster_config(args.cluster)
    source = cluster.get_node(args.source)
    destination = cluster.get_node(args.destination)
    source_api = base_url(source, cluster.api_port)
    destination_api = base_url(destination, cluster.api_port)

    preflight = {
        args.source: idle_preflight(source_api, args.source, args.expected_carbon_metric, args.expected_state_token),
        args.destination: idle_preflight(destination_api, args.destination, args.expected_carbon_metric, args.expected_state_token),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workload_label = args.benchmark if args.workload == "benchmark" else "dendro"
    variant = args.size if args.workload == "benchmark" else f"r{args.resolution}-t{args.time_end:g}"
    measurement_id = args.measurement_id or f"stage4a2-{workload_label}-{variant}-{timestamp}-{uuid4().hex[:8]}"
    bundle = Path(args.measurements_root) / measurement_id
    if bundle.exists():
        raise FileExistsError(f"Measurement bundle already exists: {bundle}")
    bundle.mkdir(parents=True)

    definition_id = f"{measurement_id}-definition"
    definition = build_definition(args, definition_id, [args.source, args.destination])
    created = request_json(f"{source_api}/task-definitions", method="POST", payload=definition, timeout=30)
    for api in (source_api, destination_api):
        wait_definition(api, created["definition_id"], int(created["revision"]), created["digest"], min(180, args.timeout_seconds))

    run_request = {
        "definition_id": created["definition_id"],
        "revision": created["revision"],
        "initial_owner_node_id": args.source,
        "idempotency_key": f"{measurement_id}-run",
        "auto_start": True,
        "labels": {
            "purpose": "stage4a2-workload-calibration",
            "measurement_id": measurement_id,
            "workload": workload_label,
            "variant": variant,
        },
    }
    run_view = request_json(f"{source_api}/task-runs", method="POST", payload=run_request, timeout=args.timeout_seconds)
    run_id = run_view["run"]["run_id"]
    print(f"[run] {run_id} workload={workload_label} variant={variant} source={args.source}")

    samples: list[dict[str, Any]] = []
    migration_response: dict[str, Any] | None = None
    migration_event: dict[str, Any] | None = None
    resumed_state: dict[str, Any] | None = None
    try:
        samples, profile_state = profile_task(
            api=source_api,
            run_id=run_id,
            profile_seconds=args.profile_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            minimum_progress=args.minimum_progress,
            timeout_seconds=args.timeout_seconds,
        )
        progress_before = float(profile_state.get("progress_completed_units") or 0.0)
        edge_before = request_json(f"{source_api}/telemetry/edges/{args.destination}/refresh", method="POST", timeout=120)
        event_status = request_json(f"{source_api}/experiment/events/status")
        event_start = int(event_status.get("last_sequence", 0))
        migration_response = request_json(
            f"{source_api}/tasks/{run_id}/migrate/{args.destination}",
            method="POST",
            timeout=args.timeout_seconds,
        )
        if migration_response.get("migrated") is not True:
            raise RuntimeError(f"Operator migration was not accepted: {migration_response}")
        events = query_events(source_api, event_start, run_id)
        completed = [event for event in events if event.get("event_type") == "migration_completed"]
        if not completed:
            raise RuntimeError("No migration_completed experiment event found")
        migration_event = completed[-1]
        resumed_state = wait_resumed(
            api=destination_api,
            run_id=run_id,
            minimum_progress=progress_before,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        actual = migration_event["payload"]
        candidate = ((migration_response.get("bid") or {}).get("candidate") or {})
        predicted = candidate.get("details") or {}
        predicted_checkpoint = float(predicted.get("checkpoint_seconds", 0.0))
        predicted_transfer = float(predicted.get("transfer_seconds", 0.0))
        predicted_restore = float(predicted.get("restore_seconds", 0.0))
        predicted_overhead = float(predicted.get("migration_overhead_seconds", 0.0))
        predicted_downtime = float(
            predicted.get(
                "predicted_downtime_seconds",
                predicted_checkpoint + predicted_transfer + predicted_restore + predicted_overhead,
            )
        )
        actual_checkpoint = float(actual["checkpoint_seconds"])
        actual_transfer = float(actual["transfer_seconds"])
        actual_restore = float(actual["restore_seconds"])
        actual_overhead = float(actual.get("migration_overhead_seconds", 0.0))
        actual_downtime = float(actual["total_downtime_seconds"])

        migration_row = {
            "measurement_id": measurement_id,
            "run_id": run_id,
            "workload": workload_label,
            "variant": variant,
            "source_node_id": args.source,
            "destination_node_id": args.destination,
            "progress_before_migration": progress_before,
            "progress_after_resume": resumed_state.get("progress_completed_units"),
            "resumed_from_checkpoint": resumed_state.get("resumed_from_checkpoint"),
            "actual_checkpoint_bytes": actual.get("checkpoint_bytes"),
            "candidate_calibration_source": predicted.get("calibration_source"),
            "candidate_transfer_model": predicted.get("transfer_model"),
            "candidate_transfer_model_source": predicted.get("transfer_model_source"),
            "predicted_checkpoint_seconds": predicted_checkpoint,
            "actual_checkpoint_seconds": actual_checkpoint,
            "checkpoint_error_percent": signed_percent_error(predicted_checkpoint, actual_checkpoint),
            "predicted_transfer_seconds": predicted_transfer,
            "actual_transfer_seconds": actual_transfer,
            "transfer_error_percent": signed_percent_error(predicted_transfer, actual_transfer),
            "predicted_restore_seconds": predicted_restore,
            "actual_restore_seconds": actual_restore,
            "restore_error_percent": signed_percent_error(predicted_restore, actual_restore),
            "predicted_migration_overhead_seconds": predicted_overhead,
            "actual_migration_overhead_seconds": actual_overhead,
            "predicted_downtime_seconds": predicted_downtime,
            "actual_downtime_seconds": actual_downtime,
            "downtime_error_percent": signed_percent_error(predicted_downtime, actual_downtime),
        }
        write_csv(bundle / "profile_samples.csv", samples, list(samples[0].keys()))
        write_csv(bundle / "migration.csv", [migration_row], list(migration_row.keys()))
        profile_summary = summarize_profile_samples(samples)
        summary = {
            "measurement_id": measurement_id,
            "workload": workload_label,
            "variant": variant,
            "source_node_id": args.source,
            "destination_node_id": args.destination,
            "run_id": run_id,
            "profile": profile_summary,
            "migration": migration_row,
            "resume_validation_passed": bool(resumed_state.get("resumed_from_checkpoint"))
            and float(resumed_state.get("progress_completed_units") or 0) >= progress_before,
            "passed": True,
        }
        metadata = {
            "format_version": 1,
            "measurement_type": "stage4a2_workload_migration_calibration",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_id": measurement_id,
            "workload": workload_label,
            "variant": variant,
            "source_node_id": args.source,
            "destination_node_id": args.destination,
            "profile_seconds": args.profile_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
            "preflight": preflight,
            "definition": definition,
            "created_definition": created,
            "run": run_view,
            "edge_before": edge_before,
            "migration_response": migration_response,
            "migration_event": migration_event,
            "resumed_state": resumed_state,
            "methodology": {
                "migration_path": (
                    "Operator endpoint forces the selected calibration edge, while candidate scoring, "
                    "destination bid/reservation, checkpoint, rsync, activation, restore, ownership, "
                    "and telemetry all use the production Magellan path."
                ),
                "accuracy_policy": "Record prediction error descriptively; no samples are dropped by an accuracy threshold.",
            },
        }
        write_json(bundle / "metadata.json", metadata)
        write_json(bundle / "summary.json", summary)
        write_checksums(bundle)
        print("STAGE_4A2_WORKLOAD_MEASUREMENT_PASS")
        print(f"bundle: {bundle}")
        print(f"run_id: {run_id}")
        print(f"checkpoint_bytes: {migration_row['actual_checkpoint_bytes']}")
        print(f"downtime_seconds: {actual_downtime:.3f}")
        return 0
    finally:
        # A calibration task is intentionally long-running; leave the cluster idle
        # even when a validation step raises after the migration has completed.
        for api in (destination_api, source_api):
            try:
                state = task_state(api, run_id)
                if state is not None and state.get("owner_node_id") in {args.source, args.destination} and state.get("status") in {"running", "paused", "failed", "recovering", "migrating"}:
                    owner_api = destination_api if state.get("owner_node_id") == args.destination else source_api
                    request_json(f"{owner_api}/tasks/{run_id}/stop", method="POST", timeout=min(120, args.timeout_seconds))
                    break
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
