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

from magellan.config.loader import load_cluster_config
from magellan.experiments.bundle import sha256_file, write_checksums, write_csv, write_json
from magellan.experiments.measurement import absolute_percent_error, signed_percent_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled application-checkpoint migrations and compare the exact "
            "Magellan prediction used for each operator-triggered migration with measured timing. "
            "Run from Boston (or another node with private API/SSH reachability)."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--local-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--state-root", default="runtime-state-gcp-measurement")
    parser.add_argument(
        "--edge",
        action="append",
        default=None,
        help="Directed source:destination edge. Repeat for multiple edges.",
    )
    parser.add_argument(
        "--checkpoint-bytes",
        action="append",
        type=int,
        default=None,
        help="Controlled payload size in bytes. Repeat for multiple sizes.",
    )
    parser.add_argument("--samples-per-case", type=int, default=1)
    parser.add_argument(
        "--skip-edge-preflight",
        action="store_true",
        help=(
            "Do not force a live directed-edge telemetry refresh immediately "
            "before each migration candidate is evaluated."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--measurement-id", default=None)
    parser.add_argument("--expected-carbon-metric", default="lifecycle")
    parser.add_argument(
        "--expected-state-token",
        default="runtime-state-gcp-measurement",
        help="Substring required in each node's telemetry state path.",
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
    *, local_node_id: str, node: Any, ssh_user: str, remote_repo: str
) -> int:
    result = run_on_node(
        local_node_id=local_node_id,
        node=node,
        ssh_user=ssh_user,
        command=f"{remote_cd(remote_repo)} && df -Pk . | tail -1 | awk '{{print $4}}'",
        timeout=20.0,
    )
    return int(result.stdout.strip().splitlines()[-1]) * 1024


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


def wait_running_with_checkpoint(
    api: str,
    run_id: str,
    minimum_checkpoint_bytes: int,
    timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = task_state(api, run_id)
        telemetry = try_json(f"{api}/telemetry/tasks/{run_id}")
        if (
            state is not None
            and state.get("status") == "running"
            and isinstance(telemetry, dict)
            and int(telemetry.get("checkpoint_bytes") or 0) >= minimum_checkpoint_bytes
        ):
            return telemetry
        if state is not None and state.get("status") in {"failed", "completed"}:
            raise RuntimeError(f"Task terminated before migration: {state}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Task {run_id} did not expose requested checkpoint payload")


def wait_completed(api: str, run_id: str, timeout: float, poll_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Production scheduling epochs can be long (900 s). Completion
        # finalization is normally reconciled at an epoch boundary, but a
        # microbenchmark should not wait up to one full scheduling interval
        # after the migrated process has already exited. The operator
        # reconcile endpoint performs the same LocalProcessRuntime.reconcile()
        # step without invoking another scheduling decision.
        request_json(f"{api}/runtime/reconcile", method="POST")
        state = task_state(api, run_id)
        if state is not None and state.get("status") == "completed":
            return state
        if state is not None and state.get("status") == "failed":
            raise RuntimeError(f"Migrated task failed: {state.get('last_error')}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Task {run_id} did not complete on destination")


def build_definition(
    *,
    definition_id: str,
    source_node_id: str,
    payload_bytes: int,
    node_ids: list[str],
) -> dict[str, Any]:
    return {
        "definition_id": definition_id,
        "profile": {
            "workload_type": "migration-measurement-counter",
            "power_kw": 0.1,
            "checkpoint_bytes": payload_bytes,
            "data_bytes": 0,
            "prestaged_node_ids": node_ids,
            "estimated_remaining_seconds": 7200,
            "accumulated_cost_usd": 0,
            "cost_cap_usd": 10.0,
            "priority": 50,
            "deadline_at_utc": None,
            "resource_request": {
                "cpu_cores": 1,
                "memory_mb": 256,
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
                "--checkpoint-file",
                "{checkpoint_file}",
                "--checkpoint-padding-bytes",
                str(payload_bytes),
                "--interval-seconds",
                "0.2",
                "--max-value",
                "100000",
                "--initial-node-id",
                source_node_id,
                "--complete-after-migration-steps",
                "5",
                "--progress-file",
                "{progress_file}",
                "--completion-file",
                "{completion_file}",
                "--output-dir",
                "{output_directory}",
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


def parse_edge(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Invalid edge {value!r}; expected source:destination")
    source, destination = value.split(":", 1)
    if not source or not destination or source == destination:
        raise ValueError(f"Invalid directed edge: {value!r}")
    return source, destination


def main() -> int:
    args = parse_args()
    edges = [parse_edge(value) for value in (args.edge or ["boston:virginia"])]
    checkpoint_sizes = args.checkpoint_bytes or [10 * 1024 * 1024]
    if args.samples_per_case < 1:
        raise ValueError("--samples-per-case must be positive")
    if any(size <= 0 for size in checkpoint_sizes):
        raise ValueError("Checkpoint sizes must be positive")

    cluster = load_cluster_config(args.cluster)
    local = cluster.get_node(args.local_node_id)
    node_ids = [node.id for node in cluster.nodes]
    node_by_id = {node.id: node for node in cluster.nodes}
    for source_id, destination_id in edges:
        cluster.get_node(source_id)
        cluster.get_node(destination_id)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurement_id = args.measurement_id or f"migration-{timestamp}-{uuid4().hex[:8]}"
    bundle = Path(args.measurements_root) / measurement_id
    if bundle.exists():
        raise FileExistsError(f"Measurement bundle already exists: {bundle}")
    (bundle / "raw").mkdir(parents=True)

    print("== Controlled migration-model measurement ==")
    print(f"measurement_id={measurement_id}")
    print(f"edges={', '.join(f'{a}->{b}' for a, b in edges)}")
    print(f"checkpoint_bytes={checkpoint_sizes}")
    print(f"samples_per_case={args.samples_per_case}")

    health: dict[str, dict[str, Any]] = {}
    print("== Verify isolated measurement mode ==")
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        value = request_json(f"{api}/health")
        events = request_json(f"{api}/experiment/events/status")
        if value.get("node_id") != node.id:
            raise RuntimeError(f"Node identity mismatch for {node.id}")
        if (
            args.expected_carbon_metric
            and value.get("carbon_metric") != args.expected_carbon_metric
        ):
            raise RuntimeError(
                f"{node.id} carbon metric={value.get('carbon_metric')} "
                f"expected={args.expected_carbon_metric}"
            )
        state_file = str(value.get("telemetry_state_file", ""))
        if args.expected_state_token and args.expected_state_token not in state_file:
            raise RuntimeError(
                f"{node.id} is not in isolated measurement state: {state_file}"
            )
        if events.get("node_id") != node.id:
            raise RuntimeError(f"Experiment event stream mismatch on {node.id}")
        health[node.id] = value
        print(f"[OK] {node.id:16} state={state_file}")

    definitions: dict[tuple[str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for source_id, destination_id in edges:
        source = node_by_id[source_id]
        destination = node_by_id[destination_id]
        source_api = base_url(source, cluster.api_port)
        destination_api = base_url(destination, cluster.api_port)

        for payload_bytes in checkpoint_sizes:
            required_free = payload_bytes * 3 + 512 * 1024 * 1024
            for node in (source, destination):
                free = available_bytes(
                    local_node_id=local.id,
                    node=node,
                    ssh_user=args.ssh_user,
                    remote_repo=args.remote_repo,
                )
                if free < required_free:
                    raise RuntimeError(
                        f"Insufficient disk on {node.id}: free={free} required={required_free} "
                        f"for payload={payload_bytes}"
                    )

            key = (source_id, payload_bytes)
            if key not in definitions:
                definition_id = (
                    f"migration-measure-{source_id}-{payload_bytes}-{measurement_id[-8:]}"
                )
                definition_payload = build_definition(
                    definition_id=definition_id,
                    source_node_id=source_id,
                    payload_bytes=payload_bytes,
                    node_ids=node_ids,
                )
                created = request_json(
                    f"{source_api}/task-definitions",
                    method="POST",
                    payload=definition_payload,
                )
                definitions[key] = {"payload": definition_payload, "created": created}
                for node in cluster.nodes:
                    wait_definition(
                        base_url(node, cluster.api_port),
                        created["definition_id"],
                        int(created["revision"]),
                        created["digest"],
                        min(120.0, args.timeout_seconds),
                    )
                print(
                    f"[definition] {created['definition_id']}@{created['revision']} converged"
                )

            created = definitions[key]["created"]
            for sample in range(1, args.samples_per_case + 1):
                event_status = request_json(f"{source_api}/experiment/events/status")
                event_start = int(event_status.get("last_sequence", 0))
                run_request = {
                    "definition_id": created["definition_id"],
                    "revision": created["revision"],
                    "initial_owner_node_id": source_id,
                    "idempotency_key": (
                        f"{measurement_id}-{source_id}-{destination_id}-"
                        f"{payload_bytes}-{sample}"
                    ),
                    "auto_start": True,
                    "labels": {
                        "purpose": "migration-model-measurement",
                        "measurement_id": measurement_id,
                    },
                }
                run_view = request_json(
                    f"{source_api}/task-runs", method="POST", payload=run_request
                )
                run_id = run_view["run"]["run_id"]
                print(
                    f"[run] {source_id}->{destination_id} bytes={payload_bytes} "
                    f"sample={sample} run={run_id}"
                )

                source_task_telemetry = wait_running_with_checkpoint(
                    source_api,
                    run_id,
                    payload_bytes,
                    min(args.timeout_seconds, 300.0),
                    args.poll_seconds,
                )
                if args.skip_edge_preflight:
                    edge_telemetry = request_json(
                        f"{source_api}/telemetry/edges/{destination_id}"
                    )
                else:
                    edge_telemetry = request_json(
                        f"{source_api}/telemetry/edges/{destination_id}/ensure",
                        method="POST",
                        timeout=min(60.0, args.timeout_seconds),
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

                migration_response = request_json(
                    f"{source_api}/tasks/{run_id}/migrate/{destination_id}",
                    method="POST",
                    timeout=args.timeout_seconds,
                )
                if not migration_response.get("migrated"):
                    raise RuntimeError(
                        f"Migration was not accepted/completed: {migration_response}"
                    )
                final_state = wait_completed(
                    destination_api,
                    run_id,
                    min(120.0, args.timeout_seconds),
                    args.poll_seconds,
                )

                events = query_events(source_api, event_start, run_id)
                completed_events = [
                    item for item in events if item.get("event_type") == "migration_completed"
                ]
                if not completed_events:
                    raise RuntimeError(f"No migration_completed event for {run_id}")
                migration_event = completed_events[-1]
                actual = migration_event["payload"]
                bid = migration_response.get("bid") or {}
                candidate = bid.get("candidate") or {}
                predicted = candidate.get("details") or {}

                predicted_checkpoint = float(predicted.get("checkpoint_seconds", 0.0))
                predicted_transfer = float(predicted.get("transfer_seconds", 0.0))
                predicted_restore = float(predicted.get("restore_seconds", 0.0))
                predicted_overhead = float(
                    predicted.get("migration_overhead_seconds", 0.0)
                )
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
                actual_activation = float(actual["activation_seconds"])
                actual_downtime = float(actual["total_downtime_seconds"])
                actual_overhead = float(
                    actual.get(
                        "migration_overhead_seconds",
                        max(
                            0.0,
                            actual_downtime
                            - actual_checkpoint
                            - actual_transfer
                            - actual_restore,
                        ),
                    )
                )

                row = {
                    "measurement_id": measurement_id,
                    "run_id": run_id,
                    "source_node_id": source_id,
                    "destination_node_id": destination_id,
                    "sample": sample,
                    "requested_payload_bytes": payload_bytes,
                    "actual_checkpoint_bytes": actual.get("checkpoint_bytes"),
                    "actual_transfer_bytes": actual.get("checkpoint_transfer_bytes"),
                    "telemetry_checkpoint_bytes_before": source_task_telemetry.get(
                        "checkpoint_bytes"
                    ),
                    "telemetry_bandwidth_mbps_before": edge_telemetry.get(
                        "effective_bandwidth_mbps"
                    ),
                    "telemetry_bandwidth_source_before": edge_telemetry.get(
                        "bandwidth_source"
                    ),
                    "telemetry_latency_ms_before": edge_telemetry.get(
                        "effective_latency_ms"
                    ),
                    "telemetry_latency_source_before": edge_telemetry.get("latency_source"),
                    "calibration_sample_count_before": (
                        edge_calibration.get("sample_count")
                        if edge_calibration is not None
                        else 0
                    ),
                    "calibration_checkpoint_seconds_before": (
                        edge_calibration.get("checkpoint_seconds_ema")
                        if edge_calibration is not None
                        else None
                    ),
                    "calibration_restore_seconds_before": (
                        edge_calibration.get("restore_seconds_ema")
                        if edge_calibration is not None
                        else None
                    ),
                    "calibration_overhead_seconds_before": (
                        max(
                            0.0,
                            float(
                                edge_calibration.get(
                                    "total_downtime_seconds_ema"
                                )
                                or 0.0
                            )
                            - float(
                                edge_calibration.get(
                                    "checkpoint_seconds_ema"
                                )
                                or 0.0
                            )
                            - float(
                                edge_calibration.get("transfer_seconds_ema")
                                or 0.0
                            )
                            - float(
                                edge_calibration.get("restore_seconds_ema")
                                or 0.0
                            ),
                        )
                        if edge_calibration is not None
                        else None
                    ),
                    "candidate_calibration_source": predicted.get("calibration_source"),
                    "candidate_bandwidth_source": predicted.get("bandwidth_source"),
                    "candidate_latency_source": predicted.get("latency_source"),
                    "candidate_transfer_model": predicted.get("transfer_model"),
                    "edge_preflight_enabled": not args.skip_edge_preflight,
                    "predicted_checkpoint_seconds": predicted_checkpoint,
                    "actual_checkpoint_seconds": actual_checkpoint,
                    "checkpoint_error_percent": signed_percent_error(
                        predicted_checkpoint, actual_checkpoint
                    ),
                    "checkpoint_absolute_error_percent": absolute_percent_error(
                        predicted_checkpoint, actual_checkpoint
                    ),
                    "predicted_transfer_seconds": predicted_transfer,
                    "actual_transfer_seconds": actual_transfer,
                    "actual_transfer_bandwidth_mbps": (
                        float(actual.get("checkpoint_transfer_bytes") or 0)
                        * 8.0
                        / max(actual_transfer, 1e-9)
                        / 1_000_000.0
                    ),
                    "transfer_error_percent": signed_percent_error(
                        predicted_transfer, actual_transfer
                    ),
                    "transfer_absolute_error_percent": absolute_percent_error(
                        predicted_transfer, actual_transfer
                    ),
                    "predicted_restore_seconds": predicted_restore,
                    "actual_restore_seconds": actual_restore,
                    "restore_error_percent": signed_percent_error(
                        predicted_restore, actual_restore
                    ),
                    "restore_absolute_error_percent": absolute_percent_error(
                        predicted_restore, actual_restore
                    ),
                    "predicted_migration_overhead_seconds": predicted_overhead,
                    "actual_migration_overhead_seconds": actual_overhead,
                    "predicted_downtime_seconds": predicted_downtime,
                    "actual_pre_checkpoint_seconds": actual.get(
                        "pre_checkpoint_seconds"
                    ),
                    "actual_post_checkpoint_seconds": actual.get(
                        "post_checkpoint_seconds"
                    ),
                    "actual_transfer_setup_seconds": actual.get(
                        "transfer_setup_seconds"
                    ),
                    "actual_transfer_wall_seconds": actual.get(
                        "transfer_wall_seconds"
                    ),
                    "actual_transfer_call_wall_seconds": actual.get(
                        "transfer_call_wall_seconds"
                    ),
                    "actual_post_transfer_seconds": actual.get(
                        "post_transfer_seconds"
                    ),
                    "actual_activation_seconds": actual_activation,
                    "actual_destination_activation_seconds": actual.get(
                        "destination_activation_seconds"
                    ),
                    "actual_activation_transport_seconds": actual.get(
                        "activation_transport_seconds"
                    ),
                    "actual_activation_non_restore_seconds": actual.get(
                        "activation_non_restore_seconds"
                    ),
                    "actual_instrumented_wall_seconds": actual.get(
                        "instrumented_wall_seconds"
                    ),
                    "actual_timing_residual_seconds": actual.get(
                        "timing_residual_seconds"
                    ),
                    "actual_downtime_seconds": actual_downtime,
                    "downtime_error_percent": signed_percent_error(
                        predicted_downtime, actual_downtime
                    ),
                    "downtime_absolute_error_percent": absolute_percent_error(
                        predicted_downtime, actual_downtime
                    ),
                    "candidate_score": candidate.get("score"),
                    "candidate_carbon_grams": candidate.get("carbon_grams"),
                    "candidate_cost_usd": candidate.get("cost_usd"),
                    "candidate_time_seconds": candidate.get("time_seconds"),
                    "final_generation": final_state.get("generation"),
                    "final_status": final_state.get("status"),
                }
                rows.append(row)
                raw = {
                    "run_request": run_request,
                    "source_task_telemetry_before": source_task_telemetry,
                    "edge_telemetry_before": edge_telemetry,
                    "calibration_before": calibration_before,
                    "migration_response": migration_response,
                    "migration_event": migration_event,
                    "final_state": final_state,
                }
                write_json(bundle / "raw" / f"{run_id}.json", raw)
                print(
                    f"[measured] checkpoint={actual_checkpoint:.3f}s "
                    f"transfer={actual_transfer:.3f}s restore={actual_restore:.3f}s "
                    f"overhead={actual_overhead:.3f}s "
                    f"downtime={actual_downtime:.3f}s "
                    f"pred_downtime={predicted_downtime:.3f}s"
                )

                cleanup_relative = (
                    f"{args.state_root}/tasks/{shlex.quote(run_id)}/checkpoint/payload.bin"
                )
                for node in cluster.nodes:
                    try:
                        run_on_node(
                            local_node_id=local.id,
                            node=node,
                            ssh_user=args.ssh_user,
                            command=(
                                f"{remote_cd(args.remote_repo)} && "
                                f"rm -f {cleanup_relative}"
                            ),
                            timeout=20.0,
                        )
                    except Exception:
                        pass

    metadata = {
        "format_version": 1,
        "measurement_type": "controlled_migration_model",
        "measurement_id": measurement_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster": {
            "path": args.cluster,
            "sha256": sha256_file(args.cluster),
            "node_ids": node_ids,
            "local_node_id": local.id,
        },
        "parameters": {
            "edges": [f"{source}:{destination}" for source, destination in edges],
            "checkpoint_bytes": checkpoint_sizes,
            "samples_per_case": args.samples_per_case,
            "state_root": args.state_root,
            "expected_carbon_metric": args.expected_carbon_metric,
            "expected_state_token": args.expected_state_token,
            "edge_preflight_enabled": not args.skip_edge_preflight,
            "checkpoint_semantics": (
                "Payload is pre-existing application checkpoint state. Actual checkpoint_seconds "
                "measures quiesce/stop plus checkpoint validation, not "
                "serialization of the payload."
            ),
        },
        "initial_health": health,
        "definitions": [
            {
                "source_node_id": source_id,
                "checkpoint_bytes": payload_bytes,
                **definition,
            }
            for (source_id, payload_bytes), definition in sorted(definitions.items())
        ],
    }
    write_json(bundle / "metadata.json", metadata)
    write_csv(bundle / "migration_samples.csv", rows, list(rows[0].keys()))
    write_checksums(bundle)

    print("\nCONTROLLED MIGRATION MEASUREMENT PASSED")
    print(f"bundle: {bundle}")
    print(f"samples: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
