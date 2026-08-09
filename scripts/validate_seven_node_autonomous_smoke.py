#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one checkpointable task to Boston and prove that the seven-node "
            "scheduler makes autonomous live decisions, migrates it, and completes it."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.smoke.json")
    parser.add_argument(
        "--definition",
        default="config/submissions/gcp-seven-node-smoke-definition.json",
    )
    parser.add_argument("--initial-node-id", default="boston")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
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
    if payload is None:
        return None
    for item in payload.get("tasks", []):
        state = item.get("state", {})
        if state.get("task_id") == run_id:
            return state
    return None


def policy_state(api: str, run_id: str) -> dict[str, Any] | None:
    value = try_json(f"{api}/policy/tasks/{run_id}")
    return value if isinstance(value, dict) else None


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    initial = cluster.get_node(args.initial_node_id)
    initial_api = base_url(initial, cluster.api_port)

    print("== Verify seven-node smoke mode ==")
    for node in cluster.nodes:
        api = base_url(node, cluster.api_port)
        health = request_json(f"{api}/health")
        policy = request_json(f"{api}/policy")
        if health.get("node_id") != node.id:
            raise RuntimeError(f"Health identity mismatch for {node.id}: {health}")
        weights = policy.get("baseline_weights", {})
        if weights != {"time": 0.05, "carbon": 0.9, "cost": 0.05}:
            raise RuntimeError(f"{node.id} is not in smoke policy mode: {weights}")
        print(f"[OK] {node.id:16} smoke policy active")

    definition = json.loads(Path(args.definition).read_text(encoding="utf-8"))
    definition_id = definition["definition_id"]
    created = request_json(
        f"{initial_api}/task-definitions",
        method="POST",
        payload=definition,
    )
    revision = created["revision"]
    print(f"definition={definition_id}@{revision}")

    print("== Wait for definition anti-entropy convergence ==")
    deadline = time.monotonic() + min(90.0, args.timeout_seconds)
    pending = {node.id for node in cluster.nodes}
    while pending and time.monotonic() < deadline:
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
        "idempotency_key": f"seven-node-autonomous-smoke-{uuid4()}",
        "auto_start": True,
        "labels": {"purpose": "seven-node-autonomous-smoke"},
    }
    run_view = request_json(
        f"{initial_api}/task-runs",
        method="POST",
        payload=run_request,
    )
    run_id = run_view["run"]["run_id"]
    print(f"run_id={run_id} initial_owner={initial.id}")

    observed_owners = [initial.id]
    migration_record: dict[str, Any] | None = None
    post_migration_decision = False
    completed_state: dict[str, Any] | None = None
    last_line: tuple[Any, ...] | None = None
    deadline = time.monotonic() + args.timeout_seconds

    print("== Observe autonomous decisions ==")
    while time.monotonic() < deadline:
        states: dict[str, dict[str, Any]] = {}
        policies: dict[str, dict[str, Any]] = {}
        for node in cluster.nodes:
            api = base_url(node, cluster.api_port)
            state = task_state(api, run_id)
            if state is not None:
                states[node.id] = state
            policy = policy_state(api, run_id)
            if policy is not None:
                policies[node.id] = policy

        if states:
            newest = max(states.values(), key=lambda item: int(item.get("generation", 0)))
            owner = newest.get("owner_node_id")
            generation = int(newest.get("generation", 0))
            status = newest.get("status")
            progress = newest.get("progress_completed_units")
            if owner and owner != observed_owners[-1]:
                observed_owners.append(owner)
                print(f"[ownership] generation={generation} owner={owner}")

            for policy in policies.values():
                for record in policy.get("decision_history", []):
                    if record.get("selected_action") == "migrate":
                        migration_record = record

            if migration_record is not None:
                destination = migration_record.get("selected_destination_node_id")
                destination_policy = policies.get(destination or "")
                if destination_policy is not None:
                    migration_index = int(migration_record.get("decision_index", 0))
                    if int(destination_policy.get("decision_count", 0)) > migration_index:
                        post_migration_decision = True

            line = (
                owner,
                generation,
                status,
                progress,
                max(
                    [int(value.get("decision_count", 0)) for value in policies.values()]
                    or [0]
                ),
            )
            if line != last_line:
                print(
                    f"[state] owner={owner} generation={generation} status={status} "
                    f"progress={progress} max_decisions={line[-1]}"
                )
                last_line = line

            completed_candidates = [
                state
                for state in states.values()
                if state.get("status") == "completed"
            ]
            if completed_candidates:
                completed_state = max(
                    completed_candidates,
                    key=lambda item: int(item.get("generation", 0)),
                )
                break

        time.sleep(args.poll_seconds)

    if completed_state is None:
        raise RuntimeError(f"Task did not complete within {args.timeout_seconds:g}s")
    if migration_record is None:
        raise RuntimeError("No autonomous MIGRATE decision was recorded")
    destination = migration_record.get("selected_destination_node_id")
    if not destination or destination == initial.id:
        raise RuntimeError(f"Invalid autonomous migration record: {migration_record}")
    if destination not in observed_owners:
        raise RuntimeError(
            f"Migration decision targeted {destination}, but ownership never moved there: "
            f"{observed_owners}"
        )
    if int(completed_state.get("generation", 0)) < 1:
        raise RuntimeError(f"Task completed without an ownership generation change: {completed_state}")
    if not post_migration_decision:
        raise RuntimeError(
            "Destination did not record a subsequent autonomous scheduling decision after migration"
        )
    if completed_state.get("last_error"):
        raise RuntimeError(f"Task completed with error state: {completed_state}")

    print("\nAUTONOMOUS SEVEN-NODE SMOKE PASSED")
    print(f"run_id: {run_id}")
    print(f"owners observed: {' -> '.join(observed_owners)}")
    print(
        "migration decision: "
        f"{migration_record.get('selected_action')} -> {destination}; "
        f"reason={migration_record.get('reason')}"
    )
    print(f"completed generation: {completed_state.get('generation')}")
    print(f"final progress: {completed_state.get('progress_completed_units')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
