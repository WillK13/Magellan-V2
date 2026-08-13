from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from magellan.experiments.bundle import write_csv, write_json, write_jsonl


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def materialize_bundle_tables(
    bundle_dir: str | Path,
    *,
    node_evidence: dict[str, dict[str, Any]],
    observations: list[dict[str, Any]],
    run_id: str,
    final_state: dict[str, Any],
    started_at_utc: str,
    finished_at_utc: str,
) -> dict[str, Any]:
    root = Path(bundle_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[dict[str, Any]] = []
    for node_id, evidence in sorted(node_evidence.items()):
        write_json(raw_dir / f"{node_id}.json", evidence)
        all_events.extend(evidence.get("events", []))
    all_events.sort(
        key=lambda item: (
            item.get("observed_at_utc", ""),
            item.get("node_id", ""),
            int(item.get("sequence", 0)),
        )
    )
    write_jsonl(raw_dir / "events.jsonl", all_events)
    write_jsonl(raw_dir / "observations.jsonl", observations)

    decision_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    migration_rows: list[dict[str, Any]] = []

    for event in all_events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "scheduler_decision":
            decision = payload.get("decision", {})
            selected = decision.get("selected", {})
            metadata = decision.get("policy_metadata", {})
            effective = metadata.get("effective_weights", {})
            baseline = metadata.get("baseline_weights", {})
            decision_rows.append(
                {
                    "node_id": event.get("node_id"),
                    "sequence": event.get("sequence"),
                    "observed_at_utc": event.get("observed_at_utc"),
                    "trace_time_utc": event.get("trace_time_utc"),
                    "task_id": event.get("task_id"),
                    "generation": event.get("generation"),
                    "selected_action": selected.get("action"),
                    "selected_destination_node_id": selected.get("destination_node_id"),
                    "selected_score": selected.get("score"),
                    "selected_time_seconds": selected.get("time_seconds"),
                    "selected_carbon_grams": selected.get("carbon_grams"),
                    "selected_cost_usd": selected.get("cost_usd"),
                    "baseline_time_weight": baseline.get("time"),
                    "baseline_carbon_weight": baseline.get("carbon"),
                    "baseline_cost_weight": baseline.get("cost"),
                    "effective_time_weight": effective.get("time"),
                    "effective_carbon_weight": effective.get("carbon"),
                    "effective_cost_weight": effective.get("cost"),
                    "reason": decision.get("reason"),
                    "task_profile_json": _json(payload.get("task_profile", {})),
                    "policy_metadata_json": _json(metadata),
                }
            )
            for rank, action in enumerate(decision.get("ranked_actions", []), start=1):
                candidate_rows.append(
                    {
                        "node_id": event.get("node_id"),
                        "sequence": event.get("sequence"),
                        "trace_time_utc": event.get("trace_time_utc"),
                        "task_id": event.get("task_id"),
                        "generation": event.get("generation"),
                        "rank": rank,
                        "action": action.get("action"),
                        "source_node_id": action.get("source_node_id"),
                        "destination_node_id": action.get("destination_node_id"),
                        "time_seconds": action.get("time_seconds"),
                        "carbon_grams": action.get("carbon_grams"),
                        "cost_usd": action.get("cost_usd"),
                        "normalized_time": action.get("normalized_time"),
                        "normalized_carbon": action.get("normalized_carbon"),
                        "normalized_cost": action.get("normalized_cost"),
                        "score": action.get("score"),
                        "details_json": _json(action.get("details", {})),
                    }
                )
        elif event_type in {"migration_completed", "migration_failed"}:
            migration_rows.append(
                {
                    "node_id": event.get("node_id"),
                    "sequence": event.get("sequence"),
                    "observed_at_utc": event.get("observed_at_utc"),
                    "trace_time_utc": event.get("trace_time_utc"),
                    "task_id": event.get("task_id"),
                    "generation": event.get("generation"),
                    "status": "completed" if event_type == "migration_completed" else "failed",
                    "migration_id": payload.get("migration_id"),
                    "bid_id": payload.get("bid_id"),
                    "source_node_id": payload.get("source_node_id"),
                    "destination_node_id": payload.get("destination_node_id"),
                    "checkpoint_bytes": payload.get("checkpoint_bytes"),
                    "missing_artifact_bytes": payload.get("missing_artifact_bytes"),
                    "checkpoint_transfer_bytes": payload.get("checkpoint_transfer_bytes"),
                    "total_accounted_transfer_bytes": payload.get("total_accounted_transfer_bytes"),
                    "checkpoint_seconds": payload.get("checkpoint_seconds"),
                    "transfer_seconds": payload.get("transfer_seconds"),
                    "restore_seconds": payload.get("restore_seconds"),
                    "activation_seconds": payload.get("activation_seconds"),
                    "total_downtime_seconds": payload.get("total_downtime_seconds"),
                    "error": payload.get("error"),
                }
            )

    write_csv(
        root / "decisions.csv",
        decision_rows,
        [
            "node_id", "sequence", "observed_at_utc", "trace_time_utc", "task_id",
            "generation", "selected_action", "selected_destination_node_id",
            "selected_score", "selected_time_seconds", "selected_carbon_grams",
            "selected_cost_usd", "baseline_time_weight", "baseline_carbon_weight",
            "baseline_cost_weight", "effective_time_weight", "effective_carbon_weight",
            "effective_cost_weight", "reason", "task_profile_json", "policy_metadata_json",
        ],
    )
    write_csv(
        root / "decision_candidates.csv",
        candidate_rows,
        [
            "node_id", "sequence", "trace_time_utc", "task_id", "generation", "rank",
            "action", "source_node_id", "destination_node_id", "time_seconds",
            "carbon_grams", "cost_usd", "normalized_time", "normalized_carbon",
            "normalized_cost", "score", "details_json",
        ],
    )
    write_csv(
        root / "migrations.csv",
        migration_rows,
        [
            "node_id", "sequence", "observed_at_utc", "trace_time_utc", "task_id",
            "generation", "status", "migration_id", "bid_id", "source_node_id",
            "destination_node_id", "checkpoint_bytes", "missing_artifact_bytes",
            "checkpoint_transfer_bytes", "total_accounted_transfer_bytes",
            "checkpoint_seconds", "transfer_seconds", "restore_seconds",
            "activation_seconds", "total_downtime_seconds", "error",
        ],
    )

    ownership_rows: list[dict[str, Any]] = []
    observations_by_time: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_time[str(observation.get("observed_at_utc"))].append(observation)

    status_rank = {
        "failed": 5,
        "completed": 5,
        "running": 4,
        "paused": 3,
        "migrating": 3,
        "stopped": 2,
        "remote": 1,
    }
    last_by_owner_generation: tuple[Any, ...] | None = None
    for observed_at, batch in sorted(observations_by_time.items()):
        candidates = [item for item in batch if item.get("state")]
        if not candidates:
            continue
        observation = max(
            candidates,
            key=lambda item: (
                int(item["state"].get("generation", 0)),
                status_rank.get(str(item["state"].get("status")), 0),
                item.get("node_id") == item["state"].get("owner_node_id"),
            ),
        )
        state = observation["state"]
        key = (
            state.get("owner_node_id"),
            state.get("generation"),
            state.get("status"),
        )
        if key == last_by_owner_generation:
            continue
        ownership_rows.append(
            {
                "observed_at_utc": observed_at,
                "reporting_node_id": observation.get("node_id"),
                "task_id": state.get("task_id"),
                "owner_node_id": state.get("owner_node_id"),
                "generation": state.get("generation"),
                "status": state.get("status"),
                "progress_completed_units": state.get("progress_completed_units"),
            }
        )
        last_by_owner_generation = key
    write_csv(
        root / "ownership.csv",
        ownership_rows,
        [
            "observed_at_utc", "reporting_node_id", "task_id", "owner_node_id",
            "generation", "status", "progress_completed_units",
        ],
    )

    task_result = {
        "task_id": run_id,
        "owner_node_id": final_state.get("owner_node_id"),
        "generation": final_state.get("generation"),
        "status": final_state.get("status"),
        "completed_at_utc": final_state.get("completed_at_utc"),
        "last_error": final_state.get("last_error"),
        "progress_completed_units": final_state.get("progress_completed_units"),
        "progress_total_units": final_state.get("progress_total_units"),
        "accumulated_runtime_seconds": final_state.get("accumulated_runtime_seconds"),
        "accumulated_paused_seconds": final_state.get("accumulated_paused_seconds"),
        "accumulated_migration_seconds": final_state.get("accumulated_migration_seconds"),
        "accumulated_compute_cost_usd": final_state.get("accumulated_compute_cost_usd"),
        "accumulated_transfer_cost_usd": final_state.get("accumulated_transfer_cost_usd"),
        "accumulated_cost_usd": final_state.get("accumulated_cost_usd"),
        "accumulated_compute_carbon_grams": final_state.get("accumulated_compute_carbon_grams"),
        "accumulated_transfer_carbon_grams": final_state.get("accumulated_transfer_carbon_grams"),
        "accumulated_carbon_grams": final_state.get("accumulated_carbon_grams"),
    }
    write_csv(root / "task_results.csv", [task_result], list(task_result))

    owners = []
    for row in ownership_rows:
        owner = row.get("owner_node_id")
        if owner and (not owners or owners[-1] != owner):
            owners.append(owner)

    event_counts: dict[str, int] = defaultdict(int)
    for event in all_events:
        event_counts[str(event.get("event_type"))] += 1

    summary = {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "status": final_state.get("status"),
        "final_owner_node_id": final_state.get("owner_node_id"),
        "final_generation": final_state.get("generation"),
        "owners_observed": owners,
        "decision_count": len(decision_rows),
        "candidate_count": len(candidate_rows),
        "successful_migration_count": sum(row["status"] == "completed" for row in migration_rows),
        "failed_migration_count": sum(row["status"] == "failed" for row in migration_rows),
        "event_counts": dict(sorted(event_counts.items())),
        "final_accounting": {
            key: value
            for key, value in task_result.items()
            if key.startswith("accumulated_")
        },
        "final_progress_completed_units": final_state.get("progress_completed_units"),
        "final_progress_total_units": final_state.get("progress_total_units"),
        "last_error": final_state.get("last_error"),
    }
    write_json(root / "summary.json", summary)
    return summary
