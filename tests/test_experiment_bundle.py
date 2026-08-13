from __future__ import annotations

import csv
import json

from magellan.experiments.bundle import validate_checksums, write_checksums, write_json
from magellan.experiments.collector import materialize_bundle_tables


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_materialized_experiment_bundle_contains_decision_and_migration_evidence(tmp_path) -> None:
    run_id = "run-1"
    decision_event = {
        "sequence": 1,
        "event_id": "event-1",
        "node_id": "boston",
        "event_type": "scheduler_decision",
        "observed_at_utc": "2026-08-13T12:00:00+00:00",
        "trace_time_utc": "2024-01-01T00:00:00+00:00",
        "task_id": run_id,
        "generation": 0,
        "payload": {
            "task_profile": {"task_id": run_id, "current_node_id": "boston"},
            "decision": {
                "selected": {
                    "action": "migrate",
                    "source_node_id": "boston",
                    "destination_node_id": "france",
                    "time_seconds": 10,
                    "carbon_grams": 1,
                    "cost_usd": 0.1,
                    "normalized_time": 0.2,
                    "normalized_carbon": 0.1,
                    "normalized_cost": 0.3,
                    "score": 0.15,
                    "details": {"bandwidth_mbps": 100},
                },
                "ranked_actions": [
                    {
                        "action": "migrate",
                        "source_node_id": "boston",
                        "destination_node_id": "france",
                        "time_seconds": 10,
                        "carbon_grams": 1,
                        "cost_usd": 0.1,
                        "normalized_time": 0.2,
                        "normalized_carbon": 0.1,
                        "normalized_cost": 0.3,
                        "score": 0.15,
                        "details": {"bandwidth_mbps": 100},
                    }
                ],
                "reason": "test",
                "policy_metadata": {
                    "baseline_weights": {"time": 0.3, "carbon": 0.6, "cost": 0.1},
                    "effective_weights": {"time": 0.2, "carbon": 0.7, "cost": 0.1},
                },
            },
        },
    }
    migration_event = {
        "sequence": 2,
        "event_id": "event-2",
        "node_id": "boston",
        "event_type": "migration_completed",
        "observed_at_utc": "2026-08-13T12:00:05+00:00",
        "trace_time_utc": "2024-01-01T00:00:05+00:00",
        "task_id": run_id,
        "generation": 1,
        "payload": {
            "migration_id": "migration-1",
            "bid_id": "bid-1",
            "source_node_id": "boston",
            "destination_node_id": "france",
            "checkpoint_bytes": 1024,
            "missing_artifact_bytes": 0,
            "checkpoint_transfer_bytes": 1024,
            "total_accounted_transfer_bytes": 1024,
            "checkpoint_seconds": 1.0,
            "transfer_seconds": 2.0,
            "restore_seconds": 1.0,
            "activation_seconds": 1.2,
            "total_downtime_seconds": 4.2,
        },
    }
    node_evidence = {
        "boston": {"events": [decision_event, migration_event]},
        "france": {"events": []},
    }
    observations = [
        {
            "observed_at_utc": "2026-08-13T12:00:00+00:00",
            "node_id": "boston",
            "state": {"task_id": run_id, "owner_node_id": "boston", "generation": 0, "status": "running"},
        },
        {
            "observed_at_utc": "2026-08-13T12:00:06+00:00",
            "node_id": "france",
            "state": {"task_id": run_id, "owner_node_id": "france", "generation": 1, "status": "completed", "progress_completed_units": 300},
        },
    ]
    final_state = {
        "task_id": run_id,
        "owner_node_id": "france",
        "generation": 1,
        "status": "completed",
        "progress_completed_units": 300,
        "progress_total_units": 300,
        "accumulated_runtime_seconds": 10,
        "accumulated_paused_seconds": 0,
        "accumulated_migration_seconds": 4.2,
        "accumulated_compute_cost_usd": 0.01,
        "accumulated_transfer_cost_usd": 0.001,
        "accumulated_cost_usd": 0.011,
        "accumulated_compute_carbon_grams": 2.0,
        "accumulated_transfer_carbon_grams": 0.1,
        "accumulated_carbon_grams": 2.1,
    }

    summary = materialize_bundle_tables(
        tmp_path,
        node_evidence=node_evidence,
        observations=observations,
        run_id=run_id,
        final_state=final_state,
        started_at_utc="2026-08-13T12:00:00+00:00",
        finished_at_utc="2026-08-13T12:00:10+00:00",
    )
    write_json(tmp_path / "manifest.json", {"run_id": run_id})
    write_checksums(tmp_path)

    assert summary["decision_count"] == 1
    assert summary["successful_migration_count"] == 1
    assert summary["owners_observed"] == ["boston", "france"]
    assert read_csv(tmp_path / "decisions.csv")[0]["selected_action"] == "migrate"
    assert read_csv(tmp_path / "migrations.csv")[0]["destination_node_id"] == "france"
    assert not validate_checksums(tmp_path)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["final_accounting"]["accumulated_carbon_grams"] == 2.1
