from __future__ import annotations

from types import SimpleNamespace

from magellan.experiments.stage5e1 import (
    STAGE5E1_CASES,
    case_by_id,
    stage5e1_passes,
    summarize_case,
)
from scripts.run_stage5e1_real_workload_smokes import case_command


def test_stage5e1_cases_cover_three_frozen_workload_classes() -> None:
    assert [case.class_id for case in STAGE5E1_CASES] == [
        "benchmark-json-medium",
        "llm-distilgpt2",
        "dendro-r9-t1p0",
    ]
    assert len({case.source_node_id for case in STAGE5E1_CASES}) == 3
    assert all(case.source_node_id != case.destination_node_id for case in STAGE5E1_CASES)


def test_summarize_benchmark_case() -> None:
    case = case_by_id("benchmark-json-medium")
    row = summarize_case(
        case,
        {
            "passed": True,
            "workload": "json",
            "variant": "medium",
            "resume_validation_passed": True,
            "migration": {
                "actual_checkpoint_bytes": 123,
                "actual_downtime_seconds": 4.5,
            },
        },
    )
    assert row["migration_count"] == 1
    assert row["resume_validation_passed"] is True
    assert row["checkpoint_bytes"] == 123


def test_summarize_llm_case_requires_exact_one_validated_migration() -> None:
    case = case_by_id("llm-distilgpt2")
    row = summarize_case(
        case,
        {
            "passed": True,
            "model": "experiment-assets/models/distilgpt2",
            "migration_count": 1,
            "resume_validations_passed": 1,
            "checkpoint_bytes_median": 456,
            "downtime_seconds_median": 7.5,
        },
    )
    assert row["resume_validation_passed"] is True
    assert row["migration_count"] == 1


def test_stage5e1_passes_all_three_cases() -> None:
    rows = []
    for case in STAGE5E1_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "class_id": case.class_id,
                "source_node_id": case.source_node_id,
                "destination_node_id": case.destination_node_id,
                "child_passed": True,
                "resume_validation_passed": True,
                "migration_count": 1,
            }
        )
    assert stage5e1_passes(rows)
    rows[1]["resume_validation_passed"] = False
    assert not stage5e1_passes(rows)


def test_case_commands_use_hardened_state_root_and_exact_workloads() -> None:
    args = SimpleNamespace(
        cluster="config/cluster.gcp.json",
        ssh_user="WILL",
        timeout_seconds=2400.0,
        profile_seconds=10.0,
        sample_interval_seconds=2.0,
    )
    root = __import__("pathlib").Path("/tmp/cases")

    benchmark = case_command(case_by_id("benchmark-json-medium"), case_root=root, args=args)
    assert "runtime-state-gcp" in benchmark
    assert "runtime-state-gcp-measurement" not in benchmark
    assert "json" in benchmark and "medium" in benchmark

    llm = case_command(case_by_id("llm-distilgpt2"), case_root=root, args=args)
    assert "experiment-assets/models/distilgpt2" in llm
    assert "--state-root" in llm
    assert "runtime-state-gcp" in llm

    dendro = case_command(case_by_id("dendro-r9-t1p0"), case_root=root, args=args)
    assert "--resolution" in dendro and "9" in dendro
    assert "--time-end" in dendro and "1.0" in dendro
