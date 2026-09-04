from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    class_id: str
    source_node_id: str
    destination_node_id: str
    runner: str


STAGE5E1_CASES = (
    SmokeCase(
        case_id="benchmark-json-medium",
        class_id="benchmark-json-medium",
        source_node_id="boston",
        destination_node_id="california",
        runner="stage4a2-workload",
    ),
    SmokeCase(
        case_id="llm-distilgpt2",
        class_id="llm-distilgpt2",
        source_node_id="california",
        destination_node_id="france",
        runner="real-llm-migration",
    ),
    SmokeCase(
        case_id="dendro-r9-t1p0",
        class_id="dendro-r9-t1p0",
        source_node_id="virginia",
        destination_node_id="nepal",
        runner="stage4a2-workload",
    ),
)


def case_by_id(case_id: str) -> SmokeCase:
    for case in STAGE5E1_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def summarize_case(case: SmokeCase, child_summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "class_id": case.class_id,
        "source_node_id": case.source_node_id,
        "destination_node_id": case.destination_node_id,
        "runner": case.runner,
        "child_passed": bool(child_summary.get("passed")),
        "resume_validation_passed": False,
        "migration_count": 0,
        "checkpoint_bytes": None,
        "downtime_seconds": None,
        "details": "",
    }

    if case.class_id == "benchmark-json-medium":
        migration = child_summary.get("migration") or {}
        row.update(
            {
                "resume_validation_passed": bool(
                    child_summary.get("resume_validation_passed")
                ),
                "migration_count": 1 if migration else 0,
                "checkpoint_bytes": migration.get("actual_checkpoint_bytes"),
                "downtime_seconds": migration.get("actual_downtime_seconds"),
                "details": (
                    f"workload={child_summary.get('workload')} "
                    f"variant={child_summary.get('variant')}"
                ),
            }
        )
    elif case.class_id == "dendro-r9-t1p0":
        migration = child_summary.get("migration") or {}
        row.update(
            {
                "resume_validation_passed": bool(
                    child_summary.get("resume_validation_passed")
                ),
                "migration_count": 1 if migration else 0,
                "checkpoint_bytes": migration.get("actual_checkpoint_bytes"),
                "downtime_seconds": migration.get("actual_downtime_seconds"),
                "details": (
                    f"workload={child_summary.get('workload')} "
                    f"variant={child_summary.get('variant')}"
                ),
            }
        )
    elif case.class_id == "llm-distilgpt2":
        row.update(
            {
                "resume_validation_passed": (
                    int(child_summary.get("resume_validations_passed") or 0)
                    == int(child_summary.get("migration_count") or 0)
                    and int(child_summary.get("migration_count") or 0) == 1
                ),
                "migration_count": int(child_summary.get("migration_count") or 0),
                "checkpoint_bytes": child_summary.get("checkpoint_bytes_median"),
                "downtime_seconds": child_summary.get("downtime_seconds_median"),
                "details": f"model={child_summary.get('model')}",
            }
        )
    return row


def case_passes(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("child_passed"))
        and bool(row.get("resume_validation_passed"))
        and int(row.get("migration_count") or 0) == 1
        and row.get("source_node_id") != row.get("destination_node_id")
    )


def stage5e1_passes(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != len(STAGE5E1_CASES):
        return False
    by_case = {str(row.get("case_id")): row for row in rows}
    if set(by_case) != {case.case_id for case in STAGE5E1_CASES}:
        return False
    for case in STAGE5E1_CASES:
        row = by_case[case.case_id]
        if row.get("class_id") != case.class_id:
            return False
        if row.get("source_node_id") != case.source_node_id:
            return False
        if row.get("destination_node_id") != case.destination_node_id:
            return False
        if not case_passes(row):
            return False
    return True
