from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from magellan.experiments.bundle import validate_checksums
from magellan.submission.models import TaskDefinitionSubmission


BENCHMARKS = ("nbody", "json", "matmul")
SIZES = ("small", "medium", "large")
DENDRO_VARIANT_RE = re.compile(r"^r(?P<resolution>\d+)-t(?P<time_end>\d+(?:\.\d+)?)$")
REPRESENTATIVE_EQUIVALENCE_CLASS = "benchmark-matmul-medium"


@dataclass(frozen=True)
class StaticCase:
    class_id: str
    workload: str
    variant: str
    benchmark: str | None = None
    size: str | None = None
    benchmark_iterations: int | None = None
    resolution: int | None = None
    time_end: float | None = None
    llm_max_steps: int | None = None


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _positive_float(row: dict[str, str], field: str) -> float:
    raw = row.get(field)
    if raw in (None, ""):
        raise ValueError(f"Missing {field} for {row.get('class_id')}")
    value = float(raw)
    if value <= 0:
        raise ValueError(f"Non-positive {field} for {row.get('class_id')}: {value}")
    return value


def _rounded_work_units(rate: float, target_seconds: float, *, quantum: int, minimum: int) -> int:
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    raw = rate * target_seconds
    rounded = int(round(raw / quantum) * quantum)
    return max(minimum, rounded)


def build_static_cases(
    profile_classes_csv: str | Path,
    *,
    target_seconds: float = 100.0,
) -> list[StaticCase]:
    """Build finite Stage 4A.4 cases from the validated Stage 4A.3 profiles.

    Synthetic benchmark iteration counts and the LLM step count are sized from
    measured progress rates so each finite static run is long enough for stable
    end-to-end accounting without inheriting the deliberately huge long-running
    defaults used by migration experiments. Dendro keeps its physical
    resolution/time-end variants unchanged.
    """

    rows = _read_csv(profile_classes_csv)
    by_id = {row["class_id"]: row for row in rows}
    expected = {
        *(f"benchmark-{benchmark}-{size}" for benchmark in BENCHMARKS for size in SIZES),
        "dendro-r8-t3p0",
        "dendro-r9-t1p0",
        "dendro-r10-t2p0",
        "llm-distilgpt2",
    }
    missing = sorted(expected - set(by_id))
    extra = sorted(set(by_id) - expected)
    if missing or extra:
        raise ValueError(f"Unexpected Stage 4A.3 class set: missing={missing} extra={extra}")

    cases: list[StaticCase] = []
    for benchmark in BENCHMARKS:
        for size in SIZES:
            class_id = f"benchmark-{benchmark}-{size}"
            rate = _positive_float(by_id[class_id], "progress_rate_median_units_per_second")
            cases.append(
                StaticCase(
                    class_id=class_id,
                    workload="benchmark",
                    variant=size,
                    benchmark=benchmark,
                    size=size,
                    benchmark_iterations=_rounded_work_units(
                        rate,
                        target_seconds,
                        quantum=100,
                        minimum=1_000,
                    ),
                )
            )

    for class_id in ("dendro-r8-t3p0", "dendro-r9-t1p0", "dendro-r10-t2p0"):
        row = by_id[class_id]
        match = DENDRO_VARIANT_RE.fullmatch(row["variant"])
        if match is None:
            raise ValueError(f"Unexpected Dendro variant for {class_id}: {row['variant']}")
        cases.append(
            StaticCase(
                class_id=class_id,
                workload="dendro",
                variant=row["variant"],
                resolution=int(match.group("resolution")),
                time_end=float(match.group("time_end")),
            )
        )

    llm_row = by_id["llm-distilgpt2"]
    llm_rate = _positive_float(llm_row, "progress_rate_median_units_per_second")
    cases.append(
        StaticCase(
            class_id="llm-distilgpt2",
            workload="llm",
            variant=llm_row["variant"],
            llm_max_steps=_rounded_work_units(
                llm_rate,
                target_seconds,
                quantum=1,
                minimum=5,
            ),
        )
    )
    return cases


def llm_training_definition(
    *,
    definition_id: str,
    model: str,
    node_ids: list[str],
    max_steps: int,
    checkpoint_every: int,
    sleep_per_step: float,
    torch_threads: int,
) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    training_text = (
        "Magellan migrates stateful workloads between carbon-aware "
        "geographically distributed computing regions while preserving optimizer state."
    )
    payload = {
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
            "adapter": "python_module",
            "module": "magellan.workloads.llm_train",
            "arguments": [
                "--checkpoint-dir", "{checkpoint_directory}",
                "--ready-file", "{readiness_file}",
                "--progress-file", "{progress_file}",
                "--checkpoint-metrics-file", "{task_directory}/runtime/checkpoint-metrics.jsonl",
                "--model", model,
                "--max-steps", str(max_steps),
                "--sleep-per-step", str(sleep_per_step),
                "--checkpoint-every", str(checkpoint_every),
                "--learning-rate", "0.00005",
                "--device", "cpu",
                "--torch-threads", str(torch_threads),
                "--text", training_text,
                "--completion-file", "{completion_file}",
                "--output-dir", "{output_directory}",
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
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def successful_static_bundle(path: str | Path, *, minimum_samples: int = 3) -> bool:
    summary_path = Path(path) / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if validate_checksums(path):
        return False
    return bool(
        summary.get("passed") is True
        and summary.get("status") == "completed"
        and int(summary.get("telemetry_sample_count") or 0) >= minimum_samples
        and int(summary.get("generation") or 0) == 0
        and float(summary.get("accumulated_paused_seconds") or 0.0) <= 1e-9
        and float(summary.get("accumulated_migration_seconds") or 0.0) <= 1e-9
        and float(summary.get("accumulated_transfer_cost_usd") or 0.0) <= 1e-12
        and float(summary.get("accumulated_transfer_carbon_grams") or 0.0) <= 1e-12
    )


def summarize_canonical_runs(rows: list[dict[str, Any]], *, trials: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["class_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for class_id, items in sorted(grouped.items()):
        if len(items) != trials:
            raise ValueError(f"{class_id} has {len(items)} canonical trials, expected {trials}")
        output.append(
            {
                "class_id": class_id,
                "workload": items[0]["workload"],
                "variant": items[0]["variant"],
                "trial_count": len(items),
                "wall_seconds_median": median(float(item["wall_seconds"]) for item in items),
                "runtime_seconds_median": median(float(item["accumulated_runtime_seconds"]) for item in items),
                "cost_usd_median": median(float(item["accumulated_cost_usd"]) for item in items),
                "carbon_grams_median": median(float(item["accumulated_carbon_grams"]) for item in items),
                "telemetry_samples_median": median(int(item["telemetry_sample_count"]) for item in items),
            }
        )
    return output


def summarize_node_equivalence(
    rows: list[dict[str, Any]],
    *,
    canonical_node_id: str,
    trials: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["node_id"]), []).append(row)
    if canonical_node_id not in grouped:
        raise ValueError(f"Missing canonical node {canonical_node_id} in equivalence rows")
    medians: dict[str, float] = {}
    for node_id, items in grouped.items():
        if len(items) != trials:
            raise ValueError(f"{node_id} has {len(items)} equivalence trials, expected {trials}")
        medians[node_id] = median(float(item["accumulated_runtime_seconds"]) for item in items)
    baseline = medians[canonical_node_id]
    if baseline <= 0:
        raise ValueError("Canonical runtime must be positive")
    return [
        {
            "node_id": node_id,
            "trial_count": len(grouped[node_id]),
            "runtime_seconds_median": medians[node_id],
            "slowdown_vs_canonical": medians[node_id] / baseline,
        }
        for node_id in sorted(grouped)
    ]
