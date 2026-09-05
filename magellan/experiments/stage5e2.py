from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from magellan.config.models import NodeResourceCapacity
from magellan.models.types import TaskResourceRequest
from magellan.submission.models import TaskDefinitionSubmission
from magellan.experiments.workload_population import benchmark_definition, dendro_definition


BENCHMARK_CLASS_ID = "benchmark-json-medium"
LLM_CLASS_ID = "llm-distilgpt2"
DENDRO_CLASS_ID = "dendro-r9-t1p0"
STAGE5E2_CLASSES = (BENCHMARK_CLASS_ID, DENDRO_CLASS_ID, LLM_CLASS_ID)

# Physical realization of the exact 11-task Stage 4D.2/4D.3 umax mix.
# Every node uses one of the maximal packings frozen by Stage 4D.1:
#   2 benchmark, benchmark+LLM, 2 LLM, or 1 Dendro.
STAGE5E2_LAYOUT: dict[str, tuple[str, ...]] = {
    "boston": (BENCHMARK_CLASS_ID, BENCHMARK_CLASS_ID),
    "california": (BENCHMARK_CLASS_ID, LLM_CLASS_ID),
    "south-australia": (LLM_CLASS_ID, LLM_CLASS_ID),
    "nepal": (DENDRO_CLASS_ID,),
    "ethiopia": (DENDRO_CLASS_ID,),
    "france": (DENDRO_CLASS_ID,),
    "virginia": (BENCHMARK_CLASS_ID, LLM_CLASS_ID),
}

EXPECTED_CLASS_COUNTS = {
    BENCHMARK_CLASS_ID: 4,
    DENDRO_CLASS_ID: 3,
    LLM_CLASS_ID: 4,
}


@dataclass(frozen=True)
class PhysicalTaskSpec:
    task_index: int
    node_id: str
    class_id: str
    node_ordinal: int


def packing_signature(classes: Iterable[str]) -> tuple[int, int, int]:
    values = list(classes)
    return (
        values.count(BENCHMARK_CLASS_ID),
        values.count(DENDRO_CLASS_ID),
        values.count(LLM_CLASS_ID),
    )


def physical_task_specs() -> list[PhysicalTaskSpec]:
    rows: list[PhysicalTaskSpec] = []
    index = 0
    for node_id, classes in STAGE5E2_LAYOUT.items():
        class_ordinals: Counter[str] = Counter()
        for class_id in classes:
            class_ordinals[class_id] += 1
            rows.append(
                PhysicalTaskSpec(
                    task_index=index,
                    node_id=node_id,
                    class_id=class_id,
                    node_ordinal=class_ordinals[class_id],
                )
            )
            index += 1
    return rows


def layout_class_counts() -> dict[str, int]:
    counts: Counter[str] = Counter()
    for classes in STAGE5E2_LAYOUT.values():
        counts.update(classes)
    return {class_id: counts[class_id] for class_id in STAGE5E2_CLASSES}


def resource_usage(
    classes: Iterable[str],
    requests: dict[str, TaskResourceRequest],
) -> tuple[float, int, int]:
    cpu = 0.0
    memory = 0
    gpu = 0
    for class_id in classes:
        request = requests[class_id]
        cpu += float(request.cpu_cores)
        memory += int(request.memory_mb)
        gpu += int(request.gpu_count)
    return cpu, memory, gpu


def validate_physical_layout(
    *,
    capacities: dict[str, NodeResourceCapacity],
    requests: dict[str, TaskResourceRequest],
    maximal_signatures: dict[str, set[tuple[int, int, int]]],
) -> list[dict[str, Any]]:
    if set(STAGE5E2_LAYOUT) != set(capacities):
        raise ValueError(
            "Stage 5E.2 layout nodes do not match frozen Stage 4D.1 capacities"
        )
    if set(STAGE5E2_CLASSES) - set(requests):
        raise ValueError("Stage 5E.2 resource model is missing workload classes")
    if layout_class_counts() != EXPECTED_CLASS_COUNTS:
        raise ValueError("Stage 5E.2 layout no longer has the frozen 4/3/4 class mix")

    rows: list[dict[str, Any]] = []
    for node_id, classes in STAGE5E2_LAYOUT.items():
        capacity = capacities[node_id]
        signature = packing_signature(classes)
        if signature not in maximal_signatures.get(node_id, set()):
            raise ValueError(
                f"{node_id} packing {signature} is not maximal in Stage 4D.1"
            )
        cpu, memory, gpu = resource_usage(classes, requests)
        if capacity.cpu_cores is not None and cpu > float(capacity.cpu_cores) + 1e-9:
            raise ValueError(f"{node_id} planned CPU exceeds capacity")
        if capacity.memory_mb is not None and memory > int(capacity.memory_mb):
            raise ValueError(f"{node_id} planned memory exceeds capacity")
        if capacity.gpu_count is not None and gpu > int(capacity.gpu_count):
            raise ValueError(f"{node_id} planned GPU exceeds capacity")
        rows.append(
            {
                "node_id": node_id,
                "packing_signature": "/".join(str(value) for value in signature),
                "task_count": len(classes),
                "benchmark_count": signature[0],
                "dendro_count": signature[1],
                "llm_count": signature[2],
                "expected_reserved_cpu_cores": cpu,
                "expected_reserved_memory_mb": memory,
                "expected_reserved_gpu_count": gpu,
                "effective_cpu_cores": capacity.cpu_cores,
                "effective_memory_mb": capacity.memory_mb,
                "effective_gpu_count": capacity.gpu_count,
                "cpu_fraction": (
                    None
                    if capacity.cpu_cores in (None, 0)
                    else cpu / float(capacity.cpu_cores)
                ),
                "memory_fraction": (
                    None
                    if capacity.memory_mb in (None, 0)
                    else memory / int(capacity.memory_mb)
                ),
                "is_frozen_maximal_packing": True,
            }
        )
    return rows


def llm_definition(
    *,
    definition_id: str,
    node_ids: list[str],
    model: str,
    checkpoint_every: int = 1,
    sleep_per_step: float = 2.0,
    torch_threads: int = 2,
) -> dict[str, Any]:
    training_text = (
        "Magellan migrates long-running stateful machine-learning workloads across "
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
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def physical_definition(
    *,
    class_id: str,
    definition_id: str,
    request: TaskResourceRequest,
    node_ids: list[str],
    seed: int,
    benchmark_iterations: int,
    llm_model: str,
    dendro_template: dict[str, Any],
    dendro_solver: str,
    dendro_parameter_template: str,
) -> dict[str, Any]:
    if class_id == BENCHMARK_CLASS_ID:
        payload = benchmark_definition(
            definition_id=definition_id,
            benchmark="json",
            size="medium",
            seed=seed,
            iterations=benchmark_iterations,
            node_ids=node_ids,
        )
    elif class_id == LLM_CLASS_ID:
        payload = llm_definition(
            definition_id=definition_id,
            node_ids=node_ids,
            model=llm_model,
            checkpoint_every=1,
            sleep_per_step=2.0,
            torch_threads=2,
        )
    elif class_id == DENDRO_CLASS_ID:
        payload = dendro_definition(
            definition_id=definition_id,
            template=dendro_template,
            solver_path=dendro_solver,
            parameter_template_path=dendro_parameter_template,
            resolution=9,
            time_end=1.0,
            eligible_nodes=node_ids,
        )
    else:
        raise KeyError(class_id)

    # Admission uses the exact frozen Stage 4D.1 p95 request.  The runtime,
    # compatibility contract, checkpoint format, and workload implementation are
    # otherwise unchanged from the real workload harnesses used in Stage 5E.1.
    payload["profile"]["resource_request"] = request.model_dump(mode="json")
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def stage5e2_passes(
    task_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
) -> bool:
    specs = physical_task_specs()
    if len(task_rows) != len(specs) or len(node_rows) != len(STAGE5E2_LAYOUT):
        return False

    if Counter(str(row.get("class_id")) for row in task_rows) != Counter(
        EXPECTED_CLASS_COUNTS
    ):
        return False
    if Counter(str(row.get("node_id")) for row in task_rows) != Counter(
        spec.node_id for spec in specs
    ):
        return False

    for row in task_rows:
        if not _truthy(row.get("launched")):
            return False
        if not _truthy(row.get("steady_running")):
            return False
        if int(float(row.get("telemetry_sample_count") or 0)) < 1:
            return False
        if int(float(row.get("cpu_sample_count") or 0)) < 1:
            return False
        if int(float(row.get("rss_sample_count") or 0)) < 1:
            return False
        if int(float(row.get("min_process_count") or 0)) < 1:
            return False
        if float(row.get("max_memory_rss_mb") or 0) <= 0:
            return False
        if float(row.get("progress_min") or 0) < 0:
            return False
        if float(row.get("progress_max") or 0) < float(row.get("progress_min") or 0):
            return False
        if not _truthy(row.get("cleanup_ok")):
            return False

    expected_nodes = set(STAGE5E2_LAYOUT)
    if {str(row.get("node_id")) for row in node_rows} != expected_nodes:
        return False
    for row in node_rows:
        if not _truthy(row.get("is_frozen_maximal_packing")):
            return False
        if not _truthy(row.get("all_tasks_running")):
            return False
        if not _truthy(row.get("reservation_matches_expected")):
            return False
        if not _truthy(row.get("capacity_respected")):
            return False
        if float(row.get("max_resource_busy_fraction") or 0) > 1.0 + 1e-9:
            return False
        if int(float(row.get("actual_cpu_sample_count") or 0)) < 1:
            return False
        if int(float(row.get("actual_rss_sample_count") or 0)) < 1:
            return False

    return True
