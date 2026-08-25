from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

from magellan.config.models import ClusterConfig
from magellan.submission.models import (
    TaskDefinitionSubmission,
    TaskRunSubmission,
)


SUPPORTED_WORKLOADS = {"nbody", "json", "matmul", "dendro", "llm"}
BENCHMARK_WORKLOADS = {"nbody", "json", "matmul"}


@dataclass(frozen=True)
class PopulationTask:
    index: int
    workload: str
    variant: str
    initial_node_id: str
    scheduled_offset_seconds: float
    definition: dict[str, Any]
    run: dict[str, Any]


def parse_weighted_mix(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid workload mix item {item!r}; expected workload=weight"
            )
        name, value = item.split("=", 1)
        name = name.strip().lower()
        if name not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"Unsupported workload {name!r}; choose from "
                f"{sorted(SUPPORTED_WORKLOADS)}"
            )
        weight = float(value)
        if weight < 0:
            raise ValueError("Workload mix weights must be non-negative")
        if weight > 0:
            weights[name] = weight
    if not weights:
        raise ValueError("Workload mix must contain at least one positive weight")
    return weights


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("Expected a comma-separated list of positive integers")
    return values


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected a comma-separated list of positive numbers")
    return values


def parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated list")
    return values


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"Cannot create identifier from {value!r}")
    return cleaned


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    target = rng.random() * total
    cumulative = 0.0
    for name in sorted(weights):
        cumulative += weights[name]
        if target <= cumulative:
            return name
    return sorted(weights)[-1]


def _benchmark_resources(benchmark: str, size: str) -> dict[str, Any]:
    cpu = {
        "nbody": {"small": 0.5, "medium": 0.75, "large": 1.0},
        "json": {"small": 0.25, "medium": 0.5, "large": 0.75},
        "matmul": {"small": 0.5, "medium": 1.0, "large": 1.5},
    }[benchmark][size]
    memory = {
        "nbody": {"small": 128, "medium": 192, "large": 256},
        "json": {"small": 128, "medium": 256, "large": 512},
        "matmul": {"small": 256, "medium": 512, "large": 1024},
    }[benchmark][size]
    return {
        "cpu_cores": cpu,
        "memory_mb": memory,
        "gpu_count": 0,
        "accelerator_type": None,
    }


def benchmark_definition(
    *,
    definition_id: str,
    benchmark: str,
    size: str,
    seed: int,
    iterations: int,
    node_ids: list[str],
) -> dict[str, Any]:
    if benchmark not in BENCHMARK_WORKLOADS:
        raise ValueError(f"Not a benchmark workload: {benchmark}")
    resources = _benchmark_resources(benchmark, size)
    payload = {
        "definition_id": definition_id,
        "profile": {
            "workload_type": f"benchmark-{benchmark}",
            "power_kw": 0.08,
            "checkpoint_bytes": 4096,
            "data_bytes": 0,
            "prestaged_node_ids": node_ids,
            "estimated_remaining_seconds": 1800,
            "accumulated_cost_usd": 0,
            "cost_cap_usd": 5,
            "priority": 10,
            "deadline_at_utc": None,
            "resource_request": resources,
            "compatibility": {
                "architectures": [],
                "operating_systems": [],
                "minimum_cpu_cores": max(1, int(resources["cpu_cores"])),
                "minimum_memory_mb": resources["memory_mb"],
                "required_commands": ["python3"],
                "required_runtimes": {"python": ">=3.11,<4"},
                "required_features": [
                    "python-module",
                    "process-group",
                    "application-checkpoint",
                ],
                "checkpoint_architecture_independent": True,
                "requires_same_mpi_world_size": False,
            },
        },
        "runtime": {
            "adapter": "python_module",
            "module": "magellan.workloads.benchmark",
            "arguments": [
                "--benchmark",
                benchmark,
                "--size",
                size,
                "--iterations",
                str(iterations),
                "--seed",
                str(seed),
                "--checkpoint-file",
                "{checkpoint_file}",
                "--progress-file",
                "{progress_file}",
                "--completion-file",
                "{completion_file}",
                "--output-dir",
                "{output_directory}",
            ],
            "environment": {},
            "working_directory": ".",
            "checkpoint_relative_path": "checkpoint/benchmark.json",
            "progress_relative_path": "runtime/progress.json",
            "completion_relative_path": "runtime/completion.json",
            "output_relative_directory": "output",
            "stop_timeout_seconds": 10,
            "minimum_process_count": 1,
        },
        "artifacts": [],
    }
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def dendro_definition(
    *,
    definition_id: str,
    template: dict[str, Any],
    solver_path: str,
    parameter_template_path: str,
    resolution: int,
    time_end: float,
    eligible_nodes: list[str],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(template))
    payload["definition_id"] = definition_id
    payload["profile"]["prestaged_node_ids"] = eligible_nodes
    payload["profile"]["resource_request"] = {
        "cpu_cores": 2,
        "memory_mb": 8192,
        "gpu_count": 0,
        "accelerator_type": None,
    }
    compatibility = payload["profile"].setdefault("compatibility", {})
    compatibility["minimum_cpu_cores"] = 2
    compatibility["minimum_memory_mb"] = 8192
    payload["runtime"]["arguments"] = [
        solver_path,
        parameter_template_path,
        "--world-size",
        "2",
        "--ts-mode",
        "1",
        "--resolution",
        str(resolution),
        "--time-end",
        str(time_end),
    ]
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def cloned_definition(
    *,
    definition_id: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(template))
    payload["definition_id"] = definition_id
    return TaskDefinitionSubmission.model_validate(payload).model_dump(mode="json")


def generate_population(
    *,
    cluster: ClusterConfig,
    count: int,
    seed: int,
    mix: dict[str, float],
    population_id: str,
    benchmark_iterations: int = 1000,
    mean_interarrival_seconds: float = 0.0,
    initial_nodes: list[str] | None = None,
    dendro_template: dict[str, Any] | None = None,
    dendro_solver: str | None = None,
    dendro_parameter_template: str | None = None,
    dendro_resolutions: list[int] | None = None,
    dendro_time_ends: list[float] | None = None,
    dendro_nodes: list[str] | None = None,
    llm_template: dict[str, Any] | None = None,
    llm_nodes: list[str] | None = None,
) -> list[PopulationTask]:
    if count < 1:
        raise ValueError("count must be positive")
    if benchmark_iterations < 1:
        raise ValueError("benchmark_iterations must be positive")
    if mean_interarrival_seconds < 0:
        raise ValueError("mean_interarrival_seconds must be non-negative")

    cluster_ids = [node.id for node in cluster.nodes]
    allowed_nodes = initial_nodes or cluster_ids
    for node_id in allowed_nodes:
        cluster.get_node(node_id)

    if "dendro" in mix:
        if dendro_template is None or not dendro_solver or not dendro_parameter_template:
            raise ValueError(
                "Dendro population requires template, solver, and parameter template"
            )
    if "llm" in mix and llm_template is None:
        raise ValueError("LLM population requires an LLM definition template")

    # Stage 4 pre-stages the Dendro runtime on every experiment node, so
    # Dendro populations should be geographically unrestricted by default.
    d_nodes = dendro_nodes or list(cluster_ids)
    l_nodes = llm_nodes or [
        node_id for node_id in ("boston", "virginia") if node_id in cluster_ids
    ]
    if "dendro" in mix and not d_nodes:
        raise ValueError("No eligible Dendro nodes configured")
    if "llm" in mix and not l_nodes:
        raise ValueError("No eligible LLM nodes configured")

    resolutions = dendro_resolutions or [8, 9, 10]
    time_ends = dendro_time_ends or [0.5, 1.0, 2.0]
    rng = random.Random(seed)
    population_slug = _slug(population_id)
    tasks: list[PopulationTask] = []
    scheduled_offset = 0.0

    for index in range(count):
        workload = _weighted_choice(rng, mix)
        task_seed = rng.randrange(0, 2**31)
        definition_id = f"{population_slug}-{index:04d}-{workload}"

        if workload in BENCHMARK_WORKLOADS:
            size = rng.choice(["small", "medium", "large"])
            variant = size
            owner = rng.choice(allowed_nodes)
            definition = benchmark_definition(
                definition_id=definition_id,
                benchmark=workload,
                size=size,
                seed=task_seed,
                iterations=benchmark_iterations,
                node_ids=cluster_ids,
            )
        elif workload == "dendro":
            assert dendro_template is not None
            assert dendro_solver is not None
            assert dendro_parameter_template is not None
            resolution = rng.choice(resolutions)
            time_end = rng.choice(time_ends)
            variant = f"maxdepth-{resolution}-tend-{time_end:g}"
            owner = rng.choice(d_nodes)
            definition = dendro_definition(
                definition_id=definition_id,
                template=dendro_template,
                solver_path=dendro_solver,
                parameter_template_path=dendro_parameter_template,
                resolution=resolution,
                time_end=time_end,
                eligible_nodes=d_nodes,
            )
        else:
            assert workload == "llm"
            assert llm_template is not None
            variant = "template"
            owner = rng.choice(l_nodes)
            definition = cloned_definition(
                definition_id=definition_id,
                template=llm_template,
            )

        if mean_interarrival_seconds > 0 and index > 0:
            scheduled_offset += rng.expovariate(1.0 / mean_interarrival_seconds)

        run = TaskRunSubmission(
            definition_id=definition_id,
            initial_owner_node_id=owner,
            idempotency_key=(
                f"population-{population_slug}-{seed}-{index:04d}-{task_seed}"
            ),
            auto_start=False,
            labels={
                "population_id": population_id,
                "population_seed": str(seed),
                "population_index": str(index),
                "workload": workload,
                "variant": variant,
            },
        ).model_dump(mode="json")

        tasks.append(
            PopulationTask(
                index=index,
                workload=workload,
                variant=variant,
                initial_node_id=owner,
                scheduled_offset_seconds=scheduled_offset,
                definition=definition,
                run=run,
            )
        )

    return tasks


def write_population_plan(
    *,
    output_directory: Path,
    population_id: str,
    seed: int,
    mix: dict[str, float],
    tasks: list[PopulationTask],
) -> Path:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Population output directory is not empty: {output_directory}"
        )
    definitions_dir = output_directory / "definitions"
    runs_dir = output_directory / "runs"
    definitions_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_tasks: list[dict[str, Any]] = []
    for task in tasks:
        definition_path = definitions_dir / f"{task.index:04d}.json"
        run_path = runs_dir / f"{task.index:04d}.json"
        definition_path.write_text(
            json.dumps(task.definition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_path.write_text(
            json.dumps(task.run, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_tasks.append(
            {
                "index": task.index,
                "workload": task.workload,
                "variant": task.variant,
                "initial_node_id": task.initial_node_id,
                "scheduled_offset_seconds": task.scheduled_offset_seconds,
                "definition_file": definition_path.relative_to(output_directory).as_posix(),
                "run_file": run_path.relative_to(output_directory).as_posix(),
            }
        )

    manifest = {
        "format_version": 1,
        "population_id": population_id,
        "seed": seed,
        "mix": mix,
        "task_count": len(tasks),
        "tasks": manifest_tasks,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
