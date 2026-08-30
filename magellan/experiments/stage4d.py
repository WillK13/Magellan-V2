from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from magellan.config.models import NodeResourceCapacity
from magellan.models.types import TaskResourceRequest


CORE_RESOURCE_CLASSES = (
    "benchmark-json-medium",
    "dendro-r9-t1p0",
    "llm-distilgpt2",
)


@dataclass(frozen=True)
class NodeResourceEvidence:
    node_id: str
    machine_type: str
    configured: NodeResourceCapacity
    observed: NodeResourceCapacity
    effective: NodeResourceCapacity


@dataclass(frozen=True)
class WorkloadResourceEvidence:
    class_id: str
    cpu_p95_percent: float
    memory_p95_mb: float
    request: TaskResourceRequest
    source: str = "stage4a3_p95"


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _capacity_from_mapping(value: dict[str, Any], *, label: str) -> NodeResourceCapacity:
    cpu = value.get("cpu_cores")
    memory = value.get("memory_mb")
    gpu = value.get("gpu_count")
    if cpu is None or memory is None or gpu is None:
        raise ValueError(f"Incomplete {label} resource capacity: {value}")
    return NodeResourceCapacity(
        cpu_cores=float(cpu),
        memory_mb=int(memory),
        gpu_count=int(gpu),
        accelerator_types=list(value.get("accelerator_types") or []),
    )


def _minimum_capacity(
    configured: NodeResourceCapacity,
    observed: NodeResourceCapacity,
) -> NodeResourceCapacity:
    configured_accelerators = set(configured.accelerator_types or [])
    observed_accelerators = set(observed.accelerator_types or [])
    return NodeResourceCapacity(
        cpu_cores=min(float(configured.cpu_cores), float(observed.cpu_cores)),
        memory_mb=min(int(configured.memory_mb), int(observed.memory_mb)),
        gpu_count=min(int(configured.gpu_count), int(observed.gpu_count)),
        accelerator_types=sorted(configured_accelerators & observed_accelerators),
    )


def load_node_resource_evidence(
    stage4a1_bundle: str | Path,
) -> dict[str, NodeResourceEvidence]:
    import json

    root = Path(stage4a1_bundle)
    hardware_path = root / "hardware.json"
    if not hardware_path.is_file():
        raise FileNotFoundError(hardware_path)
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("Stage 4A.1 hardware.json must contain node records")

    output: dict[str, NodeResourceEvidence] = {}
    for node_id, record in sorted(hardware.items()):
        configured_record = (record.get("configured") or {}).get("resources") or {}
        observed_record = (record.get("capabilities") or {}).get("observed") or {}
        configured = _capacity_from_mapping(
            configured_record,
            label=f"configured {node_id}",
        )
        observed = _capacity_from_mapping(
            observed_record,
            label=f"observed {node_id}",
        )
        output[node_id] = NodeResourceEvidence(
            node_id=node_id,
            machine_type=str((record.get("configured") or {}).get("machine_type") or ""),
            configured=configured,
            observed=observed,
            effective=_minimum_capacity(configured, observed),
        )
    return output


def load_workload_resource_evidence(
    stage4a3_bundle: str | Path,
    *,
    class_ids: Iterable[str] = CORE_RESOURCE_CLASSES,
) -> dict[str, WorkloadResourceEvidence]:
    rows = {
        row["class_id"]: row
        for row in read_csv(Path(stage4a3_bundle) / "profile_classes.csv")
    }
    output: dict[str, WorkloadResourceEvidence] = {}
    for class_id in class_ids:
        if class_id not in rows:
            raise ValueError(f"Missing Stage 4A.3 profile class {class_id}")
        row = rows[class_id]
        cpu_p95 = float(row.get("cpu_p95_percent") or 0.0)
        memory_p95 = float(row.get("memory_p95_mb") or 0.0)
        if not math.isfinite(cpu_p95) or cpu_p95 <= 0:
            raise ValueError(f"Invalid Stage 4A.3 CPU p95 for {class_id}: {cpu_p95}")
        if not math.isfinite(memory_p95) or memory_p95 <= 0:
            raise ValueError(f"Invalid Stage 4A.3 memory p95 for {class_id}: {memory_p95}")
        output[class_id] = WorkloadResourceEvidence(
            class_id=class_id,
            cpu_p95_percent=cpu_p95,
            memory_p95_mb=memory_p95,
            request=TaskResourceRequest(
                cpu_cores=cpu_p95 / 100.0,
                memory_mb=int(math.ceil(memory_p95)),
                gpu_count=0,
                accelerator_type=None,
            ),
        )
    return output


def request_fits_capacity(
    request: TaskResourceRequest,
    capacity: NodeResourceCapacity,
) -> bool:
    if capacity.cpu_cores is not None and request.cpu_cores > capacity.cpu_cores + 1e-12:
        return False
    if capacity.memory_mb is not None and request.memory_mb > capacity.memory_mb:
        return False
    if capacity.gpu_count is not None and request.gpu_count > capacity.gpu_count:
        return False
    if request.accelerator_type is not None:
        if request.accelerator_type not in set(capacity.accelerator_types or []):
            return False
    return True


def maximum_homogeneous_tasks(
    capacity: NodeResourceCapacity,
    request: TaskResourceRequest,
) -> int:
    limits: list[int] = []
    if capacity.cpu_cores is not None:
        limits.append(int(math.floor((capacity.cpu_cores + 1e-12) / request.cpu_cores)))
    if capacity.memory_mb is not None and request.memory_mb > 0:
        limits.append(int(capacity.memory_mb // request.memory_mb))
    if capacity.gpu_count is not None and request.gpu_count > 0:
        limits.append(int(capacity.gpu_count // request.gpu_count))
    if request.accelerator_type is not None and request.accelerator_type not in set(capacity.accelerator_types or []):
        return 0
    if not limits:
        raise ValueError("At least one finite node resource limit is required")
    return max(0, min(limits))


def packing_usage(
    counts: dict[str, int],
    requests: dict[str, WorkloadResourceEvidence],
) -> tuple[float, int, int]:
    cpu = 0.0
    memory = 0
    gpu = 0
    for class_id, count in counts.items():
        if count < 0:
            raise ValueError("Packing counts must be non-negative")
        request = requests[class_id].request
        cpu += request.cpu_cores * count
        memory += request.memory_mb * count
        gpu += request.gpu_count * count
    return cpu, memory, gpu


def enumerate_maximal_packings(
    capacity: NodeResourceCapacity,
    requests: dict[str, WorkloadResourceEvidence],
) -> list[dict[str, Any]]:
    class_ids = list(requests)
    maxima = {
        class_id: maximum_homogeneous_tasks(capacity, evidence.request)
        for class_id, evidence in requests.items()
    }
    feasible: list[dict[str, Any]] = []

    def visit(index: int, counts: dict[str, int]) -> None:
        if index == len(class_ids):
            if not any(counts.values()):
                return
            cpu, memory, gpu = packing_usage(counts, requests)
            if capacity.cpu_cores is not None and cpu > capacity.cpu_cores + 1e-12:
                return
            if capacity.memory_mb is not None and memory > capacity.memory_mb:
                return
            if capacity.gpu_count is not None and gpu > capacity.gpu_count:
                return
            feasible.append({
                "counts": dict(counts),
                "used_cpu_cores": cpu,
                "used_memory_mb": memory,
                "used_gpu_count": gpu,
            })
            return
        class_id = class_ids[index]
        for count in range(maxima[class_id] + 1):
            counts[class_id] = count
            visit(index + 1, counts)
        counts.pop(class_id, None)

    visit(0, {})

    maximal: list[dict[str, Any]] = []
    for candidate in feasible:
        counts = candidate["counts"]
        can_add = False
        for class_id in class_ids:
            expanded = dict(counts)
            expanded[class_id] += 1
            cpu, memory, gpu = packing_usage(expanded, requests)
            fits = True
            if capacity.cpu_cores is not None and cpu > capacity.cpu_cores + 1e-12:
                fits = False
            if capacity.memory_mb is not None and memory > capacity.memory_mb:
                fits = False
            if capacity.gpu_count is not None and gpu > capacity.gpu_count:
                fits = False
            if fits:
                can_add = True
                break
        if not can_add:
            maximal.append(candidate)

    return sorted(
        maximal,
        key=lambda row: (
            -sum(row["counts"].values()),
            -row["used_cpu_cores"],
            tuple(-row["counts"][class_id] for class_id in class_ids),
        ),
    )


def node_capacity_rows(
    capacities: dict[str, NodeResourceEvidence],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id, evidence in sorted(capacities.items()):
        rows.append({
            "node_id": node_id,
            "machine_type": evidence.machine_type,
            "configured_cpu_cores": evidence.configured.cpu_cores,
            "observed_cpu_cores": evidence.observed.cpu_cores,
            "effective_cpu_cores": evidence.effective.cpu_cores,
            "configured_memory_mb": evidence.configured.memory_mb,
            "observed_memory_mb": evidence.observed.memory_mb,
            "effective_memory_mb": evidence.effective.memory_mb,
            "configured_gpu_count": evidence.configured.gpu_count,
            "observed_gpu_count": evidence.observed.gpu_count,
            "effective_gpu_count": evidence.effective.gpu_count,
            "capacity_source": "min(stage4a1_configured,stage4a1_observed)",
        })
    return rows


def workload_request_rows(
    requests: dict[str, WorkloadResourceEvidence],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, evidence in sorted(requests.items()):
        rows.append({
            "class_id": class_id,
            "cpu_p95_percent": evidence.cpu_p95_percent,
            "cpu_request_cores": evidence.request.cpu_cores,
            "memory_p95_mb": evidence.memory_p95_mb,
            "memory_request_mb": evidence.request.memory_mb,
            "gpu_request_count": evidence.request.gpu_count,
            "request_source": evidence.source,
        })
    return rows


def homogeneous_capacity_rows(
    capacities: dict[str, NodeResourceEvidence],
    requests: dict[str, WorkloadResourceEvidence],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_id, node in sorted(capacities.items()):
        for class_id, workload in sorted(requests.items()):
            rows.append({
                "node_id": node_id,
                "class_id": class_id,
                "max_concurrent_tasks": maximum_homogeneous_tasks(
                    node.effective,
                    workload.request,
                ),
                "individually_feasible": request_fits_capacity(
                    workload.request,
                    node.effective,
                ),
            })
    return rows


def maximal_packing_rows(
    capacities: dict[str, NodeResourceEvidence],
    requests: dict[str, WorkloadResourceEvidence],
) -> list[dict[str, Any]]:
    class_ids = sorted(requests)
    rows: list[dict[str, Any]] = []
    for node_id, node in sorted(capacities.items()):
        for index, packing in enumerate(
            enumerate_maximal_packings(node.effective, requests),
            start=1,
        ):
            row: dict[str, Any] = {
                "node_id": node_id,
                "packing_index": index,
                "total_tasks": sum(packing["counts"].values()),
                "used_cpu_cores": packing["used_cpu_cores"],
                "used_memory_mb": packing["used_memory_mb"],
                "used_gpu_count": packing["used_gpu_count"],
                "cpu_fraction": (
                    packing["used_cpu_cores"] / node.effective.cpu_cores
                    if node.effective.cpu_cores
                    else 0.0
                ),
                "memory_fraction": (
                    packing["used_memory_mb"] / node.effective.memory_mb
                    if node.effective.memory_mb
                    else 0.0
                ),
            }
            for class_id in class_ids:
                row[f"count_{class_id}"] = packing["counts"][class_id]
            rows.append(row)
    return rows
