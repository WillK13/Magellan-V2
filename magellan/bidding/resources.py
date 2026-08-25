from __future__ import annotations

from dataclasses import dataclass
from math import inf

from magellan.config.models import NodeResourceCapacity
from magellan.models.types import TaskResourceRequest


@dataclass(frozen=True)
class ResourceVector:
    cpu_cores: float = 0.0
    memory_mb: int = 0
    gpu_count: int = 0

    @classmethod
    def from_request(
        cls,
        request: TaskResourceRequest,
    ) -> "ResourceVector":
        return cls(
            cpu_cores=float(request.cpu_cores),
            memory_mb=int(request.memory_mb),
            gpu_count=int(request.gpu_count),
        )

    def __add__(self, other: "ResourceVector") -> "ResourceVector":
        return ResourceVector(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_mb=self.memory_mb + other.memory_mb,
            gpu_count=self.gpu_count + other.gpu_count,
        )


@dataclass
class ResourceLedger:
    cpu_cores: float
    memory_mb: float
    gpu_count: float
    accelerator_types: set[str]

    @classmethod
    def from_capacity(
        cls,
        capacity: NodeResourceCapacity,
        used: ResourceVector | None = None,
    ) -> "ResourceLedger":
        used_value = used or ResourceVector()
        return cls(
            cpu_cores=(
                inf
                if capacity.cpu_cores is None
                else max(0.0, capacity.cpu_cores - used_value.cpu_cores)
            ),
            memory_mb=(
                inf
                if capacity.memory_mb is None
                else max(0.0, capacity.memory_mb - used_value.memory_mb)
            ),
            gpu_count=(
                inf
                if capacity.gpu_count is None
                else max(0.0, capacity.gpu_count - used_value.gpu_count)
            ),
            accelerator_types=set(capacity.accelerator_types),
        )

    def compatible(
        self,
        request: TaskResourceRequest,
    ) -> tuple[bool, str | None]:
        if request.gpu_count > 0 and self.gpu_count <= 0:
            return False, "Destination has no available GPU capacity"

        if (
            request.gpu_count > 0
            and request.accelerator_type is not None
            and self.accelerator_types
            and request.accelerator_type not in self.accelerator_types
        ):
            return (
                False,
                "Destination does not provide accelerator type "
                f"{request.accelerator_type}",
            )

        if request.cpu_cores > self.cpu_cores:
            return False, "Insufficient unreserved CPU cores"

        if request.memory_mb > self.memory_mb:
            return False, "Insufficient unreserved memory"

        if request.gpu_count > self.gpu_count:
            return False, "Insufficient unreserved GPU capacity"

        return True, None

    def consume(self, request: TaskResourceRequest) -> None:
        if self.cpu_cores != inf:
            self.cpu_cores = max(0.0, self.cpu_cores - request.cpu_cores)
        if self.memory_mb != inf:
            self.memory_mb = max(0.0, self.memory_mb - request.memory_mb)
        if self.gpu_count != inf:
            self.gpu_count = max(0.0, self.gpu_count - request.gpu_count)

    def snapshot(self) -> dict[str, float | int | None | list[str]]:
        return {
            "available_cpu_cores": (
                None if self.cpu_cores == inf else self.cpu_cores
            ),
            "available_memory_mb": (
                None if self.memory_mb == inf else int(self.memory_mb)
            ),
            "available_gpu_count": (
                None if self.gpu_count == inf else int(self.gpu_count)
            ),
            "accelerator_types": sorted(self.accelerator_types),
        }


def sum_requests(
    requests: list[TaskResourceRequest],
) -> ResourceVector:
    total = ResourceVector()
    for request in requests:
        total = total + ResourceVector.from_request(request)
    return total


def dominant_resource_share(
    request: TaskResourceRequest,
    capacity: NodeResourceCapacity,
) -> float:
    shares: list[float] = []

    if capacity.cpu_cores is not None:
        shares.append(request.cpu_cores / capacity.cpu_cores)
    if capacity.memory_mb is not None and request.memory_mb > 0:
        shares.append(request.memory_mb / capacity.memory_mb)
    if capacity.gpu_count is not None and request.gpu_count > 0:
        if capacity.gpu_count == 0:
            return inf
        shares.append(request.gpu_count / capacity.gpu_count)

    if not shares:
        # CPU demand is the most useful fallback when a deployment has not
        # yet configured explicit node resource limits.
        return max(float(request.cpu_cores), 1.0)

    return max(shares)


def resource_busy_fraction(
    used: ResourceVector,
    capacity: NodeResourceCapacity,
) -> float:
    """Return dominant reserved-resource utilization for a node.

    This is based on task declarations/reservations rather than instantaneous
    CPU usage, so admission remains deterministic and stable. Resources with
    no configured limit are ignored.
    """
    shares: list[float] = []
    if capacity.cpu_cores is not None:
        shares.append(used.cpu_cores / capacity.cpu_cores)
    if capacity.memory_mb is not None:
        shares.append(used.memory_mb / capacity.memory_mb)
    if capacity.gpu_count is not None and capacity.gpu_count > 0:
        shares.append(used.gpu_count / capacity.gpu_count)
    return max(shares, default=0.0)
