from __future__ import annotations

from pydantic import BaseModel, Field


class NodeRuntimeCapabilities(BaseModel):
    """Capabilities advertised by one Magellan location.

    The configured record is the scheduling contract. Local discovery is
    exposed separately so operators can detect drift without making task
    placement depend on an asynchronous probe during every scoring epoch.
    """

    architecture: str | None = None
    operating_system: str | None = None
    cpu_cores: float | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    gpu_count: int | None = Field(default=None, ge=0)
    accelerator_types: set[str] = Field(default_factory=set)
    commands: set[str] = Field(default_factory=set)
    runtimes: dict[str, str] = Field(default_factory=dict)
    features: set[str] = Field(default_factory=set)


class TaskCompatibilityRequirements(BaseModel):
    architectures: set[str] = Field(default_factory=set)
    operating_systems: set[str] = Field(default_factory=set)
    minimum_cpu_cores: float | None = Field(default=None, gt=0)
    minimum_memory_mb: int | None = Field(default=None, gt=0)
    minimum_gpu_count: int | None = Field(default=None, ge=0)
    accelerator_types: set[str] = Field(default_factory=set)
    required_commands: set[str] = Field(default_factory=set)
    required_runtimes: dict[str, str] = Field(default_factory=dict)
    required_features: set[str] = Field(default_factory=set)
    checkpoint_architecture_independent: bool = True
    requires_same_mpi_world_size: bool = False


class CompatibilityResult(BaseModel):
    compatible: bool
    reasons: list[str] = Field(default_factory=list)
    checked_requirements: dict = Field(default_factory=dict)
    advertised_capabilities: dict = Field(default_factory=dict)
