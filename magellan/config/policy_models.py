from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ObjectiveWeights(BaseModel):
    time: float = Field(ge=0)
    carbon: float = Field(ge=0)
    cost: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_nonzero(self) -> "ObjectiveWeights":
        if self.time + self.carbon + self.cost <= 0:
            raise ValueError("At least one objective weight must be positive")
        return self

    def normalized(self) -> tuple[float, float, float]:
        total = self.time + self.carbon + self.cost
        return (
            self.time / total,
            self.carbon / total,
            self.cost / total,
        )


class PausePolicy(BaseModel):
    pause_seconds: float = Field(ge=0)
    idle_seconds: float = Field(ge=0)
    resume_seconds: float = Field(ge=0)
    max_pause_window_seconds: float = Field(gt=0)


class MigrationPolicy(BaseModel):
    min_migration_gap_seconds: float = Field(ge=0)
    required_improvement_fraction: float = Field(ge=0, le=1)

    # Explicit energy units.
    network_energy_kwh_per_gb_base: float = Field(default=0.0, ge=0)
    network_energy_kwh_per_gb_km: float = Field(default=0.0, ge=0)
    activation_timeout_seconds: float = Field(
        default=600.0,
        gt=0,
    )

class RecoveryPolicy(BaseModel):
    enabled: bool = True
    max_restart_attempts: int = Field(default=3, ge=0)
    initial_backoff_seconds: float = Field(default=5.0, ge=0)
    max_backoff_seconds: float = Field(default=60.0, ge=0)
    scan_interval_seconds: float = Field(default=1.0, gt=0)


class ClockPolicy(BaseModel):
    mode: Literal["wall", "trace"]
    trace_start_utc: str | None = None
    trace_seconds_per_real_second: float = Field(default=1.0, gt=0)


class ScoringPolicy(BaseModel):
    horizon_seconds: int = Field(gt=0)
    weights: ObjectiveWeights
    pause: PausePolicy
    migration: MigrationPolicy
    recovery: RecoveryPolicy = Field(default_factory=RecoveryPolicy)
    clock: ClockPolicy
