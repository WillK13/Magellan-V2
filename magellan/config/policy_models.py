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
    candidate_idle_seconds: list[float] = Field(default_factory=list)
    resume_seconds: float = Field(ge=0)
    max_pause_window_seconds: float = Field(gt=0)
    min_pause_gap_seconds: float = Field(default=0.0, ge=0)
    scan_interval_seconds: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_pause_candidates(self) -> "PausePolicy":
        candidates = self.idle_candidates()
        if any(value < 0 for value in candidates):
            raise ValueError("Pause candidate durations must be non-negative")
        if any(value > self.max_pause_window_seconds for value in candidates):
            raise ValueError(
                "Pause candidate duration exceeds max_pause_window_seconds"
            )
        return self

    def idle_candidates(self) -> list[float]:
        raw = self.candidate_idle_seconds or [self.idle_seconds]
        if self.idle_seconds not in raw:
            raw = [self.idle_seconds, *raw]
        return sorted(set(float(value) for value in raw))


class CarbonForecastPolicy(BaseModel):
    enabled: bool = True
    provider: Literal["linear_trend", "persistence"] = "linear_trend"
    history_points: int = Field(default=8, ge=1, le=1000)
    minimum_points: int = Field(default=4, ge=1, le=1000)
    sample_interval_seconds: float = Field(default=900.0, gt=0)
    horizon_seconds: float = Field(default=3600.0, gt=0)
    forecast_sample_seconds: float = Field(default=300.0, gt=0)
    maximum_change_per_hour: float = Field(default=100.0, ge=0)
    stale_after_seconds: float = Field(default=1800.0, gt=0)
    configured_fallback_g_per_kwh: float | None = Field(default=None, ge=0)
    confidence_floor: float = Field(default=0.1, ge=0, le=1)
    persistence_confidence: float = Field(default=0.5, ge=0, le=1)
    fallback_confidence: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_forecast_window(self) -> "CarbonForecastPolicy":
        if self.minimum_points > self.history_points:
            raise ValueError("minimum_points cannot exceed history_points")
        return self


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


class AccountingPolicy(BaseModel):
    scan_interval_seconds: float = Field(default=1.0, gt=0)
    progress_ema_alpha: float = Field(default=0.5, gt=0, le=1)


class TelemetryPolicy(BaseModel):
    enabled: bool = True
    task_scan_interval_seconds: float = Field(default=1.0, gt=0)
    edge_probe_interval_seconds: float = Field(default=15.0, gt=0)
    edge_bandwidth_probe_interval_seconds: float = Field(default=60.0, gt=0)
    edge_bandwidth_probe_bytes: int = Field(default=1_048_576, ge=65_536)
    refresh_edges_before_decision: bool = True
    task_stale_after_seconds: float = Field(default=10.0, gt=0)
    edge_stale_after_seconds: float = Field(default=120.0, gt=0)
    calibration_stale_after_seconds: float = Field(default=3600.0, gt=0)
    ema_alpha: float = Field(default=0.35, gt=0, le=1)
    power_idle_fraction: float = Field(default=0.2, ge=0, le=1)
    cpu_power_confidence: float = Field(default=0.75, ge=0, le=1)
    fallback_power_confidence: float = Field(default=0.25, ge=0, le=1)


class ReconciliationPolicy(BaseModel):
    enabled: bool = True
    scan_interval_seconds: float = Field(default=5.0, gt=0)
    activation_resolution_timeout_seconds: float = Field(
        default=30.0, gt=0
    )
    activation_resolution_poll_seconds: float = Field(
        default=1.0, gt=0
    )


class AuctionPolicy(BaseModel):
    """Destination-local ranking policy for task bids."""

    strategy: Literal[
        "lowest_score",
        "shortest_remaining",
        "longest_remaining",
        "credit_fair",
        "highest_regret",
        "priority_deadline",
        "resource_efficiency",
    ] = "lowest_score"

    credit_increment: float = Field(default=1.0, gt=0)
    credit_max: float = Field(default=100.0, gt=0)
    accepted_credit_decay: float = Field(default=0.0, ge=0, le=1)
    deadline_urgency_window_seconds: float = Field(
        default=3600.0,
        gt=0,
    )
    resource_efficiency_floor: float = Field(
        default=0.01,
        gt=0,
    )


class AdaptivePolicy(BaseModel):
    """Bounded runtime adaptation of time, carbon, and cost weights."""

    enabled: bool = True
    multiplier_bound_fraction: float = Field(default=0.25, ge=0, le=0.5)
    rolling_window_epochs: int = Field(default=24, ge=1, le=1000)
    decision_history_limit: int = Field(default=50, ge=1, le=1000)
    confidence_floor: float = Field(default=0.25, ge=0, le=1)


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
    accounting: AccountingPolicy = Field(default_factory=AccountingPolicy)
    telemetry: TelemetryPolicy = Field(default_factory=TelemetryPolicy)
    reconciliation: ReconciliationPolicy = Field(
        default_factory=ReconciliationPolicy
    )
    auction: AuctionPolicy = Field(default_factory=AuctionPolicy)
    adaptive: AdaptivePolicy = Field(default_factory=AdaptivePolicy)
    carbon_forecast: CarbonForecastPolicy = Field(
        default_factory=CarbonForecastPolicy
    )
    clock: ClockPolicy
