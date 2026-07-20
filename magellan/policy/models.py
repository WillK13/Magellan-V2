from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WeightVector(BaseModel):
    time: float = Field(ge=0)
    carbon: float = Field(ge=0)
    cost: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_nonzero(self) -> "WeightVector":
        if self.time + self.carbon + self.cost <= 0:
            raise ValueError("At least one policy weight must be positive")
        return self

    def normalized(self) -> "WeightVector":
        total = self.time + self.carbon + self.cost
        return WeightVector(
            time=self.time / total,
            carbon=self.carbon / total,
            cost=self.cost / total,
        )


class WeightMultipliers(BaseModel):
    time: float = Field(default=1.0, gt=0)
    carbon: float = Field(default=1.0, gt=0)
    cost: float = Field(default=1.0, gt=0)


class AdaptationSignals(BaseModel):
    budget_slack_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    deadline_slack_ratio: float | None = Field(default=None, ge=0)
    carbon_opportunity_fraction: float = Field(default=0.0, ge=0, le=1)
    telemetry_confidence: float = Field(default=0.0, ge=0, le=1)
    cost_cap_exhausted: bool = False
    deadline_at_risk: bool = False


class MetricRangeSample(BaseModel):
    minimum: float = Field(ge=0)
    maximum: float = Field(ge=0)
    observed_at_utc: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_range(self) -> "MetricRangeSample":
        if self.maximum < self.minimum:
            raise ValueError("Metric range maximum must be >= minimum")
        return self


class RollingMetricState(BaseModel):
    samples: list[MetricRangeSample] = Field(default_factory=list)

    def bounds(self) -> tuple[float, float] | None:
        if not self.samples:
            return None
        return (
            min(item.minimum for item in self.samples),
            max(item.maximum for item in self.samples),
        )


class RollingNormalizationState(BaseModel):
    time: RollingMetricState = Field(default_factory=RollingMetricState)
    carbon: RollingMetricState = Field(default_factory=RollingMetricState)
    cost: RollingMetricState = Field(default_factory=RollingMetricState)


class NormalizationBounds(BaseModel):
    time_min: float = Field(ge=0)
    time_max: float = Field(ge=0)
    carbon_min: float = Field(ge=0)
    carbon_max: float = Field(ge=0)
    cost_min: float = Field(ge=0)
    cost_max: float = Field(ge=0)
    source: str = "rolling_window"


class PolicyDecisionRecord(BaseModel):
    task_id: str = Field(min_length=1)
    decision_index: int = Field(ge=1)
    evaluated_at_utc: datetime
    selected_action: str
    selected_destination_node_id: str | None = None
    selected_score: float = Field(ge=0)
    baseline_weights: WeightVector
    effective_weights: WeightVector
    multipliers: WeightMultipliers
    signals: AdaptationSignals
    normalization_bounds: NormalizationBounds
    hard_constraints: dict[str, bool | float | str | None] = Field(
        default_factory=dict
    )
    reason: str


class AdaptiveTaskPolicyState(BaseModel):
    task_id: str = Field(min_length=1)
    baseline_weights: WeightVector
    effective_weights: WeightVector
    multipliers: WeightMultipliers = Field(default_factory=WeightMultipliers)
    signals: AdaptationSignals = Field(default_factory=AdaptationSignals)
    normalization: RollingNormalizationState = Field(
        default_factory=RollingNormalizationState
    )
    decision_count: int = Field(default=0, ge=0)
    last_decision: PolicyDecisionRecord | None = None
    decision_history: list[PolicyDecisionRecord] = Field(default_factory=list)
    created_at_utc: datetime = Field(default_factory=utc_now)
    updated_at_utc: datetime = Field(default_factory=utc_now)


class AdaptiveDecisionContext(BaseModel):
    task_id: str
    baseline_weights: WeightVector
    effective_weights: WeightVector
    multipliers: WeightMultipliers
    signals: AdaptationSignals
    normalization_bounds: NormalizationBounds
    hard_constraints: dict[str, bool | float | str | None] = Field(
        default_factory=dict
    )
