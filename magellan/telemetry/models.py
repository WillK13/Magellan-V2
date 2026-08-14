from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TelemetryFreshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class ProcessMeasurement(BaseModel):
    pid: int = Field(ge=1)
    process_count: int = Field(default=1, ge=1)
    process_state: str | None = None
    cpu_time_seconds: float = Field(ge=0)
    memory_rss_mb: float = Field(ge=0)
    sampled_at_utc: datetime = Field(default_factory=utc_now)


class TaskTelemetryRecord(BaseModel):
    task_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    pid: int | None = Field(default=None, ge=1)
    process_count: int = Field(default=0, ge=0)
    process_state: str | None = None

    cpu_utilization_percent: float | None = Field(default=None, ge=0)
    memory_rss_mb: float | None = Field(default=None, ge=0)
    checkpoint_bytes: int | None = Field(default=None, ge=0)

    measured_power_kw: float | None = Field(default=None, ge=0)
    power_source: str = "configured_fallback"
    power_confidence: float = Field(default=0.0, ge=0, le=1)

    progress_rate_units_per_second: float | None = Field(default=None, gt=0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)

    sample_count: int = Field(default=0, ge=0)
    last_sample_at_utc: datetime | None = None
    last_error: str | None = None


class TaskTelemetryView(TaskTelemetryRecord):
    freshness: TelemetryFreshness
    age_seconds: float | None = Field(default=None, ge=0)
    effective_power_kw: float = Field(gt=0)
    effective_power_source: str


class EdgeTelemetrySampleRequest(BaseModel):
    """Operator-supplied real edge measurement for experiment preflight."""

    latency_ms: float | None = Field(default=None, ge=0)
    transfer_bytes: int | None = Field(default=None, gt=0)
    transfer_duration_seconds: float | None = Field(default=None, gt=0)


class EdgeTelemetryRecord(BaseModel):
    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    latency_ms_ema: float | None = Field(default=None, ge=0)
    bandwidth_mbps_ema: float | None = Field(default=None, gt=0)

    latency_sample_count: int = Field(default=0, ge=0)
    bandwidth_sample_count: int = Field(default=0, ge=0)

    last_latency_sample_at_utc: datetime | None = None
    last_bandwidth_sample_at_utc: datetime | None = None
    last_success_at_utc: datetime | None = None
    last_failure_at_utc: datetime | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    last_error: str | None = None


class EdgeTelemetryView(EdgeTelemetryRecord):
    configured_latency_ms: float = Field(ge=0)
    configured_bandwidth_mbps: float = Field(gt=0)
    effective_latency_ms: float = Field(ge=0)
    effective_bandwidth_mbps: float = Field(gt=0)
    latency_source: str
    bandwidth_source: str
    latency_freshness: TelemetryFreshness
    bandwidth_freshness: TelemetryFreshness
    latency_age_seconds: float | None = Field(default=None, ge=0)
    bandwidth_age_seconds: float | None = Field(default=None, ge=0)


class MigrationCalibrationRecord(BaseModel):
    source_node_id: str = Field(min_length=1)
    destination_node_id: str = Field(min_length=1)

    checkpoint_seconds_ema: float | None = Field(default=None, ge=0)
    transfer_seconds_ema: float | None = Field(default=None, ge=0)
    restore_seconds_ema: float | None = Field(default=None, ge=0)
    activation_seconds_ema: float | None = Field(default=None, ge=0)
    total_downtime_seconds_ema: float | None = Field(default=None, ge=0)

    last_transfer_bytes: int | None = Field(default=None, ge=0)
    sample_count: int = Field(default=0, ge=0)
    last_sample_at_utc: datetime | None = None


class MigrationCalibrationView(MigrationCalibrationRecord):
    freshness: TelemetryFreshness
    age_seconds: float | None = Field(default=None, ge=0)


class TelemetryDocument(BaseModel):
    format_version: int = 1
    task_records: dict[str, TaskTelemetryRecord] = Field(default_factory=dict)
    edge_records: dict[str, EdgeTelemetryRecord] = Field(default_factory=dict)
    migration_calibrations: dict[str, MigrationCalibrationRecord] = Field(
        default_factory=dict
    )
    updated_at_utc: datetime = Field(default_factory=utc_now)
