from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from magellan.telemetry.models import (
    EdgeTelemetryRecord,
    EdgeTelemetryView,
    MigrationCalibrationRecord,
    MigrationCalibrationView,
    TaskTelemetryRecord,
    TaskTelemetryView,
    TelemetryDocument,
    TelemetryFreshness,
)


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _age_seconds(value: datetime | None, now: datetime) -> float | None:
    if value is None:
        return None
    return max(0.0, (now - _utc(value)).total_seconds())


def _freshness(
    value: datetime | None,
    stale_after_seconds: float,
    now: datetime,
) -> TelemetryFreshness:
    age = _age_seconds(value, now)
    if age is None:
        return TelemetryFreshness.UNAVAILABLE
    if age > stale_after_seconds:
        return TelemetryFreshness.STALE
    return TelemetryFreshness.FRESH


def _ema(previous: float | None, observed: float, alpha: float) -> float:
    if previous is None:
        return observed
    return alpha * observed + (1.0 - alpha) * previous


class TelemetryStore:
    def __init__(self, state_root: str | Path, ema_alpha: float = 0.35) -> None:
        self._path = Path(state_root) / "control" / "telemetry.json"
        self._ema_alpha = ema_alpha
        self._lock = RLock()
        self._document = self._load()

    @staticmethod
    def edge_key(source_node_id: str, destination_node_id: str) -> str:
        return f"{source_node_id}->{destination_node_id}"

    def _load(self) -> TelemetryDocument:
        if not self._path.exists():
            return TelemetryDocument()
        try:
            return TelemetryDocument.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return TelemetryDocument()

    def _persist(self) -> None:
        self._document.updated_at_utc = datetime.now(timezone.utc)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._document.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self._path)

    @property
    def path(self) -> Path:
        return self._path

    def update_task(self, record: TaskTelemetryRecord) -> TaskTelemetryRecord:
        with self._lock:
            self._document.task_records[record.task_id] = record.model_copy(deep=True)
            self._persist()
            return record.model_copy(deep=True)

    def task_record(self, task_id: str) -> TaskTelemetryRecord | None:
        with self._lock:
            record = self._document.task_records.get(task_id)
            return None if record is None else record.model_copy(deep=True)

    def task_view(
        self,
        task_id: str,
        configured_power_kw: float,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> TaskTelemetryView:
        now = _utc(now)
        record = self.task_record(task_id) or TaskTelemetryRecord(
            task_id=task_id,
            node_id="unknown",
        )
        freshness = _freshness(
            record.last_sample_at_utc,
            stale_after_seconds,
            now,
        )
        use_measured = (
            freshness == TelemetryFreshness.FRESH
            and record.measured_power_kw is not None
            and record.measured_power_kw > 0
        )
        return TaskTelemetryView(
            **record.model_dump(),
            freshness=freshness,
            age_seconds=_age_seconds(record.last_sample_at_utc, now),
            effective_power_kw=(
                record.measured_power_kw if use_measured else configured_power_kw
            ),
            effective_power_source=(
                record.power_source if use_measured else "configured_fallback"
            ),
        )

    def list_task_records(self) -> list[TaskTelemetryRecord]:
        with self._lock:
            return [
                self._document.task_records[key].model_copy(deep=True)
                for key in sorted(self._document.task_records)
            ]

    def record_latency(
        self,
        source_node_id: str,
        destination_node_id: str,
        latency_ms: float,
        sampled_at_utc: datetime | None = None,
    ) -> EdgeTelemetryRecord:
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.edge_records.get(key) or EdgeTelemetryRecord(
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
            )
            record.latency_ms_ema = _ema(
                record.latency_ms_ema,
                max(0.0, latency_ms),
                self._ema_alpha,
            )
            record.latency_sample_count += 1
            record.last_latency_sample_at_utc = _utc(sampled_at_utc)
            record.last_success_at_utc = record.last_latency_sample_at_utc
            record.consecutive_failures = 0
            record.last_error = None
            self._document.edge_records[key] = record
            self._persist()
            return record.model_copy(deep=True)

    def record_transfer(
        self,
        source_node_id: str,
        destination_node_id: str,
        transfer_bytes: int,
        duration_seconds: float,
        sampled_at_utc: datetime | None = None,
        *,
        sample_source: str = "migration_transfer",
    ) -> EdgeTelemetryRecord:
        if transfer_bytes <= 0 or duration_seconds <= 0:
            raise ValueError("A bandwidth sample needs positive bytes and duration")
        bandwidth_mbps = transfer_bytes * 8.0 / duration_seconds / 1_000_000.0
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.edge_records.get(key) or EdgeTelemetryRecord(
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
            )
            record.bandwidth_mbps_ema = _ema(
                record.bandwidth_mbps_ema,
                bandwidth_mbps,
                self._ema_alpha,
            )
            record.bandwidth_sample_count += 1
            record.last_bandwidth_sample_source = sample_source
            record.last_bandwidth_sample_at_utc = _utc(sampled_at_utc)
            record.last_success_at_utc = record.last_bandwidth_sample_at_utc
            record.consecutive_failures = 0
            record.last_error = None
            self._document.edge_records[key] = record
            self._persist()
            return record.model_copy(deep=True)

    def record_edge_failure(
        self,
        source_node_id: str,
        destination_node_id: str,
        error: str,
        sampled_at_utc: datetime | None = None,
    ) -> EdgeTelemetryRecord:
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.edge_records.get(key) or EdgeTelemetryRecord(
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
            )
            record.last_failure_at_utc = _utc(sampled_at_utc)
            record.consecutive_failures += 1
            record.last_error = error
            self._document.edge_records[key] = record
            self._persist()
            return record.model_copy(deep=True)

    def edge_record(
        self, source_node_id: str, destination_node_id: str
    ) -> EdgeTelemetryRecord | None:
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.edge_records.get(key)
            return None if record is None else record.model_copy(deep=True)

    def edge_view(
        self,
        source_node_id: str,
        destination_node_id: str,
        configured_bandwidth_mbps: float,
        configured_latency_ms: float,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> EdgeTelemetryView:
        now = _utc(now)
        record = self.edge_record(source_node_id, destination_node_id) or EdgeTelemetryRecord(
            source_node_id=source_node_id,
            destination_node_id=destination_node_id,
        )
        latency_freshness = _freshness(
            record.last_latency_sample_at_utc, stale_after_seconds, now
        )
        bandwidth_freshness = _freshness(
            record.last_bandwidth_sample_at_utc, stale_after_seconds, now
        )
        use_latency = (
            latency_freshness == TelemetryFreshness.FRESH
            and record.latency_ms_ema is not None
        )
        use_bandwidth = (
            bandwidth_freshness == TelemetryFreshness.FRESH
            and record.bandwidth_mbps_ema is not None
        )
        return EdgeTelemetryView(
            **record.model_dump(),
            configured_latency_ms=configured_latency_ms,
            configured_bandwidth_mbps=configured_bandwidth_mbps,
            effective_latency_ms=(
                record.latency_ms_ema if use_latency else configured_latency_ms
            ),
            effective_bandwidth_mbps=(
                record.bandwidth_mbps_ema
                if use_bandwidth
                else configured_bandwidth_mbps
            ),
            latency_source=("measured_http_rtt" if use_latency else "configured_fallback"),
            bandwidth_source=(
                "measured_migration_transport_ema"
                if use_bandwidth
                else "configured_fallback"
            ),
            latency_freshness=latency_freshness,
            bandwidth_freshness=bandwidth_freshness,
            latency_age_seconds=_age_seconds(record.last_latency_sample_at_utc, now),
            bandwidth_age_seconds=_age_seconds(
                record.last_bandwidth_sample_at_utc, now
            ),
        )

    def list_edge_records(self) -> list[EdgeTelemetryRecord]:
        with self._lock:
            return [
                self._document.edge_records[key].model_copy(deep=True)
                for key in sorted(self._document.edge_records)
            ]

    def record_migration_calibration(
        self,
        source_node_id: str,
        destination_node_id: str,
        checkpoint_seconds: float,
        transfer_seconds: float,
        restore_seconds: float,
        activation_seconds: float,
        total_downtime_seconds: float,
        transfer_bytes: int,
        sampled_at_utc: datetime | None = None,
    ) -> MigrationCalibrationRecord:
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.migration_calibrations.get(key) or (
                MigrationCalibrationRecord(
                    source_node_id=source_node_id,
                    destination_node_id=destination_node_id,
                )
            )
            values = {
                "checkpoint_seconds_ema": checkpoint_seconds,
                "transfer_seconds_ema": transfer_seconds,
                "restore_seconds_ema": restore_seconds,
                "activation_seconds_ema": activation_seconds,
                "total_downtime_seconds_ema": total_downtime_seconds,
            }
            for field, observed in values.items():
                setattr(
                    record,
                    field,
                    _ema(getattr(record, field), max(0.0, observed), self._ema_alpha),
                )
            record.last_transfer_bytes = max(0, transfer_bytes)
            record.sample_count += 1
            record.last_sample_at_utc = _utc(sampled_at_utc)
            self._document.migration_calibrations[key] = record
            self._persist()
            return record.model_copy(deep=True)

    def calibration_record(
        self, source_node_id: str, destination_node_id: str
    ) -> MigrationCalibrationRecord | None:
        key = self.edge_key(source_node_id, destination_node_id)
        with self._lock:
            record = self._document.migration_calibrations.get(key)
            return None if record is None else record.model_copy(deep=True)

    def calibration_view(
        self,
        source_node_id: str,
        destination_node_id: str,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> MigrationCalibrationView:
        now = _utc(now)
        record = self.calibration_record(source_node_id, destination_node_id) or (
            MigrationCalibrationRecord(
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
            )
        )
        return MigrationCalibrationView(
            **record.model_dump(),
            freshness=_freshness(record.last_sample_at_utc, stale_after_seconds, now),
            age_seconds=_age_seconds(record.last_sample_at_utc, now),
        )

    def list_calibrations(self) -> list[MigrationCalibrationRecord]:
        with self._lock:
            return [
                self._document.migration_calibrations[key].model_copy(deep=True)
                for key in sorted(self._document.migration_calibrations)
            ]
