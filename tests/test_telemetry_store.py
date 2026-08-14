from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from magellan.telemetry.models import TaskTelemetryRecord, TelemetryFreshness
from magellan.telemetry.store import TelemetryStore


def test_task_telemetry_persists_and_stale_power_falls_back(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    store = TelemetryStore(tmp_path, ema_alpha=0.5)
    store.update_task(
        TaskTelemetryRecord(
            task_id="task-1",
            node_id="boston",
            pid=123,
            measured_power_kw=0.08,
            power_source="procfs_cpu_utilization_model",
            power_confidence=0.75,
            sample_count=2,
            last_sample_at_utc=now,
        )
    )

    restarted = TelemetryStore(tmp_path, ema_alpha=0.5)
    fresh = restarted.task_view("task-1", 0.2, 10, now + timedelta(seconds=1))
    assert fresh.freshness == TelemetryFreshness.FRESH
    assert fresh.effective_power_kw == pytest.approx(0.08)
    assert fresh.effective_power_source == "procfs_cpu_utilization_model"

    stale = restarted.task_view("task-1", 0.2, 10, now + timedelta(seconds=20))
    assert stale.freshness == TelemetryFreshness.STALE
    assert stale.effective_power_kw == pytest.approx(0.2)
    assert stale.effective_power_source == "configured_fallback"


def test_edge_telemetry_uses_ema_and_survives_restart(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    store = TelemetryStore(tmp_path, ema_alpha=0.5)
    store.record_latency("boston", "virginia", 40, now)
    store.record_latency("boston", "virginia", 20, now + timedelta(seconds=1))
    store.record_transfer("boston", "virginia", 10_000_000, 1.0, now)
    store.record_transfer("boston", "virginia", 10_000_000, 2.0, now)

    restarted = TelemetryStore(tmp_path, ema_alpha=0.5)
    view = restarted.edge_view(
        "boston",
        "virginia",
        configured_bandwidth_mbps=100,
        configured_latency_ms=50,
        stale_after_seconds=30,
        now=now + timedelta(seconds=2),
    )
    assert view.effective_latency_ms == pytest.approx(30)
    # 80 Mbps then 40 Mbps with alpha .5.
    assert view.effective_bandwidth_mbps == pytest.approx(60)
    assert view.latency_source == "measured_http_rtt"
    assert view.bandwidth_source == "measured_migration_transport_ema"


def test_migration_calibration_persists_and_has_freshness(tmp_path) -> None:
    now = datetime.now(timezone.utc)
    store = TelemetryStore(tmp_path, ema_alpha=0.5)
    store.record_migration_calibration(
        "boston",
        "virginia",
        checkpoint_seconds=4,
        transfer_seconds=8,
        restore_seconds=2,
        activation_seconds=3,
        total_downtime_seconds=15,
        transfer_bytes=1_000,
        sampled_at_utc=now,
    )
    store.record_migration_calibration(
        "boston",
        "virginia",
        checkpoint_seconds=2,
        transfer_seconds=4,
        restore_seconds=1,
        activation_seconds=2,
        total_downtime_seconds=9,
        transfer_bytes=2_000,
        sampled_at_utc=now + timedelta(seconds=1),
    )

    restarted = TelemetryStore(tmp_path, ema_alpha=0.5)
    view = restarted.calibration_view(
        "boston", "virginia", 30, now + timedelta(seconds=2)
    )
    assert view.freshness == TelemetryFreshness.FRESH
    assert view.checkpoint_seconds_ema == pytest.approx(3)
    assert view.restore_seconds_ema == pytest.approx(1.5)
    assert view.total_downtime_seconds_ema == pytest.approx(12)
    assert view.last_transfer_bytes == 2_000
