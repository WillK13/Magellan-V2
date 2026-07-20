from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import (
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
    TelemetryPolicy,
)
from magellan.graph.topology import ClusterGraph
from magellan.models.types import TaskProfile
from magellan.runtime.accounting import RuntimeAccountingService
from magellan.runtime.clock import MagellanClock
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition
from magellan.telemetry.models import TaskTelemetryRecord
from magellan.telemetry.store import TelemetryStore


class ConstantCarbon:
    def average(self, *_args):
        return 100.0

    def value_at(self, *_args):
        return 100.0


def test_accounting_uses_fresh_telemetry_power(tmp_path) -> None:
    node = NodeConfig(
        id="boston",
        name="Boston",
        vm_name="boston",
        zone="a",
        internal_ip="10.0.0.1",
        carbon_region="Boston",
        dataset_file="unused",
        latitude=42,
        longitude=-71,
        pue=1.0,
        compute_price_usd_per_hour=0,
    )
    cluster = ClusterConfig(nodes=[node])
    policy = ScoringPolicy(
        horizon_seconds=3600,
        weights=ObjectiveWeights(time=1, carbon=1, cost=1),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=0,
            resume_seconds=0,
            max_pause_window_seconds=60,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        telemetry=TelemetryPolicy(task_stale_after_seconds=30),
        clock=ClockPolicy(mode="wall"),
    )
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="power-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.5,
            checkpoint_bytes=0,
        ),
        runtime=LocalProcessSpec(module="example.module"),
    )
    registry = PersistentTaskRegistry([definition], tmp_path, "boston")
    registry.mark_running("power-task", os.getpid())
    now = datetime.now(timezone.utc)
    state = registry.get_state("power-task")
    state.last_accounted_at_utc = now - timedelta(hours=1)
    registry.set_state(state)

    telemetry = TelemetryStore(tmp_path)
    telemetry.update_task(
        TaskTelemetryRecord(
            task_id="power-task",
            node_id="boston",
            measured_power_kw=0.1,
            power_source="procfs_cpu_utilization_model",
            power_confidence=0.75,
            sample_count=2,
            last_sample_at_utc=now,
        )
    )
    service = RuntimeAccountingService(
        local_node=node,
        cluster=cluster,
        policy=policy,
        graph=ClusterGraph(cluster, telemetry, policy.telemetry),
        carbon_store=ConstantCarbon(),
        clock=MagellanClock(policy.clock),
        registry=registry,
        telemetry_store=telemetry,
    )
    service.settle_task(
        "power-task",
        now,
        pd.Timestamp("2024-01-01T01:00:00Z"),
    )
    state = registry.get_state("power-task")
    assert state.accumulated_compute_carbon_grams == pytest.approx(10.0)
