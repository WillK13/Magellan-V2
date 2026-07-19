import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from magellan.config.models import ClusterConfig, NodeConfig
from magellan.config.policy_models import (
    AccountingPolicy,
    ClockPolicy,
    MigrationPolicy,
    ObjectiveWeights,
    PausePolicy,
    ScoringPolicy,
)
from magellan.graph.topology import ClusterGraph
from magellan.models.types import TaskProfile
from magellan.runtime.accounting import RuntimeAccountingService
from magellan.runtime.clock import MagellanClock
from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.state.task_models import LocalProcessSpec, TaskDefinition


class ConstantCarbonStore:
    def average(self, _node_id, _start, _duration):
        return 100.0

    def value_at(self, _node_id, _at):
        return 100.0


def write_progress(path, completed, total, updated_at):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "task_id": "account-task",
                "completed_units": completed,
                "total_units": total,
                "updated_at_utc": updated_at.isoformat(),
            }
        )
    )


def policy() -> ScoringPolicy:
    return ScoringPolicy(
        horizon_seconds=3600,
        weights=ObjectiveWeights(time=0.25, carbon=0.5, cost=0.25),
        pause=PausePolicy(
            pause_seconds=0,
            idle_seconds=60,
            resume_seconds=0,
            max_pause_window_seconds=7200,
        ),
        migration=MigrationPolicy(
            min_migration_gap_seconds=0,
            required_improvement_fraction=0,
        ),
        recovery={},
        accounting=AccountingPolicy(
            scan_interval_seconds=1,
            progress_ema_alpha=0.5,
        ),
        clock=ClockPolicy(mode="wall"),
    )


def test_runtime_accounting_updates_cost_carbon_and_remaining_time(
    tmp_path,
) -> None:
    node = NodeConfig(
        id="boston",
        name="Boston",
        vm_name="boston",
        zone="us-east1-c",
        internal_ip="10.0.0.1",
        carbon_region="Boston",
        dataset_file="unused.csv",
        latitude=42,
        longitude=-71,
        pue=1.2,
        compute_price_usd_per_hour=1.0,
    )
    cluster = ClusterConfig(nodes=[node])
    definition = TaskDefinition(
        profile=TaskProfile(
            task_id="account-task",
            workload_type="test",
            current_node_id="boston",
            power_kw=0.5,
            checkpoint_bytes=0,
            estimated_remaining_seconds=1000,
        ),
        runtime=LocalProcessSpec(
            module="example.module",
            progress_relative_path="runtime/progress.json",
        ),
    )
    registry = PersistentTaskRegistry(
        definitions=[definition],
        state_root=tmp_path,
        local_node_id="boston",
    )
    registry.mark_running("account-task", pid=os.getpid())

    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    state = registry.get_state("account-task")
    state.last_accounted_at_utc = t0
    registry.set_state(state)

    progress = registry.progress_file("account-task")
    assert progress is not None
    write_progress(progress, 10, 100, t0)

    service = RuntimeAccountingService(
        local_node=node,
        cluster=cluster,
        policy=policy(),
        graph=ClusterGraph(cluster),
        carbon_store=ConstantCarbonStore(),
        clock=MagellanClock(ClockPolicy(mode="wall")),
        registry=registry,
    )
    trace = pd.Timestamp("2024-01-01T01:00:00Z")
    service.settle_task("account-task", t0 + timedelta(hours=1), trace)

    write_progress(
        progress,
        20,
        100,
        t0 + timedelta(seconds=10),
    )
    service.settle_task(
        "account-task",
        t0 + timedelta(hours=1, seconds=10),
        trace + pd.Timedelta(seconds=10),
    )

    state = registry.get_state("account-task")
    assert state.accumulated_runtime_seconds == pytest.approx(3610, rel=1e-4)
    assert state.accumulated_compute_cost_usd == pytest.approx(
        3610 / 3600,
        rel=1e-4,
    )
    assert state.accumulated_compute_carbon_grams == pytest.approx(
        0.5 * 1.2 * 100 * (3610 / 3600),
        rel=1e-4,
    )
    assert state.progress_rate_units_per_second == pytest.approx(1.0)
    assert state.progress_fraction == pytest.approx(0.2)
    assert state.estimated_remaining_seconds == pytest.approx(80.0)

    scoring = registry.scoring_profile("account-task")
    assert scoring.accumulated_cost_usd == pytest.approx(
        state.accumulated_cost_usd
    )
    assert scoring.estimated_remaining_seconds == pytest.approx(80.0)
