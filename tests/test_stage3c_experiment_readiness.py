from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from magellan.bidding.arbiter import BidArbiter
from magellan.bidding.models import BidRequest, BidStatus, TaskBidContext
from magellan.bidding.store import BidStore
from magellan.config.models import ClusterConfig, NodeResourceCapacity
from magellan.experiments.workload_population import (
    generate_population,
    parse_weighted_mix,
)
from magellan.models.types import ActionType, ScoredAction, TaskResourceRequest
from magellan.state.task_registry import TaskRegistry


def _cluster() -> ClusterConfig:
    return ClusterConfig.model_validate(
        {
            "nodes": [
                {
                    "id": "boston",
                    "name": "Boston",
                    "vm_name": "boston-vm",
                    "zone": "us-east1-c",
                    "internal_ip": "10.0.0.1",
                    "carbon_region": "boston",
                    "dataset_file": "boston.csv",
                    "latitude": 42.36,
                    "longitude": -71.06,
                    "capacity": None,
                    "resources": {
                        "cpu_cores": 2,
                        "memory_mb": 16384,
                        "gpu_count": 0,
                    },
                },
                {
                    "id": "virginia",
                    "name": "Virginia",
                    "vm_name": "virginia-vm",
                    "zone": "northamerica-northeast1-c",
                    "internal_ip": "10.0.0.2",
                    "carbon_region": "virginia",
                    "dataset_file": "virginia.csv",
                    "latitude": 38.0,
                    "longitude": -78.0,
                    "capacity": None,
                    "resources": {
                        "cpu_cores": 2,
                        "memory_mb": 16384,
                        "gpu_count": 0,
                    },
                },
            ]
        }
    )


def _bid(bid_id: str, cpu: float) -> BidRequest:
    return BidRequest(
        bid_id=bid_id,
        epoch_id="stage3c",
        task_id=f"task-{bid_id}",
        source_node_id="boston",
        destination_node_id="virginia",
        task_context=TaskBidContext(
            workload_type="benchmark",
            estimated_remaining_seconds=100,
            resource_request=TaskResourceRequest(
                cpu_cores=cpu,
                memory_mb=256,
            ),
        ),
        submitted_at_utc=datetime.now(timezone.utc),
        candidate=ScoredAction(
            action=ActionType.MIGRATE,
            source_node_id="boston",
            destination_node_id="virginia",
            time_seconds=1,
            carbon_grams=1,
            cost_usd=0,
            normalized_time=0,
            normalized_carbon=0,
            normalized_cost=0,
            score=0.1,
        ),
    )


@pytest.mark.asyncio
async def test_resource_only_admission_packs_until_cpu_is_full() -> None:
    store = BidStore()
    arbiter = BidArbiter(
        store=store,
        registry=TaskRegistry([]),
        local_node_id="virginia",
        capacity=None,
        bid_window_seconds=1,
        node_resources=NodeResourceCapacity(
            cpu_cores=2,
            memory_mb=16384,
            gpu_count=0,
        ),
    )
    await store.submit(_bid("a", 1))
    await store.submit(_bid("b", 1))
    await store.submit(_bid("c", 1))
    await arbiter.run_once(datetime.now(timezone.utc) + timedelta(seconds=2))

    records = {name: await store.get(name) for name in ("a", "b", "c")}
    accepted = [
        record
        for record in records.values()
        if record and record.status == BidStatus.ACCEPTED
    ]
    rejected = [
        record
        for record in records.values()
        if record and record.status == BidStatus.REJECTED
    ]
    assert len(accepted) == 2
    assert len(rejected) == 1
    assert rejected[0].resource_fit is False
    assert "CPU" in (rejected[0].decision_reason or "")

    status = await arbiter.status()
    assert status["task_slot_capacity"] is None
    assert status["available_task_slots"] is None
    assert status["reserved_cpu_cores"] == 2
    assert status["available_cpu_cores"] == 0
    assert status["resource_busy_fraction"] == pytest.approx(1.0)


def test_population_is_seeded_and_heterogeneous() -> None:
    kwargs = dict(
        cluster=_cluster(),
        count=12,
        seed=42,
        mix=parse_weighted_mix("nbody=1,json=1,matmul=1"),
        population_id="test-population",
        benchmark_iterations=25,
        mean_interarrival_seconds=2.0,
    )
    first = generate_population(**kwargs)
    second = generate_population(**kwargs)
    assert first == second
    assert len(first) == 12
    assert {task.workload for task in first}.issubset({"nbody", "json", "matmul"})
    assert len({task.workload for task in first}) >= 2
    assert all(
        task.definition["profile"]["resource_request"]["cpu_cores"] <= 1.5
        for task in first
    )
    assert all(
        task.run["labels"]["population_seed"] == "42"
        for task in first
    )
    offsets = [task.scheduled_offset_seconds for task in first]
    assert offsets == sorted(offsets)


def test_population_generates_dendro_resolution_and_time_variants() -> None:
    template = json.loads(
        Path("config/submissions/dendro-bssn-template.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = generate_population(
        cluster=_cluster(),
        count=4,
        seed=7,
        mix={"dendro": 1.0},
        population_id="dendro-pop",
        dendro_template=template,
        dendro_solver="/opt/dendro/bssnSolver",
        dendro_parameter_template="/opt/dendro/q1.toml",
        dendro_resolutions=[8, 10],
        dendro_time_ends=[0.5, 2.0],
        dendro_nodes=["boston", "virginia"],
    )
    assert len(tasks) == 4
    for task in tasks:
        args = task.definition["runtime"]["arguments"]
        assert "--resolution" in args
        assert int(args[args.index("--resolution") + 1]) in {8, 10}
        assert "--time-end" in args
        assert float(args[args.index("--time-end") + 1]) in {0.5, 2.0}
        assert task.definition["profile"]["resource_request"]["cpu_cores"] == 2
        assert task.definition["profile"]["resource_request"]["memory_mb"] == 8192


def test_checkpointable_benchmark_resumes_and_completes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    progress = tmp_path / "progress.json"
    completion = tmp_path / "completion.json"
    output = tmp_path / "output"
    checkpoint.write_text(
        json.dumps(
            {
                "completed_iterations": 2,
                "checksum": 1.25,
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["MAGELLAN_TASK_ID"] = "benchmark-test"
    env["MAGELLAN_NODE_ID"] = "boston"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "magellan.workloads.benchmark",
            "--benchmark",
            "json",
            "--size",
            "small",
            "--iterations",
            "4",
            "--seed",
            "123",
            "--checkpoint-file",
            str(checkpoint),
            "--progress-file",
            str(progress),
            "--completion-file",
            str(completion),
            "--output-dir",
            str(output),
        ],
        check=True,
        env=env,
    )
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    progress_payload = json.loads(progress.read_text(encoding="utf-8"))
    completion_payload = json.loads(completion.read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert state["completed_iterations"] == 4
    assert progress_payload["completed_units"] == 4
    assert progress_payload["total_units"] == 4
    assert completion_payload["success"] is True
    assert result["benchmark"] == "json"
    assert result["iterations"] == 4
