from __future__ import annotations

import pandas as pd

from magellan.config.loader import load_cluster_config
from magellan.experiments.stage4e1 import (
    CLASS_SEQUENCE,
    SCALE_SIZES,
    build_scale_population,
    class_counts,
    node_counts,
)
from magellan.models.types import TaskResourceRequest


def _requests() -> dict[str, TaskResourceRequest]:
    # Frozen Stage 4D.1 values, duplicated only as test fixtures so local unit
    # tests do not depend on the GCP measurement directory being present.
    return {
        "benchmark-json-medium": TaskResourceRequest(
            cpu_cores=0.9972222178769694, memory_mb=13, gpu_count=0
        ),
        "dendro-r9-t1p0": TaskResourceRequest(
            cpu_cores=1.7763255932687552, memory_mb=1379, gpu_count=0
        ),
        "llm-distilgpt2": TaskResourceRequest(
            cpu_cores=0.7630787942682616, memory_mb=1572, gpu_count=0
        ),
    }


def test_scaling_populations_are_exact_and_balanced() -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    requests = _requests()
    node_ids = [node.id for node in cluster.nodes]

    for size in SCALE_SIZES:
        specs = build_scale_population(
            task_count=size,
            node_ids=node_ids,
            requests=requests,
            start_utc=pd.Timestamp("2024-08-20T12:00:00Z"),
            arrival_window_seconds=3 * 3600,
            epoch_seconds=float(cluster.epoch_seconds),
        )
        assert len(specs) == size
        classes = class_counts(specs)
        homes = node_counts(specs)
        assert max(classes.values()) - min(classes.values()) <= 1
        assert max(homes.values()) - min(homes.values()) <= 1


def test_scaling_population_uses_only_frozen_resource_classes() -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    requests = _requests()
    specs = build_scale_population(
        task_count=100,
        node_ids=[node.id for node in cluster.nodes],
        requests=requests,
        start_utc=pd.Timestamp("2024-08-20T12:00:00Z"),
        arrival_window_seconds=3 * 3600,
        epoch_seconds=float(cluster.epoch_seconds),
    )
    assert set(task.class_id for task in specs) == set(CLASS_SEQUENCE)
    for task in specs:
        assert task.resource_request == requests[task.class_id]


def test_arrivals_fit_fixed_window_and_epoch_grid() -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    requests = _requests()
    start = pd.Timestamp("2024-08-20T12:00:00Z")
    specs = build_scale_population(
        task_count=100,
        node_ids=[node.id for node in cluster.nodes],
        requests=requests,
        start_utc=start,
        arrival_window_seconds=3 * 3600,
        epoch_seconds=float(cluster.epoch_seconds),
    )
    offsets = [(task.arrival_utc - start).total_seconds() for task in specs]
    assert min(offsets) == 0
    assert max(offsets) < 3 * 3600
    assert all(offset % float(cluster.epoch_seconds) == 0 for offset in offsets)
    assert offsets == sorted(offsets)


def test_scale_ids_are_stable_across_policies() -> None:
    cluster = load_cluster_config("config/cluster.gcp.json")
    requests = _requests()
    specs = build_scale_population(
        task_count=25,
        node_ids=[node.id for node in cluster.nodes],
        requests=requests,
        start_utc=pd.Timestamp("2024-08-20T12:00:00Z"),
        arrival_window_seconds=3 * 3600,
        epoch_seconds=float(cluster.epoch_seconds),
    )
    assert specs[0].task_id == "scale-025-001"
    assert specs[-1].task_id == "scale-025-025"
    assert len({task.task_id for task in specs}) == 25
