#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.carbon.store import CarbonMetric, as_utc_timestamp
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage4b import (
    CORE_WORKLOADS,
    FrozenCalibrationGraph,
    load_node_slowdowns,
    load_stage4a1_edges,
    load_workload_calibrations,
)
from magellan.experiments.stage4c import runtime_scales_for_target
from magellan.experiments.stage4d2 import ReplayCarbonStore, read_resource_model
from magellan.experiments.stage4e1 import SCALE_SIZES, build_scale_population
from magellan.experiments.stage4e2 import benchmark_control_plane, benchmark_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4E.2 control-plane 25/50/100-task scaling benchmark."
    )
    parser.add_argument("--stage4e1-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    return parser.parse_args()


def require_bundle(path: Path, label: str, *, require_passed: bool = True) -> dict:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum validation failed: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if require_passed and summary.get("passed") is not True:
        raise RuntimeError(f"{label} summary passed=false")
    return summary


def main() -> int:
    args = parse_args()
    e1 = Path(args.stage4e1_bundle)
    e1_summary = require_bundle(e1, "Stage 4E.1")

    d44 = Path(str(e1_summary.get("source_stage4d4_bundle") or ""))
    d44_summary = require_bundle(d44, "Stage 4D.4")
    d43 = Path(str(d44_summary.get("source_stage4d3_bundle") or ""))
    d43_summary = require_bundle(d43, "Stage 4D.3")
    d42 = Path(str(d43_summary.get("source_stage4d2_bundle") or ""))
    d42_summary = require_bundle(d42, "Stage 4D.2")
    d41 = Path(str(d42_summary.get("source_stage4d1_bundle") or ""))
    require_bundle(d41, "Stage 4D.1")

    a1 = Path(str(e1_summary.get("stage4a1_bundle") or ""))
    a2 = Path(str(e1_summary.get("stage4a2_bundle") or ""))
    a3 = Path(str(e1_summary.get("stage4a3_bundle") or ""))
    a4 = Path(str(e1_summary.get("stage4a4_bundle") or ""))
    a5 = Path(str(e1_summary.get("stage4a5_bundle") or ""))
    a1_summary = require_bundle(a1, "Stage 4A.1", require_passed=False)
    require_bundle(a2, "Stage 4A.2")
    require_bundle(a3, "Stage 4A.3")
    require_bundle(a4, "Stage 4A.4")
    require_bundle(a5, "Stage 4A.5")

    cluster = load_cluster_config(args.cluster)
    policy = load_policy_config(args.policy)
    capacities, requests = read_resource_model(d41)
    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    node_slowdowns = load_node_slowdowns(a4)
    target_seconds = float(e1_summary["target_boston_hours"]) * 3600.0
    runtime_scales = runtime_scales_for_target(
        calibrations,
        node_slowdowns=node_slowdowns,
        target_boston_runtime_seconds=target_seconds,
    )
    edge_rows = load_stage4a1_edges(a1, a1_summary)
    graphs = {
        class_id: FrozenCalibrationGraph(
            cluster=cluster,
            edge_rows=edge_rows,
            workload=calibration,
        )
        for class_id, calibration in calibrations.items()
    }
    carbon_store = ReplayCarbonStore(
        cluster,
        args.datasets,
        carbon_metric=CarbonMetric.LIFECYCLE,
    )
    at_utc = as_utc_timestamp(e1_summary["trace_start_utc"])
    arrival_window_seconds = float(e1_summary["arrival_window_hours"]) * 3600.0
    node_ids = [node.id for node in cluster.nodes]
    all_node_ids = set(node_ids)

    comparison_id = args.comparison_id or (
        f"stage4e2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4E.2 control-plane scaling benchmark ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4e1={e1}")
    print(f"sizes={','.join(str(value) for value in SCALE_SIZES)}")
    print(f"repetitions={args.repetitions} warmups={args.warmups}")
    print(
        "timed_path=production evaluate_task + best migration bid construction + "
        "production rank_bids + measured resource-ledger admission"
    )
    print(
        "latency_mode=steady-state after carbon-cache warmup; cold first-pass "
        "epoch is reported separately"
    )
    print(
        "memory_mode=separate tracemalloc probe so memory instrumentation does "
        "not contaminate reported latency samples"
    )

    summary_rows = []
    sample_rows = []

    for size in SCALE_SIZES:
        specs = build_scale_population(
            task_count=size,
            node_ids=node_ids,
            requests=requests,
            start_utc=at_utc,
            arrival_window_seconds=arrival_window_seconds,
            epoch_seconds=float(cluster.epoch_seconds),
        )
        tasks = benchmark_tasks(
            specs=specs,
            calibrations=calibrations,
            runtime_scales=runtime_scales,
            node_slowdowns=node_slowdowns,
            graphs=graphs,
            all_node_ids=all_node_ids,
        )

        print(f"\n[n={size}] benchmarking {size} production decisions + {size} bids", flush=True)
        before_cache = carbon_store.cache_summary()
        row, samples = benchmark_control_plane(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
            repetitions=args.repetitions,
            warmups=args.warmups,
        )
        after_cache = carbon_store.cache_summary()
        row.update(
            {
                "forecast_cache_hits_delta": (
                    after_cache["forecast_hits"] - before_cache["forecast_hits"]
                ),
                "forecast_cache_entries_delta": (
                    after_cache["forecast_entries"] - before_cache["forecast_entries"]
                ),
                "average_cache_hits_delta": (
                    after_cache["average_hits"] - before_cache["average_hits"]
                ),
                "average_cache_entries_delta": (
                    after_cache["average_entries"] - before_cache["average_entries"]
                ),
            }
        )
        summary_rows.append(row)
        sample_rows.extend(samples)

        print(
            f"  decision={row['decision_wall_ms_median']:.3f}ms "
            f"({row['decision_per_task_ms_median']:.3f}ms/task) "
            f"auction={row['auction_wall_ms_median']:.3f}ms "
            f"({row['auction_bids_per_second']:.1f} bids/s) "
            f"epoch={row['epoch_wall_ms_median']:.3f}ms "
            f"p95={row['epoch_wall_ms_p95']:.3f}ms "
            f"peak_mem={row['peak_incremental_tracemalloc_kb']:.1f}KiB",
            flush=True,
        )

    # No fixed threshold is used as a PASS condition: performance is a measured
    # result, not something the benchmark tunes itself to pass.
    passed = (
        len(summary_rows) == len(SCALE_SIZES)
        and len(sample_rows) == len(SCALE_SIZES) * args.repetitions
        and all(int(row["bid_count"]) == int(row["task_count"]) for row in summary_rows)
        and all(float(row["epoch_wall_ms_median"]) > 0 for row in summary_rows)
        and all(float(row["peak_incremental_tracemalloc_kb"]) > 0 for row in summary_rows)
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4e1_bundle": str(e1),
        "source_stage4d4_bundle": str(d44),
        "source_stage4d3_bundle": str(d43),
        "source_stage4d2_bundle": str(d42),
        "source_stage4d1_bundle": str(d41),
        "stage4a1_bundle": str(a1),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "stage4a5_bundle": str(a5),
        "scale_sizes": list(SCALE_SIZES),
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "benchmark_timestamp_utc": at_utc.isoformat(),
        "summary_row_count": len(summary_rows),
        "sample_row_count": len(sample_rows),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4e2_control_plane_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Measure actual wall-clock and CPU overhead of Magellan's control-plane "
                "decision and destination-auction code at 25, 50 and 100 task batch sizes."
            ),
            "decision_path": (
                "Every task invokes the production evaluate_task function using the "
                "same frozen resource classes, workload calibration, node slowdown, "
                "summer carbon timestamp, policy, and seven-node graph as Stage 4E.1."
            ),
            "auction_path": (
                "Each evaluated task contributes one best migration candidate bid. "
                "Bids are grouped by their scored destination and processed through "
                "production rank_bids(lowest_score), followed by measured ResourceLedger "
                "compatibility and reservation."
            ),
            "adaptive_policy": (
                "Each repetition receives a fresh production AdaptivePolicyService store "
                "so mutable adaptive state does not leak between repeated measurements. "
                "Store creation itself occurs outside the timed epoch."
            ),
            "cache": (
                "The first epoch is reported as a cold measurement. Warmup epochs then "
                "populate the process-local ReplayCarbonStore cache, and the reported "
                "median/p95 latency values represent steady-state repeated scheduler use."
            ),
            "memory": (
                "tracemalloc peak incremental allocation is measured in a separate epoch "
                "because tracemalloc instrumentation materially perturbs execution time. "
                "It is a Python allocation metric, not total VM RSS."
            ),
            "scope": (
                "This is a single-process offline control-plane microbenchmark. It does "
                "not measure FastAPI/network transport, GCP RPC latency, checkpoint I/O, "
                "or physical workload execution. Those are evaluated separately by the "
                "system and migration experiments."
            ),
            "pass_condition": (
                "PASS checks coverage and benchmark integrity only. No latency, throughput "
                "or memory threshold is required, so the measurement cannot tune itself "
                "to a desired scalability conclusion."
            ),
        },
    }

    write_csv(
        root / "control_plane_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        root / "latency_samples.csv",
        sample_rows,
        list(sample_rows[0].keys()),
    )
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4E2_CONTROL_PLANE_SCALING_PASS" if passed else "STAGE_4E2_CONTROL_PLANE_SCALING_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"sizes: {len(summary_rows)}/{len(SCALE_SIZES)}")
    print(f"samples: {len(sample_rows)}/{len(SCALE_SIZES) * args.repetitions}")
    print("\nControl-plane curve:")
    for row in summary_rows:
        print(
            f"  n={int(row['task_count']):3d} "
            f"decision={float(row['decision_wall_ms_median']):9.3f}ms "
            f"auction={float(row['auction_wall_ms_median']):8.3f}ms "
            f"epoch={float(row['epoch_wall_ms_median']):9.3f}ms "
            f"p95={float(row['epoch_wall_ms_p95']):9.3f}ms "
            f"decision_rate={float(row['decision_tasks_per_second']):8.1f}/s "
            f"bid_rate={float(row['auction_bids_per_second']):9.1f}/s "
            f"peak_mem={float(row['peak_incremental_tracemalloc_kb']):9.1f}KiB"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
