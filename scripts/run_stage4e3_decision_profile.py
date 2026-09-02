#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from magellan.experiments.stage4e2 import benchmark_tasks
from magellan.experiments.stage4e3 import profile_control_plane_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Stage 4E decision-engine hotspots at 25/50/100 tasks."
    )
    parser.add_argument("--stage4e2-bundle", required=True)
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument(
        "--top-functions",
        type=int,
        default=40,
        help="Number of highest cumulative-time functions retained per scale.",
    )
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    if args.top_functions <= 0:
        raise ValueError("--top-functions must be positive")

    e2 = Path(args.stage4e2_bundle)
    e2_summary = require_bundle(e2, "Stage 4E.2")
    e1 = Path(str(e2_summary.get("source_stage4e1_bundle") or ""))
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

    e2_rows = read_csv(e2 / "control_plane_summary.csv")
    e2_by_size = {int(row["task_count"]): row for row in e2_rows}

    comparison_id = args.comparison_id or (
        f"stage4e3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4E.3 decision-engine hotspot attribution ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4e2={e2}")
    print(f"sizes={','.join(str(value) for value in SCALE_SIZES)}")
    print(
        "method=one warm cProfile epoch per scale using the same production "
        "evaluate_task + auction path as Stage 4E.2"
    )
    print(
        "interpretation=cProfile adds overhead, so Stage 4E.2 remains the canonical "
        "latency measurement; Stage 4E.3 attributes where execution time is spent"
    )
    print(
        "hypothesis_to_test=AdaptivePolicyStore rewrites the complete task-state "
        "dictionary on every put; no code is modified by this profiler"
    )

    summary_rows = []
    function_rows = []
    category_rows = []

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

        print(f"\n[n={size}] warming cache then profiling one full epoch", flush=True)
        row, functions, categories = profile_control_plane_epoch(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
        )
        baseline = e2_by_size[size]
        baseline_ms = float(baseline["epoch_wall_ms_median"])
        row["stage4e2_epoch_wall_ms_median"] = baseline_ms
        row["cprofile_overhead_ratio_vs_e2_median"] = (
            float(row["profiled_epoch_wall_ms"]) / baseline_ms
            if baseline_ms > 0
            else 0.0
        )
        row["stage4e2_decision_per_task_ms_median"] = float(
            baseline["decision_per_task_ms_median"]
        )
        summary_rows.append(row)
        function_rows.extend(functions[: args.top_functions])
        category_rows.extend(categories)

        print(
            f"  profiled_wall={row['profiled_epoch_wall_ms']:.1f}ms "
            f"E2_median={baseline_ms:.1f}ms "
            f"store_put_calls={row['adaptive_store_put_calls']} "
            f"persist_calls={row['adaptive_store_persist_calls']} "
            f"persist_cum={row['adaptive_store_persist_cumulative_ms']:.1f}ms "
            f"persist_fraction={100*row['adaptive_store_persist_fraction_of_profiled_wall']:.1f}% "
            f"dominant_self={row['dominant_self_category']}",
            flush=True,
        )

    passed = (
        len(summary_rows) == len(SCALE_SIZES)
        and all(int(row["evaluate_task_calls"]) == int(row["task_count"]) for row in summary_rows)
        and all(int(row["adaptive_store_put_calls"]) > 0 for row in summary_rows)
        and all(float(row["profiled_epoch_wall_ms"]) > 0 for row in summary_rows)
        and bool(function_rows)
        and bool(category_rows)
    )

    n100 = next(row for row in summary_rows if int(row["task_count"]) == 100)
    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4e2_bundle": str(e2),
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
        "summary_row_count": len(summary_rows),
        "function_row_count": len(function_rows),
        "category_row_count": len(category_rows),
        "top_functions_per_scale": args.top_functions,
        "n100_adaptive_store_put_calls": n100["adaptive_store_put_calls"],
        "n100_adaptive_store_persist_calls": n100["adaptive_store_persist_calls"],
        "n100_adaptive_store_persist_fraction_of_profiled_wall": (
            n100["adaptive_store_persist_fraction_of_profiled_wall"]
        ),
        "n100_dominant_self_category": n100["dominant_self_category"],
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4e3_decision_engine_hotspot_profile",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Attribute the superlinear per-task decision cost observed in canonical "
                "Stage 4E.2 without changing scheduler behavior."
            ),
            "profile_scope": (
                "One cProfile-instrumented warm control-plane epoch is run at each of "
                "25, 50 and 100 tasks. The timed path is exactly the Stage 4E.2 "
                "execute_control_plane_epoch production decision + auction path."
            ),
            "latency_boundary": (
                "cProfile perturbs timing. Canonical median/p95 latency remains Stage 4E.2. "
                "Stage 4E.3 uses profiled time only for attribution and reports the "
                "instrumentation overhead relative to Stage 4E.2."
            ),
            "cache": (
                "Each size receives an unprofiled warmup epoch using the same ReplayCarbonStore, "
                "followed by a profiled epoch with a fresh adaptive-policy state directory."
            ),
            "categories": (
                "category_profile.csv aggregates cProfile self time into non-overlapping "
                "scheduler, estimator, forecast, adaptive store/policy, auction, Pydantic, "
                "pandas, filesystem, JSON and runtime categories."
            ),
            "functions": (
                "function_profile.csv contains the highest cumulative-time functions per "
                "scale. Cumulative times overlap by call stack and are used to locate "
                "specific expensive call paths; category self times do not overlap."
            ),
            "hypothesis": (
                "The current AdaptivePolicyStore.put path atomically persists the complete "
                "per-task state dictionary. evaluate_task invokes adaptive prepare and "
                "record_decision, each of which writes state. The profile records put, "
                "_persist and fsync call counts/times but PASS does not require that this "
                "hypothesis be correct."
            ),
            "pass_condition": (
                "PASS validates profile coverage and integrity only. It does not require "
                "a particular hotspot, scaling slope, or optimization opportunity."
            ),
        },
    }

    write_csv(
        root / "profile_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_csv(
        root / "category_profile.csv",
        category_rows,
        list(category_rows[0].keys()),
    )
    write_csv(
        root / "function_profile.csv",
        function_rows,
        list(function_rows[0].keys()),
    )
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4E3_DECISION_PROFILE_PASS" if passed else "STAGE_4E3_DECISION_PROFILE_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"sizes: {len(summary_rows)}/{len(SCALE_SIZES)}")
    print("\nHotspot curve:")
    for row in summary_rows:
        print(
            f"  n={int(row['task_count']):3d} "
            f"E2={float(row['stage4e2_epoch_wall_ms_median']):8.1f}ms "
            f"profiled={float(row['profiled_epoch_wall_ms']):9.1f}ms "
            f"put_calls={int(row['adaptive_store_put_calls']):4d} "
            f"persist_calls={int(row['adaptive_store_persist_calls']):4d} "
            f"persist_cum={float(row['adaptive_store_persist_cumulative_ms']):9.1f}ms "
            f"persist_share={100*float(row['adaptive_store_persist_fraction_of_profiled_wall']):6.1f}% "
            f"dominant_self={row['dominant_self_category']}"
        )

    print("\nTop cumulative functions at n=100:")
    for row in [
        item for item in function_rows if int(item["task_count"]) == 100
    ][:12]:
        print(
            f"  {float(row['cumulative_ms']):9.1f}ms "
            f"self={float(row['self_ms']):8.1f}ms "
            f"calls={int(row['total_calls']):7d} "
            f"{Path(str(row['filename'])).name}:{row['line_number']} "
            f"{row['function']}"
        )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
