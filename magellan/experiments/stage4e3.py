from __future__ import annotations

import cProfile
import pstats
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.models import ClusterConfig, NodeResourceCapacity
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.stage4e2 import (
    BenchmarkTask,
    execute_control_plane_epoch,
    make_adaptive_service,
)


def classify_profile_function(filename: str, function: str) -> str:
    normalized = filename.replace("\\", "/")

    if normalized.endswith("/magellan/policy/store.py"):
        return "adaptive_store"
    if normalized.endswith("/magellan/policy/adaptive.py"):
        return "adaptive_policy"
    if normalized.endswith("/magellan/scheduler/scoring.py"):
        return "scheduler_scoring"
    if normalized.endswith("/magellan/models/continue_model.py"):
        return "continue_estimator"
    if normalized.endswith("/magellan/models/pause_model.py"):
        return "pause_estimator"
    if normalized.endswith("/magellan/models/migrate_model.py"):
        return "migration_estimator"
    if normalized.endswith("/magellan/carbon/forecast.py"):
        return "carbon_forecast"
    if normalized.endswith("/magellan/carbon/store.py"):
        return "carbon_store"
    if "/magellan/bidding/" in normalized:
        return "bidding_auction"
    if "/pydantic/" in normalized or "/pydantic_core/" in normalized:
        return "pydantic"
    if "/pandas/" in normalized:
        return "pandas"
    if "/json/" in normalized or function in {"dump", "dumps"}:
        return "json_serialization"
    if (
        function.startswith("<built-in method posix.")
        or "/tempfile.py" in normalized
        or "/pathlib" in normalized
    ):
        return "filesystem_io"
    if "/magellan/" in normalized:
        return "other_magellan"
    return "python_runtime"


def function_rows_from_stats(
    stats: dict[tuple[str, int, str], tuple[int, int, float, float, Any]],
    *,
    task_count: int,
    profile_wall_seconds: float,
) -> list[dict[str, Any]]:
    rows = []
    denominator = max(profile_wall_seconds, 1e-12)
    for (filename, line_number, function), values in stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        rows.append(
            {
                "task_count": task_count,
                "filename": filename,
                "line_number": line_number,
                "function": function,
                "category": classify_profile_function(filename, function),
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_ms": self_seconds * 1000.0,
                "cumulative_ms": cumulative_seconds * 1000.0,
                "self_fraction_of_profiled_wall": self_seconds / denominator,
                "cumulative_fraction_of_profiled_wall": cumulative_seconds / denominator,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["cumulative_ms"]),
            -float(row["self_ms"]),
            str(row["filename"]),
            int(row["line_number"]),
            str(row["function"]),
        ),
    )


def category_rows(
    function_rows: Iterable[dict[str, Any]],
    *,
    task_count: int,
    profile_wall_seconds: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in function_rows:
        category = str(row["category"])
        item = grouped.setdefault(
            category,
            {
                "self_ms": 0.0,
                "primitive_calls": 0.0,
                "total_calls": 0.0,
            },
        )
        item["self_ms"] += float(row["self_ms"])
        item["primitive_calls"] += int(row["primitive_calls"])
        item["total_calls"] += int(row["total_calls"])

    denominator_ms = max(profile_wall_seconds * 1000.0, 1e-12)
    output = [
        {
            "task_count": task_count,
            "category": category,
            "self_ms": values["self_ms"],
            "self_fraction_of_profiled_wall": values["self_ms"] / denominator_ms,
            "primitive_calls": int(values["primitive_calls"]),
            "total_calls": int(values["total_calls"]),
        }
        for category, values in grouped.items()
    ]
    return sorted(output, key=lambda row: -float(row["self_ms"]))


def _matching_stats(
    function_rows: list[dict[str, Any]],
    *,
    filename_suffix: str | None = None,
    function: str | None = None,
    function_contains: str | None = None,
) -> list[dict[str, Any]]:
    output = []
    for row in function_rows:
        filename = str(row["filename"]).replace("\\", "/")
        func = str(row["function"])
        if filename_suffix is not None and not filename.endswith(filename_suffix):
            continue
        if function is not None and func != function:
            continue
        if function_contains is not None and function_contains not in func:
            continue
        output.append(row)
    return output


def cumulative_metric(
    function_rows: list[dict[str, Any]],
    *,
    filename_suffix: str | None = None,
    function: str | None = None,
    function_contains: str | None = None,
) -> tuple[int, float]:
    matches = _matching_stats(
        function_rows,
        filename_suffix=filename_suffix,
        function=function,
        function_contains=function_contains,
    )
    calls = sum(int(row["total_calls"]) for row in matches)
    cumulative_ms = sum(float(row["cumulative_ms"]) for row in matches)
    return calls, cumulative_ms


def profile_control_plane_epoch(
    *,
    tasks: list[BenchmarkTask],
    capacities: dict[str, NodeResourceCapacity],
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    carbon_store: CarbonStore,
    at_utc: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="magellan-stage4e3-") as directory:
        root = Path(directory)

        # Populate the existing process-local carbon cache without carrying any
        # adaptive task state into the profiled epoch.
        warm_service = make_adaptive_service(
            policy=policy,
            root=root / "warmup",
        )
        execute_control_plane_epoch(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
            adaptive_service=warm_service,
        )

        profile_service = make_adaptive_service(
            policy=policy,
            root=root / "profile",
        )
        profiler = cProfile.Profile()
        wall_start = time.perf_counter()
        profiler.enable()
        result = execute_control_plane_epoch(
            tasks=tasks,
            capacities=capacities,
            cluster=cluster,
            policy=policy,
            carbon_store=carbon_store,
            at_utc=at_utc,
            adaptive_service=profile_service,
        )
        profiler.disable()
        profile_wall_seconds = time.perf_counter() - wall_start

    stats = pstats.Stats(profiler)
    functions = function_rows_from_stats(
        stats.stats,
        task_count=len(tasks),
        profile_wall_seconds=profile_wall_seconds,
    )
    categories = category_rows(
        functions,
        task_count=len(tasks),
        profile_wall_seconds=profile_wall_seconds,
    )

    evaluate_calls, evaluate_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/scheduler/scoring.py",
        function="evaluate_task",
    )
    raw_calls, raw_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/scheduler/scoring.py",
        function="build_raw_actions",
    )
    continue_calls, continue_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/models/continue_model.py",
        function="estimate_continue",
    )
    pause_calls, pause_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/models/pause_model.py",
        function="estimate_pause",
    )
    migrate_calls, migrate_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/models/migrate_model.py",
        function="estimate_migrate",
    )
    forecast_calls, forecast_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/carbon/forecast.py",
        function="forecast_or_average",
    )
    prepare_calls, prepare_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/policy/adaptive.py",
        function="prepare",
    )
    record_calls, record_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/policy/adaptive.py",
        function="record_decision",
    )
    put_calls, put_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/policy/store.py",
        function="put",
    )
    persist_calls, persist_ms = cumulative_metric(
        functions,
        filename_suffix="/magellan/policy/store.py",
        function="_persist",
    )
    fsync_calls, fsync_ms = cumulative_metric(
        functions,
        function_contains="posix.fsync",
    )

    summary = {
        "task_count": len(tasks),
        "profiled_epoch_wall_ms": profile_wall_seconds * 1000.0,
        "profile_total_self_ms": stats.total_tt * 1000.0,
        "profile_total_calls": stats.total_calls,
        "profile_primitive_calls": stats.prim_calls,
        "decision_wall_ms_instrumented": result["decision_wall_ns"] / 1e6,
        "auction_wall_ms_instrumented": result["auction_wall_ns"] / 1e6,
        "evaluate_task_calls": evaluate_calls,
        "evaluate_task_cumulative_ms": evaluate_ms,
        "build_raw_actions_calls": raw_calls,
        "build_raw_actions_cumulative_ms": raw_ms,
        "estimate_continue_calls": continue_calls,
        "estimate_continue_cumulative_ms": continue_ms,
        "estimate_pause_calls": pause_calls,
        "estimate_pause_cumulative_ms": pause_ms,
        "estimate_migrate_calls": migrate_calls,
        "estimate_migrate_cumulative_ms": migrate_ms,
        "forecast_or_average_calls": forecast_calls,
        "forecast_or_average_cumulative_ms": forecast_ms,
        "adaptive_prepare_calls": prepare_calls,
        "adaptive_prepare_cumulative_ms": prepare_ms,
        "adaptive_record_calls": record_calls,
        "adaptive_record_cumulative_ms": record_ms,
        "adaptive_store_put_calls": put_calls,
        "adaptive_store_put_cumulative_ms": put_ms,
        "adaptive_store_persist_calls": persist_calls,
        "adaptive_store_persist_cumulative_ms": persist_ms,
        "adaptive_store_persist_fraction_of_profiled_wall": (
            persist_ms / max(profile_wall_seconds * 1000.0, 1e-12)
        ),
        "fsync_calls": fsync_calls,
        "fsync_cumulative_ms": fsync_ms,
        "dominant_self_category": categories[0]["category"] if categories else "",
        "dominant_self_category_ms": categories[0]["self_ms"] if categories else 0.0,
    }
    return summary, functions, categories
