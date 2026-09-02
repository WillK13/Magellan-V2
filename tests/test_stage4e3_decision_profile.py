from __future__ import annotations

from magellan.experiments.stage4e3 import (
    category_rows,
    classify_profile_function,
    cumulative_metric,
    function_rows_from_stats,
)


def test_profile_categories_identify_magellan_hotspot_modules() -> None:
    assert (
        classify_profile_function(
            "/repo/magellan/policy/store.py",
            "_persist",
        )
        == "adaptive_store"
    )
    assert (
        classify_profile_function(
            "/repo/magellan/scheduler/scoring.py",
            "evaluate_task",
        )
        == "scheduler_scoring"
    )
    assert (
        classify_profile_function(
            "/repo/magellan/models/migrate_model.py",
            "estimate_migrate",
        )
        == "migration_estimator"
    )
    assert (
        classify_profile_function(
            "/repo/magellan/carbon/forecast.py",
            "forecast_or_average",
        )
        == "carbon_forecast"
    )


def test_profile_rows_and_cumulative_lookup() -> None:
    stats = {
        ("/repo/magellan/policy/store.py", 35, "_persist"): (
            4,
            4,
            0.2,
            0.8,
            {},
        ),
        ("/repo/magellan/scheduler/scoring.py", 305, "evaluate_task"): (
            2,
            2,
            0.1,
            1.0,
            {},
        ),
    }
    rows = function_rows_from_stats(
        stats,
        task_count=2,
        profile_wall_seconds=1.25,
    )
    calls, cumulative_ms = cumulative_metric(
        rows,
        filename_suffix="/magellan/policy/store.py",
        function="_persist",
    )
    assert calls == 4
    assert cumulative_ms == 800.0

    categories = category_rows(
        rows,
        task_count=2,
        profile_wall_seconds=1.25,
    )
    by_category = {row["category"]: row for row in categories}
    assert by_category["adaptive_store"]["self_ms"] == 200.0
    assert by_category["scheduler_scoring"]["self_ms"] == 100.0
