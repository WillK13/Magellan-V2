from __future__ import annotations

import pandas as pd

from magellan.carbon.forecast import LinearTrendForecastProvider
from magellan.config.policy_models import CarbonForecastPolicy


def series(values: list[float]) -> pd.Series:
    index = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=len(values),
        freq="15min",
    )
    return pd.Series(values, index=index)


def test_linear_forecast_uses_history_without_future_samples() -> None:
    values = series([10, 20, 30, 40, 50, 60, 70, 80])
    observed = values.index[-1]
    forecast = LinearTrendForecastProvider().forecast(
        node_id="boston",
        series=values,
        observed_at_utc=observed,
        forecast_start_utc=observed,
        duration_seconds=3600,
        policy=CarbonForecastPolicy(
            history_points=8,
            minimum_points=4,
            maximum_change_per_hour=100,
        ),
    )

    assert forecast.source == "linear_trend"
    assert forecast.freshness == "fresh"
    assert forecast.history_points == 8
    assert forecast.average_g_per_kwh > forecast.current_g_per_kwh
    assert 39 < forecast.slope_g_per_kwh_per_hour < 41
    assert forecast.confidence > 0.5


def test_forecast_uses_persistence_when_history_is_short() -> None:
    values = series([20, 30])
    observed = values.index[-1]
    forecast = LinearTrendForecastProvider().forecast(
        node_id="boston",
        series=values,
        observed_at_utc=observed,
        forecast_start_utc=observed,
        duration_seconds=3600,
        policy=CarbonForecastPolicy(
            history_points=8,
            minimum_points=4,
        ),
    )

    assert forecast.source == "persistence"
    assert forecast.average_g_per_kwh == 30


def test_stale_forecast_uses_configured_fallback() -> None:
    values = series([20, 30, 40, 50])
    observed = values.index[-1] + pd.Timedelta(hours=1)
    forecast = LinearTrendForecastProvider().forecast(
        node_id="boston",
        series=values,
        observed_at_utc=observed,
        forecast_start_utc=observed,
        duration_seconds=3600,
        policy=CarbonForecastPolicy(
            stale_after_seconds=60,
            configured_fallback_g_per_kwh=42,
        ),
    )

    assert forecast.source == "configured_fallback"
    assert forecast.freshness == "stale"
    assert forecast.average_g_per_kwh == 42


def test_carbon_store_uses_node_specific_fallback(tmp_path) -> None:
    from magellan.carbon.store import CARBON_COLUMN, TIME_COLUMN, CarbonStore
    from magellan.config.models import ClusterConfig, NodeConfig

    csv_path = tmp_path / "carbon.csv"
    pd.DataFrame(
        {
            TIME_COLUMN: ["2024-01-01T00:00:00Z"],
            CARBON_COLUMN: [100],
        }
    ).to_csv(csv_path, index=False)
    node = NodeConfig(
        id="boston",
        name="Boston",
        vm_name="boston",
        zone="zone",
        internal_ip="10.0.0.1",
        carbon_region="Boston",
        dataset_file="carbon.csv",
        latitude=0,
        longitude=0,
        carbon_fallback_g_per_kwh=55,
    )
    store = CarbonStore(
        cluster=ClusterConfig(nodes=[node]),
        datasets_directory=tmp_path,
    )
    forecast = store.forecast(
        node_id="boston",
        observed_at_utc="2023-12-31T23:00:00Z",
        forecast_start_utc="2023-12-31T23:00:00Z",
        duration_seconds=3600,
        policy=CarbonForecastPolicy(),
    )

    assert forecast.source == "configured_fallback"
    assert forecast.average_g_per_kwh == 55
