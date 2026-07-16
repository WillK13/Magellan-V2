from __future__ import annotations

from pathlib import Path

import pandas as pd

from magellan.config.models import ClusterConfig


TIME_COLUMN = "Datetime (UTC)"
CARBON_COLUMN = "Carbon intensity gCO₂eq/kWh (direct)"


def as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


class CarbonStore:
    def __init__(
        self,
        cluster: ClusterConfig,
        datasets_directory: str | Path,
    ) -> None:
        self._cluster = cluster
        self._datasets_directory = Path(datasets_directory)
        self._series_by_node_id: dict[str, pd.Series] = {}

        file_cache: dict[str, pd.Series] = {}

        for node in cluster.nodes:
            if node.dataset_file not in file_cache:
                file_path = self._datasets_directory / node.dataset_file
                file_cache[node.dataset_file] = self._load_series(file_path)

            self._series_by_node_id[node.id] = file_cache[node.dataset_file]

    @staticmethod
    def _load_series(csv_path: Path) -> pd.Series:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Carbon CSV does not exist: {csv_path}")

        frame = pd.read_csv(csv_path)

        missing = {
            TIME_COLUMN,
            CARBON_COLUMN,
        } - set(frame.columns)

        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing)}"
            )

        timestamps = pd.to_datetime(frame[TIME_COLUMN], utc=True, errors="raise")
        values = pd.to_numeric(frame[CARBON_COLUMN], errors="raise")

        series = pd.Series(values.to_numpy(), index=timestamps).sort_index()

        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()

        if series.empty:
            raise ValueError(f"Carbon CSV has no rows: {csv_path}")

        return series.astype(float)

    def bounds(self, node_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        series = self._series_by_node_id[node_id]
        return series.index[0], series.index[-1]

    def value_at(
        self,
        node_id: str,
        at_utc: str | pd.Timestamp,
    ) -> float:
        timestamp = as_utc_timestamp(at_utc)
        series = self._series_by_node_id[node_id]

        if timestamp < series.index[0] or timestamp > series.index[-1]:
            raise ValueError(
                f"Timestamp {timestamp} is outside data range "
                f"[{series.index[0]}, {series.index[-1]}] for node {node_id}"
            )

        if timestamp in series.index:
            return float(series.loc[timestamp])

        expanded_index = series.index.union(pd.DatetimeIndex([timestamp]))
        interpolated = (
            series.reindex(expanded_index)
            .sort_index()
            .interpolate(method="time")
        )

        return float(interpolated.loc[timestamp])

    def average(
        self,
        node_id: str,
        start_utc: str | pd.Timestamp,
        duration_seconds: float,
    ) -> float:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

        start = as_utc_timestamp(start_utc)

        if duration_seconds == 0:
            return self.value_at(node_id, start)

        end = start + pd.Timedelta(seconds=duration_seconds)
        series = self._series_by_node_id[node_id]

        if start < series.index[0] or end > series.index[-1]:
            raise ValueError(
                f"Carbon window [{start}, {end}] is outside data range "
                f"[{series.index[0]}, {series.index[-1]}] for node {node_id}"
            )

        sample_index = pd.date_range(
            start=start,
            end=end,
            freq="1min",
            inclusive="left",
        )

        if sample_index.empty:
            return self.value_at(node_id, start)

        expanded_index = series.index.union(sample_index)
        interpolated = (
            series.reindex(expanded_index)
            .sort_index()
            .interpolate(method="time")
        )

        return float(interpolated.reindex(sample_index).mean())
