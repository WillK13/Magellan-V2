from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from magellan.carbon.store import (
    DIRECT_CARBON_COLUMN,
    LIFECYCLE_CARBON_COLUMN,
    TIME_COLUMN,
    CarbonMetric,
    CarbonStore,
)
from magellan.config.models import ClusterConfig, NodeConfig


def cluster() -> ClusterConfig:
    return ClusterConfig(
        nodes=[
            NodeConfig(
                id="test",
                name="Test",
                vm_name="test-vm",
                zone="test-zone",
                internal_ip="10.0.0.1",
                carbon_region="test",
                dataset_file="test.csv",
                latitude=0.0,
                longitude=0.0,
            )
        ]
    )


def write_trace(path: Path, *, include_lifecycle: bool = True) -> None:
    data: dict[str, object] = {
        TIME_COLUMN: pd.date_range(
            "2024-01-01T00:00:00Z",
            periods=3,
            freq="1h",
        ),
        DIRECT_CARBON_COLUMN: [0.0, 10.0, 20.0],
    }
    if include_lifecycle:
        data[LIFECYCLE_CARBON_COLUMN] = [24.0, 34.0, 44.0]
    pd.DataFrame(data).to_csv(path, index=False)


def test_carbon_store_preserves_direct_default(tmp_path: Path) -> None:
    write_trace(tmp_path / "test.csv")
    store = CarbonStore(cluster(), tmp_path)

    assert store.carbon_metric is CarbonMetric.DIRECT
    assert store.carbon_column == DIRECT_CARBON_COLUMN
    assert store.value_at("test", "2024-01-01T00:00:00Z") == 0.0


def test_carbon_store_can_select_lifecycle_series(tmp_path: Path) -> None:
    write_trace(tmp_path / "test.csv")
    store = CarbonStore(cluster(), tmp_path, carbon_metric="lifecycle")

    assert store.carbon_metric is CarbonMetric.LIFECYCLE
    assert store.carbon_column == LIFECYCLE_CARBON_COLUMN
    assert store.value_at("test", "2024-01-01T00:00:00Z") == 24.0
    assert store.average("test", "2024-01-01T00:00:00Z", 3600) > 24.0


def test_selected_carbon_column_is_required(tmp_path: Path) -> None:
    write_trace(tmp_path / "test.csv", include_lifecycle=False)

    with pytest.raises(ValueError, match="Life cycle"):
        CarbonStore(cluster(), tmp_path, carbon_metric=CarbonMetric.LIFECYCLE)
