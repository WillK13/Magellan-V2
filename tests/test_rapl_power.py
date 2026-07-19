from __future__ import annotations

import pytest

from magellan.telemetry.power import RaplPowerReader


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_rapl_reader_computes_package_power(tmp_path) -> None:
    zone = tmp_path / "intel-rapl:0"
    zone.mkdir()
    energy = zone / "energy_uj"
    energy.write_text("1000000")
    (zone / "max_energy_range_uj").write_text("100000000")
    clock = Clock()
    reader = RaplPowerReader(tmp_path, monotonic=clock)

    assert reader.sample_power_kw() is None
    clock.value = 2.0
    energy.write_text("21000000")
    # 20 J over 2 seconds = 10 W = 0.01 kW.
    assert reader.sample_power_kw() == pytest.approx(0.01)
