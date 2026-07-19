from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable


class RaplPowerReader:
    """Read aggregate top-level Intel RAPL package energy when available."""

    _TOP_LEVEL = re.compile(r"intel-rapl:\d+$")

    def __init__(
        self,
        powercap_root: str | Path = "/sys/class/powercap",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = Path(powercap_root)
        self._monotonic = monotonic
        self._previous: tuple[dict[Path, int], float] | None = None

    def _zones(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return sorted(
            path
            for path in self._root.glob("intel-rapl:*")
            if path.is_dir()
            and self._TOP_LEVEL.fullmatch(path.name)
            and (path / "energy_uj").is_file()
        )

    @staticmethod
    def _read_int(path: Path) -> int:
        return int(path.read_text(encoding="utf-8").strip())

    def sample_power_kw(self) -> float | None:
        zones = self._zones()
        if not zones:
            return None
        now = self._monotonic()
        current = {zone: self._read_int(zone / "energy_uj") for zone in zones}
        previous = self._previous
        self._previous = (current, now)
        if previous is None:
            return None
        previous_values, previous_at = previous
        elapsed = now - previous_at
        if elapsed <= 0:
            return None

        delta_uj = 0
        for zone, value in current.items():
            old = previous_values.get(zone)
            if old is None:
                continue
            if value >= old:
                delta_uj += value - old
                continue
            maximum_file = zone / "max_energy_range_uj"
            if maximum_file.is_file():
                maximum = self._read_int(maximum_file)
                delta_uj += max(0, maximum - old + value)

        if delta_uj <= 0:
            return None
        # microjoules / seconds -> watts; watts / 1000 -> kilowatts.
        return delta_uj / 1_000_000.0 / elapsed / 1000.0
