from __future__ import annotations

import pandas as pd

from magellan.carbon.store import as_utc_timestamp
from magellan.config.policy_models import ClockPolicy


class MagellanClock:
    """Provides either real UTC time or accelerated trace time."""

    def __init__(self, policy: ClockPolicy) -> None:
        self._policy = policy
        self._real_start_utc = pd.Timestamp.now(tz="UTC")

        if policy.mode == "trace":
            if not policy.trace_start_utc:
                raise ValueError(
                    "trace_start_utc is required when clock mode is 'trace'"
                )

            self._trace_start_utc = as_utc_timestamp(
                policy.trace_start_utc
            )
        else:
            self._trace_start_utc = None

    @property
    def trace_seconds_per_real_second(self) -> float:
        if self._policy.mode == "wall":
            return 1.0
        return self._policy.trace_seconds_per_real_second

    def evaluation_seconds_for_wall_seconds(
        self,
        wall_seconds: float,
    ) -> float:
        return max(0.0, wall_seconds) * (
            self.trace_seconds_per_real_second
        )

    def wall_seconds_for_evaluation_seconds(
        self,
        evaluation_seconds: float,
    ) -> float:
        return max(0.0, evaluation_seconds) / (
            self.trace_seconds_per_real_second
        )

    def now(self) -> pd.Timestamp:
        wall_now = pd.Timestamp.now(tz="UTC")

        if self._policy.mode == "wall":
            return wall_now

        assert self._trace_start_utc is not None

        real_elapsed_seconds = (
            wall_now - self._real_start_utc
        ).total_seconds()

        trace_elapsed_seconds = (
            real_elapsed_seconds
            * self._policy.trace_seconds_per_real_second
        )

        return self._trace_start_utc + pd.Timedelta(
            seconds=trace_elapsed_seconds
        )
