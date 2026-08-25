from __future__ import annotations

import os
from pathlib import Path

from magellan.telemetry.models import ProcessMeasurement


class ProcfsUnavailableError(RuntimeError):
    pass


class ProcfsProcessSampler:
    """Read aggregate workload-session CPU and RSS from Linux procfs.

    Magellan starts each workload in a new session, making the leader PID the
    session ID. Descendants may create their own process groups (for example,
    MPI ranks launched by OpenMPI), so aggregate by session rather than only
    the leader's process group.
    """

    def __init__(self, proc_root: str | Path = "/proc") -> None:
        self._proc_root = Path(proc_root)
        try:
            self._clock_ticks = int(os.sysconf("SC_CLK_TCK"))
            self._page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError) as exc:
            raise ProcfsUnavailableError(str(exc)) from exc

    @staticmethod
    def _parse_stat(raw: str) -> tuple[int, str, int, int, float, int]:
        left = raw.find("(")
        right = raw.rfind(")")
        if left <= 0 or right <= left:
            raise ValueError("Malformed /proc stat record")
        pid = int(raw[:left].strip())
        fields = raw[right + 2 :].split()
        # fields[0] is state; fields[2] is process group; fields[3] is session;
        # utime/stime are original fields 14/15, represented here by indexes
        # 11/12. RSS is original field 24, represented by index 21.
        state = fields[0]
        process_group = int(fields[2])
        session_id = int(fields[3])
        cpu_ticks = float(fields[11]) + float(fields[12])
        rss_pages = int(fields[21])
        return pid, state, process_group, session_id, cpu_ticks, rss_pages

    def sample(self, session_id: int) -> ProcessMeasurement:
        if not self._proc_root.is_dir():
            raise ProcfsUnavailableError(
                f"procfs is unavailable at {self._proc_root}"
            )

        process_count = 0
        cpu_ticks = 0.0
        rss_pages = 0
        leader_state: str | None = None

        for path in self._proc_root.iterdir():
            if not path.name.isdigit():
                continue
            try:
                parsed = self._parse_stat(
                    (path / "stat").read_text(encoding="utf-8")
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
            pid, state, _process_group, process_session, item_ticks, item_rss = parsed
            if process_session != session_id:
                continue
            process_count += 1
            cpu_ticks += item_ticks
            rss_pages += max(0, item_rss)
            if pid == session_id:
                leader_state = state

        if process_count == 0:
            raise ProcessLookupError(session_id)

        return ProcessMeasurement(
            pid=session_id,
            process_count=process_count,
            process_state=leader_state,
            cpu_time_seconds=cpu_ticks / self._clock_ticks,
            memory_rss_mb=(rss_pages * self._page_size) / (1024 * 1024),
        )
