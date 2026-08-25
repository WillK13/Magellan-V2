from __future__ import annotations

from pathlib import Path

import pytest

from magellan.telemetry.process import ProcfsProcessSampler


def stat_line(
    pid: int,
    process_group: int,
    utime: int,
    stime: int,
    rss_pages: int,
    state: str = "R",
    session_id: int | None = None,
) -> str:
    fields = ["0"] * 22
    fields[0] = state
    fields[1] = "1"  # ppid
    fields[2] = str(process_group)
    fields[3] = str(process_group if session_id is None else session_id)
    fields[11] = str(utime)
    fields[12] = str(stime)
    fields[21] = str(rss_pages)
    return f"{pid} (worker {pid}) " + " ".join(fields)


def write_stat(root: Path, pid: int, content: str) -> None:
    path = root / str(pid)
    path.mkdir(parents=True)
    (path / "stat").write_text(content, encoding="utf-8")


def test_procfs_sampler_aggregates_workload_session(tmp_path) -> None:
    write_stat(tmp_path, 100, stat_line(100, 100, 100, 50, 10, session_id=100))
    # MPI-style child: same session as the workload leader, different PGID.
    write_stat(tmp_path, 101, stat_line(101, 101, 50, 25, 5, "S", session_id=100))
    write_stat(tmp_path, 200, stat_line(200, 200, 900, 100, 100, session_id=200))

    sampler = ProcfsProcessSampler(tmp_path)
    sample = sampler.sample(100)

    ticks = int(__import__("os").sysconf("SC_CLK_TCK"))
    page_size = int(__import__("os").sysconf("SC_PAGE_SIZE"))
    assert sample.process_count == 2
    assert sample.process_state == "R"
    assert sample.cpu_time_seconds == pytest.approx(225 / ticks)
    assert sample.memory_rss_mb == pytest.approx(
        15 * page_size / (1024 * 1024)
    )
