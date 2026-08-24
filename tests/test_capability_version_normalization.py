from __future__ import annotations

from magellan.capabilities.discovery import runtime_version_matches


def test_runtime_version_matches_human_readable_openmpi_banner() -> None:
    assert runtime_version_matches("4.1.4", "mpirun (Open MPI) 4.1.4")


def test_runtime_version_accepts_more_specific_patch_version() -> None:
    assert runtime_version_matches("3.11", "Python 3.11.2")


def test_runtime_version_rejects_different_version() -> None:
    assert not runtime_version_matches("4.1.4", "mpirun (Open MPI) 4.1.5")
    assert not runtime_version_matches("4.1.4", "mpirun (Open MPI) 4.1.40")
