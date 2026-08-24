from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from magellan.capabilities.models import NodeRuntimeCapabilities
from magellan.config.models import NodeConfig


_RUNTIME_VERSION_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)")


def _runtime_version_tuple(value: str) -> tuple[int, ...] | None:
    match = _RUNTIME_VERSION_RE.search(value)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def runtime_version_matches(configured: str, observed: str) -> bool:
    """Compare configured runtime versions with human-readable tool output.

    Discovery commands commonly return a banner rather than a bare version,
    for example ``mpirun (Open MPI) 4.1.4``.  A configured prefix such as
    ``3.11`` should also accept an observed patch version such as ``3.11.2``.
    """

    if observed.startswith(configured):
        return True
    configured_parts = _runtime_version_tuple(configured)
    observed_parts = _runtime_version_tuple(observed)
    if configured_parts is None or observed_parts is None:
        return False
    return observed_parts[: len(configured_parts)] == configured_parts


def _memory_mb() -> int | None:
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    return None


def _command_version(command: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else None


def discover_local_capabilities(node: NodeConfig) -> NodeRuntimeCapabilities:
    configured_commands = set(node.capabilities.commands)
    common_commands = {
        "bash",
        "python3",
        "rsync",
        "mpirun",
        "mpiexec",
    }
    discovered_commands = {
        command
        for command in configured_commands | common_commands
        if shutil.which(command) is not None
    }
    runtimes: dict[str, str] = {
        "python": platform.python_version(),
    }
    for name, command in (("openmpi", "mpirun"), ("rsync", "rsync")):
        version = _command_version(command)
        if version is not None:
            runtimes[name] = version

    features = {
        "local-command",
        "python-module",
        "process-group",
        "application-checkpoint",
        "dendro-adapter",
    }
    if "mpirun" in discovered_commands or "mpiexec" in discovered_commands:
        features.add("mpi")
    return NodeRuntimeCapabilities(
        architecture=platform.machine().lower(),
        operating_system=platform.system().lower(),
        cpu_cores=float(os.cpu_count() or 1),
        memory_mb=_memory_mb(),
        gpu_count=node.resources.gpu_count,
        accelerator_types=set(node.resources.accelerator_types),
        commands=discovered_commands,
        runtimes=runtimes,
        features=features,
    )
