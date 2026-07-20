from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from magellan.capabilities.models import NodeRuntimeCapabilities
from magellan.config.models import NodeConfig


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
