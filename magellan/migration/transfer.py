from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from magellan.config.models import ClusterConfig
from magellan.state.persistent_registry import PersistentTaskRegistry


@dataclass(frozen=True)
class CheckpointTransferResult:
    transfer_bytes: int
    duration_seconds: float
    setup_seconds: float = 0.0
    wall_seconds: float | None = None

    @property
    def bandwidth_mbps(self) -> float | None:
        if self.transfer_bytes <= 0 or self.duration_seconds <= 0:
            return None
        return self.transfer_bytes * 8.0 / self.duration_seconds / 1_000_000.0


class RsyncCheckpointTransfer:
    def __init__(
        self,
        cluster: ClusterConfig,
        registry: PersistentTaskRegistry,
        ssh_user: str,
        remote_state_root: str | Path,
    ) -> None:
        self._cluster = cluster
        self._registry = registry
        self._ssh_user = ssh_user
        self._remote_state_root = Path(remote_state_root)

    @staticmethod
    def _directory_bytes(path: Path) -> int:
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )

    def send(
        self,
        task_id: str,
        destination_node_id: str,
        migration_id: str,
    ) -> CheckpointTransferResult:
        destination = self._cluster.get_node(destination_node_id)
        local_checkpoint = self._registry.checkpoint_directory(task_id)

        if not local_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {local_checkpoint}"
            )

        transfer_bytes = self._directory_bytes(local_checkpoint)
        remote_checkpoint = (
            self._remote_state_root
            / "incoming"
            / migration_id
            / task_id
            / "checkpoint"
        )
        target = f"{self._ssh_user}@{destination.internal_ip}"
        ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        mkdir_command = f"mkdir -p {shlex.quote(str(remote_checkpoint))}"

        wall_started = time.perf_counter()
        setup_started = wall_started
        subprocess.run(
            ["ssh", *ssh_options, target, mkdir_command],
            check=True,
        )
        setup_seconds = max(0.0, time.perf_counter() - setup_started)

        started = time.perf_counter()
        subprocess.run(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
                f"{local_checkpoint}/",
                f"{target}:{remote_checkpoint}/",
            ],
            check=True,
        )
        duration = max(1e-9, time.perf_counter() - started)
        wall_seconds = max(1e-9, time.perf_counter() - wall_started)

        result = CheckpointTransferResult(
            transfer_bytes=transfer_bytes,
            duration_seconds=duration,
            setup_seconds=setup_seconds,
            wall_seconds=wall_seconds,
        )
        print(
            f"[checkpoint-transfer] task={task_id} "
            f"destination={destination_node_id} migration={migration_id} "
            f"bytes={transfer_bytes} duration={duration:.6f}s "
            f"setup={setup_seconds:.6f}s wall={wall_seconds:.6f}s "
            f"bandwidth_mbps={result.bandwidth_mbps or 0.0:.3f}",
            flush=True,
        )
        return result
