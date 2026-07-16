from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from magellan.config.models import ClusterConfig
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)


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

    def send(
        self,
        task_id: str,
        destination_node_id: str,
        migration_id: str,
    ) -> None:
        destination = self._cluster.get_node(
            destination_node_id
        )

        local_checkpoint = (
            self._registry.checkpoint_directory(task_id)
        )

        if not local_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: "
                f"{local_checkpoint}"
            )

        remote_checkpoint = (
            self._remote_state_root
            / "incoming"
            / migration_id
            / task_id
            / "checkpoint"
        )

        target = (
            f"{self._ssh_user}@"
            f"{destination.internal_ip}"
        )

        ssh_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]

        mkdir_command = (
            f"mkdir -p "
            f"{shlex.quote(str(remote_checkpoint))}"
        )

        subprocess.run(
            [
                "ssh",
                *ssh_options,
                target,
                mkdir_command,
            ],
            check=True,
        )

        subprocess.run(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                (
                    "ssh -o BatchMode=yes "
                    "-o StrictHostKeyChecking=accept-new"
                ),
                f"{local_checkpoint}/",
                f"{target}:{remote_checkpoint}/",
            ],
            check=True,
        )

        print(
            f"[checkpoint-transfer] task={task_id} "
            f"destination={destination_node_id} "
            f"migration={migration_id}",
            flush=True,
        )
