from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from magellan.artifacts.manager import ArtifactManager
from magellan.config.models import ClusterConfig


class RsyncArtifactTransfer:
    def __init__(
        self,
        cluster: ClusterConfig,
        manager: ArtifactManager,
        ssh_user: str,
        remote_state_root: str | Path,
    ) -> None:
        self._cluster = cluster
        self._manager = manager
        self._ssh_user = ssh_user
        self._remote_state_root = Path(
            remote_state_root
        )

    def send(
        self,
        digest: str,
        destination_node_id: str,
        migration_id: str,
    ) -> None:
        destination = self._cluster.get_node(
            destination_node_id
        )

        local_directory = (
            self._manager.cache_directory(digest)
        )

        if not local_directory.is_dir():
            raise FileNotFoundError(
                f"Local artifact cache is missing: "
                f"{local_directory}"
            )

        remote_directory = (
            self._remote_state_root
            / "artifact-incoming"
            / migration_id
            / digest
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

        subprocess.run(
            [
                "ssh",
                *ssh_options,
                target,
                (
                    "mkdir -p "
                    f"{shlex.quote(str(remote_directory))}"
                ),
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
                f"{local_directory}/",
                f"{target}:{remote_directory}/",
            ],
            check=True,
        )
