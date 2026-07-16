from __future__ import annotations

import asyncio

from magellan.artifacts.client import ArtifactClient
from magellan.artifacts.manager import ArtifactManager
from magellan.artifacts.models import (
    ArtifactBinding,
    ArtifactCommitRequest,
)
from magellan.artifacts.transfer import (
    RsyncArtifactTransfer,
)


class ArtifactPrefetchService:
    def __init__(
        self,
        manager: ArtifactManager,
        client: ArtifactClient,
        transfer: RsyncArtifactTransfer,
    ) -> None:
        self._manager = manager
        self._client = client
        self._transfer = transfer

    async def missing_bytes(
        self,
        task_id: str,
        destination_node_id: str,
    ) -> int:
        bindings = await asyncio.to_thread(
            self._manager.ensure_task_artifacts,
            task_id,
        )

        try:
            status = await self._client.status(
                task_id=task_id,
                destination_node_id=destination_node_id,
                bindings=bindings,
            )
        except Exception as exc:
            print(
                f"[artifact-status-warning] "
                f"task={task_id} "
                f"destination={destination_node_id} "
                f"error={exc}",
                flush=True,
            )

            # Conservative fallback: assume everything must move.
            return sum(
                binding.size_bytes
                for binding in bindings
            )

        missing = set(status.missing_digests)

        return sum(
            binding.size_bytes
            for binding in bindings
            if binding.digest in missing
        )

    async def prefetch(
        self,
        task_id: str,
        destination_node_id: str,
        migration_id: str,
    ) -> list[ArtifactBinding]:
        bindings = await asyncio.to_thread(
            self._manager.ensure_task_artifacts,
            task_id,
        )

        status = await self._client.status(
            task_id=task_id,
            destination_node_id=destination_node_id,
            bindings=bindings,
        )

        missing = set(status.missing_digests)

        print(
            f"[prefetch-start] task={task_id} "
            f"destination={destination_node_id} "
            f"missing_artifacts={len(missing)} "
            f"missing_bytes="
            f"{sum(b.size_bytes for b in bindings if b.digest in missing)}",
            flush=True,
        )

        for binding in bindings:
            if binding.digest not in missing:
                continue

            await asyncio.to_thread(
                self._transfer.send,
                binding.digest,
                destination_node_id,
                migration_id,
            )

            result = await self._client.commit(
                destination_node_id=destination_node_id,
                request=ArtifactCommitRequest(
                    migration_id=migration_id,
                    task_id=task_id,
                    artifact_id=binding.artifact_id,
                    digest=binding.digest,
                ),
            )

            if not result.committed:
                raise RuntimeError(
                    result.error
                    or (
                        "Destination failed to commit "
                        f"artifact {binding.artifact_id}"
                    )
                )

            print(
                f"[artifact-prefetched] "
                f"task={task_id} "
                f"artifact={binding.artifact_id} "
                f"bytes={result.size_bytes}",
                flush=True,
            )

        print(
            f"[prefetch-complete] task={task_id} "
            f"destination={destination_node_id}",
            flush=True,
        )

        return bindings
