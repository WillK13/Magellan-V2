from __future__ import annotations

import httpx

from magellan.artifacts.models import (
    ArtifactBinding,
    ArtifactCommitRequest,
    ArtifactCommitResponse,
    ArtifactStatusRequest,
    ArtifactStatusResponse,
)
from magellan.config.models import ClusterConfig


class ArtifactClient:
    def __init__(
        self,
        cluster: ClusterConfig,
    ) -> None:
        self._cluster = cluster

    def _base_url(self, node_id: str) -> str:
        node = self._cluster.get_node(node_id)

        return (
            f"http://{node.internal_ip}:"
            f"{self._cluster.api_port}"
        )

    async def status(
        self,
        task_id: str,
        destination_node_id: str,
        bindings: list[ArtifactBinding],
    ) -> ArtifactStatusResponse:
        timeout = httpx.Timeout(
            self._cluster.request_timeout_seconds
        )

        request = ArtifactStatusRequest(
            task_id=task_id,
            artifacts=bindings,
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.post(
                (
                    f"{self._base_url(destination_node_id)}"
                    "/artifacts/status"
                ),
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()

        return ArtifactStatusResponse.model_validate(
            response.json()
        )

    async def commit(
        self,
        destination_node_id: str,
        request: ArtifactCommitRequest,
    ) -> ArtifactCommitResponse:
        timeout = httpx.Timeout(
            connect=self._cluster.request_timeout_seconds,
            read=600,
            write=self._cluster.request_timeout_seconds,
            pool=self._cluster.request_timeout_seconds,
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.post(
                (
                    f"{self._base_url(destination_node_id)}"
                    "/artifacts/commit"
                ),
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()

        return ArtifactCommitResponse.model_validate(
            response.json()
        )
