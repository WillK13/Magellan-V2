from __future__ import annotations

import asyncio
import httpx

from magellan.config.models import ClusterConfig
from magellan.submission.models import TaskCatalogSnapshot


class TaskCatalogClient:
    def __init__(self, cluster: ClusterConfig, local_node_id: str) -> None:
        self._cluster = cluster
        self._local_node_id = local_node_id

    async def fetch_all(self) -> list[TaskCatalogSnapshot]:
        timeout = httpx.Timeout(self._cluster.request_timeout_seconds)

        async def fetch(node) -> TaskCatalogSnapshot | None:
            if node.id == self._local_node_id:
                return None
            url = (
                f"http://{node.internal_ip}:"
                f"{self._cluster.api_port}/catalog/snapshot"
            )
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return TaskCatalogSnapshot.model_validate(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                print(
                    f"[catalog-peer-warning] node={node.id} error={exc}",
                    flush=True,
                )
                return None

        results = await asyncio.gather(
            *(fetch(node) for node in self._cluster.nodes)
        )
        return [item for item in results if item is not None]
