from __future__ import annotations

import asyncio

import httpx

from magellan.config.models import ClusterConfig
from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    OwnershipUpdate,
)


class ActivationOutcomeUnknownError(RuntimeError):
    pass


class MigrationClient:
    def __init__(
        self,
        cluster: ClusterConfig,
        activation_timeout_seconds: float,
    ) -> None:
        self._cluster = cluster
        self._activation_timeout_seconds = (
            activation_timeout_seconds
        )

    async def activate(
        self,
        request: MigrationActivationRequest,
    ) -> MigrationActivationResponse:
        destination = self._cluster.get_node(
            request.destination_node_id
        )

        url = (
            f"http://{destination.internal_ip}:"
            f"{self._cluster.api_port}/migrations/activate"
        )

        timeout = httpx.Timeout(
            connect=self._cluster.request_timeout_seconds,
            read=self._activation_timeout_seconds,
            write=self._cluster.request_timeout_seconds,
            pool=self._cluster.request_timeout_seconds,
        )

        last_error: Exception | None = None

        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout
                ) as client:
                    response = await client.post(
                        url,
                        json=request.model_dump(mode="json"),
                    )
                    response.raise_for_status()

                    return (
                        MigrationActivationResponse.model_validate(
                            response.json()
                        )
                    )

            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc

                if attempt < 3:
                    await asyncio.sleep(1.0)

        raise ActivationOutcomeUnknownError(
            "Destination activation outcome is unknown after retries: "
            f"{last_error}"
        )


class OwnershipBroadcaster:
    def __init__(
        self,
        cluster: ClusterConfig,
        local_node_id: str,
    ) -> None:
        self._cluster = cluster
        self._local_node_id = local_node_id

    async def broadcast(
        self,
        update: OwnershipUpdate,
    ) -> None:
        timeout = httpx.Timeout(
            self._cluster.request_timeout_seconds
        )

        async def send(node) -> None:
            if node.id == self._local_node_id:
                return

            url = (
                f"http://{node.internal_ip}:"
                f"{self._cluster.api_port}/ownership"
            )

            try:
                async with httpx.AsyncClient(
                    timeout=timeout
                ) as client:
                    response = await client.post(
                        url,
                        json=update.model_dump(mode="json"),
                    )
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                print(
                    f"[ownership-warning] node={node.id} "
                    f"error={exc}",
                    flush=True,
                )

        await asyncio.gather(
            *(send(node) for node in self._cluster.nodes)
        )
