from __future__ import annotations

import asyncio
import os

import httpx

from magellan.config.models import ClusterConfig
from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    MigrationRecord,
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

                    result = MigrationActivationResponse.model_validate(
                        response.json()
                    )
                    if os.getenv(
                        "MAGELLAN_TEST_FORCE_ACTIVATION_RESPONSE_LOSS",
                        "",
                    ).lower() in {"1", "true", "yes"}:
                        raise ActivationOutcomeUnknownError(
                            "Injected activation response loss after "
                            "destination committed"
                        )
                    return result

            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc

                if attempt < 3:
                    await asyncio.sleep(1.0)

        raise ActivationOutcomeUnknownError(
            "Destination activation outcome is unknown after retries: "
            f"{last_error}"
        )


    async def status(
        self,
        destination_node_id: str,
        migration_id: str,
    ) -> MigrationRecord | None:
        destination = self._cluster.get_node(destination_node_id)
        url = (
            f"http://{destination.internal_ip}:"
            f"{self._cluster.api_port}/migrations/{migration_id}"
        )
        timeout = httpx.Timeout(self._cluster.request_timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return MigrationRecord.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ActivationOutcomeUnknownError(
                f"Could not query migration {migration_id}: {exc}"
            ) from exc


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
