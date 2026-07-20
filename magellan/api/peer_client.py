from __future__ import annotations

import asyncio

import httpx

from magellan.config.models import ClusterConfig, NodeConfig


async def check_peer(
    client: httpx.AsyncClient,
    peer: NodeConfig,
    api_port: int,
) -> dict:
    url = f"http://{peer.internal_ip}:{api_port}/health"

    try:
        response = await client.get(url)
        response.raise_for_status()
        body = response.json()

        return {
            "node_id": peer.id,
            "reachable": True,
            "url": url,
            "health": body,
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "node_id": peer.id,
            "reachable": False,
            "url": url,
            "error": str(exc),
        }


async def check_all_peers(
    config: ClusterConfig,
    local_node_id: str,
) -> list[dict]:
    peers = [node for node in config.nodes if node.id != local_node_id]

    timeout = httpx.Timeout(config.request_timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(
            *[
                check_peer(client, peer, config.api_port)
                for peer in peers
            ]
        )
