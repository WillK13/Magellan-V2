from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from magellan.config.loader import load_cluster_config
from magellan.state.node_state import LocalNodeState

from magellan.api.peer_client import check_all_peers

CONFIG_PATH = Path(os.getenv("MAGELLAN_CONFIG", "config/cluster.json"))
NODE_ID = os.getenv("MAGELLAN_NODE_ID", "").strip()

cluster = load_cluster_config(CONFIG_PATH)

if not NODE_ID:
    raise RuntimeError("MAGELLAN_NODE_ID must be set before starting the daemon")

try:
    local_node = cluster.get_node(NODE_ID)
except KeyError as exc:
    raise RuntimeError(str(exc)) from exc

state = LocalNodeState(node_id=NODE_ID)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(
        f"[magellan] starting node={local_node.id} "
        f"zone={local_node.zone} ip={local_node.internal_ip}",
        flush=True,
    )
    yield
    print(f"[magellan] stopping node={local_node.id}", flush=True)


app = FastAPI(
    title="Magellan V2 Peer API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "node_id": local_node.id,
        "node_name": local_node.name,
        "zone": local_node.zone,
        "carbon_region": local_node.carbon_region,
        "capacity": local_node.capacity,
        "active_tasks": state.active_tasks,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/nodes/{node_id}")
async def node_details(node_id: str) -> dict:
    try:
        node = cluster.get_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return node.model_dump(mode="json")

@app.get("/peers")
async def peers() -> dict:
    results = await check_all_peers(cluster, local_node.id)

    return {
        "local_node_id": local_node.id,
        "reachable_count": sum(item["reachable"] for item in results),
        "peer_count": len(results),
        "peers": results,
    }
