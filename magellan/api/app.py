from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from magellan.api.peer_client import check_all_peers
from magellan.bidding.models import (
    BidRecord,
    BidRequest,
    BidStatus,
)
from magellan.daemon.context import build_daemon_context


context = build_daemon_context()


@asynccontextmanager
async def lifespan(_: FastAPI):
    stop_event = asyncio.Event()

    background_tasks = [
        asyncio.create_task(
            context.bid_arbiter.run(stop_event),
            name="magellan-bid-arbiter",
        ),
        asyncio.create_task(
            context.scheduler_service.run(stop_event),
            name="magellan-scheduler",
        ),
    ]

    print(
        f"[magellan] started node={context.local_node.id} "
        f"zone={context.local_node.zone} "
        f"ip={context.local_node.internal_ip} "
        f"owned_tasks="
        f"{context.registry.count_owned(context.local_node.id)}",
        flush=True,
    )

    try:
        yield
    finally:
        stop_event.set()

        await asyncio.gather(
            *background_tasks,
            return_exceptions=True,
        )

        print(
            f"[magellan] stopped node="
            f"{context.local_node.id}",
            flush=True,
        )


app = FastAPI(
    title="Magellan V2 Peer API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    records = await context.bid_store.list_records()

    return {
        "status": "ok",
        "node_id": context.local_node.id,
        "node_name": context.local_node.name,
        "vm_name": context.local_node.vm_name,
        "zone": context.local_node.zone,
        "internal_ip": str(context.local_node.internal_ip),
        "carbon_region": context.local_node.carbon_region,
        "capacity": context.local_node.capacity,
        "owned_task_count": context.registry.count_owned(
            context.local_node.id
        ),
        "pending_bid_count": sum(
            record.status == BidStatus.PENDING
            for record in records
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/nodes/{node_id}")
async def node_details(node_id: str) -> dict:
    try:
        node = context.cluster.get_node(node_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return node.model_dump(mode="json")


@app.get("/peers")
async def peers() -> dict:
    results = await check_all_peers(
        context.cluster,
        context.local_node.id,
    )

    return {
        "local_node_id": context.local_node.id,
        "reachable_count": sum(
            item["reachable"]
            for item in results
        ),
        "peer_count": len(results),
        "peers": results,
    }


@app.get("/tasks")
async def tasks() -> dict:
    return {
        "local_node_id": context.local_node.id,
        "owned_tasks": [
            task.model_dump(mode="json")
            for task in context.registry.owned_tasks(
                context.local_node.id
            )
        ],
        "known_tasks": [
            task.model_dump(mode="json")
            for task in context.registry.all_tasks()
        ],
    }


@app.post("/bids", response_model=BidRecord)
async def submit_bid(
    request: BidRequest,
) -> BidRecord:
    if (
        request.destination_node_id
        != context.local_node.id
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Bid destination is "
                f"{request.destination_node_id}, "
                f"but this daemon is "
                f"{context.local_node.id}"
            ),
        )

    try:
        context.cluster.get_node(
            request.source_node_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    record = await context.bid_store.submit(request)

    print(
        f"[bid-received] node={context.local_node.id} "
        f"bid={record.bid_id} "
        f"task={record.task_id} "
        f"source={record.source_node_id} "
        f"score={record.candidate.score:.6f}",
        flush=True,
    )

    return record


@app.get("/bids", response_model=list[BidRecord])
async def list_bids() -> list[BidRecord]:
    return await context.bid_store.list_records()


@app.get(
    "/bids/{bid_id}",
    response_model=BidRecord,
)
async def get_bid(bid_id: str) -> BidRecord:
    record = await context.bid_store.get(bid_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown bid: {bid_id}",
        )

    return record
