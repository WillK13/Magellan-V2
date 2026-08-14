from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from magellan.api.peer_client import check_all_peers
from magellan.bidding.models import (
    BidRecord,
    BidRequest,
    BidStatus,
)
from magellan.daemon.context import build_daemon_context
from magellan.policy.models import AdaptiveTaskPolicyState
from magellan.telemetry.models import (
    EdgeTelemetrySampleRequest,
    EdgeTelemetryView,
    MigrationCalibrationView,
    TaskTelemetryView,
)

from magellan.migration.models import (
    MigrationActivationRequest,
    MigrationActivationResponse,
    MigrationRecord,
    OwnershipUpdate,
)

from magellan.reconciliation.models import OwnershipSnapshot


from magellan.submission.models import (
    TaskCatalogSnapshot,
    TaskDefinitionRecord,
    TaskDefinitionSubmission,
    TaskRunSubmission,
    TaskRunView,
)

from magellan.artifacts.models import (
    ArtifactCommitRequest,
    ArtifactCommitResponse,
    ArtifactStatusRequest,
    ArtifactStatusResponse,
)

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
        asyncio.create_task(
            context.recovery_service.run(stop_event),
            name="magellan-recovery",
        ),
        asyncio.create_task(
            context.pause_service.run(stop_event),
            name="magellan-pause",
        ),
        asyncio.create_task(
            context.telemetry_service.run(stop_event),
            name="magellan-telemetry",
        ),
        asyncio.create_task(
            context.accounting_service.run(stop_event),
            name="magellan-accounting",
        ),
        asyncio.create_task(
            context.reconciliation_service.run(stop_event),
            name="magellan-reconciliation",
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
    version="1.1.0",
    lifespan=lifespan,
)


@app.post("/runtime/reconcile")
async def reconcile_runtime() -> dict:
    """Run one operator-triggered local runtime reconciliation pass.

    This does not run the scheduler or select an action. It is useful for
    measurement/operations workflows that need to finalize a process that has
    already exited without waiting for the next scheduling epoch.
    """
    events = await asyncio.to_thread(context.runtime.reconcile)
    return {
        "node_id": context.local_node.id,
        "events": [
            {
                "task_id": event.task_id,
                "status": event.status.value,
                "exit_code": event.exit_code,
                "error": event.error,
            }
            for event in events
        ],
    }


@app.get("/health")
async def health() -> dict:
    records = await context.bid_store.list_records()
    active_reservations = (
        await context.bid_store.active_reservation_count()
    )
    auction_status = await context.bid_arbiter.status()

    return {
        "status": "ok",
        "node_id": context.local_node.id,
        "node_name": context.local_node.name,
        "vm_name": context.local_node.vm_name,
        "zone": context.local_node.zone,
        "internal_ip": str(context.local_node.internal_ip),
        "carbon_region": context.local_node.carbon_region,
        "carbon_metric": context.carbon_store.carbon_metric.value,
        "carbon_column": context.carbon_store.carbon_column,
        "capacity": context.local_node.capacity,
        "epoch_seconds": context.cluster.epoch_seconds,
        "auction_strategy": auction_status["strategy"],
        "available_task_slots": auction_status[
            "available_task_slots"
        ],
        "resource_capacity": auction_status[
            "resource_capacity"
        ],
        "owned_task_count": context.registry.count_owned(
            context.local_node.id
        ),
        "pending_bid_count": sum(
            record.status == BidStatus.PENDING
            for record in records
        ),
        "active_reservation_count": active_reservations,
        "completed_task_count": sum(
            item["state"]["status"] == "completed"
            for item in context.registry.summaries()
        ),
        "paused_task_count": sum(
            item["state"]["status"] == "paused"
            for item in context.registry.summaries()
        ),
        "migration_record_count": len(
            context.migration_journal.list_records()
        ),
        "task_definition_count": len(
            context.task_catalog.list_definitions()
        ),
        "task_run_count": len(
            context.task_catalog.list_runs()
        ),
        "last_reconciliation_at_utc": (
            context.reconciliation_service.last_completed_at_utc
        ),
        "last_reconciliation_updates": (
            context.reconciliation_service.last_applied_updates
        ),
        "telemetry_task_record_count": len(
            context.telemetry_store.list_task_records()
        ),
        "telemetry_edge_record_count": len(
            context.telemetry_store.list_edge_records()
        ),
        "telemetry_calibration_count": len(
            context.telemetry_store.list_calibrations()
        ),
        "telemetry_state_file": str(context.telemetry_store.path),
        "adaptive_policy_enabled": context.policy.adaptive.enabled,
        "adaptive_policy_task_count": len(
            context.adaptive_policy_store.list_states()
        ),
        "adaptive_policy_state_file": str(
            context.adaptive_policy_store.path
        ),
        "runtime_adapters": ["command", "dendro", "python_module"],
        "carbon_forecast_enabled": context.policy.carbon_forecast.enabled,
        "carbon_forecast_provider": context.policy.carbon_forecast.provider,
        "pause_candidate_idle_seconds": (
            context.policy.pause.idle_candidates()
        ),
        "configured_architecture": (
            context.local_node.capabilities.architecture
        ),
        "observed_architecture": (
            context.observed_capabilities.architecture
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/capabilities")
async def capabilities() -> dict:
    configured = context.local_node.capabilities
    observed = context.observed_capabilities
    drift: list[str] = []
    if (
        configured.architecture is not None
        and observed.architecture != configured.architecture
    ):
        drift.append(
            f"configured architecture {configured.architecture}; "
            f"observed {observed.architecture}"
        )
    if (
        configured.operating_system is not None
        and observed.operating_system != configured.operating_system
    ):
        drift.append(
            f"configured operating system {configured.operating_system}; "
            f"observed {observed.operating_system}"
        )
    missing_commands = configured.commands - observed.commands
    if missing_commands:
        drift.append(
            f"configured commands missing locally: {sorted(missing_commands)}"
        )
    missing_features = configured.features - observed.features
    if missing_features:
        drift.append(
            f"configured features missing locally: {sorted(missing_features)}"
        )
    for runtime, configured_version in configured.runtimes.items():
        observed_version = observed.runtimes.get(runtime)
        if observed_version is None:
            drift.append(f"configured runtime missing locally: {runtime}")
        elif not observed_version.startswith(configured_version):
            drift.append(
                f"configured {runtime} version {configured_version}; "
                f"observed {observed_version}"
            )
    return {
        "node_id": context.local_node.id,
        "configured": configured.model_dump(mode="json"),
        "observed": observed.model_dump(mode="json"),
        "runtime_adapters": ["command", "dendro", "python_module"],
        "carbon_forecast_enabled": context.policy.carbon_forecast.enabled,
        "carbon_forecast_provider": context.policy.carbon_forecast.provider,
        "pause_candidate_idle_seconds": (
            context.policy.pause.idle_candidates()
        ),
        "drift": drift,
        "ready": not drift,
    }


@app.get("/carbon/forecast/{node_id}")
async def carbon_forecast(
    node_id: str,
    horizon_seconds: float | None = None,
    start_offset_seconds: float = 0.0,
) -> dict:
    try:
        context.cluster.get_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if start_offset_seconds < 0:
        raise HTTPException(
            status_code=400,
            detail="start_offset_seconds must be non-negative",
        )
    duration = (
        context.policy.carbon_forecast.horizon_seconds
        if horizon_seconds is None
        else horizon_seconds
    )
    if duration < 0:
        raise HTTPException(
            status_code=400,
            detail="horizon_seconds must be non-negative",
        )
    observed_at = context.clock.now()
    forecast_start = observed_at + timedelta(
        seconds=start_offset_seconds
    )
    try:
        forecast = context.carbon_store.forecast(
            node_id=node_id,
            observed_at_utc=observed_at,
            forecast_start_utc=forecast_start,
            duration_seconds=duration,
            policy=context.policy.carbon_forecast,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return forecast.model_dump(mode="json")


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
        "tasks": context.registry.summaries(),
    }


@app.post(
    "/task-definitions",
    response_model=TaskDefinitionRecord,
)
async def submit_task_definition(
    submission: TaskDefinitionSubmission,
) -> TaskDefinitionRecord:
    try:
        return await asyncio.to_thread(
            context.submission_service.submit_definition,
            submission,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(
    "/task-definitions",
    response_model=list[TaskDefinitionRecord],
)
async def list_task_definitions() -> list[TaskDefinitionRecord]:
    return context.task_catalog.list_definitions()


@app.get(
    "/task-definitions/{definition_id}",
    response_model=TaskDefinitionRecord,
)
async def get_task_definition(
    definition_id: str,
    revision: int | None = None,
) -> TaskDefinitionRecord:
    try:
        return context.task_catalog.get_definition(definition_id, revision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/task-runs", response_model=TaskRunView)
async def create_task_run(submission: TaskRunSubmission) -> TaskRunView:
    try:
        return await asyncio.to_thread(
            context.submission_service.create_run,
            submission,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/task-runs", response_model=list[TaskRunView])
async def list_task_runs() -> list[TaskRunView]:
    return context.submission_service.list_runs()


@app.get("/task-runs/{run_id}", response_model=TaskRunView)
async def get_task_run(run_id: str) -> TaskRunView:
    try:
        return context.submission_service.view_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/catalog/snapshot", response_model=TaskCatalogSnapshot)
async def catalog_snapshot() -> TaskCatalogSnapshot:
    return context.task_catalog.snapshot()

@app.get("/telemetry")
async def telemetry_summary() -> dict:
    task_views = []
    for state in context.registry.all_states():
        configured_power = context.registry.get_definition(
            state.task_id
        ).profile.power_kw
        task_views.append(
            context.telemetry_store.task_view(
                state.task_id,
                configured_power,
                context.policy.telemetry.task_stale_after_seconds,
            ).model_dump(mode="json")
        )

    edge_views = []
    for node in context.cluster.nodes:
        if node.id == context.local_node.id:
            continue
        configured = context.cluster.get_edge_override(
            context.local_node.id, node.id
        )
        bandwidth = (
            configured.bandwidth_mbps
            if configured is not None
            else context.cluster.default_bandwidth_mbps
        )
        latency = (
            configured.latency_ms
            if configured is not None
            else context.cluster.default_latency_ms
        )
        edge_views.append(
            context.telemetry_store.edge_view(
                context.local_node.id,
                node.id,
                bandwidth,
                latency,
                context.policy.telemetry.edge_stale_after_seconds,
            ).model_dump(mode="json")
        )

    return {
        "node_id": context.local_node.id,
        "task_telemetry": task_views,
        "edge_telemetry": edge_views,
        "calibration": [
            context.telemetry_store.calibration_view(
                record.source_node_id,
                record.destination_node_id,
                context.policy.telemetry.calibration_stale_after_seconds,
            ).model_dump(mode="json")
            for record in context.telemetry_store.list_calibrations()
        ],
    }


@app.get("/telemetry/tasks", response_model=list[TaskTelemetryView])
async def task_telemetry() -> list[TaskTelemetryView]:
    return [
        context.telemetry_store.task_view(
            state.task_id,
            context.registry.get_definition(state.task_id).profile.power_kw,
            context.policy.telemetry.task_stale_after_seconds,
        )
        for state in context.registry.all_states()
    ]


@app.get("/telemetry/tasks/{task_id}", response_model=TaskTelemetryView)
async def task_telemetry_detail(task_id: str) -> TaskTelemetryView:
    try:
        definition = context.registry.get_definition(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return context.telemetry_store.task_view(
        task_id,
        definition.profile.power_kw,
        context.policy.telemetry.task_stale_after_seconds,
    )


@app.get("/telemetry/edges", response_model=list[EdgeTelemetryView])
async def edge_telemetry() -> list[EdgeTelemetryView]:
    result = []
    for node in context.cluster.nodes:
        if node.id == context.local_node.id:
            continue
        configured = context.cluster.get_edge_override(
            context.local_node.id, node.id
        )
        result.append(
            context.telemetry_store.edge_view(
                context.local_node.id,
                node.id,
                (configured.bandwidth_mbps if configured else context.cluster.default_bandwidth_mbps),
                (configured.latency_ms if configured else context.cluster.default_latency_ms),
                context.policy.telemetry.edge_stale_after_seconds,
            )
        )
    return result


@app.post(
    "/telemetry/edges/{destination_node_id}/sample",
    response_model=EdgeTelemetryView,
)
async def record_edge_telemetry_sample(
    destination_node_id: str,
    sample: EdgeTelemetrySampleRequest,
) -> EdgeTelemetryView:
    """Record an operator-measured edge sample for experiment preflight."""
    try:
        context.cluster.get_node(destination_node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if destination_node_id == context.local_node.id:
        raise HTTPException(
            status_code=400, detail="Destination must be a peer"
        )
    if sample.latency_ms is None and sample.transfer_bytes is None:
        raise HTTPException(
            status_code=400, detail="At least one sample is required"
        )
    if (sample.transfer_bytes is None) != (
        sample.transfer_duration_seconds is None
    ):
        raise HTTPException(
            status_code=400,
            detail="transfer_bytes and transfer_duration_seconds must be provided together",
        )

    if sample.latency_ms is not None:
        context.telemetry_store.record_latency(
            context.local_node.id, destination_node_id, sample.latency_ms
        )
    if (
        sample.transfer_bytes is not None
        and sample.transfer_duration_seconds is not None
    ):
        context.telemetry_store.record_transfer(
            context.local_node.id,
            destination_node_id,
            sample.transfer_bytes,
            sample.transfer_duration_seconds,
        )

    configured = context.cluster.get_edge_override(
        context.local_node.id, destination_node_id
    )
    return context.telemetry_store.edge_view(
        context.local_node.id,
        destination_node_id,
        (
            configured.bandwidth_mbps
            if configured
            else context.cluster.default_bandwidth_mbps
        ),
        (
            configured.latency_ms
            if configured
            else context.cluster.default_latency_ms
        ),
        context.policy.telemetry.edge_stale_after_seconds,
    )


@app.get(
    "/telemetry/edges/{destination_node_id}",
    response_model=EdgeTelemetryView,
)
async def edge_telemetry_detail(
    destination_node_id: str,
) -> EdgeTelemetryView:
    try:
        context.cluster.get_node(destination_node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if destination_node_id == context.local_node.id:
        raise HTTPException(status_code=400, detail="Destination must be a peer")
    configured = context.cluster.get_edge_override(
        context.local_node.id, destination_node_id
    )
    return context.telemetry_store.edge_view(
        context.local_node.id,
        destination_node_id,
        (configured.bandwidth_mbps if configured else context.cluster.default_bandwidth_mbps),
        (configured.latency_ms if configured else context.cluster.default_latency_ms),
        context.policy.telemetry.edge_stale_after_seconds,
    )


@app.get(
    "/telemetry/calibration",
    response_model=list[MigrationCalibrationView],
)
async def telemetry_calibration() -> list[MigrationCalibrationView]:
    return [
        context.telemetry_store.calibration_view(
            record.source_node_id,
            record.destination_node_id,
            context.policy.telemetry.calibration_stale_after_seconds,
        )
        for record in context.telemetry_store.list_calibrations()
    ]


@app.get("/policy")
async def policy_summary() -> dict:
    return {
        "node_id": context.local_node.id,
        "baseline_weights": context.policy.weights.model_dump(),
        "adaptive": context.policy.adaptive.model_dump(),
        "carbon_forecast": context.policy.carbon_forecast.model_dump(),
        "pause_candidate_idle_seconds": (
            context.policy.pause.idle_candidates()
        ),
        "task_states": [
            state.model_dump(mode="json")
            for state in context.adaptive_policy_store.list_states()
        ],
        "state_file": str(context.adaptive_policy_store.path),
    }


@app.get(
    "/policy/tasks",
    response_model=list[AdaptiveTaskPolicyState],
)
async def policy_task_states() -> list[AdaptiveTaskPolicyState]:
    return context.adaptive_policy_store.list_states()


@app.get(
    "/policy/tasks/{task_id}",
    response_model=AdaptiveTaskPolicyState,
)
async def policy_task_state(task_id: str) -> AdaptiveTaskPolicyState:
    state = context.adaptive_policy_store.get(task_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No adaptive policy state for task: {task_id}",
        )
    return state


@app.post("/policy/tasks/{task_id}/reset")
async def reset_policy_task_state(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "deleted": context.adaptive_policy_service.reset(task_id),
    }


@app.get("/experiment/events/status")
async def experiment_event_status() -> dict:
    return {
        "node_id": context.local_node.id,
        "last_sequence": context.experiment_journal.last_sequence,
        "path": str(context.experiment_journal.path),
        "carbon_metric": context.carbon_store.carbon_metric.value,
        "carbon_column": context.carbon_store.carbon_column,
    }


@app.get("/experiment/events")
async def experiment_events(
    after_sequence: int = 0,
    task_id: str | None = None,
    event_type: str | None = None,
    limit: int = 10_000,
) -> dict:
    if after_sequence < 0:
        raise HTTPException(status_code=400, detail="after_sequence must be non-negative")
    if limit < 1 or limit > 100_000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100000")
    events = context.experiment_journal.list_events(
        after_sequence=after_sequence,
        task_id=task_id,
        event_type=event_type,
        limit=limit,
    )
    return {
        "node_id": context.local_node.id,
        "last_sequence": context.experiment_journal.last_sequence,
        "events": [event.model_dump(mode="json") for event in events],
    }


@app.get("/auction/status")
async def auction_status() -> dict:
    return await context.bid_arbiter.status()


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
        f"[task-bid-received] node={context.local_node.id} "
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


@app.post(
    "/bids/{bid_id}/renew",
    response_model=BidRecord,
)
async def renew_bid(bid_id: str) -> BidRecord:
    try:
        return await context.bid_store.renew(bid_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post(
    "/bids/{bid_id}/cancel",
    response_model=BidRecord,
)
async def cancel_bid(
    bid_id: str,
    reason: str = "Source cancelled reservation",
) -> BidRecord:
    try:
        return await context.bid_store.cancel(
            bid_id,
            reason=reason,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post("/task-runs/{run_id}/start")
async def start_task_run(run_id: str) -> dict:
    return await start_task(run_id)


@app.post("/task-runs/{run_id}/stop")
async def stop_task_run(run_id: str) -> dict:
    return await stop_task(run_id)


@app.post("/tasks/{task_id}/start")
async def start_task(task_id: str) -> dict:
    try:
        state = await asyncio.to_thread(
            context.runtime.start,
            task_id,
        )
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return state.model_dump(mode="json")


@app.post("/tasks/{task_id}/pause")
async def pause_task(
    task_id: str,
    idle_seconds: float | None = None,
) -> dict:
    try:
        return await context.scheduler_service.request_pause(
            task_id=task_id,
            idle_seconds=idle_seconds,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str) -> dict:
    try:
        return await context.scheduler_service.request_resume(
            task_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post(
    "/tasks/{task_id}/migrate/{destination_node_id}"
)
async def migrate_task(
    task_id: str,
    destination_node_id: str,
) -> dict:
    try:
        return await context.scheduler_service.request_migration(
            task_id=task_id,
            destination_node_id=destination_node_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str) -> dict:
    try:
        await asyncio.to_thread(
            context.accounting_service.settle_task,
            task_id,
        )
        state = await asyncio.to_thread(
            context.runtime.stop,
            task_id,
        )
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return state.model_dump(mode="json")


@app.post(
    "/migrations/activate",
    response_model=MigrationActivationResponse,
)
async def activate_migration(
    request: MigrationActivationRequest,
) -> MigrationActivationResponse:
    return await context.migration_service.activate_incoming(
        request
    )


@app.get("/migrations", response_model=list[MigrationRecord])
async def list_migrations() -> list[MigrationRecord]:
    return context.migration_journal.list_records()


@app.get(
    "/migrations/{migration_id}",
    response_model=MigrationRecord,
)
async def get_migration(migration_id: str) -> MigrationRecord:
    record = context.migration_journal.get(migration_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown migration: {migration_id}",
        )
    return record


@app.get(
    "/ownership/snapshot",
    response_model=OwnershipSnapshot,
)
async def ownership_snapshot() -> OwnershipSnapshot:
    updates = context.registry.ownership_updates()
    for update in updates:
        update.adaptive_policy = context.adaptive_policy_store.get(
            update.task_id
        )
    return OwnershipSnapshot(
        reporting_node_id=context.local_node.id,
        updates=updates,
    )


@app.post("/ownership")
async def ownership_update(
    update: OwnershipUpdate,
) -> dict:
    try:
        applied = context.registry.apply_ownership(
            task_id=update.task_id,
            owner_node_id=update.owner_node_id,
            generation=update.generation,
            migration_id=update.last_migration_id,
            migration_at_utc=update.migration_at_utc,
            status=update.status,
            completed_at_utc=update.completed_at_utc,
            final_output_manifest_sha256=(
                update.final_output_manifest_sha256
            ),
            final_output_bytes=update.final_output_bytes,
            accounting=update.accounting,
        )

        if update.artifact_digests:
            context.registry.set_artifact_digests(
                update.task_id,
                update.artifact_digests,
            )
        if applied and update.adaptive_policy is not None:
            context.adaptive_policy_store.merge(
                update.adaptive_policy
            )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return {
        "applied": applied,
        "task_id": update.task_id,
        "owner_node_id": update.owner_node_id,
        "generation": update.generation,
    }

@app.get("/tasks/{task_id}/outputs")
async def task_outputs(task_id: str) -> dict:
    try:
        state = context.registry.get_state(task_id)

        if state.owner_node_id != context.local_node.id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Final outputs are owned by "
                    f"{state.owner_node_id}"
                ),
            )

        manifest = context.completion_manager.load_manifest(
            task_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return manifest.model_dump(mode="json")


@app.get("/tasks/{task_id}/outputs/{relative_path:path}")
async def task_output_file(
    task_id: str,
    relative_path: str,
):
    try:
        state = context.registry.get_state(task_id)

        if state.owner_node_id != context.local_node.id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Final outputs are owned by "
                    f"{state.owner_node_id}"
                ),
            )

        path = context.completion_manager.resolve_output_file(
            task_id,
            relative_path,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return FileResponse(path)


@app.post(
    "/artifacts/status",
    response_model=ArtifactStatusResponse,
)
async def artifact_status(
    request: ArtifactStatusRequest,
) -> ArtifactStatusResponse:
    present: list[str] = []
    missing: list[str] = []

    for artifact in request.artifacts:
        if context.artifact_manager.has_artifact(
            artifact.digest
        ):
            present.append(artifact.digest)
        else:
            missing.append(artifact.digest)

    return ArtifactStatusResponse(
        task_id=request.task_id,
        present_digests=present,
        missing_digests=missing,
    )


@app.post(
    "/artifacts/commit",
    response_model=ArtifactCommitResponse,
)
async def commit_artifact(
    request: ArtifactCommitRequest,
) -> ArtifactCommitResponse:
    try:
        manifest = await asyncio.to_thread(
            context.artifact_manager.commit_incoming,
            request.migration_id,
            request.digest,
        )

        print(
            f"[artifact-committed] "
            f"task={request.task_id} "
            f"artifact={request.artifact_id} "
            f"digest={request.digest} "
            f"bytes={manifest.size_bytes}",
            flush=True,
        )

        return ArtifactCommitResponse(
            artifact_id=request.artifact_id,
            digest=request.digest,
            committed=True,
            size_bytes=manifest.size_bytes,
        )

    except Exception as exc:
        return ArtifactCommitResponse(
            artifact_id=request.artifact_id,
            digest=request.digest,
            committed=False,
            error=f"{type(exc).__name__}: {exc}",
        )