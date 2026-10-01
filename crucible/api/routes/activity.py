from __future__ import annotations

import asyncio
import time
from typing import Any

from ... import VERSION, accelerator, clock
from ...backend import CUDA_LINUX
from ...errors import ApiError
from ...jobs.queue import JobStore
from ...protocol import API_VERSION
from ..context import AppContext, Routers
from ..proxy import chat_limit_of
from ..responses import Activity


def _activity_row(store: JobStore, job: Any) -> dict[str, Any]:
    line = store.line
    waiting = None if line is None else line.get(job.id)
    if waiting is not None and (waiting.is_call or waiting.is_session):
        return _call_row(waiting)
    row = _lane_row(store, job)
    if waiting is not None:
        row["waited_s"] = waiting.waited_s(clock.now())
        row["max_wait_s"] = waiting.max_wait_s
    return row


def _call_row(waiting: Any) -> dict[str, Any]:
    call = waiting.job
    return {
        "job_id": call.id,
        "type": call.type,
        "model": call.model,
        "status": call.status,
        "position": waiting.position,
        "progress": 0.0,
        "message": None,
        "created": waiting.submitted.isoformat(),
        "started": None,
        "client": call.client,
        "waited_s": waiting.waited_s(clock.now()),
        "max_wait_s": waiting.max_wait_s,
        "kind": waiting.kind,
    }


def _lane_row(store: JobStore, job: Any) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "type": job.type,
        "model": job.model,
        "status": job.status,
        "position": store.position(job),
        "progress": job.progress,
        "message": job.message,
        "created": job.created,
        "started": job.started,
        "client": job.client,
    }


def _unattributed(state: Any, allowance: int) -> int | None:
    if state.backend != CUDA_LINUX:
        return None
    return accelerator.unattributed_bytes(state, allowance)


def _resident_brief(resident: Any) -> dict[str, Any] | None:
    if resident is None:
        return None
    return {
        "kind": resident.kind,
        "id": resident.id,
        "since": resident.loaded_at,
        "memory_bytes_estimate": resident.memory_bytes_estimate,
    }


def _holder_rows(state: Any, owned: Any) -> list[dict[str, Any]]:
    return [
        {
            "pid": holder.pid,
            "name": holder.name,
            "bytes": holder.used_bytes,
            "owned_by_crucible": holder.pid in owned,
        }
        for holder in state.compute_apps
    ]


def _accelerator_body(ctx: AppContext, state: Any) -> dict[str, Any]:
    config, backend = ctx.config, ctx.backend
    return {
        "backend": state.backend,
        "gpu": {
            "vendor": backend.gpu.vendor,
            "name": backend.gpu.name,
            "total_bytes": state.total_bytes,
        },
        "free_bytes": state.free_bytes,
        "used_bytes": state.used_bytes,
        "desktop_allowance_bytes": config.desktop_allowance_bytes,
        "unattributed_bytes": _unattributed(state, config.desktop_allowance_bytes),
        "resident": _resident_brief(ctx.residency.resident),
        "holders": _holder_rows(state, ctx.residency.owned_pids()),
        "detail": state.detail,
    }


def _accelerator_state_handler(ctx: AppContext):
    async def accelerator_state() -> dict[str, Any]:
        """What is on the card right now and which holders are Crucible's own processes.
        Reports only; it never evicts anything.
        """
        try:
            state = await asyncio.to_thread(
                accelerator.read_state, ctx.backend.kind, ctx.config.desktop_allowance_bytes
            )
        except accelerator.ProbeError as exc:
            raise ApiError(
                503,
                "accelerator_unreadable",
                f"this server cannot read its accelerator: {exc}",
            ) from None
        return _accelerator_body(ctx, state)

    return accelerator_state


def _server_section(ctx: AppContext) -> dict[str, Any]:
    return {
        "name": ctx.config.name,
        "version": VERSION,
        "api_version": API_VERSION,
        "backend": ctx.backend.kind,
        "uptime_s": round(time.monotonic() - ctx.started_at, 3),
    }


def _resident_section(ctx: AppContext) -> dict[str, Any] | None:
    residency = ctx.residency
    resident = residency.resident
    if resident is None:
        return None
    held = ctx.settlement.held_by()
    unclaimed = ctx.settlement.unheld_since()
    return {
        **_resident_brief(resident),
        "engine_exit_code": residency.engine_exit_code,
        "reference": getattr(resident, "reference", None),
        "held_by": (None if held is None else held.to_dict()),
        "unclaimed_since": (None if unclaimed is None else unclaimed.isoformat()),
    }


def _streaming_section(ctx: AppContext) -> dict[str, Any] | None:
    session = ctx.streams.session
    if session is None:
        return None
    owner = ctx.sessions.of_stream_session(session.id)
    return {
        "session_id": session.id,
        "voice": session.voice,
        "language": session.language,
        "narrator_engine": session.narrator_engine,
        "since": session.opened_at,
        "client": session.client,
        "queue_session_id": None if owner is None else owner.id,
        "progress": None,
        **session.progress_report(),
    }


def _chat_section(ctx: AppContext) -> dict[str, Any]:
    inflight = ctx.inflight
    chat_limit, chat_limit_basis = chat_limit_of(ctx.residency)
    return {
        "in_flight": len(inflight),
        "max_in_flight": chat_limit,
        "max_in_flight_basis": chat_limit_basis,
        "rows": inflight.rows(),
    }


def _slots_section(ctx: AppContext) -> dict[str, Any]:
    store = ctx.store
    running = store.running
    return {
        "accelerated": {
            "busy": 0 if running is None else 1,
            "of": 1,
            "queue_depth": store.queue_depth,
            "accepts_work": (
                running is None
                and ctx.residency.claimed_by is None
                and len(ctx.line) == 0
                and ctx.sessions.current() is None
            ),
        },
    }


def activity_body(ctx: AppContext) -> dict[str, Any]:
    residency = ctx.residency
    store = ctx.store
    running = store.running
    queued = store.queued(calls=True)
    queue_session = ctx.sessions.current()
    return {
        "server": _server_section(ctx),
        "resident": _resident_section(ctx),
        "stopping": (
            None if residency.stopping is None else residency.stopping.to_dict()
        ),
        "warming": residency.warming,
        "claim": (
            None if residency.claimed_by is None else {"held_by": residency.claimed_by}
        ),
        "streaming": _streaming_section(ctx),
        "chat": _chat_section(ctx),
        "settings": {"writes": ctx.settings_history.rows()},
        "catalog": {"removals": ctx.removals.rows()},
        "session": (
            None if queue_session is None else ctx.sessions.state(queue_session)
        ),
        "slots": _slots_section(ctx),
        "running": [] if running is None else [_activity_row(store, running)],
        "queued": [_activity_row(store, job) for job in queued],
    }


async def _accelerator_probe(ctx: AppContext) -> dict[str, Any]:
    allowance = ctx.config.desktop_allowance_bytes
    try:
        state = await asyncio.to_thread(accelerator.read_state, ctx.backend.kind, allowance)
    except accelerator.ProbeError as exc:
        return {"error": f"{exc}"}
    return {
        "total_bytes": state.total_bytes,
        "free_bytes": state.free_bytes,
        "used_bytes": state.used_bytes,
        "unattributed_bytes": _unattributed(state, allowance),
    }


def _activity_handler(ctx: AppContext):
    async def activity(accelerator_probe: bool = False) -> dict[str, Any]:
        """What this server is doing and how far along, in one read with no job id. A
        display and a preflight, never admission; `?accelerator_probe=true` adds a live
        card probe.
        """
        body = activity_body(ctx)
        if accelerator_probe:
            body["accelerator"] = await _accelerator_probe(ctx)
        return body

    return activity


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    private.get("/accelerator")(_accelerator_state_handler(ctx))
    private.get("/activity", response_model=Activity, response_model_exclude_unset=True)(
        _activity_handler(ctx)
    )
