from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Request

from ... import API_VERSION, VERSION, accelerator
from ...backend import CUDA_LINUX
from ...errors import ApiError
from ...inflight import InFlight
from ...jobs.queue import JobStore
from ...leases import Leases
from ...settle import Settlement
from ...ttsstream import StreamManager
from ..context import AppContext, Routers
from ..proxy import _chat_limit_of


def _activity_row(store: JobStore, job: Any) -> dict[str, Any]:
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


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    @private.get("/accelerator")
    async def accelerator_state(request: Request) -> dict[str, Any]:
        """What is on the card right now and which holders are Crucible's own processes.
        Reports only; it never evicts anything.
        """
        try:
            state = await asyncio.to_thread(
                accelerator.read_state, backend.kind, config.desktop_allowance_bytes
            )
        except accelerator.ProbeError as exc:
            raise ApiError(
                503,
                "accelerator_unreadable",
                f"this server cannot read its accelerator: {exc}",
            ) from None
        owned = residency.owned_pids()
        holders = [
            {
                "pid": holder.pid,
                "name": holder.name,
                "bytes": holder.used_bytes,
                "owned_by_crucible": holder.pid in owned,
            }
            for holder in state.compute_apps
        ]
        resident = residency.resident
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
            "unattributed_bytes": (
                accelerator.unattributed_bytes(state, config.desktop_allowance_bytes)
                if state.backend == CUDA_LINUX
                else None
            ),
            "resident": (
                None
                if resident is None
                else {
                    "kind": resident.kind,
                    "id": resident.id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                }
            ),
            "holders": holders,
            "detail": state.detail,
        }

    @private.get("/activity")
    async def activity(request: Request, accelerator_probe: bool = False) -> dict[str, Any]:
        """What this server is doing and how far along, in one read with no job id. A
        display and a preflight, never admission; `?accelerator_probe=true` adds a live
        card probe.
        """
        store: JobStore = request.app.state.store
        streams: StreamManager = request.app.state.streams
        inflight: InFlight = request.app.state.inflight
        leases: Leases = request.app.state.leases
        running = store.running
        queued = store.queued()
        resident = residency.resident
        session = streams.session
        lease = leases.current()
        chat_limit, chat_limit_basis = _chat_limit_of(residency)
        settlement: Settlement = request.app.state.settlement
        held = settlement.held_by()
        unclaimed = settlement.unheld_since()

        body: dict[str, Any] = {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
                "backend": backend.kind,
                "uptime_s": round(time.monotonic() - request.app.state.started_at, 3),
            },
            "resident": (
                None
                if resident is None
                else {
                    "kind": resident.kind,
                    "id": resident.id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                    "engine_exit_code": residency.engine_exit_code,
                    "reference": getattr(resident, "reference", None),
                    "held_by": (None if held is None else held.to_dict()),
                    "unclaimed_since": (
                        None if unclaimed is None else unclaimed.isoformat()
                    ),
                }
            ),
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
            "warming": residency.warming,
            "claim": (
                None
                if residency.claimed_by is None
                else {"held_by": residency.claimed_by}
            ),
            "streaming": (
                None
                if session is None
                else {
                    "session_id": session.id,
                    "voice": session.voice,
                    "language": session.language,
                    "narrator_engine": session.narrator_engine,
                    "since": session.opened_at,
                    "client": session.client,
                    "progress": None,
                    **session.progress_report(),
                }
            ),
            "chat": {
                "in_flight": len(inflight),
                "max_in_flight": chat_limit,
                "max_in_flight_basis": chat_limit_basis,
                "rows": inflight.rows(),
            },
            "settings": {"writes": request.app.state.settings_history.rows()},
            "catalog": {"removals": request.app.state.removals.rows()},
            "lease": None if lease is None else lease.to_dict(),
            "slots": {
                "accelerated": {
                    "busy": 0 if running is None else 1,
                    "of": 1,
                    "queue_depth": store.queue_depth,
                    "accepts_work": running is None and residency.claimed_by is None,
                },
            },
            "running": [] if running is None else [_activity_row(store, running)],
            "queued": [_activity_row(store, job) for job in queued],
        }

        if accelerator_probe:
            try:
                state = await asyncio.to_thread(
                    accelerator.read_state, backend.kind, config.desktop_allowance_bytes
                )
            except accelerator.ProbeError as exc:
                body["accelerator"] = {"error": f"{exc}"}
            else:
                body["accelerator"] = {
                    "total_bytes": state.total_bytes,
                    "free_bytes": state.free_bytes,
                    "used_bytes": state.used_bytes,
                    "unattributed_bytes": (
                        accelerator.unattributed_bytes(
                            state, config.desktop_allowance_bytes
                        )
                        if state.backend == CUDA_LINUX
                        else None
                    ),
                }
        return body
