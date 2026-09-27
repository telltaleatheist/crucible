from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...errors import ApiError
from ...installonsubmit import INSTALLABLE_REFUSALS, InstallOnSubmit
from ...jobs import resolve, resolve_model
from ...jobs.base import Job, validate_member_name
from ...jobs.queue import JobStore
from ...journal import InputDigest
from ...leases import CARD_EFFECTS, Leases
from ..caller import client_agent
from ..context import AppContext, Routers
from ..inputs import (
    _input_digests,
    _journal_identity,
    _materialise_inputs,
    _refuse_resume_without_a_journal,
)
from ..schemas import JobCreate
from ..sse import _event_stream, _last_event_id
from ..upstream import _refuse_lease_on_an_upstream


UPLOAD_CHUNK = 1024 * 1024


_JOB_STATE_KEYS: frozenset[str] = frozenset(
    {
        "job_id",
        "type",
        "model",
        "status",
        "progress",
        "position",
        "error",
        "artifacts",
        "created",
        "started",
        "finished",
        "client_ref",
        "interrupted_at",
        "held_by",
        "held_since",
        "chunks_done",
        "chunks_total",
        "chunk_at",
        "resume_id",
        "resumed",
    }
)


def _job_state(store: JobStore, job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "type": job.type,
        "model": job.model,
        "status": job.status,
        "progress": job.progress,
        "position": store.position(job),
        "error": job.error,
        "artifacts": list(job.artifacts),
        "created": job.created,
        "started": job.started,
        "finished": job.finished,
        "client_ref": job.client_ref,
        "interrupted_at": job.interrupted_at,
        "held_by": job.held_by,
        "held_since": job.held_since,
        "chunks_done": sorted(job.chunks_done),
        "chunks_total": job.chunks_total,
        "chunk_at": job.chunk_at,
        "resume_id": job.resume_id,
        "resumed": job.resumed,
        **{k: v for k, v in job.done_extra.items() if k not in _JOB_STATE_KEYS},
    }


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, residency, decide_here = ctx.config, ctx.residency, ctx.decide_here

    @private.post("/uploads", status_code=201)
    async def upload(request: Request, file: UploadFile) -> dict[str, Any]:
        blob_id = uuid.uuid4().hex
        target = Path(config.uploads_dir) / blob_id
        digest = hashlib.sha256()
        written = 0
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
        meta = {
            "blob_id": blob_id,
            "bytes": written,
            "sha256": digest.hexdigest(),
            "filename": file.filename,
        }
        (Path(config.uploads_dir) / f"{blob_id}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        return {"blob_id": blob_id, "bytes": written, "sha256": meta["sha256"]}

    @private.post("/jobs", status_code=202)
    async def create_job(request: Request, body: JobCreate) -> dict[str, Any]:
        """Admit one job, or refuse by name: a busy lane is `409 server_busy`, and a
        missing environment or model is installed while the job is refused `409
        installing`. `params.resume` set to a `resume_id` continues a journaled job;
        without it the job starts fresh.
        """
        store: JobStore = request.app.state.store
        leases: Leases = request.app.state.leases
        installs: InstallOnSubmit = request.app.state.installs
        try:
            plugin = resolve(store.registry, body.type, config)
        except ApiError as refusal:
            if refusal.code != "job_type_disabled":
                raise
            if (refusal.details or {}).get("reason") == "undecided" and installs.installable(
                body.type
            ):
                refusal = decide_here(body.type)
            if not config.install_on_submit:
                raise refusal from None
            raise installs.start(installs.plan(body.type, body.model, refusal)) from None
        if body.model is not None:
            _refuse_lease_on_an_upstream(body.model)
        model = resolve_model(plugin, body.model)
        _refuse_resume_without_a_journal(plugin, body.type, body.params)

        def unloads_what_is_being_cleared() -> bool:
            return (
                model is not None
                and residency.being_cleared(model)
                and CARD_EFFECTS[body.type].takes_off is not None
            )

        async with residency.settled_for(
            f"a {body.type} job", same_intent=unloads_what_is_being_cleared
        ):
            leases.refuse_if_leased(body.type, model)
            store.refuse_if_busy()
            not_installed: ApiError | None = None
            try:
                plugin.preflight(model, body.params)
            except ApiError as refusal:
                if not (
                    config.install_on_submit and refusal.code in INSTALLABLE_REFUSALS
                ):
                    raise
                not_installed = refusal
            else:
                identity = _journal_identity(plugin, model, body.params)
                digests: list[InputDigest] = []
                resuming: Any = None
                if identity is not None:
                    digests = _input_digests(config, store, body.inputs)
                    resume = body.params.get("resume")
                    if resume is not None:
                        resuming = store.journals.verify(resume, identity, digests)
                job = store.create(
                    body.type, model, body.params,
                    client=client_agent(request), client_ref=body.client_ref,
                    hold=body.hold,
                )
                fresh: Any = None
                try:
                    _materialise_inputs(config, store, job, body.inputs)
                    if identity is not None and resuming is None:
                        fresh = store.journals.create(identity, digests, job.id)
                    store.enqueue(job)
                except ApiError:
                    store.discard(job)
                    if fresh is not None:
                        store.journals.forget_new(fresh)
                    raise
                if fresh is not None:
                    store.attach_journal(job, fresh.id, resumed=False)
                elif resuming is not None:
                    store.journals.adopt(resuming, job.id)
                    store.attach_journal(job, resuming.id, resumed=True)
        if not_installed is not None:
            need = installs.need_for(body.type, model, not_installed)
            if need is None:
                raise not_installed
            raise installs.start(need)
        return {"job_id": job.id, "resume_id": job.resume_id}

    @private.get("/jobs/{job_id}")
    async def get_job(request: Request, job_id: str) -> dict[str, Any]:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        return _job_state(store, job)

    @private.delete("/jobs/{job_id}")
    async def cancel_job(request: Request, job_id: str) -> dict[str, str]:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        outcome = store.cancel(job)
        return {"job_id": job.id, "status": outcome}

    @private.get("/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str) -> StreamingResponse:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        delivered = _last_event_id(request)
        return StreamingResponse(
            _event_stream(request, store, job, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @private.post("/jobs/{job_id}/hold")
    async def hold_job(request: Request, job_id: str) -> dict[str, Any]:
        """Keep this job's artifacts for a later job's `{"artifact": {job_id, name}}`
        inputs until the hold is released or `retention_days` collects it. Idempotent.
        """
        store: JobStore = request.app.state.store
        return store.hold(store.get(job_id), client_agent(request))

    @private.delete("/jobs/{job_id}/hold", status_code=204)
    async def release_job(request: Request, job_id: str) -> Response:
        """The chain is complete: release the hold and remove the job now."""
        store: JobStore = request.app.state.store
        store.release(store.get(job_id))
        return Response(status_code=204)

    @private.get("/jobs/{job_id}/artifacts/{name}")
    async def job_artifact(request: Request, job_id: str, name: str) -> FileResponse:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        try:
            validate_member_name(name)
        except ValueError as exc:
            raise ApiError(400, "invalid_artifact_name", str(exc)) from None
        path = job.artifacts_dir / name
        if not path.is_file():
            raise ApiError(
                404,
                "unknown_artifact",
                f"job {job.id} has no artifact {name!r}; it has {job.artifacts}",
            )
        store.mark_fetched(job, name)
        media_type = (
            "application/json"
            if name.endswith(".provenance.json")
            else "application/octet-stream"
        )
        return FileResponse(path, media_type=media_type, filename=name)
