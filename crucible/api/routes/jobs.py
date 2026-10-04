from __future__ import annotations

from typing import Any

from fastapi import Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...admission import JobRequest, Refusal, admit
from ...errors import ApiError
from ...jobs.base import Job, validate_member_name
from ...jobs.line import DEFAULT_MAX_WAIT_S, MAX_MAX_WAIT_S
from ...jobs.queue import JobStore
from ...queuerequest import max_wait_of
from ...uploads import store_upload
from .. import sse
from ..caller import client_agent, queue_session
from ..context import AppContext, Routers
from ..responses import BUSY_RESPONSES, NOT_FOUND, JobStatus
from ..schemas import JobCreate

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
        "removal",
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
        "removal": job.removal,
        **{k: v for k, v in job.done_extra.items() if k not in _JOB_STATE_KEYS},
    }


# What a player needs to be told to play an artifact straight from its URL (an <audio>
# element or AVPlayer streaming with Range). Anything else is bytes.
ARTIFACT_MEDIA_TYPES: dict[str, str] = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}


def _artifact_media_type(name: str) -> str:
    if name.endswith(".provenance.json"):
        return "application/json"
    for suffix, media_type in ARTIFACT_MEDIA_TYPES.items():
        if name.endswith(suffix):
            return media_type
    return "application/octet-stream"


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private

    @private.post("/uploads", status_code=201)
    async def upload(request: Request, file: UploadFile) -> dict[str, Any]:
        blob = await store_upload(ctx.config.uploads_dir, file.read, file.filename)
        return blob.receipt()

    @private.post("/jobs", status_code=202, responses=BUSY_RESPONSES)
    async def create_job(request: Request, body: JobCreate) -> dict[str, Any]:
        """Admit one job, or refuse by name. A busy lane queues the job: it waits with
        status `queued` (up to an hour, or `queue.max_wait_s`) and its events say where
        it stands. With `"queue": false` a busy lane is refused `409 server_busy`
        instead. A missing environment or model is installed while the job is refused
        `409 installing`. `params.resume` set to a `resume_id` continues a journaled
        job; without it the job starts fresh. An item of the open queue session (named
        in the session header, or any submit from the client holding it) goes ahead of
        everything waiting, waits only behind the session's own jobs, and waits up to
        a day unless its `queue` says otherwise.
        """
        session = queue_session(request, ctx.sessions)
        queue = max_wait_of(
            body.queue, DEFAULT_MAX_WAIT_S if session is None else MAX_MAX_WAIT_S
        )
        outcome = await admit(
            JobRequest(
                type=body.type,
                model=body.model,
                params=body.params,
                inputs=body.inputs,
                client=client_agent(request),
                client_ref=body.client_ref,
                hold=body.hold,
                queue=queue,
                session=None if session is None else session.id,
            ),
            ctx.admission(),
        )
        if isinstance(outcome, Refusal):
            raise outcome.error
        if session is not None:
            ctx.sessions.item_arrived(session)
        return outcome.receipt()

    @private.get(
        "/jobs/{job_id}",
        response_model=JobStatus,
        response_model_exclude_unset=True,
        responses=NOT_FOUND,
    )
    async def get_job(job_id: str) -> dict[str, Any]:
        store = ctx.store
        job = store.get(job_id)
        ctx.line.touch(job_id=job.id)
        return _job_state(store, job)

    @private.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, str]:
        """Cancel a job; a job still waiting in the queue is removed (reason
        `client`) and answers `status: removed`.
        """
        store = ctx.store
        job = store.get(job_id)
        return {"job_id": job.id, "status": store.cancel(job)}

    @private.get("/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str) -> StreamingResponse:
        store = ctx.store
        job = store.get(job_id)
        ctx.line.touch(job_id=job.id)
        return sse.job_events(request, store, job, sse.last_event_id(request))

    @private.post("/jobs/{job_id}/hold")
    async def hold_job(request: Request, job_id: str) -> dict[str, Any]:
        """Keep this job's artifacts for a later job's `{"artifact": {job_id, name}}`
        inputs until the hold is released or `retention_days` collects it. Idempotent.
        """
        store = ctx.store
        return store.hold(store.get(job_id), client_agent(request))

    @private.delete("/jobs/{job_id}/hold", status_code=204)
    async def release_job(job_id: str) -> Response:
        """The chain is complete: release the hold and remove the job now."""
        store = ctx.store
        store.release(store.get(job_id))
        return Response(status_code=204)

    @private.get("/jobs/{job_id}/artifacts/{name}")
    async def job_artifact(job_id: str, name: str) -> FileResponse:
        store = ctx.store
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
        return FileResponse(path, media_type=_artifact_media_type(name), filename=name)
