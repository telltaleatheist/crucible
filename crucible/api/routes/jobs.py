from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ...errors import ApiError
from ...installonsubmit import PULLABLE_REFUSALS, InstallOnSubmit
from ...jobs import resolve, resolve_model
from ...jobs.base import Job, validate_member_name
from ...jobs.queue import JobStore
from ...journal import InputDigest
from ...leases import CARD_EFFECTS, Leases
from ..caller import client_agent
from ..inputs import _input_digests, _journal_identity, _materialise_inputs, _refuse_resume_without_a_journal
from ..schemas import JobCreate
from ..sse import _event_stream, _last_event_id
from ..upstream import _refuse_lease_on_an_upstream
from ..context import AppContext, Routers


UPLOAD_CHUNK = 1024 * 1024


#: The keys `_job_state` owns. A job type's `done_extra` may add to the record
#: and may never rewrite these.
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
        # THE RESTART'S OWN FIELDS. `client_ref` is what the caller named this
        # work; `interrupted_at` is when this server was found to have stopped
        # while it was running (null for every other outcome); `chunks_done` is
        # the chunk index of every artifact published, so a resume is a set
        # difference rather than every client parsing `<index>.flac` for itself.
        "client_ref": job.client_ref,
        "interrupted_at": job.interrupted_at,
        # HELD FOR A CHAIN (2026-09-25): the client that holds this job's
        # artifacts, and since when; both null when nothing does.
        "held_by": job.held_by,
        "held_since": job.held_since,
        "chunks_done": sorted(job.chunks_done),
        # DONE/TOTAL AND A PACE, FROM THE RECORD ALONE (2026-09-21, the ladder's
        # ask): the denominator the job type stated, and when the last chunk
        # landed. Both null for a job whose artifacts are not chunks.
        "chunks_total": job.chunks_total,
        "chunk_at": job.chunk_at,
        # THE JOURNAL THIS JOB WRITES (2026-09-27, `crucible/journal.py`): the
        # id to send as `resume` if it does not finish, and whether this job
        # was itself a resume. Null for a job type that keeps no journal.
        "resume_id": job.resume_id,
        "resumed": job.resumed,
        # THE TERMINAL FACTS, READABLE AFTER THE STREAM IS GONE (2026-09-20).
        # `done_extra` is what a job adds to its own `done` event — `resident`
        # for a loader, and since the lease moved onto the load, `lease_id`. A
        # client that lost the events stream could not read those back, and a
        # `lease_id` a client cannot recover is a hold nobody can release: the
        # exact shape of the strandings this release exists to end. Same rule as
        # the cancel door's receipt — the frame is not the fact.
        #
        # Merged UNDER the keys above so a job type can never rename `status` or
        # `error` by accident; a collision keeps this function's answer.
        **{k: v for k, v in job.done_extra.items() if k not in _JOB_STATE_KEYS},
    }


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, residency, decide_here = ctx.config, ctx.residency, ctx.decide_here

    # --------------------------------------------------------------- uploads

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

    # ------------------------------------------------------------------ jobs

    @private.post("/jobs", status_code=202)
    async def create_job(request: Request, body: JobCreate) -> dict[str, Any]:
        """Admit one job, or refuse with the facts about the one already here.

        **EXPLICIT RESUME ONLY** (Owen, 2026-09-27: *"if the user doesnt send
        the resume flag then it starts fresh. if they do send a resume flag, it
        continues from where they left off"*). A job type that keeps a journal
        (`crucible/journal.py`, docs/RESUMABLE-JOBS.md) answers `resume_id`
        beside `job_id`: the id to send as `params.resume` if this job does not
        finish. Without `resume` a job starts a NEW journal and never reads an
        old one. With it, the journal is checked against this submission (job
        type, model revision, output-affecting params, every input's sha256,
        the type's format version) and anything different is refused
        `resume_mismatch` naming it, before the job exists; an unknown id is
        `unknown_resume_id` and a collected one `resume_expired`. A type that
        keeps no journal refuses `resume` as `resume_unsupported`.

        **This door refuses when the lane is busy (ARCHITECTURE.md section 3).**
        It used to queue, which made Crucible answer the same question two ways:
        the streaming door has always refused with `409 stream_session_open`
        naming the holder, while this one accepted and appended. Same server,
        same card, two policies. Now both refuse and both name who has it.

        The order of the checks is the order of their cost and their specificity,
        and it is deliberate. The type and model are resolved first, because
        `unknown_job_type` is true whether or not anything is running and a client
        with a typo should be told about the typo rather than about somebody
        else's render. Admission comes next, before `preflight` — preflight
        shells out (`ffmpeg -version`), reads manifests and probes the card with
        `nvidia-smi`, and spending that on a request that cannot be admitted is
        work done for a 409. It also comes before `store.create`, so a refused
        submission never makes a directory, and before the inputs are
        materialised, so it never writes a client's megabytes to disk to delete
        them again.

        **A lease is refused ahead of both** (PHASE7-LANES.md section 5.2). A
        chat completion holds nothing, so a server mid-way through a
        two-thousand-block translation looks idle between two blocks; a client
        that says it intends a run takes a lease, and while one is open this
        door refuses the jobs that would move the leased thing off the card. It
        does not refuse anything else — a lease is not a reservation, and the
        lane is still free for work that leaves the card alone, INCLUDING the
        work the lease was taken for: a `tts` render of the leased voice and an
        `align` on the leased aligner are admitted, because they run against
        what is already resident rather than loading it again.

        **A MISSING ENVIRONMENT OR MODEL IS INSTALLED FOR THE CALLER, and the
        job is refused while it installs** (Owen, 2026-09-26: *"yes, we need to
        install a missing environment when a job is submitted"*; 2026-09-27:
        *"Crucible isn't responsible for queuing. The apps that use it are."*).
        A type this card can run and has not installed, or a declared model or
        voice (and `rvc`'s base assets) not yet pulled, starts the operator
        page's install as a task and answers `409 installing`: a sentence
        saying what is being installed, how big, and to submit again after it,
        with the task to watch in `details.task_id`. No job is created and
        nothing waits here; the app's queue retries. A second submit while it
        runs gets the same answer with the same task, never a second install.
        A server that never decided its card decides and records it first. A
        type the card cannot run is refused as before. `[jobs]
        install_on_submit = false` turns the install off (the refusal then
        carries `details.install`). `crucible/installonsubmit.py`.
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
                # Nothing ever measured this card: decide it now (2026-09-27).
                refusal = decide_here(body.type)
            if not config.install_on_submit:
                raise refusal from None
            # Raises `refusal` itself for every case an install cannot fix, a
            # sharper refusal for a request or card it cannot serve, and
            # otherwise starts (or finds) the install and raises `installing`.
            raise installs.start(installs.plan(body.type, body.model, refusal)) from None
        if body.model is not None:
            # An upstream model is never resident and never on the lane, so
            # `load-model` naming one is the same mistake a lease on one is,
            # and gets the same name (PHASE15-HOST.md section 3.4). Checked
            # before `resolve_model`, whose refusal would be
            # `unknown_model` — true of a local catalog and wrong about what
            # the caller actually did.
            _refuse_lease_on_an_upstream(body.model)
        model = resolve_model(plugin, body.model)
        _refuse_resume_without_a_journal(plugin, body.type, body.params)

        def unloads_what_is_being_cleared() -> bool:
            # T6 (2026-09-15): an `unload-...` of the very subject the
            # settlement is clearing is the same intent, admitted at once; its
            # `run` waits the clearance out and ends `done`. Everything else
            # waits at this door instead.
            return (
                model is not None
                and residency.being_cleared(model)
                and CARD_EFFECTS[body.type].takes_off is not None
            )

        # A CLEARANCE IS WAITED OUT, NOT REFUSED (2026-09-24, Briefcase). A
        # `load-model` that arrives while the settlement is SIGTERMing the last
        # engine used to be `409 engine_in_use` "held by the settlement
        # clearing the card" — refusing the very request that would have put
        # the card right. It now waits (off the loop, within
        # `CLEARANCE_TIMEOUT_SECONDS`) and is admitted against the SETTLED
        # card, so its preflight's accelerator guard counts no VRAM of an
        # engine that is leaving. Everything from the first refusal to
        # `enqueue` runs under the card's lock, which is the lock the
        # settlement's check-and-claim takes: a clearance cannot begin between
        # this preflight and this job reaching the lane, and a job on the lane
        # is a holder it will see (crucible/residency.py, `settled_for`).
        async with residency.settled_for(
            f"a {body.type} job", same_intent=unloads_what_is_being_cleared
        ):
            # Would this take the leased thing off the card while somebody has
            # said they are mid-run on it? Asked BEFORE the lane, and before
            # `server_busy`, because the two refusals have different lifetimes:
            # the lane frees in minutes and a client told "busy" will rightly
            # come back, while a lease will still be there when it does. Telling
            # it the transient reason first would send it away to be refused
            # again for the durable one (PHASE7-LANES.md section 5.2).
            #
            # The resolved `model` goes with the type because the answer is not
            # a property of the type alone: `tts` of the leased voice reuses
            # what is resident and is admitted, `tts` of any other voice evicts
            # it and is not (`Lease.evicted_by`).
            leases.refuse_if_leased(body.type, model)
            # Is there room right now? The one question the server answers
            # about scheduling; the queue is the client's (ARCHITECTURE.md
            # section 3).
            store.refuse_if_busy()
            # Every refusal a job type can make about host state happens here,
            # before the job exists, so the client is told by name instead of
            # watching a job fail (PHASE2-LLM.md section 5). The lane being free
            # is not the only way to be busy: a streaming session holds the
            # resident engine without occupying the lane, and the job types that
            # would talk to it or move it refuse `engine_in_use` from here
            # (crucible/residency.py).
            #
            # ITS WEIGHTS MAY NOT BE HERE (Owen, 2026-09-27: *"Yes, it should
            # try to pull the model"*). On a missing-weights refusal the pull is
            # worked out AFTER this block, which holds the card's lock and must
            # stay short: a declared, pullable model, voice or base asset this
            # card can run is pulled for the caller and the job is refused
            # `installing`, exactly as a missing env is. Only these refusals
            # ask, so a job whose weights are here pays nothing for it.
            missing_weights: ApiError | None = None
            try:
                plugin.preflight(model, body.params)
            except ApiError as refusal:
                if not (
                    config.install_on_submit and refusal.code in PULLABLE_REFUSALS
                ):
                    raise
                missing_weights = refusal
            else:
                # Which journal identity this job has, if its type keeps one:
                # asked after preflight, which has validated the params.
                identity = _journal_identity(plugin, model, body.params)
                digests: list[InputDigest] = []
                resuming: Any = None
                if identity is not None:
                    digests = _input_digests(config, store, body.inputs)
                    resume = body.params.get("resume")
                    if resume is not None:
                        # Refused HERE, before the job exists and before any
                        # upload is moved: `resume_mismatch` names what differs.
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
                        # A NEW journal, never an old one reused: forgetting
                        # the flag destroys nothing.
                        fresh = store.journals.create(identity, digests, job.id)
                    # `enqueue` asks admission again and is the authority on it;
                    # nothing awaits between here and the check above — the body of
                    # `settled_for` must not — so the two are one atomic stretch on
                    # the event loop. Inside the same `try` so that a refusal from
                    # either leaves no half-built job behind.
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
        if missing_weights is not None:
            need = installs.pulls_for(body.type, model, missing_weights)
            if need is None:
                raise missing_weights
            raise installs.start(need)
        # `resume_id` is null for a job type that keeps no journal.
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
        """Keep this done job's artifacts for a later job (Owen, 2026-09-25).

        *"keep all working files on the crucible side until the chain is
        complete. then remove them"*. Held, the job is not reaped for having
        been fetched: it stays until `DELETE` on this route releases it (and
        removes it at once), or until the `retention_days` collector takes it
        (`gc_at`). A later job names its files as `{"artifact": {job_id,
        name}}` inputs. Survives a restart. Idempotent.
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
        # THE FETCH IS RECORDED HERE, and here is the only place that knows it
        # happened (Owen's ruling, 2026-09-18). A job whose every artifact and
        # sidecar has been through this line is a job whose directory is a
        # second copy of what the client now holds, and `JobStore.reap` deletes
        # it on the next idle tick. Recorded after the file check, so a 404 for
        # a name that is not there never counts as a collection.
        store.mark_fetched(job, name)
        media_type = (
            "application/json"
            if name.endswith(".provenance.json")
            else "application/octet-stream"
        )
        return FileResponse(path, media_type=media_type, filename=name)
