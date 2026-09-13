"""API v1 — exactly the surface in DESIGN.md section 4.

Base path is `/v1`. Every route except `GET /v1/ping` needs
`Authorization: Bearer <token>` and `X-Crucible-Api: 1`. The checks run in that
order, so a request with a bad token and a missing version header is answered 401.
Errors are always `{"error": {"code", "message", "details"?}}`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import API_VERSION, VERSION, accelerator
from .backend import CUDA_LINUX, Backend
from .config import Config
from .errors import ApiError
from .jobs import (
    ALL_JOB_TYPES,
    build_registry,
    model_rows,
    resolve,
    resolve_model,
    voice_rows,
)
from .jobs.base import Job, validate_member_name
from .jobs.queue import JobStore
from .jobs.tts.common import known_voice
from .residency import Residency
from .ttsstream import (
    StreamManager,
    StreamSession,
    require_sayable,
    require_streamable,
)

API_HEADER = "X-Crucible-Api"
TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})
KEEPALIVE_SECONDS = 15.0
UPLOAD_CHUNK = 1024 * 1024

#: The proxy waits on the engine, not on a clock it invented. A streamed
#: completion has no read timeout at all (the engine emits a token at a time and
#: may think for a while before the first one); a non-streamed one gets a long
#: but finite ceiling so a wedged engine surfaces as an error rather than a hang.
PROXY_CONNECT_TIMEOUT = 10.0
PROXY_READ_TIMEOUT = 900.0

#: The proxy sends the client's own bytes, so it declares the type itself rather
#: than letting httpx serialise a document and label it.
JSON_HEADERS = {"Content-Type": "application/json"}

#: How often a non-streamed completion checks whether its caller is still there.
#: `Request.is_disconnected()` is a poll and not a wait — it reads `receive`
#: inside an already-cancelled scope and answers at once — so something has to
#: hold the clock. A quarter of a second is far below the seconds a completion
#: takes and far above what asking costs.
DISCONNECT_POLL_SECONDS = 0.25


# --------------------------------------------------------------------- schemas


class JobInput(BaseModel):
    """One named input: either an uploaded blob or bytes inline in the request."""

    model_config = ConfigDict(extra="forbid")

    blob_id: str | None = None
    inline_base64: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "JobInput":
        given = [name for name, value in
                 (("blob_id", self.blob_id), ("inline_base64", self.inline_base64))
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                "each input needs exactly one of blob_id or inline_base64, got "
                f"{given if given else 'neither'}"
            )
        return self


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, JobInput] = Field(default_factory=dict)


class StreamOpen(BaseModel):
    """`POST /v1/tts/stream` — PHASE3-TTS.md section 7.

    Nothing has a default, for the render door's reason: a session opened in the
    wrong language, or on a voice the client did not choose, is a silent
    substitution and a whole afternoon of listening in the wrong accent.
    """

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)


class StreamOp(BaseModel):
    """`POST /v1/tts/stream/{id}` — one op.

        {"op": "say",    "id": "r12", "text": "...", "take": 0}
        {"op": "cancel", "id": "r12"}
        {"op": "cancel_all"}
        {"op": "close"}

    `take` is **required** on `say` and has no default here, even though every
    voice in this build declares exactly one take and anything above 0 is refused
    as `sampling_not_wired`. The SDK's `say(id, text, take?)` defaults it to 0 in
    the caller's own code, which is a client choosing; a default on the wire would
    be the server choosing, and the day a ladder is wired that becomes a render at
    a take nobody asked for.
    """

    model_config = ConfigDict(extra="forbid")

    op: str
    id: str | None = None
    text: str | None = None
    take: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def the_op_carries_what_it_needs(self) -> "StreamOp":
        allowed = ("say", "cancel", "cancel_all", "close")
        if self.op not in allowed:
            raise ValueError(f"op must be one of {list(allowed)}, got {self.op!r}")
        if self.op == "say":
            if not (self.id or "").strip():
                raise ValueError("say needs an id; it is how every frame names its row")
            if not (self.text or "").strip():
                # narrator answers an empty generate with a whole-request error,
                # which would take the rest of the batch with it. Refused here.
                raise ValueError(
                    "say needs text that is not blank; narrator refuses an empty "
                    "generate with a whole-request error, which would end the batch"
                )
            if self.take is None:
                raise ValueError("say needs a take; there is no default on the wire")
        elif self.op == "cancel":
            if not (self.id or "").strip():
                raise ValueError("cancel needs the id of the row to cancel")
        else:
            if self.id is not None or self.text is not None or self.take is not None:
                raise ValueError(f"{self.op} takes no id, text or take")
        return self


# ------------------------------------------------------------------ app wiring


def _error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.body())


def require_auth(request: Request) -> None:
    config: Config = request.app.state.config
    header = request.headers.get("authorization")
    if header is None:
        raise ApiError(
            401,
            "unauthorized",
            "missing Authorization header; send `Authorization: Bearer <token>`",
        )
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or presented == "":
        raise ApiError(
            401, "unauthorized", "Authorization header must be `Bearer <token>`"
        )
    if not secrets.compare_digest(presented.strip(), config.token):
        raise ApiError(401, "unauthorized", "bearer token is not this server's token")


def require_api_version(request: Request) -> None:
    presented = request.headers.get(API_HEADER)
    if presented is None:
        raise ApiError(
            426,
            "api_version_required",
            f"send `{API_HEADER}: {API_VERSION}`; this server speaks API version "
            f"{API_VERSION}",
            {"server_api_version": API_VERSION, "client_api_version": None},
        )
    major_text = presented.strip().split(".")[0]
    try:
        major = int(major_text)
    except ValueError:
        raise ApiError(
            426,
            "api_version_unreadable",
            f"{API_HEADER}: {presented!r} is not a version; this server speaks API "
            f"version {API_VERSION}",
            {"server_api_version": API_VERSION, "client_api_version": presented},
        ) from None
    if major != API_VERSION:
        raise ApiError(
            426,
            "api_version_mismatch",
            f"client speaks API version {major}, this server speaks {API_VERSION}",
            {"server_api_version": API_VERSION, "client_api_version": major},
        )


def create_app(config: Config, backend: Backend) -> FastAPI:
    """Build the ASGI app for one server instance."""
    residency = Residency(config)
    registry = build_registry(config, backend, residency)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: JobStore = app.state.store
        # MONOTONIC, not a wall clock: uptime is a duration, and a duration
        # computed across an NTP correction or a DST jump is how a bench ends up
        # reporting that a server has been up for minus four minutes.
        app.state.started_at = time.monotonic()
        store.start()
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=PROXY_CONNECT_TIMEOUT,
                read=PROXY_READ_TIMEOUT,
                write=60.0,
                pool=10.0,
            )
        )
        try:
            yield
        finally:
            await store.stop()
            await app.state.http.aclose()
            # Before the residency, and that order is load-bearing: a streaming
            # session holds the resident engine's exclusive claim, and
            # `Residency.unload` refuses by name while somebody holds it. Closing
            # the sessions first is what makes the shutdown below able to reach
            # the card at all.
            await asyncio.to_thread(app.state.streams.shutdown)
            # A resident engine is this process's child. Leaving one holding the
            # card after the server exits would be exactly the thing the guard
            # refuses to do to somebody else.
            await asyncio.to_thread(residency.shutdown)

    app = FastAPI(
        title="Crucible",
        version=VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config
    app.state.backend = backend
    app.state.residency = residency
    app.state.store = JobStore(config, backend, registry)
    app.state.streams = StreamManager(residency)

    Path(config.jobs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.uploads_dir).mkdir(parents=True, exist_ok=True)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # exc.errors() can carry exception objects in `ctx`; keep only what is
        # JSON-safe and actually useful to the client.
        problems = [
            {
                "location": [str(part) for part in problem["loc"]],
                "type": problem["type"],
                "message": problem["msg"],
            }
            for problem in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "the request body is not a valid job request",
                    "details": {"problems": problems},
                }
            },
        )

    public = APIRouter(prefix="/v1")
    private = APIRouter(
        prefix="/v1", dependencies=[Depends(require_auth), Depends(require_api_version)]
    )

    # ------------------------------------------------------------------ ping

    @public.get("/ping")
    async def ping() -> dict[str, Any]:
        """Unauthenticated. Lets a client tell "wrong token" from "not a Crucible"."""
        return {"crucible": True, "name": config.name, "api_version": API_VERSION}

    # ------------------------------------------------------------------ info

    @private.get("/info")
    async def info(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store

        # ONE CAPABILITY PER CAPABILITY, not one per POSTable job type.
        #
        # This listed every registered job type as its own capability until
        # 2026-09-13, when running the merged server and reading its answer
        # showed `qwen3.5-9b` appearing three times — under `load-model`, under
        # `unload-model`, and under `llm` — in TWO different shapes, because the
        # lifecycle types describe a model with `ModelDescriptor.to_dict()` and
        # the `llm` capability uses the far richer `/v1/models` row. That is
        # exactly the thing PHASE2-LLM.md section 5 was written to forbid: "One
        # model, one description: a client reads a model's standing in one shape
        # wherever it finds it, and never reconciles two."
        #
        # `ALL_JOB_TYPES` already maps a job type to the capability it operates
        # (`load-model` and `unload-model` both to `llm`), so the grouping is not
        # a new table anybody has to keep in step — it is the one the registry is
        # already built from. What you can POST is answered by `job_types`, which
        # is what the lifecycle rows were really there to tell anyone.
        rows_for: dict[str, list[dict[str, Any]]] = {}
        for name, plugin in sorted(store.registry.items()):
            capability = ALL_JOB_TYPES[name]
            if capability in rows_for:
                continue
            rows_for[capability] = [m.to_dict() for m in plugin.describe_models()]
        if config.enable_llm:
            # PHASE2-LLM.md section 5: the `llm` rows are `/v1/models`' rows,
            # from the same producer, revision included.
            rows_for["llm"] = model_rows(config, backend, residency)
        if config.enable_tts:
            # PHASE3-TTS.md section 8: the `tts` rows are `/v1/voices`' rows
            # VERBATIM, for the same reason `llm`'s are.
            #
            # This assignment REPLACES whatever the loop above produced, and for
            # `tts` that is not a no-op the way it is for `llm`. `llm` is a
            # capability name and nothing else — the types you POST are
            # `load-model` and `unload-model` — while `tts` is both: the render
            # door's job type is literally `tts`, so the loop has already filled
            # this key from its `describe_models()`. The richer row wins, which
            # is the same rule applied one level down.
            rows_for["tts"] = voice_rows(config, backend, residency)
        capabilities = [
            {"job_type": capability, "models": rows}
            for capability, rows in sorted(rows_for.items())
        ]
        return {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
            },
            "host": {
                "platform": backend.platform,
                "arch": backend.arch,
                "backend": backend.kind,
                "gpu": {
                    "vendor": backend.gpu.vendor,
                    "name": backend.gpu.name,
                    "vram_bytes": backend.gpu.vram_bytes,
                },
            },
            # What this server will accept in `POST /v1/jobs`. A capability
            # above says what it can serve; this says what to ask it with, and
            # the two are not the same list: `llm` is served through
            # `load-model` and `unload-model`, neither of which is a capability.
            "job_types": sorted(store.registry),
            "capabilities": capabilities,
        }

    @private.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        store: JobStore = request.app.state.store
        if residency.warming is not None:
            status = "warming"
        elif store.running_id is not None:
            status = "busy"
        else:
            status = "ok"
        return {
            "status": status,
            "queue_depth": store.queue_depth,
            # Keeps its name and its shape — a list of ids — and gains the kind
            # beside it, because since PHASE3-TTS.md section 5 the one thing on
            # the card may be a voice, and a client has to know which door to
            # knock on. `resident_models` is not renamed: every phase-2 client
            # reads it, and one id is one id whatever kind of thing it names.
            "resident_models": residency.ids(),
            "resident_kind": residency.resident_kind,
        }

    # ----------------------------------------------------------- accelerator

    @private.get("/accelerator")
    async def accelerator_state(request: Request) -> dict[str, Any]:
        """What is on the card right now, and which of it is Crucible's.

        PHASE4-AUDIO.md section 5. This is the same `nvidia-smi
        --query-compute-apps` the load guard runs, plus the free/total figures,
        plus the resident set, plus a flag saying which holders are this server's
        own processes — and it exists because BookForge arbitrates the GPU three
        incompatible ways at once (a queue slot, an in-process mutex whose
        timeout *proceeds without the lock*, and nothing at all for the hosted
        page reader), on top of a lock file with no producer inside the app. One
        call here answers the question all three were guessing at.

        **It never evicts anybody, ever.** It reports, and that is the whole of
        it. The rule is PHASE2-LLM.md section 4's and it does not soften because
        more job types now depend on the answer.

        It is private like every other route here: the bearer token and the
        version header, in that order. A probe of somebody's hardware is not
        public information, and `GET /v1/ping` already exists for "is this a
        Crucible".
        """
        # nvidia-smi is a subprocess and takes tens of milliseconds; off the
        # event loop, or a poll of this route stalls every job's event stream.
        try:
            state = await asyncio.to_thread(
                accelerator.read_state, backend.kind, config.desktop_allowance_bytes
            )
        except accelerator.ProbeError as exc:
            # 503 and not 409: nothing was asked for and refused, the server
            # simply cannot see its own card at the moment. A client polling for
            # a free GPU must read this as "ask again", never as "it is free" —
            # which is why the probe raises rather than returning zeroes.
            raise ApiError(
                503,
                "accelerator_unreadable",
                f"this server cannot read its accelerator: {exc}",
            ) from None

        owned = residency.owned_pids()
        holders = [
            {
                "pid": app.pid,
                "name": app.name,
                # None where the driver will not say (WDDM, permissions). That is
                # not zero and must not be rendered as zero.
                "bytes": app.used_bytes,
                "owned_by_crucible": app.pid in owned,
            }
            for app in state.compute_apps
        ]
        resident = residency.resident
        return {
            "backend": state.backend,
            "gpu": {
                "vendor": backend.gpu.vendor,
                "name": backend.gpu.name,
                # The live figure from the probe, not the one detection recorded
                # at start-up. They agree on a real host; where they would not,
                # the live one is the one a caller is about to make a decision on.
                "total_bytes": state.total_bytes,
            },
            "free_bytes": state.free_bytes,
            "used_bytes": state.used_bytes,
            "desktop_allowance_bytes": config.desktop_allowance_bytes,
            # VRAM in use that no listed compute app accounts for, past the
            # declared desktop allowance. Under WSL2 the driver shim answers the
            # compute-app query with an EMPTY LIST even while a process inside
            # that same VM holds 17 GB (measured on Owen's PC, 2026-09-12), so on
            # that host this number is the only honest report of the card being
            # busy and `holders` will be misleadingly empty. Null on mlx-darwin,
            # where "used unified memory" is the OS doing its job and attributing
            # it to compute processes is not a question vm_stat can answer.
            "unattributed_bytes": (
                accelerator.unattributed_bytes(state, config.desktop_allowance_bytes)
                if state.backend == CUDA_LINUX
                else None
            ),
            "resident": (
                None
                if resident is None
                else {
                    # `kind` is the family of thing that is resident, not the job
                    # type that put it there — and it is ASKED rather than
                    # assumed. This said `"llm"` and read `resident.model_id`
                    # until 2026-09-13, which was true while a model was the only
                    # thing a card could hold and became a 500 the moment
                    # PHASE3-TTS.md section 5's generalised residency landed: a
                    # `ResidentVoice` has a `voice_id` and an `id`, and no
                    # `model_id` at all. `model_rows()` had already learned to ask
                    # for `resident_model`; this route had not caught up, so the
                    # route whose whole job is to say what is on the card was the
                    # one that could not say a voice was.
                    #
                    # `id` and `kind` are what every Resident has in common, by
                    # design: phase 4's aligner is a third kind and needs no
                    # change here.
                    "kind": resident.kind,
                    "id": resident.id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                }
            ),
            "holders": holders,
            "detail": state.detail,
        }

    # -------------------------------------------------------------- activity

    def _activity_row(store: JobStore, job: Any) -> dict[str, Any]:
        """One job, as a bench reads it. Never its params: a chat prompt or a
        chapter of a book is not something a whole-server read should spray at
        anyone holding the token."""
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

    @private.get("/activity")
    async def activity(request: Request, accelerator_probe: bool = False) -> dict[str, Any]:
        """What is on this server and how far along — one read, no job id.

        PHASE7-LANES.md section 5. Owen, 2026-09-13: *"Crucible will have to have
        an api endpoint that will report what's on it and its progress so
        Bookforge can hit that endpoint and fill that gpu slot with that data."*

        WHY THIS IS A POLL AND NOT THE SSE IT ALREADY HAS. Per-job events are
        push, fine-grained and exactly right for the step that owns a job. This
        answers a different question, asked by a bench widget that owns no job
        and may never own one: *what is this machine doing?* Opening a stream per
        job per server to render one line of text is the wrong shape. The two do
        not compete — the step reads the stream, the bench reads this.

        IT REPORTS AND NOTHING ELSE. It does not admit, reserve, claim or lock. A
        client that reads "free" and submits is racing every other client, and
        that race is settled at the door: `POST /v1/jobs` admits one and refuses
        the other `server_busy`, naming the winner (ARCHITECTURE.md section 3).
        The loser has lost nothing but a round trip, because it never gave up
        ownership of its own queue — which is the point of the ruling. A
        reservation here would be a second place to arbitrate, and a stale one.

        **So this route is a bench display and a preflight, never admission.** It
        is the honest answer to "how long until that finishes"; it is not
        permission to submit, and a client must be able to be refused after
        reading it. Only `POST /v1/jobs` can say yes.

        THE PROBE IS OPT-IN, and that is the one design decision in this route.
        `nvidia-smi` is a subprocess costing tens of milliseconds, and a bench
        polling three servers every few seconds would spawn one per server per
        tick forever to render a number nobody is reading. `resident` below
        already says what is loaded and roughly what it costs, in memory, for
        free. A caller that genuinely wants the live figure asks for it with
        `?accelerator_probe=true` and pays for it; `GET /v1/accelerator` remains
        the full answer.
        """
        store: JobStore = request.app.state.store
        running = store.running
        queued = store.queued()
        resident = residency.resident

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
                }
            ),
            # `warming` is neither running nor queued and a bench that ignored it
            # would draw an idle machine that is in fact spending two minutes
            # loading a model. It is the reason a slot is unavailable, so it is
            # reported where the slot is.
            "warming": residency.warming,
            "slots": {
                # ONE LANE TODAY, and it is named rather than counted so the
                # ancillary lane (PHASE7-LANES.md section 3) can appear beside it
                # without changing this one's meaning. A key that is absent means
                # this build has no such lane — never that the lane is idle.
                "accelerated": {
                    "busy": 0 if running is None else 1,
                    "of": 1,
                    "queue_depth": store.queue_depth,
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
                # NOT a 503 for the whole route, unlike `/v1/accelerator`. The
                # caller asked for the bench and additionally for a probe; a card
                # the driver will not talk about right now must not blank out the
                # progress of a render that is plainly still going. The failure is
                # named in place and everything else stands.
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

    # ---------------------------------------------------------------- models

    @private.get("/models")
    async def models(request: Request) -> list[dict[str, Any]]:
        """Every model this build has a manifest for, and where it stands here."""
        if not config.enable_llm:
            raise ApiError(
                400,
                "job_type_disabled",
                "job type 'llm' is not enabled on this server "
                "(set [jobs] enable_llm = true in config.toml)",
            )
        return model_rows(config, backend, residency)

    # ---------------------------------------------------------------- voices

    @private.get("/voices")
    async def voices(request: Request) -> list[dict[str, Any]]:
        """Every voice this build has a manifest for, and where it stands here."""
        if not config.enable_tts:
            raise ApiError(
                400,
                "job_type_disabled",
                "job type 'tts' is not enabled on this server "
                "(set [jobs] enable_tts = true in config.toml)",
            )
        return voice_rows(config, backend, residency)

    # -------------------------------------------------------- tts streaming
    #
    # PHASE3-TTS.md section 7. Four routes and no socket: the Listen path, the
    # in-app Play button and the browser extension, built out of the two things
    # this server already does well. The session's own machinery — the rows, the
    # replay buffer, the grace window, the per-row cancel — is
    # `crucible/ttsstream.py`; what is here is the wire.

    def _streaming_voice(voice: str) -> Any:
        """The manifest for a voice this server may be asked to stream."""
        if not config.enable_tts:
            raise ApiError(
                400,
                "job_type_disabled",
                "job type 'tts' is not enabled on this server "
                "(set [jobs] enable_tts = true in config.toml)",
            )
        manifest = known_voice(voice)
        require_streamable(manifest, config.backend_kind)
        return manifest

    @private.post("/tts/stream", status_code=201)
    async def open_stream(request: Request, body: StreamOpen) -> dict[str, Any]:
        """Open the one streaming session this server will hold at a time."""
        streams: StreamManager = request.app.state.streams
        manifest = _streaming_voice(body.voice)
        session = streams.open(
            voice=body.voice,
            language=body.language,
            manifest=manifest,
            loop=asyncio.get_running_loop(),
        )
        return {
            "session_id": session.id,
            "voice": session.voice,
            "fingerprint": session.fingerprint,
            "sample_rate": session.sample_rate,
            "backend": session.backend,
        }

    @private.get("/tts/stream/{session_id}/events")
    async def stream_events(request: Request, session_id: str) -> StreamingResponse:
        """The session's SSE stream — everything it has to say, audio included.

        `Last-Event-ID` is the reattach: a connection that dropped in a tunnel
        comes back here inside the grace window, is replayed what it missed and
        follows live from there. It is the one behaviour a WebSocket could not
        have given for free, which is why this door is not one.
        """
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        delivered = _last_event_id(request)
        # Asked before the response is built, so an unreplayable resume is a 409
        # with a body rather than a 200 that ends at once.
        session.check_replayable(delivered)
        return StreamingResponse(
            _session_event_stream(request, session, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @private.post("/tts/stream/{session_id}", status_code=202)
    async def stream_op(
        request: Request, session_id: str, body: StreamOp
    ) -> dict[str, Any]:
        """One op: `say`, `cancel`, `cancel_all` or `close`.

        **`say` answers with the row's id and not the audio.** A client that
        wants the audio reads the stream; a client that never opened one is
        refused by name rather than generating into nothing.
        """
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        if body.op == "say":
            manifest = known_voice(session.voice)
            require_sayable(manifest, body.take)
            if len(body.text) > session.max_chars:
                # The cap certificate, refused rather than re-split: chunking is
                # the client's (PHASE3-TTS.md section 1), and a server that
                # quietly cut a sentence in half would stream two rows where one
                # was asked for and retire an id the client never sees again.
                raise ApiError(
                    400,
                    "chunk_too_long",
                    f"this row is {len(body.text)} characters and the cap for "
                    f"{session.voice!r} on {session.backend} is "
                    f"{session.max_chars}. Chunking is the client's, so this is "
                    "a refusal and not a re-split",
                    {"voice": session.voice, "max_chars": session.max_chars},
                )
            return {"id": session.say(body.id, body.text, body.take)}
        if body.op == "cancel":
            return {"id": body.id, "outcome": session.cancel(body.id)}
        if body.op == "cancel_all":
            return {"cancelled": session.cancel_all()}
        # `close`, which the validator has already proved is the only one left.
        closed = await asyncio.to_thread(
            streams.close, session, "the client closed the session"
        )
        return {"session_id": session.id, "closed": closed}

    @private.delete("/tts/stream/{session_id}")
    async def close_stream(request: Request, session_id: str) -> dict[str, Any]:
        """The same as `{"op": "close"}`, for a client that only has verbs."""
        streams: StreamManager = request.app.state.streams
        session = streams.get(session_id)
        closed = await asyncio.to_thread(
            streams.close, session, "the client closed the session"
        )
        return {"session_id": session.id, "closed": closed}

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
    async def create_job(request: Request, body: JobCreate) -> dict[str, str]:
        """Admit one job, or refuse with the facts about the one already here.

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
        """
        store: JobStore = request.app.state.store
        plugin = resolve(store.registry, body.type)
        model = resolve_model(plugin, body.model)
        # Is there room right now? The one question the server answers about
        # scheduling; the queue is the client's (ARCHITECTURE.md section 3).
        store.refuse_if_busy()
        # Every refusal a job type can make about host state happens here, before
        # the job exists, so the client is told by name instead of watching a job
        # fail (PHASE2-LLM.md section 5). The lane being free is not the only way
        # to be busy: a streaming session holds the resident engine without
        # occupying the lane, and the job types that would talk to it or move it
        # refuse `engine_in_use` from here (crucible/residency.py).
        plugin.preflight(model, body.params)

        # The SDK sends `<clientName> crucible-client/<version>`; anything else
        # speaking to this server may send whatever it likes, or nothing.
        # Truncated because it is a header, and a header is attacker-controlled
        # length even inside one trust domain.
        agent = (request.headers.get("user-agent") or "").strip()[:200] or None
        job = store.create(body.type, model, body.params, client=agent)
        try:
            _materialise_inputs(config, job, body.inputs)
            # `enqueue` asks admission again and is the authority on it; nothing
            # awaits between here and the check above, so the two are one atomic
            # stretch on the event loop. Inside the same `try` so that a refusal
            # from either leaves no half-built job behind.
            store.enqueue(job)
        except ApiError:
            store.discard(job)
            raise
        return {"job_id": job.id}

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
        media_type = (
            "application/json"
            if name.endswith(".provenance.json")
            else "application/octet-stream"
        )
        return FileResponse(path, media_type=media_type, filename=name)

    # --------------------------------------------------------------- openai

    @private.get("/openai/models")
    async def openai_models(request: Request) -> dict[str, Any]:
        """The resident model in OpenAI's list shape, or an empty list.

        `resident_model` rather than `resident`: with a voice on the card there
        is no model to list, and narrator answers no OpenAI route.
        """
        resident = residency.resident_model
        if resident is None:
            return {"object": "list", "data": []}
        return {
            "object": "list",
            "data": [
                {
                    "id": resident.model_id,
                    "object": "model",
                    "created": int(
                        datetime.fromisoformat(resident.loaded_at).timestamp()
                    ),
                    "owned_by": config.name,
                    # Crucible's id is the contract; this is the name the engine
                    # itself answers to. They differ on mlx-lm, which has no
                    # --served-model-name (crucible/engines/mlx_lm.py).
                    "engine_model_name": resident.engine_model_name,
                    # This entry describes the ENGINE, not the manifest, so both
                    # of these are what was actually loaded. `/v1/models`' row
                    # for the same model reports the manifest's pin, and the two
                    # differ only if somebody edited the manifest while the
                    # engine was up — in which case a client recording what it
                    # talked to wants this one.
                    "revision": resident.revision,
                    "fingerprint": resident.fingerprint,
                    # The context this engine was started with, under OpenAI's
                    # own field name. This is the door Foundry reads — it asks
                    # the OpenAI-shaped listing, not `/v1/models` — and it is the
                    # one that must not lie, because `capFor` subtracts the
                    # prompt from this number to size `max_tokens` and skips the
                    # clamp entirely when it is absent (CLIENT-SURFACES.md
                    # section 6.1).
                    "max_model_len": resident.max_model_len,
                }
            ],
        }

    @private.post("/openai/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """Proxied to the resident engine. Never loads one (section 5)."""
        raw = await request.body()
        body = _chat_body(raw)
        requested = body.get("model")
        if not isinstance(requested, str) or requested == "":
            raise ApiError(
                400,
                "model_required",
                "a chat request must name a model; this server proxies only to the "
                "model that is resident",
            )
        # The resident MODEL: a voice on the card is not something a chat
        # request can be proxied to, so this door's honest answer is the same
        # `model_not_resident` it gives for an empty card.
        resident = residency.resident_model
        if resident is None or resident.model_id != requested:
            raise ApiError(
                409,
                "model_not_resident",
                f"{requested!r} is not resident on this server; "
                + (
                    f"{resident.model_id!r} is. "
                    if resident is not None
                    else "no model is. "
                )
                + "Crucible never loads a model to answer a chat request — submit "
                'a {"type": "load-model"} job first.',
                {"requested": requested, "resident": None if resident is None
                 else resident.model_id},
            )

        forwarded = _forward_body(raw, body, resident)
        url = f"{resident.base_url}/v1/chat/completions"
        client: httpx.AsyncClient = request.app.state.http

        if body.get("stream") is True:
            return await _proxy_stream(client, url, forwarded, resident)
        try:
            upstream = await _post_unless_the_caller_leaves(
                client, url, forwarded, request
            )
        except httpx.HTTPError as exc:
            raise _engine_unreachable(resident, exc) from None
        if upstream is None:
            return _caller_gone(resident)
        content = upstream.content
        if upstream.status_code == 200:
            content = _restore_model_id(content, resident)
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    app.include_router(public)
    app.include_router(private)
    return app


def _chat_body(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(
            400, "invalid_request", f"the chat request body is not JSON: {exc}"
        ) from None
    if not isinstance(body, dict):
        raise ApiError(
            400,
            "invalid_request",
            f"the chat request body must be a JSON object, got "
            f"{type(body).__name__}",
        )
    return body


def _forward_body(raw: bytes, body: dict[str, Any], resident: Any) -> bytes:
    """The client's chat body on its way to the engine.

    The proxy owns exactly one field (PHASE2-LLM.md section 5), so where the
    engine already answers to the Crucible id — vLLM, which takes
    `--served-model-name` — there is nothing to substitute and the bytes the
    client sent are the bytes the engine reads. That is worth more than
    tidiness: `response_format.json_schema.schema` is a grammar Foundry hands to
    the guided-decoding backend (CLIENT-SURFACES.md section 6.2), and
    re-encoding somebody else's grammar on the way past is not the proxy's job.

    Where the two names differ — mlx-lm has no `--served-model-name` and answers
    to the resolved weights directory — one field has to change, so the document
    is re-serialised with `model` replaced in the position it already held.
    Nothing else is added, removed or reordered.
    """
    if resident.engine_model_name == resident.model_id:
        return raw
    return json.dumps({**body, "model": resident.engine_model_name}).encode("utf-8")


async def _watch_for_disconnect(request: Request) -> None:
    """Return once the caller's connection has gone away."""
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_SECONDS)


async def _post_unless_the_caller_leaves(
    client: httpx.AsyncClient, url: str, body: bytes, request: Request
) -> httpx.Response | None:
    """The upstream POST, raced against the caller hanging up.

    A bare `await client.post(...)` is not enough, and the gap is not cosmetic:
    nothing inside it watches the caller's own socket, so somebody who gives up
    after five seconds leaves the engine generating to the end of its token
    budget with no one to hand the answer to. Crucible runs one job at a time
    (DESIGN.md section 6), so that is not wasted time in the abstract — it is the
    next job's time. Dropping the connection is also the *only* cancel either app
    has for a chat (CLIENT-SURFACES.md, closing section), which makes this the
    cancel path rather than a refinement of one.

    Returns the engine's response, or None when the caller went first. In that
    case the upstream request has already been cancelled, and cancelling it is
    what closes the socket the engine is writing to — which is how the engine
    learns to stop.
    """
    post = asyncio.create_task(client.post(url, content=body, headers=JSON_HEADERS))
    watch = asyncio.create_task(_watch_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({post, watch}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watch.cancel()
    if post in done:
        return post.result()

    post.cancel()
    try:
        await post
    except asyncio.CancelledError:
        # Ours, not this handler's: the task was cancelled two lines above, and
        # awaiting it is how the cancellation is given time to reach httpx and
        # close the connection. Letting it propagate would report the caller's
        # own departure as this request being cancelled.
        pass
    return None


def _caller_gone(resident: Any) -> JSONResponse:
    """What the proxy answers a caller who is no longer there to read it.

    Nothing reads this: the socket it would travel down is closed. It exists
    because the handler still has to return something, and returning a body
    shaped like a completion would be a lie told to the log. 499 is nginx's code
    for a client that closed the request, and Crucible borrows it rather than
    inventing one.
    """
    return JSONResponse(
        status_code=499,
        content=ApiError(
            499,
            "client_disconnected",
            f"the caller closed the connection before the engine serving "
            f"{resident.model_id!r} answered; the engine's request was cancelled "
            "with it",
        ).body(),
    )


def _engine_unreachable(resident: Any, exc: Exception) -> ApiError:
    return ApiError(
        502,
        "engine_unreachable",
        f"the engine serving {resident.model_id!r} at {resident.base_url} did not "
        f"answer: {type(exc).__name__}: {exc}. Its log is {resident.log_path}",
    )


def _restore_model_id(raw: bytes, resident: Any) -> bytes:
    """Put Crucible's id back where the proxy substituted the engine's name.

    The request's `model` is rewritten on the way in to the name the engine
    answers to, because mlx-lm has no `--served-model-name` and answers to the
    resolved weights directory (`crucible/engines/mlx_lm.py`). OpenAI engines
    echo the name they were asked for, so without this the completion would come
    back naming a directory on the server's disk — and would name the Crucible id
    on vLLM, which *does* take a served name, so the answer to "what did I just
    talk to" would depend on the backend. Crucible's id is the contract in both
    directions; this is the second half of the one substitution the proxy makes,
    and nothing else in the body is touched.

    A backend whose engine already answers to the id (vLLM) is relayed byte for
    byte, as before.
    """
    if resident.engine_model_name == resident.model_id:
        return raw
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(
            502,
            "engine_response_unreadable",
            f"the engine serving {resident.model_id!r} answered 200 with a body "
            f"that is not JSON: {exc}. Its log is {resident.log_path}",
        ) from None
    if not isinstance(body, dict):
        raise ApiError(
            502,
            "engine_response_unreadable",
            f"the engine serving {resident.model_id!r} answered 200 with a JSON "
            f"{type(body).__name__}, not a completion object. Its log is "
            f"{resident.log_path}",
        )
    body["model"] = resident.model_id
    return json.dumps(body).encode("utf-8")


def _restore_model_id_in_frame(frame: bytes, resident: Any) -> bytes:
    """The same substitution inside one SSE frame of a streamed completion.

    Only `data:` lines carrying a JSON chunk are touched, and only their `model`
    field. `data: [DONE]`, comments, and any line the engine frames some other
    way are passed through as they arrived: mid-stream there is no way to raise,
    and a frame Crucible does not recognise is the engine's to explain.
    """
    lines = frame.split(b"\n")
    changed = False
    for index, line in enumerate(lines):
        if not line.startswith(b"data: "):
            continue
        payload = line[len(b"data: "):]
        if payload.strip() == b"[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(chunk, dict) or "model" not in chunk:
            continue
        chunk["model"] = resident.model_id
        lines[index] = b"data: " + json.dumps(chunk).encode("utf-8")
        changed = True
    return b"\n".join(lines) if changed else frame


class _RelayResponse(StreamingResponse):
    """A streamed relay whose upstream is closed however the relay ends.

    A caller who drops a streamed completion has to reach the engine, or it goes
    on producing tokens for nobody — and on one exclusive lane that is the next
    job's time. What closes the engine's end is closing the upstream response, so
    the only question is who is certain to do it.

    Not the relay generator, is the answer. Measured 2026-09-13 against uvicorn
    0.52 and starlette 1.6: uvicorn advertises ASGI `spec_version` 2.3, so
    `StreamingResponse.__call__` takes its task-group branch, a disconnect
    cancels the task pulling from the generator, the `CancelledError` lands in
    the generator's frame and a `finally` there would run. Read the 2.4 branch of
    that same function, though, and the disconnect arrives as an `OSError` out of
    `send` — raised *outside* the generator, which is then left suspended at its
    `yield` until asyncio's async-generator finalizer gets to it, whenever that
    is. Same proxy, same engine, two different answers depending on which branch
    Starlette takes.

    Which is not a thing this proxy should depend on. The upstream's lifetime
    belongs to the response that owns it, not to the generator that happens to be
    reading from it, and `__call__`'s `finally` runs on every one of those paths.
    """

    def __init__(self, *args: Any, upstream: httpx.Response, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._upstream = upstream

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._upstream.aclose()


async def _proxy_stream(
    client: httpx.AsyncClient, url: str, body: bytes, resident: Any
) -> Response:
    """Forward a streamed completion, SSE framing intact.

    The upstream response is opened before anything is returned, so a refusal
    from the engine comes back with the engine's own status code and body rather
    than as a 200 whose stream turns out to be an error.

    The bytes are relayed unchanged except for the one field the proxy
    substituted on the way in (see `_restore_model_id`); on a backend whose
    engine answers to the Crucible id there is nothing to undo and the relay is
    byte for byte.
    """
    log_path = resident.log_path
    request = client.build_request(
        "POST",
        url,
        content=body,
        headers=JSON_HEADERS,
        # A streamed completion emits a token at a time and may think for a long
        # while before the first one; there is no honest read deadline here.
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
        upstream = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        raise ApiError(
            502,
            "engine_unreachable",
            f"the resident engine at {url} did not answer: {type(exc).__name__}: "
            f"{exc}. Its log is {log_path}",
        ) from None

    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def relay() -> AsyncIterator[bytes]:
        if resident.engine_model_name == resident.model_id:
            async for chunk in upstream.aiter_bytes():
                yield chunk
            return
        # SSE frames end at a blank line, so the relay holds a partial frame
        # until it has one. Whatever is left when the engine stops is
        # forwarded as it stands rather than swallowed.
        buffer = b""
        async for chunk in upstream.aiter_bytes():
            buffer += chunk
            while b"\n\n" in buffer:
                frame, buffer = buffer.split(b"\n\n", 1)
                yield _restore_model_id_in_frame(frame, resident) + b"\n\n"
        if buffer:
            yield _restore_model_id_in_frame(buffer, resident)

    return _RelayResponse(
        relay(),
        upstream=upstream,
        status_code=200,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------- helpers


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
    }


def _materialise_inputs(
    config: Config, job: Job, inputs: dict[str, JobInput]
) -> None:
    """Write every declared input into the job's scratch dir before it is queued."""
    for name, declared in inputs.items():
        try:
            validate_member_name(name)
        except ValueError as exc:
            raise ApiError(400, "invalid_input_name", str(exc)) from None
        target = job.inputs_dir / name
        if declared.blob_id is not None:
            try:
                validate_member_name(declared.blob_id)
            except ValueError as exc:
                raise ApiError(400, "invalid_blob_id", str(exc)) from None
            source = Path(config.uploads_dir) / declared.blob_id
            if not source.is_file():
                raise ApiError(
                    400,
                    "unknown_blob",
                    f"input {name!r} names blob {declared.blob_id!r}, which this "
                    "server does not hold",
                )
            shutil.copyfile(source, target)
        elif declared.inline_base64 is not None:
            try:
                payload = base64.b64decode(declared.inline_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(
                    400,
                    "invalid_inline_input",
                    f"input {name!r} is not valid base64: {exc}",
                ) from None
            target.write_bytes(payload)
        else:  # unreachable: JobInput's validator requires exactly one source
            raise ApiError(
                400,
                "invalid_input",
                f"input {name!r} names neither a blob nor inline bytes",
            )


def _last_event_id(request: Request) -> int:
    raw = request.headers.get("last-event-id")
    if raw is None:
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        raise ApiError(
            400,
            "invalid_last_event_id",
            f"Last-Event-ID must be an integer, got {raw!r}",
        ) from None
    if value < 0:
        raise ApiError(
            400, "invalid_last_event_id", f"Last-Event-ID must not be negative: {value}"
        )
    return value


def _format_event(event: dict[str, Any]) -> str:
    return (
        f"id: {event['id']}\n"
        f"event: {event['event']}\n"
        f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
    )


async def _session_event_stream(
    request: Request, session: StreamSession, last_event_id: int
) -> AsyncIterator[str]:
    """The job stream's shape, over a session's log instead of a job's events.

    Deliberately a second function rather than a parameterised one. The two look
    alike and are not the same: a job's events end at a terminal status and its
    log lives as long as the job does, while a session's end at `closed` and its
    log is pruned behind the readers (`StreamSession._prune`), so the cursor here
    has to be written back onto the reader rather than kept local. Folding them
    together would mean one of the two behaviours becoming a flag.
    """
    reader = session.attach(last_event_id)
    try:
        while True:
            for event in session.frames_after(reader.delivered):
                reader.delivered = event.id
                yield _format_event(
                    {"id": event.id, "event": event.event, "data": event.data}
                )
                if event.event == "closed":
                    return
            reader.waiter.clear()
            if session.frames_after(reader.delivered):
                continue
            try:
                await asyncio.wait_for(reader.waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                # The job stream's shape, byte for byte, and measured on
                # 2026-09-13 to be enough rather than assumed to be. A real
                # socket close cancels this generator through starlette's
                # disconnect listener in about 0.17 s, so this poll is for the
                # OTHER kind of departure: a tunnel that died without closing
                # anything, where nothing but a write that fails can discover
                # it. A shorter `is_disconnected()` tick was written, measured
                # to change neither case, and taken back out.
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        # Detaching is what starts the grace window. A dropped stream does not
        # cancel immediately — that is the whole reason this door is SSE — so
        # this marks the session unattended and the watchdog does the rest.
        session.detach(reader)


async def _event_stream(
    request: Request, store: JobStore, job: Job, last_event_id: int
) -> AsyncIterator[str]:
    """Replay everything after `last_event_id`, then follow live until terminal."""
    waiter = store.subscribe(job)
    index = last_event_id
    try:
        while True:
            while index < len(job.events):
                event = job.events[index]
                index += 1
                yield _format_event(event)
                if event["event"] in TERMINAL_EVENTS:
                    return
            waiter.clear()
            if index < len(job.events):
                continue
            try:
                await asyncio.wait_for(waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        store.unsubscribe(job, waiter)
