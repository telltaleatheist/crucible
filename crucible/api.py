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
from .residency import Residency

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
                    # type that put it there. Today the only resident thing is an
                    # LLM engine; phase 3's generalised residency adds tts voices
                    # and phase 4's aligner, and they land here beside it.
                    "kind": "llm",
                    "id": resident.model_id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                }
            ),
            "holders": holders,
            "detail": state.detail,
        }

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
        store: JobStore = request.app.state.store
        plugin = resolve(store.registry, body.type)
        model = resolve_model(plugin, body.model)
        # Every refusal a job type can make about host state happens here, before
        # the job exists, so the client is told by name instead of watching a job
        # fail (PHASE2-LLM.md section 5).
        plugin.preflight(model, body.params)

        job = store.create(body.type, model, body.params)
        try:
            _materialise_inputs(config, job, body.inputs)
        except ApiError:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise
        store.enqueue(job)
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
