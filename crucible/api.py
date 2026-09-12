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

from . import API_VERSION, VERSION
from .backend import Backend
from .config import Config
from .errors import ApiError
from .jobs import Residency, build_registry, model_rows, resolve, resolve_model
from .jobs.base import Job, validate_member_name
from .jobs.queue import JobStore

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
    registry = build_registry(config, residency)

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
        capabilities = [
            {
                "job_type": name,
                "models": [m.to_dict() for m in plugin.describe_models()],
            }
            for name, plugin in sorted(store.registry.items())
        ]
        if config.enable_llm:
            # PHASE2-LLM.md section 5: `/info` gains an `llm` capability whose
            # models are the `/v1/models` rows. `load-model` and `unload-model`
            # are listed above as themselves, because they are what you POST.
            capabilities.append(
                {
                    "job_type": "llm",
                    "models": model_rows(config, backend.kind, residency),
                }
            )
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
            "resident_models": residency.ids(),
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
        return model_rows(config, backend.kind, residency)

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
        """The resident model in OpenAI's list shape, or an empty list."""
        resident = residency.resident
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
                }
            ],
        }

    @private.post("/openai/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """Proxied to the resident engine. Never loads one (section 5)."""
        body = _chat_body(await request.body())
        requested = body.get("model")
        if not isinstance(requested, str) or requested == "":
            raise ApiError(
                400,
                "model_required",
                "a chat request must name a model; this server proxies only to the "
                "model that is resident",
            )
        resident = residency.resident
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

        forwarded = dict(body)
        forwarded["model"] = resident.engine_model_name
        url = f"{resident.base_url}/v1/chat/completions"
        client: httpx.AsyncClient = request.app.state.http

        if body.get("stream") is True:
            return await _proxy_stream(client, url, forwarded, resident.log_path)
        try:
            upstream = await client.post(url, json=forwarded)
        except httpx.HTTPError as exc:
            raise _engine_unreachable(resident, exc) from None
        return Response(
            content=upstream.content,
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


def _engine_unreachable(resident: Any, exc: Exception) -> ApiError:
    return ApiError(
        502,
        "engine_unreachable",
        f"the engine serving {resident.model_id!r} at {resident.base_url} did not "
        f"answer: {type(exc).__name__}: {exc}. Its log is {resident.log_path}",
    )


async def _proxy_stream(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], log_path: Any
) -> Response:
    """Forward a streamed completion byte for byte, SSE framing intact.

    The upstream response is opened before anything is returned, so a refusal
    from the engine comes back with the engine's own status code and body rather
    than as a 200 whose stream turns out to be an error.
    """
    request = client.build_request(
        "POST",
        url,
        json=body,
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
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        relay(),
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
