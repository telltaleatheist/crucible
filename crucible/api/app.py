from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from .. import VERSION, catalog, upstreams
from .. import peer as peer_module
from .. import settings as settings_module
from ..backend import Backend
from ..config import Config, load_config
from ..connect import PairingRequests
from ..errors import ApiError, ConfigError
from ..inflight import InFlight
from ..installonsubmit import InstallOnSubmit
from ..jobs import build_registry, disabled_error
from ..jobs.queue import JobStore
from ..leases import Leases
from ..residency import Residency
from ..settle import Settlement
from ..tasks import ReloadRefused, TaskStore
from ..ttsstream import StreamManager
from .context import AppContext, Routers
from .deps import (
    BeforeEveryRequest,
    _error_response,
    require_api_version,
    require_auth,
    require_peer_api_version,
    require_peer_auth,
)
from .proxy import PROXY_CONNECT_TIMEOUT, PROXY_KEEPALIVE_EXPIRY, PROXY_READ_TIMEOUT
from .routes import (
    activity,
    capability,
    decide,
    info,
    jobs,
    openai,
    pairing,
    peer,
    resumable,
    settings,
    tasks,
    tts_stream,
    voices,
)
from .routes import catalog as catalog_routes
from .routes import leases as lease_routes

ROUTE_MODULES = (
    pairing,
    capability,
    settings,
    info,
    peer,
    catalog_routes,
    activity,
    lease_routes,
    voices,
    tts_stream,
    jobs,
    resumable,
    tasks,
    openai,
    decide,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def _validation_message(request: Request, problems: list[dict[str, Any]]) -> str:
    route = f"{request.method} {request.url.path}"
    if not problems:
        return f"the request to {route} is not valid"
    first = problems[0]
    where = ".".join(first["location"]) or "the request"
    more = len(problems) - 1
    also = "" if more == 0 else f" (and {more} more problem(s), listed in details)"
    return (
        f"the request to {route} is not valid: {where}: {first['message']}{also}"
    )


def create_app(config: Config, backend: Backend) -> FastAPI:
    residency = Residency(config)
    leases = Leases()
    registry = build_registry(config, backend, residency, leases)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: JobStore = app.state.store
        app.state.started_at = time.monotonic()
        store.restore()
        residency.start_reclaiming()
        store.start()
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=PROXY_CONNECT_TIMEOUT,
                read=PROXY_READ_TIMEOUT,
                write=60.0,
                pool=10.0,
            ),
            limits=httpx.Limits(keepalive_expiry=PROXY_KEEPALIVE_EXPIRY),
        )
        try:
            yield
        finally:
            await app.state.tasks.stop()
            await store.stop()
            await app.state.http.aclose()
            await asyncio.to_thread(app.state.streams.shutdown)
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

    def take_up_enabled_types(live: Config) -> None:
        try:
            wanted = build_registry(live, backend, residency, leases)
        except Exception as exc:
            print(
                f"crucible: config.toml turned a job type on, but its plugin could "
                f"not be built: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return
        added = sorted(name for name in wanted if name not in registry)
        for name in added:
            registry[name] = wanted[name]
        if added:
            print(f"crucible: took up newly enabled job type(s) {added}", file=sys.stderr)

    def follow_the_config_file() -> None:
        live: Config = app.state.config
        try:
            if live.follow_file():
                print("crucible: config.toml moved on disk; the server adopted it", file=sys.stderr)
                take_up_enabled_types(live)
        except ConfigError as exc:
            failed = getattr(app.state, "config_follow_failed", None)
            if failed != str(exc):
                app.state.config_follow_failed = str(exc)
                print(
                    f"crucible: config.toml could not be re-read; serving the last "
                    f"good document: {exc}",
                    file=sys.stderr,
                )

    app.add_middleware(BeforeEveryRequest, step=follow_the_config_file)
    app.state.bind_host = config.host
    app.state.bind_port = config.port
    app.state.store = JobStore(config, backend, registry)
    app.state.streams = StreamManager(residency)
    app.state.inflight = InFlight()
    app.state.ollama_contexts = upstreams.OllamaContexts()
    app.state.settings_history = settings_module.History()
    app.state.removals = catalog.Removals()
    app.state.peer = peer_module.PeerState()
    app.state.pairing_requests = PairingRequests(open_pairing=config.open_pairing)
    app.state.leases = leases
    app.state.settlement = Settlement(
        residency=residency,
        store=app.state.store,
        leases=app.state.leases,
        inflight=app.state.inflight,
    )
    app.state.store.attach_settlement(app.state.settlement)
    app.state.streams.when_closed(app.state.settlement.settle_quietly)

    def reload_registry() -> list[str]:
        held = app.state.settlement.holder()
        if held is not None:
            raise ReloadRefused(held)
        config.adopt(load_config(config.home))
        rebuilt = build_registry(config, backend, residency, leases)
        registry.clear()
        registry.update(rebuilt)
        return sorted(registry)

    def take_up_installed() -> list[str]:
        config.adopt(load_config(config.home))
        take_up_enabled_types(config)
        return sorted(registry)

    app.state.tasks = TaskStore(
        config,
        backend,
        reload=reload_registry,
        holder=app.state.settlement.holder,
        take_up=take_up_installed,
    )

    def decide_here(job_type: str) -> ApiError:
        from .. import cli

        try:
            cli._write_capability(config, backend, cli._decide_here(config, backend), {})
            config.adopt(load_config(config.home))
            print(
                "crucible: no capability record; decided this card and recorded it",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"crucible: could not decide this card: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return disabled_error(job_type, config)

    app.state.installs = InstallOnSubmit(config, backend, app.state.tasks)

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
                    "message": _validation_message(request, problems),
                    "details": {"problems": problems},
                }
            },
        )

    public = APIRouter(prefix="/v1")
    private = APIRouter(
        prefix="/v1", dependencies=[Depends(require_auth), Depends(require_api_version)]
    )
    openai_router = APIRouter(
        prefix="/openai/v1",
        dependencies=[Depends(require_auth), Depends(require_api_version)],
    )
    peer_router = APIRouter(
        prefix="/v1/peer",
        dependencies=[Depends(require_peer_auth), Depends(require_peer_api_version)],
    )

    routers = Routers(public=public, private=private, openai=openai_router, peer=peer_router)
    ctx = AppContext(
        app=app,
        config=config,
        backend=backend,
        residency=residency,
        decide_here=decide_here,
    )
    for module in ROUTE_MODULES:
        module.register(routers, ctx)

    app.include_router(public)
    app.include_router(private)
    app.include_router(openai_router)
    app.include_router(peer_router)

    @app.get("/", include_in_schema=False)
    async def operator_page() -> Response:
        index = UI_DIR / "index.html"
        if not index.is_file():
            raise ApiError(
                503,
                "ui_missing",
                f"this build has no operator page: {index} is not there. The "
                "page ships as package data (`crucible/ui/`); a wheel built "
                "without it serves the API and nothing else",
            )
        return RedirectResponse("/ui/", status_code=307)

    if UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
