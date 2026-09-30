from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

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
from ..capabilitystore import decide_for, write_capability
from ..config import Config, load_config
from ..connect import PairingRequests
from ..errors import ApiError, ConfigError
from ..inflight import InFlight
from ..installonsubmit import InstallOnSubmit
from ..jobs import build_registry, disabled_error
from ..jobs.queue import JobStore
from ..jobs.line import WaitingLine
from ..leases import Leases
from ..queuepump import QueuePump
from ..residency import Residency
from ..settle import Settlement
from ..tasks import TaskStore
from ..tasks.states import ReloadRefused
from ..ttsstream import StreamManager
from ..voicecatalog import seed_unresolved
from .context import AppContext, Routers, Services
from .deps import (
    BeforeEveryRequest,
    error_response,
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
from .routes import queue as queue_routes

ROUTE_MODULES = (
    pairing,
    capability,
    settings,
    info,
    peer,
    catalog_routes,
    activity,
    lease_routes,
    queue_routes,
    voices,
    tts_stream,
    jobs,
    resumable,
    tasks,
    openai,
    decide,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def _say(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr)


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


def _validation_refusal(request: Request, exc: RequestValidationError) -> ApiError:
    problems = [
        {
            "location": [str(part) for part in problem["loc"]],
            "type": problem["type"],
            "message": problem["msg"],
        }
        for problem in exc.errors()
    ]
    return ApiError(
        400,
        "invalid_request",
        _validation_message(request, problems),
        {"problems": problems},
    )


def _http_refusal(exc: StarletteHTTPException) -> ApiError:
    code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
    return ApiError(exc.status_code, code, str(exc.detail))


class RegistryKeeper:
    def __init__(
        self, config: Config, backend: Backend, residency: Residency, leases: Leases
    ) -> None:
        self._config = config
        self._backend = backend
        self._residency = residency
        self._leases = leases
        self.registry = self._built(config)

    def _built(self, config: Config) -> dict[str, Any]:
        return build_registry(config, self._backend, self._residency, self._leases)

    def take_up_enabled_types(self, live: Config) -> None:
        try:
            wanted = self._built(live)
        except Exception as exc:
            _say(
                "config.toml turned a job type on, but its plugin could not be "
                f"built: {type(exc).__name__}: {exc}"
            )
            return
        added = sorted(name for name in wanted if name not in self.registry)
        for name in added:
            self.registry[name] = wanted[name]
        if added:
            _say(f"took up newly enabled job type(s) {added}")

    def reload(self, holder: Callable[[], Any]) -> list[str]:
        held = holder()
        if held is not None:
            raise ReloadRefused(held)
        self._config.adopt(load_config(self._config.home))
        rebuilt = self._built(self._config)
        self.registry.clear()
        self.registry.update(rebuilt)
        return sorted(self.registry)

    def take_up_installed(self) -> list[str]:
        self._config.adopt(load_config(self._config.home))
        self.take_up_enabled_types(self._config)
        return sorted(self.registry)


class ConfigFollower:
    def __init__(self, config: Config, keeper: RegistryKeeper) -> None:
        self._config = config
        self._keeper = keeper
        self._failed: str | None = None

    def __call__(self) -> None:
        try:
            if self._config.follow_file():
                _say("config.toml moved on disk; the server adopted it")
                self._keeper.take_up_enabled_types(self._config)
        except ConfigError as exc:
            if self._failed != str(exc):
                self._failed = str(exc)
                _say(
                    "config.toml could not be re-read; serving the last good "
                    f"document: {exc}"
                )


def _decider(config: Config, backend: Backend) -> Callable[[str], ApiError]:
    def decide_here(job_type: str) -> ApiError:
        try:
            write_capability(config, backend, decide_for(config, backend), {})
            config.adopt(load_config(config.home))
            _say("no capability record; decided this card and recorded it")
        except Exception as exc:
            _say(f"could not decide this card: {type(exc).__name__}: {exc}")
        return disabled_error(job_type, config)

    return decide_here


def _services(
    config: Config,
    backend: Backend,
    residency: Residency,
    leases: Leases,
    keeper: RegistryKeeper,
) -> Services:
    store = JobStore(config, backend, keeper.registry)
    line = WaitingLine(store, lambda: _lease_client(leases))
    streams = StreamManager(residency)
    inflight = InFlight()
    settlement = Settlement(
        residency=residency, store=store, leases=leases, inflight=inflight
    )
    store.attach_settlement(settlement)
    streams.when_closed(settlement.settle_quietly)
    task_store = TaskStore(
        config,
        backend,
        reload=lambda: keeper.reload(settlement.holder),
        holder=settlement.holder,
        take_up=keeper.take_up_installed,
        in_use=lambda subject: catalog_routes.held_on_card(residency, leases, subject),
    )
    return Services(
        leases=leases,
        store=store,
        line=line,
        streams=streams,
        inflight=inflight,
        ollama_contexts=upstreams.OllamaContexts(),
        settings_history=settings_module.History(),
        removals=catalog.Removals(),
        peer=peer_module.PeerState(),
        pairing_requests=PairingRequests(open_pairing=config.open_pairing),
        settlement=settlement,
        tasks=task_store,
        installs=InstallOnSubmit(config, backend, task_store),
    )


def _lease_client(leases: Leases) -> str | None:
    lease = leases.current()
    return None if lease is None else lease.client


def _attach_queue_pump(app: FastAPI, ctx: AppContext) -> None:
    pump = QueuePump(app.state.line, ctx.admission)
    app.state.store.when_idle(pump.wake)
    app.state.line.when_changed(pump.wake)
    app.state.queue_pump = pump


def _proxy_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT,
            read=PROXY_READ_TIMEOUT,
            write=60.0,
            pool=10.0,
        ),
        limits=httpx.Limits(keepalive_expiry=PROXY_KEEPALIVE_EXPIRY),
    )


def _look_up_unresolved_voice_tags(home: Path) -> None:
    def look_up() -> None:
        try:
            for found in seed_unresolved(home):
                _say(
                    f"voice tag {found.hf_repo}@{found.ref}: "
                    + (found.revision or f"not resolved ({found.error})")
                )
        except Exception as exc:
            _say(
                f"could not look up the voices' tags ({type(exc).__name__}: {exc}); "
                "run `crucible voices check-updates`"
            )

    threading.Thread(target=look_up, name="crucible-voice-tags", daemon=True).start()


def _lifespan(residency: Residency) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: JobStore = app.state.store
        app.state.started_at = time.monotonic()
        store.restore()
        residency.start_reclaiming()
        store.start()
        app.state.queue_pump.start()
        if app.state.config.enable_tts:
            _look_up_unresolved_voice_tags(app.state.config.home)
        app.state.http = _proxy_client()
        try:
            yield
        finally:
            await app.state.tasks.stop()
            await app.state.queue_pump.stop()
            await store.stop()
            await app.state.http.aclose()
            await asyncio.to_thread(app.state.streams.shutdown)
            await asyncio.to_thread(residency.shutdown)

    return lifespan


def _answer_refusals(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(_http_refusal(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(_validation_refusal(request, exc))


def _routers() -> Routers:
    behind_the_token = [Depends(require_auth), Depends(require_api_version)]
    return Routers(
        public=APIRouter(prefix="/v1"),
        private=APIRouter(prefix="/v1", dependencies=behind_the_token),
        openai=APIRouter(prefix="/openai/v1", dependencies=behind_the_token),
        peer=APIRouter(
            prefix="/v1/peer",
            dependencies=[Depends(require_peer_auth), Depends(require_peer_api_version)],
        ),
    )


def _mount_routes(app: FastAPI, ctx: AppContext) -> None:
    routers = _routers()
    for module in ROUTE_MODULES:
        module.register(routers, ctx)
    for router in (routers.public, routers.private, routers.openai, routers.peer):
        app.include_router(router)


def _mount_operator_page(app: FastAPI) -> None:
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


def create_app(config: Config, backend: Backend) -> FastAPI:
    residency = Residency(config)
    leases = Leases()
    keeper = RegistryKeeper(config, backend, residency, leases)
    app = FastAPI(
        title="Crucible",
        version=VERSION,
        lifespan=_lifespan(residency),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config
    app.state.backend = backend
    app.state.residency = residency
    app.state.bind_host = config.host
    app.state.bind_port = config.port
    app.add_middleware(BeforeEveryRequest, step=ConfigFollower(config, keeper))
    _services(config, backend, residency, leases, keeper).publish(app)
    Path(config.jobs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.uploads_dir).mkdir(parents=True, exist_ok=True)
    _answer_refusals(app)
    ctx = AppContext(
        app=app,
        config=config,
        backend=backend,
        residency=residency,
        decide_here=_decider(config, backend),
    )
    _attach_queue_pump(app, ctx)
    _mount_routes(app, ctx)
    _mount_operator_page(app)
    return app
