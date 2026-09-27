from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

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

#: Where the operator page lives, INSIDE the package, so one path works from a
#: checkout and from an installed wheel alike. It ships as package data
#: (`[tool.setuptools.package-data]` in pyproject.toml) rather than being found
#: relative to a repo root, because a wheel installed on the Mac has no repo
#: root and the page has to travel with the code that serves it.
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def create_app(config: Config, backend: Backend) -> FastAPI:
    """Build the ASGI app for one server instance."""
    residency = Residency(config)
    # CONSTRUCTED HERE, not at `app.state.leases` below, because the registry
    # needs it: since 2026-09-20 a `load-model`/`load-voice` can be asked to
    # hold what it made resident (`params.lease`), and the loaders are built in
    # `build_registry`. One register, handed to both, so the lease a load opens
    # and the lease `POST /v1/models/{id}/lease` opens are the same one — a
    # second instance would be two servers disagreeing about who holds the card.
    leases = Leases()
    registry = build_registry(config, backend, residency, leases)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: JobStore = app.state.store
        # MONOTONIC, not a wall clock: uptime is a duration, and a duration
        # computed across an NTP correction or a DST jump is how a bench ends up
        # reporting that a server has been up for minus four minutes.
        app.state.started_at = time.monotonic()
        # BEFORE THE LANE AND BEFORE THE FIRST REQUEST. A client asking about
        # its job during the restore must not be told 404 about a job that is
        # about to exist, and the lane must not reap a directory whose record
        # has not been read yet.
        store.restore()
        store.start()
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=PROXY_CONNECT_TIMEOUT,
                read=PROXY_READ_TIMEOUT,
                write=60.0,
                pool=10.0,
            ),
            # Below the engine's keep-alive; see `PROXY_KEEPALIVE_EXPIRY`.
            limits=httpx.Limits(keepalive_expiry=PROXY_KEEPALIVE_EXPIRY),
        )
        try:
            yield
        finally:
            # Before the job lane, because an operator task can be holding a
            # download thread and a pip subprocess, and both want telling
            # before the process goes. A task's cancel is cooperative and
            # returns in milliseconds (`crucible/tasks.py`).
            await app.state.tasks.stop()
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

    def take_up_enabled_types(live: Config) -> None:
        """Add the plugin of every type the config now enables and the registry lacks.

        2026-09-26, #42; `follow_the_config_file` says why. A build that fails
        is logged, not raised: the request goes on, and the refusal a client
        then gets (`disabled_error`) says the type is on but not taken up.
        """
        try:
            wanted = build_registry(live, backend, residency, leases)
        except Exception as exc:  # noqa: BLE001 - reported; the request proceeds
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
        """Every request sees the config.toml that is on disk NOW.

        `Config.follow_file()` — one stat per request, a re-read only when
        the file moved. The record `GET /v1/capability` serves, the `[jobs]`
        flags the doors refuse on, the routes and the upstreams all follow,
        because `adopt()` replaces the one Config object's fields in place
        and every route, the residency and the store close over that object.

        A JOB TYPE TURNED ON IS TAKEN UP HERE TOO (2026-09-26, fresh-install
        #42). The registry is built from the flags at start, and this used to
        say a flag going on "needs a start, which `crucible install` performs
        anyway". It does not: on kylies-pc `crucible install rvc` wrote
        `enable_rvc = true` and the running server answered
        `job_type_disabled` until somebody restarted it by hand. So when the
        adopted file turns a type on, its plugin is built from the same
        config, backend, residency and leases and ADDED to the registry. Only
        added: an instance already there is never replaced, so a job running
        on it is untouched, which is `reload_registry`'s concern and the
        reason that one refuses while the card is held. A flag turned OFF is
        honoured at once, by the doors.

        A file that will not read is REPORTED AND NOT SERVED: the last good
        document stays, the failure is logged once per stamp rather than per
        request, and the request proceeds. A half-written config.toml is
        weather (the writer stages and `os.replace`s, so it should not happen);
        a broken one is misconfiguration the operator can repair, and refusing
        every request over it would take `/v1/ping` down with the record.
        """
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

    # A PLAIN ASGI STEP, NEVER `@app.middleware("http")` (2026-09-24). That
    # decorator is starlette's `BaseHTTPMiddleware`, and it hands every route a
    # `receive` of its own making: a task group around the server's `receive`.
    # `Request.is_disconnected()` asks `receive` inside an already-cancelled
    # scope, and through that task group the answer is always lost to the
    # cancellation — measured: 116 polls over thirty seconds after the caller
    # had gone, every one `False`. From 1.0.18 (7babb40, which added this step
    # as that decorator) until this fix, no non-streamed chat and no decision
    # ever noticed its caller leave, and the Mac went on answering a stopped
    # Foundry run for 45 s on 2026-09-24. This step reads a file; it has no
    # business wrapping the request's channel, so it does not.
    app.add_middleware(BeforeEveryRequest, step=follow_the_config_file)
    # WHERE THIS SERVER IS REALLY LISTENING, which the config alone cannot say:
    # `crucible serve --host 0.0.0.0` overrides `[server] host` for that run, and
    # `GET /v1/setup` would otherwise hand out pairing lines for the address the
    # file remembers rather than the one uvicorn bound. `cmd_serve` overwrites
    # these when it is given a flag; the config's values are the truth when it is
    # not, which is every service-managed server (`crucible service install`
    # bakes the config's host and port into the unit).
    app.state.bind_host = config.host
    app.state.bind_port = config.port
    app.state.store = JobStore(config, backend, registry)
    app.state.streams = StreamManager(residency)
    app.state.inflight = InFlight()
    # What each Ollama tag's own context is, remembered per digest
    # (`upstreams.OllamaContexts`, PHASE15-HOST.md section 3.4a).
    app.state.ollama_contexts = upstreams.OllamaContexts()
    # The last few settings writes, for `/v1/activity` (PHASE15-HOST.md section
    # 3.2). In memory and a restart forgets, like a task's record: this is a
    # display of "who changed what just now" when two apps and a page all edit
    # one server, not an audit log, and it never holds a key.
    app.state.settings_history = settings_module.History()
    # The last few subject removals, for `/v1/activity` (3.5a). In memory and
    # a restart forgets, like the settings history: deleting gigabytes is the
    # one catalog act nobody can undo, so it is the one that most needs to say
    # who asked.
    app.state.removals = catalog.Removals()
    # WHO MANAGES THIS ENGINE (PHASE17-ORCHESTRATOR.md 2.3). In memory, and a
    # restart forgets — that is the design and not a shortcut. A claim written
    # to disk would outlive the orchestrator that made it: uninstall the tray,
    # reboot, and the engine still names a door that will never answer again,
    # which is `docs/ARCHITECTURE.md`'s one shape. The relation is
    # RE-ASSERTED instead, on the orchestrator's next watch tick.
    app.state.peer = peer_module.PeerState()
    # The POLICY comes from the config, not from this module's idea of a default:
    # an operator who wrote `open_pairing = false` must not have it re-opened by
    # the construction site.
    app.state.pairing_requests = PairingRequests(open_pairing=config.open_pairing)
    # In memory, and a restart forgets: a lease protects a resident model, and a
    # restarted server holds none (crucible/leases.py). Built at the top of
    # `create_app` because the loaders need it too — see there.
    app.state.leases = leases
    # OWEN'S RULING, 2026-09-14: *"Models should always be unloaded when we're
    # done with them. Every time."* The settlement is the one place that decides
    # nothing holds the card any more, and it is wired to all four of the things
    # that can hold it — the lane, the lease, the claim and the chats in flight
    # (crucible/settle.py). Built last because it reads every one of them.
    app.state.settlement = Settlement(
        residency=residency,
        store=app.state.store,
        leases=app.state.leases,
        inflight=app.state.inflight,
    )
    app.state.store.attach_settlement(app.state.settlement)
    app.state.streams.when_closed(app.state.settlement.settle_quietly)

    def reload_registry() -> list[str]:
        """Make an installed job type reachable, in place. **Event loop only.**

        PHASE13-OPERATOR.md section 3.4, which is where the decision and its
        reason are written. The whole of it is here rather than in
        `crucible/tasks.py` because it is a statement about how a server is
        ASSEMBLED — the config object, the registry, the residency — and a task
        module that reached for those would be a second owner of that.

        THE FOUR FACTS ARE READ AGAIN, HERE, and this is the second read (the
        first gated the task at POST). Minutes of pip have passed since then and
        a job may have been admitted; swapping the registry underneath it would
        hand its `_restamp_provenance` a plugin instance built from a different
        config, and a capability step that turned a flag OFF would remove the
        very type it is running. So a holder found here is a refusal that fails
        the task, naming it — a loud wrong answer rather than a quiet one (R3).

        Nothing awaits between the read and the swap, so the two are one atomic
        stretch on the loop, which is `JobStore.enqueue`'s property and its
        reason.
        """
        held = app.state.settlement.holder()
        if held is not None:
            raise ReloadRefused(held)
        # The SAME Config object, re-read from the same file. See
        # `Config.adopt`: every route, the residency, the store and every
        # plugin holds a reference to this one object, and handing half of them
        # a second one is R1's defect built on purpose.
        config.adopt(load_config(config.home))
        # The SAME residency, so what was resident stays resident and the new
        # plugin instances hold the live engine. The dict's CONTENTS are
        # replaced rather than the dict, because `JobStore` holds it by
        # reference and a store pointed at the old mapping would accept exactly
        # the job types the rest of the server had stopped offering.
        rebuilt = build_registry(config, backend, residency, leases)
        registry.clear()
        registry.update(rebuilt)
        return sorted(registry)

    def take_up_installed() -> list[str]:
        """The additive reload an install-on-submit ends with. **Event loop only.**

        2026-09-26, Owen's ruling (`crucible/installonsubmit.py`). Not
        `reload_registry`: that swap refuses while anything holds the card,
        and this install started while a job may be running. It re-reads the
        file and ADDS the plugins it now enables, replacing none, which is
        `take_up_enabled_types` (#42) and exactly what this server already
        does when `crucible install` runs beside it.
        """
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
        """Decide this card and record it, for a server that never had. **Loop only.**

        Owen, 2026-09-27, on a server with no capability record: *"Yes, it
        should automatically be checked"*. `job_type_disabled` / `undecided`
        used to send the operator to `crucible capability`, a command. The
        server now makes the walk that command and `crucible install` make
        (`cli._decide_here`: this card's size, vendor, the ladder's measured
        facts via `ladder.card_for`, the config's choices) and writes the record
        the way they do (`cli._write_capability`). No flag is changed: like
        `crucible capability --write`, deciding turns nothing on; installing
        does. Returns the refusal the recorded card now gives.
        """
        from .. import cli  # cli imports this module

        try:
            cli._write_capability(config, backend, cli._decide_here(config, backend), {})
            config.adopt(load_config(config.home))
            print(
                "crucible: no capability record; decided this card and recorded it",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - reported; the old refusal stands
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
    # THE OPENAI-COMPATIBLE SURFACE, WHERE OPENAI CLIENTS LOOK FOR IT. The two
    # OpenAI-shaped routes below are also mounted at `/openai/v1/...`, because
    # that is the shape every OpenAI client composes: a base URL, then `/v1/models`
    # and `/v1/chat/completions`. Foundry's engine does exactly that (its
    # `normaliseVllmEndpoint` appends `/v1` unless the base already ends in a
    # version), and on 2026-09-13 the first real Foundry act against a Crucible
    # asked for `/v1/openai/v1/models` and got a 404 — the door existed and no
    # OpenAI client could reach it. Same handlers, same auth, same version
    # header; nothing is duplicated but the path, and the path is the other
    # protocol's convention rather than this API's. The SDK keeps `/v1/openai/*`,
    # which is Crucible's own namespace for the same door.
    openai_router = APIRouter(
        prefix="/openai/v1",
        dependencies=[Depends(require_auth), Depends(require_api_version)],
    )
    # THE RELATION'S OWN DOOR (PHASE17-ORCHESTRATOR.md section 2). Same token,
    # same version header, two refusals with the relation's names on them —
    # `require_peer_auth` says why that is not duplication.
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
        leases=leases,
        registry=registry,
        decide_here=decide_here,
    )
    for module in ROUTE_MODULES:
        module.register(routers, ctx)

    app.include_router(public)
    app.include_router(private)
    app.include_router(openai_router)
    app.include_router(peer_router)

    # ------------------------------------------------------- the static page
    #
    # PHASE13-OPERATOR.md section 1 and section 4. `GET /` and `GET /ui/*` are
    # the ONLY public surface besides `/v1/ping`, and they are public because
    # there is no secret in any of them: the page asks for the token, or reads
    # it out of the URL fragment a pairing line put there, and keeps it in the
    # browser's own storage. A fragment never reaches this server, which is why
    # the token travels in one.
    #
    # Mounted LAST, after every router, so nothing it serves can shadow a
    # route. `/ui` cannot collide with `/v1` in any case — the two prefixes are
    # disjoint and a test asserts that `/ui/v1/info` is a 404 from the static
    # files rather than the API with an extra path segment.

    @app.get("/", include_in_schema=False)
    async def operator_page() -> Response:
        """The door, or a named refusal saying the build is missing its page.

        A wheel built without `[tool.setuptools.package-data]` would have an
        API and no page, and the honest report of that is a 503 that names the
        directory — not a 404, which reads as "there is no page here", and not
        a crash at start-up, which would take the whole API down because a
        static file is missing. `tests/test_ui_mount.py` asserts this build
        HAS the directory, so the refusal below can only ever mean a broken
        package rather than a normal state.

        **WHY THIS REDIRECTS RATHER THAN SERVING THE BYTES HERE (2026-09-14,
        building section 4).** The page is three files and the other two are
        its own: `index.html` asks for `app.css` and `app.js` by RELATIVE name,
        which is what lets the same three bytes be served from any mount. Sent
        from `/`, those names resolve to `/app.css` and `/app.js`, which are
        not mounted — so the page would arrive unstyled and inert. The three
        ways out were: serve the page at `/` and register two more routes for
        its assets (one file reachable at two URLs, and a third place to
        remember when a fourth file is added); put `/ui/` into the HTML
        (an absolute path, which pins the page to this mount and makes it
        unservable from anywhere else); or make `/ui/` the page's one home and
        have `/` say so. The last is the only one that leaves a single owner of
        where the page lives.

        The pairing line's fragment survives it: a redirect whose target
        carries no fragment of its own keeps the request's, in every browser,
        so `http://host:7100/#token=…` lands on `/ui/#token=…` signed in.
        """
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
        # `html=True` so `/ui/` is the page rather than a 404 — it is the
        # directory the page lives in, and the door above sends every visitor
        # to it. It does NOT make a missing file fall back to the index: a path
        # under `/ui` that is not a file is still a 404, which is what keeps
        # `/ui/v1/info` from answering with HTML.
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
