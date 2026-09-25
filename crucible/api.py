"""API v1 — exactly the surface in DESIGN.md section 4.

Base path is `/v1`. Protected routes need `Authorization: Bearer <token>` and
`X-Crucible-Api: 1`, checked in that order. Public discovery (`GET /v1/ping`)
and the limited pairing start/poll exchange do not require an existing token.
Pairing approval always requires authentication; start/poll require the version header.
Errors are always `{"error": {"code", "message", "details"?}}`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.background import BackgroundTask
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import (
    API_HEADER,
    API_VERSION,
    VERSION,
    accelerator,
    catalog,
    pages as pages_module,
    pairing,
    upstreams,
    weights,
)
import re

from . import CLIENT_NAME_HEADER
from . import capability as capability_classes
from . import peer as peer_module
from . import settings as settings_module
from .backend import CUDA_LINUX, Backend
from .config import Config, load_config
from .connect import PairingRequests, StartPairing, PollPairing, DecidePairing
#: `X-Crucible-Client`'s value, validated exactly as `connect.py` validates a
#: pairing `client_name`: 1-80 characters, no control characters. One shape
#: for "a name a person will read", so a client cannot be called one thing at
#: the pairing door and another on a job row.
_CLIENT_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")

from .errors import ApiError, ConfigError, CrucibleError
from .interfaces import InterfaceError
from .jobs import (
    ALL_JOB_TYPES,
    build_registry,
    disabled_error,
    model_rows,
    resolve,
    resolve_model,
    voice_rows,
)
from .jobs.base import Job, validate_member_name
from .manifests import ManifestError, load_manifest
from .jobs.queue import JobStore
from .jobs.tts.common import known_voice
from . import decide as decide_core
from .decide import DecideRequest, DecideResponse
from .engines import chat_admission, decide_reading
from .inflight import Entry, InFlight, read_act, require_act_name
from .leases import CARD_EFFECTS, Leases, require_ttl
from .residency import KIND_NOUNS, Residency
from .settle import Settlement
from .sampling import SAMPLING_HEADER, Applied, apply_defaults
from .tasks import TASK_TYPES, ReloadRefused, Task, TaskStore
# NAMES, NOT THE MODULE. `from . import voices` shadows -- the `GET /v1/voices`
# handler inside `create_app` is itself called `voices`, so a module handle of
# that name resolves to the route function from anywhere below it and every
# `voices.X` is an AttributeError on a function. Caught by the new tests; the
# fix is to never hold the module by a name something else in this file uses.
from .voicerepo import (
    home_pins_path,
    remove_home_pin,
    voice_for_pin,
    write_home_pin,
)
from .voicerepo import Pin as VoicePin
from .voices import (
    NARRATOR_ENGINE_SAMPLING,
    VoiceError,
    load_all_voices,
    remove_home_voice,
    write_home_voice,
)
from .weights import WeightsError, resolve_revision
from .ttsstream import (
    StreamManager,
    StreamSession,
    require_sayable,
    require_streamable,
)

TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})
KEEPALIVE_SECONDS = 15.0
UPLOAD_CHUNK = 1024 * 1024

#: Where the operator page lives, INSIDE the package, so one path works from a
#: checkout and from an installed wheel alike. It ships as package data
#: (`[tool.setuptools.package-data]` in pyproject.toml) rather than being found
#: relative to a repo root, because a wheel installed on the Mac has no repo
#: root and the page has to travel with the code that serves it.
UI_DIR = Path(__file__).resolve().parent / "ui"

#: The proxy waits on the engine, not on a clock it invented. A streamed
#: completion has no read timeout at all (the engine emits a token at a time and
#: may think for a while before the first one); a non-streamed one gets a long
#: but finite ceiling so a wedged engine surfaces as an error rather than a hang.
PROXY_CONNECT_TIMEOUT = 10.0
PROXY_READ_TIMEOUT = 900.0

#: How long the proxy keeps an idle socket to an engine before letting it go.
#: BELOW the engine's own keep-alive, and the gap is the whole point: vLLM's
#: uvicorn closes an idle connection after `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` = 5 s
#: (vllm 0.29.0, vllm/envs.py:109), and httpx's pool keeps one for the same
#: 5.0 s by default (`httpx.Limits().keepalive_expiry`), so at that boundary the
#: proxy could hand a request to a socket the engine had just closed. On
#: 2026-09-21 22:03:20 it did: one `ReadError` with an empty message from an
#: engine that answered nine other completions in the same window, a 502 the
#: client took as fatal, and a book's page read dropped its lease over it. At
#: two seconds no pooled socket is ever older than the engine's patience.
PROXY_KEEPALIVE_EXPIRY = 2.0

#: Transport faults that mean the request was LOST ON THE WAY, not answered and
#: not refused: a reset, a peer that closed a keep-alive socket under us, a
#: half-written request. That is weather, not misconfiguration (CLAUDE.md,
#: "harden transients"), so the proxy sends the request once more on a fresh
#: socket — httpx discards the connection a network error happened on, and
#: `PROXY_KEEPALIVE_EXPIRY` keeps the pool clear of the next stale one. Timeouts
#: are NOT in this set on purpose: an engine that has not answered inside
#: `PROXY_READ_TIMEOUT` is wedged, and repeating that would be a second 900 s.
LOST_ON_THE_WIRE: tuple[type[Exception], ...] = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
#: The whole budget, stated: the first attempt and one more. Not a loop.
WIRE_ATTEMPTS = 2

#: The proxy sends the client's own bytes, so it declares the type itself rather
#: than letting httpx serialise a document and label it.
JSON_HEADERS = {"Content-Type": "application/json"}


class BeforeEveryRequest:
    """A synchronous step run before every HTTP request, and nothing else.

    Pure ASGI on purpose: `receive` and `send` go to the app exactly as the
    server made them. A middleware that wraps them decides what a route can
    learn about its own caller, and the one this replaced (starlette's
    `BaseHTTPMiddleware`) made every caller look present forever —
    `_watch_for_disconnect` says how that was found.
    """

    def __init__(self, app: ASGIApp, *, step: Callable[[], None]) -> None:
        self.app = app
        self.step = step

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            self.step()
        await self.app(scope, receive, send)



# --------------------------------------------------------------------- schemas


class ArtifactRef(BaseModel):
    """A previous job's artifact on THIS server, taken as an input.

    Owen, 2026-09-25: a render's 2,510 chunk FLACs were downloaded, then read
    back off a share and uploaded again (2.5 minutes, 1.25 GB) to the server
    that made them, for the align. A reference takes them where they already
    are. Usually of a HELD job (`POST /v1/jobs/{id}/hold`); an unheld one works
    while its directory still exists.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str


class JobInput(BaseModel):
    """One named input: an uploaded blob, bytes inline in the request, or an
    artifact of a previous job on this server."""

    model_config = ConfigDict(extra="forbid")

    blob_id: str | None = None
    inline_base64: str | None = None
    artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "JobInput":
        given = [name for name, value in
                 (("blob_id", self.blob_id), ("inline_base64", self.inline_base64),
                  ("artifact", self.artifact))
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                "each input needs exactly one of blob_id, inline_base64 or "
                f"artifact, got {given if given else 'neither'}"
            )
        return self


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, JobInput] = Field(default_factory=dict)
    #: THE CLIENT'S OWN NAME FOR THIS WORK. Echoed back on the job record,
    #: never read by this server, never parsed.
    #:
    #: It is for the restart. An `interrupted` job has to be matched to whatever
    #: the client was doing when its own process went away too, and a job id it
    #: may have lost alongside everything else is a poor key for that.
    #: BookForge puts its queue step id here.
    #:
    #: Bounded like a client name and for the same reason (`_CLIENT_NAME`): it
    #: is printed into logs and benches, so a control character in it is the
    #: caller choosing what somebody's terminal does.
    client_ref: str | None = Field(
        default=None, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$"
    )


class StreamOpen(BaseModel):
    """`POST /v1/tts/stream` — PHASE3-TTS.md section 7.

    Nothing has a default, for the render door's reason: a session opened in the
    wrong language, or on a voice the client did not choose, is a silent
    substitution and a whole afternoon of listening in the wrong accent.
    """

    model_config = ConfigDict(extra="forbid")

    voice: str = Field(min_length=1)
    language: str = Field(min_length=1)


class TaskCreate(BaseModel):
    """`POST /v1/tasks` — one operator operation. PHASE13-OPERATOR.md 3.3.

    One model for three request shapes rather than three routes, because there
    is one lane and one refusal (`task_busy`) governing all of them, and a
    client that had to pick a path before it could be told "busy" would have to
    know which of three doors to retry.

    The validator is `StreamOp`'s in spirit: the `type` word decides which
    fields are required and which are REFUSED. A `narrator_engine` sent with a
    `pull`, or an `id` sent with an `install`, is a client that has confused two
    requests, and accepting it silently would run the wrong one.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    # pull
    kind: str | None = None
    id: str | None = None
    # install
    job_type: str | None = None
    narrator_engine: str | None = None
    # module
    module: dict[str, Any] | None = None
    # engine (PHASE15-HOST.md 4.7)
    target: str | None = None

    #: Which fields each type owns. The validator reads this rather than three
    #: hand-written branches, so a fourth task type is one row.
    FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "pull": ("kind", "id"),
        "install": ("job_type", "narrator_engine"),
        "module": ("module",),
        "engine": ("target",),
        # PHASE17-ORCHESTRATOR.md 4.2: NO fields, and that is the whole
        # request. There is exactly one engine on a machine and the
        # orchestrator knows which — a `target` here would be a client
        # naming a thing it cannot see.
        "engine-restart": (),
    }
    #: ...and which of those may not be omitted. `narrator_engine` is absent
    #: here because whether it is required depends on the job type, which is
    #: `crucible/tasks.py`'s question and not this schema's.
    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "pull": ("kind", "id"),
        "install": ("job_type",),
        "module": ("module",),
        "engine": ("target",),
        "engine-restart": (),
    }

    @model_validator(mode="after")
    def the_type_carries_what_it_needs(self) -> "TaskCreate":
        if self.type not in TASK_TYPES:
            raise ValueError(
                f"type must be one of {list(TASK_TYPES)}, got {self.type!r}"
            )
        mine = self.FIELDS[self.type]
        for name in self.REQUIRED[self.type]:
            if getattr(self, name) is None:
                raise ValueError(f"a {self.type} task needs {name!r}")
        theirs = [
            name
            for group in self.FIELDS.values()
            for name in group
            if name not in mine and getattr(self, name) is not None
        ]
        if theirs:
            raise ValueError(
                f"a {self.type} task takes {list(mine)}; it was also sent "
                f"{sorted(theirs)}, which belong to another task type"
            )
        return self

    def request(self) -> dict[str, Any]:
        """The body as the task echoes it: this type's fields and no others."""
        return {
            "type": self.type,
            **{name: getattr(self, name) for name in self.FIELDS[self.type]},
        }


class LeaseOpen(BaseModel):
    """`POST /v1/models/{id}/lease` — a client saying it intends a run.

    Both fields are required and neither has a default, for the streaming door's
    reason. A default `act` would put a name nobody chose on a bench, which is
    the thing `X-Crucible-Act` is refused for; a default `ttl_seconds` would be
    this server picking how long somebody else's run is, which is the one number
    only the client knows.

    **There is no `kind`.** The id in the path is the resident thing's, of
    whatever kind, and the card holds one thing — so the server reads the kind
    off `Residency.resident` and a client has nothing to disambiguate. A `kind`
    on the body would be a second owner of `resident.kind`, able to disagree with
    it (R1), and would let a client be refused for spelling a fact it was never
    asked to know.
    """

    model_config = ConfigDict(extra="forbid")

    act: str = Field(min_length=1)
    ttl_seconds: int


class StreamOp(BaseModel):
    """`POST /v1/tts/stream/{id}` — one op.

        {"op": "say",    "id": "r12", "text": "...", "take": 0}
        {"op": "cancel", "id": "r12"}
        {"op": "cancel_all"}
        {"op": "close"}

    `take` is **required** on `say` and has no default here. The SDK's
    `say(id, text, take?)` defaults it to 0 in the caller's own code, which is a
    client choosing; a default on the wire would be the server choosing, and now
    that the five fine-tunes declare a second rung that would be a render at a
    take nobody asked for. A take past the end of the voice's ladder is a SEED
    LANE at the voice's own sampling (2026-09-19) and is still never clamped —
    take 4 is never take 2's numbers under take 4's name.
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


def installable_job_type_rows() -> list[dict[str, Any]]:
    """Every job type this BUILD knows, and what it would take to have it.

    `GET /v1/capability`'s `job_types`, PHASE13-OPERATOR.md sections 3.2a and 4.
    One row per job type named by the capability class table, in that table's
    report order, and every field read from the module that already owns it:

    | field | owner |
    |---|---|
    | `job_type`, `classes` | `crucible/capability.py`'s `CLASSES` |
    | `installer` | `crucible/cli.py`'s `INSTALLER_FOR` |
    | `narrator_engines` | `crucible/voices.py`'s `NARRATOR_ENGINE_SAMPLING` |

    `installer` is the job type `POST /v1/tasks {"type": "install"}` must be
    given to build this one's env, which is almost always itself — `denoise` is
    the exception, because it shares `rvc`'s env, and a page offering it an
    Install button of its own would be drawing a control the task door refuses
    `unknown_job_type`. `null` means nothing installs it: `echo` is compiled in.

    `narrator_engines` is empty for every type but `tts`, and for `tts` it is
    the whole of what `narrator_engine` may be — the same list the task door
    validates against, so a page cannot offer an engine the POST will refuse.
    It is a LIST and not a default: on cuda-linux the two engines cannot share
    a venv and there is no default (`crucible/tasks.py`'s
    `require_narrator_engine`).

    Whether the type is OFFERED here is deliberately absent: that is
    `/v1/setup`'s `job_types`, which is `store.registry` and the one owner of
    it. Repeating it would make a stale second answer possible in the seconds
    around an install's reload (3.4).
    """
    from .cli import INSTALLER_FOR

    ordered: list[str] = []
    classes_of: dict[str, list[str]] = {}
    for entry in capability_classes.CLASSES:
        if entry.job_type not in classes_of:
            ordered.append(entry.job_type)
            classes_of[entry.job_type] = []
        classes_of[entry.job_type].append(entry.name)
    return [
        {
            "job_type": job_type,
            "classes": classes_of[job_type],
            "installer": INSTALLER_FOR.get(job_type),
            "narrator_engines": (
                sorted(NARRATOR_ENGINE_SAMPLING) if job_type == "tts" else []
            ),
        }
        for job_type in ordered
    ]


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


def require_peer_auth(request: Request) -> None:
    """`require_auth`, refusing under the RELATION's name. PHASE17 2.1.

    The same comparison against the same token, and a different code, because
    the caller on this door is not an app: it is an orchestrator that has just
    booted an engine and is telling it so. Told `unauthorized`, an
    orchestrator cannot tell *"the token I copied out of the guest's pairing
    file is stale"* — its own bug, and the thing its log must say — from
    *"some app's token is wrong"*, which is not its business at all. One name
    per relation, so a log line says which handshake failed.
    """
    config: Config = request.app.state.config
    header = request.headers.get("authorization")
    scheme, _, presented = (header or "").partition(" ")
    if (
        header is None
        or scheme.lower() != "bearer"
        or presented.strip() == ""
        or not secrets.compare_digest(presented.strip(), config.token)
    ):
        raise ApiError(
            401,
            peer_module.PEER_TOKEN_MISMATCH,
            "the peer door takes THIS engine's bearer token, which is the one "
            "the orchestrator already holds: it reads it from the guest's "
            "pairing line, or it is the token in the config it wrote itself. "
            "There is no second credential for the relation "
            "(PHASE17-ORCHESTRATOR.md 2.1)",
        )


def require_peer_api_version(request: Request) -> None:
    """`require_api_version`, refusing `peer_version_incompatible`. PHASE17 2.1.

    The API version travels in the header where every other call already
    carries it, rather than in the claim body: a second copy in the body would
    be a fact with two owners. What changes here is only the NAME of the
    refusal, for `require_peer_auth`'s reason.
    """
    presented = request.headers.get(API_HEADER)
    major: int | None = None
    if presented is not None:
        try:
            major = int(presented.strip().split(".")[0])
        except ValueError:
            major = None
    if major != API_VERSION:
        raise ApiError(
            426,
            peer_module.PEER_VERSION_INCOMPATIBLE,
            f"this engine speaks API version {API_VERSION} and the claim "
            f"presented {presented!r}. An orchestrator and the engine it "
            "manages are usually one install and one version; when they are "
            "not, the relation is refused rather than half-spoken",
            {"server_api_version": API_VERSION, "client_api_version": presented},
        )


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

    def follow_the_config_file() -> None:
        """Every request sees the config.toml that is on disk NOW.

        `Config.follow_file()` — one stat per request, a re-read only when
        the file moved. The record `GET /v1/capability` serves, the `[jobs]`
        flags the doors refuse on, the routes and the upstreams all follow,
        because `adopt()` replaces the one Config object's fields in place
        and every route, the residency and the store close over that object.

        THE ONE THING THAT DOES NOT FOLLOW is a job type turned ON: the
        registry is built from the flags at start (`crucible/jobs/__init__.py`
        `build_registry`), so a flag that goes from off to on needs a start —
        which `crucible install` performs anyway, since a new job type is a new
        env. A flag turned OFF is honoured at once, by the doors.

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

    app.state.tasks = TaskStore(
        config,
        backend,
        reload=reload_registry,
        holder=app.state.settlement.holder,
    )

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
    openai = APIRouter(
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

    # ------------------------------------------------------------------ ping

    @public.get("/ping")
    async def ping() -> dict[str, Any]:
        """Unauthenticated. Lets a client tell "wrong token" from "not a Crucible"."""
        return {"crucible": True, "name": config.name, "api_version": API_VERSION,
                "pairing_version": 1}

    @public.post("/pairing/start", dependencies=[Depends(require_api_version)])
    async def start_pairing(body: StartPairing, request: Request, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        address = request.client.host if request.client is not None else "unknown"
        return {"name": config.name, **app.state.pairing_requests.start(body.client_name, address)}

    @public.post("/pairing/poll", dependencies=[Depends(require_api_version)])
    async def poll_pairing(body: PollPairing, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        status = app.state.pairing_requests.poll(body.id, body.device_code)
        result = {"status": status}
        if status == "approved":
            result.update(name=config.name, token=config.token)
        return result

    @private.get("/pairing/requests")
    async def pairing_requests(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {"requests": app.state.pairing_requests.pending()}

    @private.post("/pairing/decision")
    async def decide_pairing(body: DecidePairing, response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return app.state.pairing_requests.decide(body.id, body.user_code, body.allow)

    # ------------------------------------------------------------ capability

    @private.get("/capability")
    async def capability(request: Request) -> dict[str, Any]:
        """What this server can hold, per capability class, and why not.

        The read a client needs before it decides what to ask for. PHASE 9 made
        the act-to-model mapping a PER-HOST fact — `crucible install` probes the
        card and picks the largest candidate that fits, so a 24 GB box serves
        `translate` with a 4-bit 27B, a bigger one serves it with something else,
        and a 12 GB box does not serve it at all. A client that was handed a model
        id by configuration would be carrying a model this server may have
        refused.

        WHY A CLASS AND NOT A JOB TYPE. `enable_llm` is one boolean and Owen
        ruled translation binary per server, so `clean` and `translate` have to be
        able to disagree. `simplify` and `analysis` became classes of their own
        too, on the naming ruling (Owen, 2026-09-13: a job is never reported
        as a different job), and `generate` is the generic chat-shaped act
        whose working context the CLIENT may state (`?class=generate&
        context_tokens=&concurrency=`, see `capability.served_rows`).

        EVERY ROW STATES ITS WORK: `work` is the working context the fit was
        computed for, with `from: "default" | "request"`, and a client-sized
        row adds `context_ceilings` — each candidate's longest servable
        request here, the smaller of the most its manifest ever starts an
        engine with on this backend (`max_context`, or `context_default` where
        none is stated) and what this host's memory affords. A size above the
        ceiling is `400 context_over_limit`, never clamped. A `load-model` with
        `params.context` is held to the same ceiling (at one in flight) and
        refused with the same body; that is how a client reaches a ceiling
        above the resident model's `max_model_len`.

        `enabled: false` IS AN ANSWER, not an error. A server that cannot
        translate says so with the number that decided it, and a client should be
        able to render "this machine cannot do that" without it looking like a
        fault.

        THIS IS A RECORD, NOT AN AUTHORITY. `[jobs] enable_*` remains the single
        owner of what this server offers; this says what the numbers were when
        somebody decided. `total_bytes` is the card the decision was made on, so a
        reader can tell a stale record from a current one — which is how a swapped
        GPU is noticed without anybody writing down a date.

        `job_types` IS NOT PART OF THE RECORD, and that is why it is added here
        rather than in `CapabilityRecord.to_dict()`. PHASE13-OPERATOR.md section
        4 draws the operator page's Job types section from this one read, and to
        draw it the page needs three things the stored record cannot carry: which
        job type each class feeds (`capability.CLASSES`), which command builds
        that type's env (`cli.INSTALLER_FOR` — `denoise` shares `rvc`'s), and
        which narrator engines a `tts` install may name
        (`voices.NARRATOR_ENGINE_SAMPLING`). All three are THIS BUILD's tables,
        read live; a record written months ago must not be able to answer them,
        because they are facts about the code, not about the card. Put in the
        record they would be a second copy that goes stale the day an engine is
        added — which is the shape R1 exists to forbid. The page holding its own
        copy is the same defect one layer out, and is what section 4 means by
        "never a hard-coded list".
        """
        live: Config = request.app.state.config
        record = live.capability
        if record is None:
            # Absent is its own answer and must not be dressed up as an empty
            # decision: a config written before `crucible capability` ran, or by a
            # build that predates it, has DECIDED NOTHING. Returning empty rows
            # would read as "probed, and nothing fit", which is the opposite news.
            raise ApiError(
                503,
                "capability_undecided",
                "this server has no capability record; nothing has probed the card "
                "on this host yet. Run `crucible capability --write` (or reinstall) "
                "to decide, and read `GET /v1/info` for what it offers meanwhile",
            )
        # EVERY ROW SAYS WHERE ITS WORK RUNS (PHASE15-HOST.md section 3.3), and
        # the answer is read off `[routes]` — the one owner of it — rather than
        # inferred from the row's `selected` carrying a slash. The two agree,
        # because `crucible/settings.py` rewrites the record from the routes on
        # every write that touches one; asking the routes is what makes them
        # unable to disagree if a record ever went stale (R1).
        # A CLIENT MAY SIZE A CLIENT-SIZED CLASS (`?class=generate&
        # context_tokens=40960&concurrency=1`), because the client is the one
        # that knows its request sizes and this server is the one that knows
        # its card (Owen, 2026-09-23: *"give it the ability to set the context
        # limit"*). Read off the raw query rather than as typed parameters so a
        # bad value is this server's named refusal (`invalid_working_context`)
        # and not the framework's 422. `capability.served_rows` owns all of it:
        # the validation, the live decision, the ceiling, and the `work` echo
        # every row now carries.
        query = request.query_params
        document = record.to_dict()
        document["classes"] = capability_classes.served_rows(
            record,
            gpu_vendor=backend.gpu.vendor,
            chosen={entry.capability: entry.model for entry in live.local_models},
            routes={entry.capability: entry.model for entry in live.routes},
            capability_class=query.get("class"),
            context_tokens=query.get(capability_classes.CONTEXT_TOKENS_PARAM),
            concurrency=query.get(capability_classes.CONCURRENCY_PARAM),
        )
        for row in document["classes"]:
            row["route"] = (
                "upstream"
                if live.route_model(row["capability"]) is not None
                else "local"
            )
        return {**document, "job_types": installable_job_type_rows()}

    # -------------------------------------------------------------- settings

    def _installed_subjects(live: Config) -> dict[str, bool]:
        """Which subjects are on this disk, by id, straight from the catalog.

        The settings document needs `installed` for every model an app may
        choose, and `GET /v1/catalog` is ALREADY the one owner of that fact.
        Asking it here rather than re-deriving is what keeps the chooser and
        the catalog from disagreeing about what is on the disk
        (ARCHITECTURE.md R1) — every kind, not only `model`, because a voice
        and an aligner are chosen the same way.
        """
        return {row["id"]: row["installed"] for row in catalog.rows(live, backend, residency)}

    @private.get("/settings")
    async def get_settings(request: Request) -> dict[str, Any]:
        """Where each class's work runs, and which upstreams are configured.

        PHASE15-HOST.md section 3.1. Owen, 2026-09-14: *"Settings live in the
        engine and nowhere else."* An app draws this document and writes
        through `PUT`; it holds no key, no route and no model list of its own.

        **A key is never in this answer.** `key_hint` is its last four
        characters, which is enough to recognise WHICH key is there — the
        question a person with two accounts asks — and nothing else. There is
        no route on this server that returns one.
        """
        live: Config = request.app.state.config
        return settings_module.document(live, installed=_installed_subjects(live))

    @private.put("/settings")
    async def put_settings(request: Request) -> dict[str, Any]:
        """A partial patch, applied whole or not at all, live without a restart.

        PHASE15-HOST.md section 3.2. The order inside one request is the
        contract's — upstreams, then routes, then the whole validated — which
        is what lets an app configure an upstream AND route a class to it in
        one call, the way section 5.2 tells it to.

        **A refusal applies nothing.** `settings.resolve` builds the candidate
        document in memory and raises before `settings.apply` writes a byte, so
        a request refused for its routes does not leave a key behind on a
        server whose operator believes it failed.

        The answer is the whole `GET /v1/settings` document AFTER the write, so
        a window never has to guess what took.
        """
        live: Config = request.app.state.config
        # Read BEFORE the work, like the chat door: an unknown act is a 400
        # rather than a write that happened and was then recorded under a name
        # nobody knows.
        act = read_act(request.headers)
        try:
            patch = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the settings body is not JSON: {exc}"
            ) from None
        resolved = settings_module.resolve(live, patch)
        # Off the event loop: this writes a file and re-reads it, and a settings
        # write must not stall a job's event stream.
        await asyncio.to_thread(
            lambda: settings_module.apply(
                live, resolved, gpu_vendor=backend.gpu.vendor
            )
        )
        if resolved.changed:
            request.app.state.settings_history.record(
                act=act,
                client=_client_agent(request),
                changed=resolved.changed,
            )
        return settings_module.document(live, installed=_installed_subjects(live))

    @private.post("/settings/upstreams/{name}/test")
    async def test_upstream(request: Request, name: str) -> dict[str, Any]:
        """Ask an upstream what it serves, with a key that may not be saved yet.

        PHASE15-HOST.md section 3.2. The body is optional and carries
        `{"key": …}` or `{"url": …}` to test BEFORE saving, which is the order a
        person actually works in: paste, check it works, then save. With no
        body the stored record is used.

        **Unbilled, and never cached.** The answer is somebody else's and
        changes without telling us; a stale list shown beside a key the
        operator pasted ten seconds ago is exactly the moment they would
        believe it.

        `POST` and not `GET` because it takes a body carrying a secret, and a
        secret in a query string is a secret in a log.
        """
        live: Config = request.app.state.config
        upstreams.require_name(name, "the path")
        raw = await request.body()
        if raw.strip() == b"":
            probe = None
        else:
            try:
                probe = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(
                    400, "invalid_request", f"the test body is not JSON: {exc}"
                ) from None
        if probe is None or probe == {}:
            record = live.upstream(name)
            if record is None:
                raise ApiError(
                    400,
                    "upstream_unconfigured",
                    f"{name} is not configured on this server and the request "
                    f"carried no {upstreams.UPSTREAM_FIELD[name]!r} to test "
                    "with. Send one to check it before saving it",
                    {
                        "field": f"upstreams.{name}."
                        f"{upstreams.UPSTREAM_FIELD[name]}",
                        "upstream": name,
                    },
                )
        else:
            record = upstreams.record_from_patch(
                name, probe, f"the test body for {name}"
            )
        client: httpx.AsyncClient = request.app.state.http
        return {"models": await upstreams.list_models(client, record)}

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
            rows_for["tts"] = voice_rows(
                config, backend, residency,
                leases=app.state.leases, store=app.state.store,
            )
        capabilities = [
            {"job_type": capability, "models": rows}
            for capability, rows in sorted(rows_for.items())
        ]
        peer_state: peer_module.PeerState = request.app.state.peer
        return {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
            },
            # WHICH HALF OF THE RELATION THIS PROCESS IS
            # (PHASE17-ORCHESTRATOR.md 3.1). A property of a PROCESS, never of
            # an install: on a Windows machine with no WSL, one install runs
            # an orchestrator and an engine as two processes, and only one of
            # them answers this document.
            #
            # A client reads it all-or-nothing, exactly as it reads `route`
            # (PHASE15 3.3): a document with NO `role` comes from a server
            # that predates this phase, and such a server IS an engine with
            # `managed_by: null` — a fact the document states by its vintage,
            # not a default the client fills.
            "role": peer_module.ROLE_ENGINE,
            "managed_by": peer_state.managed_by(),
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
            # WHICH ENGINE READS A PAGE HERE, AND WHAT A PAGE REQUEST IS
            # (PHASE15-HOST.md 3.10). Two halves, and the second is the
            # load-bearing one: `engine` is for an operator looking at a
            # machine, and `request` is the prompt, the dpi, the pixel
            # budget, the ceiling and the dialect — read from
            # `crucible/pages.py` on EVERY backend, so a client builds the
            # same bytes whether vLLM, llama.cpp or mlx-vlm is behind them.
            #
            # It is on the wire because page reading has no job type of its
            # own (PHASE3-VLM.md section 1): the CLIENT builds the chat
            # completion, and it was building it out of constants pinned in
            # its own source — a prompt and a pixel budget are facts about
            # the weights, and the division-of-knowledge ruling puts those
            # here.
            "pages_engine": _pages_engine(),
        }

    def _pages_engine() -> dict[str, Any]:
        """`pages_engine` for THIS host. Null engine where none is served."""
        try:
            manifest = load_manifest(pages_module.MODEL_ID)
        except ManifestError as exc:
            # A build whose page manifest will not read has no page engine,
            # and says which file rather than answering an empty block.
            return pages_module.engine_block(
                engine=None,
                installed=False,
                detail=f"{pages_module.MODEL_ID}'s manifest will not read: {exc}",
            )
        if not manifest.supports(backend.kind):
            return pages_module.engine_block(
                engine=None,
                installed=False,
                detail=(
                    f"{manifest.path.name} has no {backend.kind} block, so this "
                    f"host reads no pages. It declares {sorted(manifest.backends)}"
                ),
            )
        spec = manifest.spec(backend.kind)
        found = weights.installed(config, manifest, spec)
        if found is None:
            return pages_module.engine_block(
                engine=spec.engine,
                installed=False,
                detail=(
                    f"{spec.engine} would serve {manifest.id} here, and its "
                    f"weights are not pulled — `crucible models pull {manifest.id}`"
                ),
            )
        return pages_module.engine_block(
            engine=spec.engine,
            installed=True,
            detail=(
                f"{spec.engine} serves {manifest.id} from {found.path} "
                f"({found.bytes / 1e9:.2f} GB, {spec.hf_repo}@{spec.revision[:12]})"
            ),
        )

    # ------------------------------------------------------------------ peer

    @peer_router.get("")
    async def read_peer(request: Request) -> dict[str, Any]:
        """`GET /v1/peer` — who manages this engine, and how old this process is.

        PHASE17 2.4: health flows ONE way. The orchestrator polls this and
        `/v1/ping`; the engine calls nothing back. An engine that phoned home
        would need to know its orchestrator's address, keep it fresh across
        restarts, and behave when it is wrong — three facts to own for a push
        a 15-second poll already delivers.
        """
        state: peer_module.PeerState = request.app.state.peer
        return state.document(_uptime_s(request))

    @peer_router.post("/claim")
    async def claim_peer(request: Request) -> dict[str, Any]:
        """`POST /v1/peer/claim` — an orchestrator says it manages this engine.

        A STATEMENT OF FACT, not a grant of permission: nothing on this server
        consults `managed_by` before doing anything, because there is nothing
        an orchestrator asks an engine to do that an app may not also ask
        (`crucible/peer.py`'s preamble). What it buys is that `/v1/info` can
        answer "who manages this".
        """
        body = await _peer_body(request)
        state: peer_module.PeerState = request.app.state.peer
        orchestrator = peer_module.Orchestrator.from_body(body.get("orchestrator"))
        force = body.get("force")
        if force is not None and not isinstance(force, bool):
            raise ApiError(
                400,
                "invalid_request",
                "`force` is a boolean. It takes an engine away from another "
                "orchestrator and is a person's act through the page, never "
                "an orchestrator's own (PHASE17-ORCHESTRATOR.md 2.1)",
            )
        claim = state.claim(orchestrator, force=bool(force))
        return {
            "role": peer_module.ROLE_ENGINE,
            "managed_by": claim.managed_by(),
            "claimed": claim.claimed,
        }

    @peer_router.delete("/claim")
    async def release_peer(request: Request) -> dict[str, Any]:
        """`DELETE /v1/peer/claim` — the orchestrator's Quit (PHASE17 2.2).

        Nothing claimed is NOT a refusal: "there is no claim" is the state the
        caller asked for. Somebody else's claim is refused, because releasing
        one by accident is how an engine ends up unmanaged with a tray still
        watching it.
        """
        body = await _peer_body(request)
        state: peer_module.PeerState = request.app.state.peer
        raw = body.get("orchestrator")
        who = None if raw is None else peer_module.Orchestrator.from_body(raw)
        state.release(who, force=bool(body.get("force")))
        return {"role": peer_module.ROLE_ENGINE, "managed_by": None}

    async def _peer_body(request: Request) -> dict[str, Any]:
        """The claim body, or `{}`. A DELETE with no body is the common one."""
        raw = await request.body()
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(
                400, "invalid_request", f"the peer body is not JSON: {exc}"
            ) from None
        if not isinstance(parsed, dict):
            raise ApiError(
                400,
                "invalid_request",
                f"the peer body is an object; a bare {type(parsed).__name__} "
                "says nothing about who is claiming",
            )
        return parsed

    def _uptime_s(request: Request) -> float:
        """MONOTONIC seconds since this process started serving.

        It is what tells an orchestrator that an engine answering again is a
        NEW process rather than the one it claimed — the signal that a
        re-claim is owed. A wall clock would make that signal lie across an
        NTP correction, which is the reason `started_at` is monotonic in the
        first place.
        """
        return time.monotonic() - request.app.state.started_at

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
            # WHAT WAS TOLD TO GO AND HAS NOT (ledger R13, Owen's ruling
            # 2026-09-18). Null when nothing is. It belongs on the smallest
            # read this server has because it is the reason every load, the
            # claim and the streaming door are refusing. `status` is untouched
            # and still reports the LANE — `ok` there has always meant "no job
            # is running", never "the card is free" — so this is the field
            # that makes the difference readable instead of a redefinition of
            # one every phase-2 client already reads. The object is
            # `Residency.stopping`'s own (`DyingResident.to_dict`), so this
            # and `/v1/activity` cannot tell two stories.
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
        }

    # ----------------------------------------------------------------- setup

    @private.get("/setup")
    async def setup(request: Request) -> dict[str, Any]:
        """Everything an app needs to be pointed at this server, in one read.

        PHASE13-OPERATOR.md section 3.1. Owen, 2026-09-14: *"crucible has its own
        ui. and it provides the token or whatever else we need to set it up on
        foundry or bookforge."* This is "whatever else we need".

        **It returns the token, and that reveals nothing.** Every `/v1/*` route
        is behind the bearer token, so the only caller who can read this is one
        who already has it. What it buys is that nobody types a secret twice:
        the operator page fetches this and draws a copyable pairing line, and
        the person pasting that line into BookForge has not seen a token at all.

        `urls` and `pairing` are the same list read two ways, and both are
        derived rather than stored — the bind address this process actually
        holds, made dialable (`crucible/pairing.py`). A wildcard bind becomes
        one entry per non-loopback IPv4 interface; a concrete bind becomes
        exactly one. Never a hostname lookup: an interface is a fact about this
        host, a name is a fact about somebody else's resolver.

        `job_types` repeats `/v1/info`'s list rather than making the page read
        twice, and it repeats it from the same producer — `store.registry` — so
        the two cannot disagree. After an install task's reload (3.4) both
        answer the new list in the same tick.
        """
        live: Config = request.app.state.config
        store: JobStore = request.app.state.store
        host = request.app.state.bind_host
        port = request.app.state.bind_port
        try:
            # `live.advertise` is the operator's statement that something
            # forwards here from elsewhere. The bind is what this host can see;
            # that is what it cannot.
            urls = pairing.reachable_urls(
                host, port, live.advertise + live.tailscale_advertise + live.lan_advertise
            )
        except InterfaceError as exc:
            # 503 and not an empty `urls`: an empty list reads as "reachable
            # from nowhere", which is a claim about this host rather than a
            # report that the question could not be asked (R3). The page shows
            # the reason and the operator can still bind a concrete address.
            raise ApiError(
                503,
                "interfaces_unreadable",
                f"this server is bound to {host!r} and cannot list its own "
                f"interfaces, so it cannot say where an app should reach it: "
                f"{exc}",
            ) from None
        return {
            "name": live.name,
            "version": VERSION,
            "backend": backend.kind,
            "bind": f"http://{host}:{port}",
            "urls": urls,
            "token": live.token,
            "pairing": pairing.pairing_lines(live.name, urls, live.token),
            "job_types": sorted(store.registry),
            "config_path": str(live.path),
        }

    # --------------------------------------------------------------- catalog

    @private.get("/catalog")
    async def catalog_route(request: Request) -> dict[str, Any]:
        """Every subject this backend can hold, installed or not.

        PHASE13-OPERATOR.md section 3.2. Every field is derived from something
        this server already owns and no row is authored here — see
        `crucible/catalog.py`, which is the whole of it.
        """
        live: Config = request.app.state.config
        # `backend_kind` on every backend, not only where the list is empty
        # (PHASE15-HOST.md section 3.5). A reader that had to infer "there are
        # no rows because there is no card" from the emptiness would be
        # guessing, and the same key on cuda-linux is what makes this a field
        # rather than a marker for one mode.
        return {"rows": catalog.rows(live, backend, residency),
                "backend_kind": backend.kind}

    @private.delete("/catalog/{kind}/{subject_id}", status_code=204)
    async def catalog_remove(
        request: Request, kind: str, subject_id: str
    ) -> Response:
        """Delete an installed subject's files. PHASE15-HOST.md 3.5a.

        THE DOOR THE WEIGHTS RULE NEEDS. 3.5: a subject is never stored twice
        on one machine, so when the guest has its own copy the Windows one
        goes — and the host must never reach into `crucible/weights.py`'s
        layout from outside to do it, because a layout with two owners is the
        shape ARCHITECTURE.md R1 is about. So the server that owns the disk
        owns the deletion, and this is how it is asked.

        THE ORDER OF THE REFUSALS IS THE JOB DOOR'S: what is wrong with the
        REQUEST first (an unknown kind or id is true whatever this server is
        doing), then what is wrong with this server's STATE. A caller who
        misspelled a subject id and was told "it is in use" would fix the
        wrong thing.
        """
        live: Config = request.app.state.config
        if kind not in catalog.KINDS:
            raise ApiError(
                404,
                "subject_unknown",
                f"{kind!r} is not a subject kind; they are {list(catalog.KINDS)}",
                {"kind": kind, "id": subject_id},
            )
        subject = catalog.find(live, backend, kind, subject_id)
        if subject is None:
            raise ApiError(
                404,
                "subject_unknown",
                f"this server has no {kind} called {subject_id!r} for "
                f"{backend.kind}. GET /v1/catalog lists every subject it can hold",
                {"kind": kind, "id": subject_id},
            )
        found = subject.installed()
        if found is None:
            raise ApiError(
                409,
                "subject_not_installed",
                f"{kind} {subject_id!r} is not installed on this server, so "
                "there is nothing to remove. Refused rather than answered 204: "
                "a caller told 'done' about a subject that was never there "
                "would believe a migration had deleted something",
                {"kind": kind, "id": subject_id},
            )
        who = _subject_holder(request, subject)
        if who is not None:
            raise ApiError(
                409,
                "subject_in_use",
                f"{kind} {subject_id!r} cannot be removed: {who['who']}. "
                "Deleting the files under a running engine would leave it "
                "serving a model that is no longer on the disk",
                who,
            )
        try:
            gone = subject.remove()
        except weights.WeightsShared as exc:
            # PHASE22 section 2.9: a base's folder is also an alias's. 409 and
            # not 500 — nothing is broken, the subject is HELD, by models this
            # refusal names so a caller can remove them first.
            raise ApiError(
                409,
                "weights_shared",
                str(exc),
                {
                    "kind": kind,
                    "id": subject_id,
                    "backend": exc.backend,
                    "aliases": list(exc.aliases),
                },
            ) from None
        except weights.RemoveFailed as exc:
            raise ApiError(
                500,
                "subject_remove_failed",
                str(exc),
                {"kind": kind, "id": subject_id, "path": str(exc.path)},
            ) from None
        except (CrucibleError, OSError) as exc:
            raise ApiError(
                500,
                "subject_remove_failed",
                f"removing {kind} {subject_id!r} failed: "
                f"{type(exc).__name__}: {exc}",
                {"kind": kind, "id": subject_id, "path": str(found.path)},
            ) from None
        request.app.state.removals.record(
            kind=kind,
            subject_id=subject_id,
            bytes_freed=found.bytes,
            act=read_act(request.headers),
            client=request.headers.get("user-agent"),
        )
        print(
            f"crucible: removed {kind} {subject_id} ({found.bytes / 1e9:.2f} GB) "
            f"from {gone}",
            file=sys.stderr,
        )
        return Response(status_code=204)

    def _subject_holder(request: Request, subject: catalog.Subject) -> dict | None:
        """Why this subject may not be deleted right now, or None.

        THREE HOLDS, and each is read from the thing that owns it rather than
        inferred: the RESIDENCY (this subject is the thing on the card), the
        LEASES (a client has said it intends a run on it), and the TASK STORE
        (a pull or a module naming it is in flight). The four facts'
        `Settlement.holder` is deliberately NOT what is asked — it answers
        "is the card busy at all", and a `denoise` separator on disk is not
        made undeletable by a `tts` render.
        """
        # WHO READS THESE FILES (PHASE22 section 2.9): the subject, and every
        # alias whose weights are its download. A `qwen3.5-9b-vl` on the card
        # is serving `qwen3.5-9b`'s folder, so it holds the base exactly as
        # the base itself would — whether or not the alias was ever pulled as
        # itself (on cuda-linux it is installed by the base's pull alone).
        readers = catalog.ids_reading(subject)

        def through(holder_id: str) -> str:
            return (
                ""
                if holder_id == subject.id
                else f" ({holder_id}, which shares its weights)"
            )

        resident = residency.resident
        if resident is not None and resident.id in readers:
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": (
                    f"it is the {KIND_NOUNS[resident.kind]} on the card right "
                    f"now{through(resident.id)}; unload it first"
                ),
                "fact": "resident",
            }
        lease = request.app.state.leases.current()
        if lease is not None and lease.subject in readers:
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": (
                    f"{lease.client or 'a client'} holds a lease on it"
                    f"{through(lease.subject)} "
                    f"until {lease.expires_at.isoformat()}"
                ),
                "fact": "lease",
                **lease.receipt(),
            }
        running = request.app.state.tasks.running
        if running is not None and any(
            _task_names(running, subject.kind, reader) for reader in readers
        ):
            return {
                "kind": subject.kind,
                "id": subject.id,
                "who": f"task {running.id} ({running.type}) names it",
                "fact": "task",
                "task_id": running.id,
            }
        return None

    def _task_names(task: Any, kind: str, subject_id: str) -> bool:
        """Does this running task name this subject? Read off its REQUEST.

        The request is the task's own echo of what was asked (`Task.request`),
        so this needs no second table of which task types touch which
        subjects — a `pull` names one, a `module` names a list, and an
        `install` names none.
        """
        body = task.request
        if body.get("kind") == kind and body.get("id") == subject_id:
            return True
        module = body.get("module")
        if isinstance(module, dict):
            for entry in module.get("subjects", []) or []:
                if (
                    isinstance(entry, dict)
                    and entry.get("kind") == kind
                    and entry.get("id") == subject_id
                ):
                    return True
        return False

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

    def _client_agent(request: Request) -> str | None:
        """Who is speaking to this server, or None because they did not say.

        The SDK sends `<clientName> crucible-client/<version>`; anything else may
        send whatever it likes, or nothing. Truncated because it is a header, and
        a header is attacker-controlled length even inside one trust domain.

        One function rather than one expression per door, because there are now
        two doors that record a holder — `POST /v1/jobs` and `POST /v1/tts/stream`
        — and a bench puts both names in the same column. Two copies of "how we
        read the User-Agent" would be two truncation limits and two spellings of
        "did not say" the day one of them was edited.

        `X-Crucible-Client` FIRST, BECAUSE A BROWSER CANNOT SET ITS UA
        (2026-09-20). The BookForge Reader extension's popup read *"Mozilla/5.0
        (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 … Chrome/154.0.0.0
        Safari/537.36 is running a load-voice job here"* — the extension
        describing its OWN job, because `User-Agent` is a forbidden header name
        in a browser: the SDK sets `clientName` and Chrome silently drops it.
        Every consumer of `Job.client` then had a 120-character browser string
        where a name belongs, and a bench's "held by" column is the worst place
        for one.

        So the SDK sends the name in a header a browser CAN set, and this reads
        that first. The User-Agent rule is unchanged and is still the answer for
        curl, for the CLI, and for anything that sends no such header.

        AN INVALID HEADER IS IGNORED, NOT REFUSED. It is the same shape
        `connect.py` validates a pairing `client_name` with — 1-80 characters,
        no control characters — and failing it means falling back to the
        User-Agent, because this is a LABEL for a bench and not an
        authorization: nothing is decided by it, so nothing is worth refusing a
        render over. Control characters are the part that matters; a name is
        printed in a terminal, a log line and a popup.

        ONE TRUNCATION, and it is still this line's: the 80-character limit
        rejects a header outright rather than trimming it, and the 200 below is
        what a User-Agent gets. Two truncations would be two answers to "how
        long is a client name".
        """
        stated = (request.headers.get(CLIENT_NAME_HEADER) or "").strip()
        if stated and _CLIENT_NAME.fullmatch(stated):
            return stated
        return (request.headers.get("user-agent") or "").strip()[:200] or None

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
                    # WHICH CLIP, for a zero-shot voice. `zeroshot` is ONE
                    # voice id and any number of recordings — BookForge keeps
                    # its clips in a userData directory and the extension
                    # keeps its own in the browser — so the id alone is two
                    # clients each assuming the resident one is theirs. Null
                    # for every other kind and for a model, and always
                    # present: an absent key would mean "this build does not
                    # say" (PHASE3-TTS.md section 5).
                    "reference": getattr(resident, "reference", None),
                    # WHAT HOLDS IT, and since when it has been held by
                    # nothing (2026-09-20). `resident` says what is on the
                    # card; these say whether anybody is coming back for it.
                    #
                    # Both read `crucible/settle.py`, which is the ONE owner of
                    # "what holds the card" — four facts, one function. Deriving
                    # them again here would be the one-fact-two-owners shape
                    # `docs/ARCHITECTURE.md` says this repo keeps finding, and
                    # the copy would drift the first time a fifth kind of
                    # holder arrived.
                    #
                    # `held_by` null with `resident` set is the stranded card: a
                    # `load-model` that succeeded and was never claimed, or a
                    # lease that lapsed with nothing asking again. Neither is
                    # hypothetical — the first happened on this server at
                    # 18:29:37Z on 2026-09-20, when a runner was stopped one
                    # second after its load reached `done`.
                    #
                    # REPORTING ONLY. Nothing here unloads anything and there is
                    # no timer: what should be DONE about a card nobody holds is
                    # a ruling (`docs/BUG-HUNT-2026-09-20.md` §F.8). This is the
                    # half that needs no ruling, because a reconciler on either
                    # side cannot act on a state the server will not say.
                    "held_by": (None if held is None else held.to_dict()),
                    "unclaimed_since": (
                        None if unclaimed is None else unclaimed.isoformat()
                    ),
                }
            ),
            # WHAT WAS TOLD TO GO AND HAS NOT (ledger R13, Owen's ruling
            # 2026-09-18). `resident` above says what may be USED; this says
            # what is still ON THE CARD after a stop that was asked for and
            # never confirmed (`crucible/residency.py`'s dying slot). The two
            # are never both set.
            #
            # It is on the bench read because it is a state a human has to
            # end: `engines/base.py` never SIGKILLs — a killed CUDA process
            # wedges WSL2 until Windows reboots — so nothing in Crucible will
            # clear this, and every load, the claim and the streaming door go
            # on refusing `engine_still_stopping` until somebody stops those
            # pids. `accepts_work` below is untouched and still true, and
            # truthfully so — the lane is free and an `echo` or a render
            # against nothing resident is still admitted. What was missing
            # was any way for a bench to explain the load that comes back
            # `engine_still_stopping` off a server that looks idle.
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
            # `warming` is neither running nor queued and a bench that ignored it
            # would draw an idle machine that is in fact spending two minutes
            # loading a model. It is the reason a slot is unavailable, so it is
            # reported where the slot is.
            "warming": residency.warming,
            # WHO HOLDS NARRATOR'S WIRE, WHICH IS NOT THE SAME QUESTION AS THE
            # LANE. `refuse_if_claimed`'s docstring is the long version: a
            # streaming session holds the resident engine *without* occupying the
            # lane, so `slots` below can say this server is free while the card is
            # not. Until this field existed, a bench polling for a free machine
            # read `busy: 0` and `running: []` **while the browser extension was
            # streaming from it**, submitted, and was refused `engine_in_use`
            # after the round trip. The refusal was right; the display was a lie,
            # and it lied in the one direction that matters (R3: nothing is ever
            # told "maybe" — and "free" when it is not is worse than "maybe").
            #
            # Reported as its own field rather than folded into `slots` because it
            # is a different fact with a different owner: the lane belongs to
            # `JobStore`, the claim belongs to `Residency`. Folding them would
            # give the composite a third owner and lose which one said no.
            "claim": (
                None
                if residency.claimed_by is None
                else {"held_by": residency.claimed_by}
            ),
            # THE OTHER KIND OF WORK. Three BookForge surfaces stream rather than
            # queue — the streaming page, the correct-sentences/re-roll page and
            # the browser extension — and they claim a server for as long as a
            # reader keeps reading. `progress` is null and always will be: see
            # `StreamSession.progress_report` for why a session has no
            # denominator and what is counted instead.
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
            # THE THIRD KIND OF WORK, and the one that was invisible longest. A
            # chat completion takes no lane, makes no job and left no record, so
            # a server grinding through a 27B translation reported `running: []`
            # and read as idle. It is counted here and it still gates nothing:
            # a vLLM engine BATCHES, so two passes on one resident model really
            # do run at once, and taking the lane to fix a reporting bug would
            # have serialised work the engine exists to overlap.
            #
            # `act` is the client's word (the `X-Crucible-Act` header, validated
            # against the capability classes). Null means it did not say, and
            # this server never guesses one: it cannot tell a simplify from a
            # translate, since both are a chat against the same 27B and the only
            # difference is a prompt it does not own.
            # `max_in_flight` is what THIS engine's door will admit at once,
            # and `max_in_flight_basis` is where that number came from, so a
            # client can size its own pool from the server instead of guessing
            # and discovering the answer as a starved socket. Null for an engine
            # that states no concurrency — the door then bounds nothing, which
            # is mlx-vlm today — and null when no model is
            # resident, because the limit belongs to the engine and there is no
            # engine to ask.
            "chat": {
                "in_flight": len(inflight),
                "max_in_flight": chat_limit,
                "max_in_flight_basis": chat_limit_basis,
                "rows": inflight.rows(),
            },
            # WHO CHANGED THIS SERVER'S SETTINGS, AND WHEN (PHASE15-HOST.md
            # section 3.2). Two apps and the operator page can all write the
            # same engine, so "why is translate suddenly on Anthropic" needs an
            # answer that is not "read three apps' logs". Newest first, the
            # last `settings.HISTORY_LIMIT` of them, in memory.
            #
            # **The field paths, never the values of an upstream.** A route's
            # model id is recorded because it is not a secret; an upstream
            # entry records `set` or `removed` and nothing more, which is the
            # whole of "a key appears in no activity record".
            "settings": {"writes": request.app.state.settings_history.rows()},
            # WHAT WAS DELETED, AND BY WHOM (PHASE15-HOST.md 3.5a). The host's
            # weights migration deletes a Windows copy once the guest has its
            # own, and a person looking at a machine with less on it than they
            # remember needs an answer that is not "read the host's log".
            # Newest first, in memory, the last `Removals.LIMIT`.
            "catalog": {"removals": request.app.state.removals.rows()},
            # THE INTENTION BEHIND THE CHATS, which no amount of looking at this
            # server could infer. A chat holds nothing and is over in seconds, so
            # between two blocks of a 2000-block translation this machine is idle
            # by every other measure here — and a `load-voice` submitted in that
            # gap used to evict the translator (crucible/leases.py). While this
            # is non-null, the thing on the card cannot be moved.
            #
            # It does NOT change `accepts_work` below. A lease is not a
            # reservation: this server will still take a job that does not need
            # the card's contents to change, and admission is still the door's.
            #
            # No `subject` field: a lease is only ever on the resident thing, and
            # `resident.id` above is already that fact's owner (R1). `kind` IS
            # carried, because the same six fields are a `409 leased`'s details —
            # a document with no `resident` beside it — and the kind is what says
            # which jobs the refusal covers.
            "lease": None if lease is None else lease.to_dict(),
            "slots": {
                # ONE LANE TODAY, and it is named rather than counted so the
                # ancillary lane (PHASE7-LANES.md section 3) can appear beside it
                # without changing this one's meaning. A key that is absent means
                # this build has no such lane — never that the lane is idle.
                #
                # `busy` counts THE LANE and nothing else — a stream does not take
                # it, and saying otherwise would redefine the lane to mean "the
                # card", which is `claim`'s job above. What a caller actually
                # wants before submitting is `accepts_work`, which is the
                # composition, derived here once so that three benches do not each
                # invent their own and disagree.
                "accelerated": {
                    "busy": 0 if running is None else 1,
                    "of": 1,
                    "queue_depth": store.queue_depth,
                    # Derived, never stored. Still not a reservation: a client
                    # that reads true and submits is racing every other client,
                    # and that race is settled at the door. See this route's
                    # "IT REPORTS AND NOTHING ELSE" note.
                    #
                    # `chat` is deliberately NOT a term here. A chat in flight
                    # does not stop this server taking a job, because the engine
                    # batches — adding it would turn an honest display into a
                    # false refusal and serialise work that overlaps today.
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
            # The same sentence the job door refuses with, from the same
            # producer: a client told "llm is off" by /v1/models and something
            # else by POST /v1/jobs would have two stories about one server
            # (PHASE9-CAPABILITY.md section 2.1).
            raise disabled_error("load-model", config)
        return model_rows(config, backend, residency)

    # ---------------------------------------------------------------- leases
    #
    # PHASE7-LANES.md section 5.2. Three routes and no state worth the name: a
    # client says it intends a run on the resident thing — model, voice or
    # aligner — heartbeats while the run is alive, and releases when it is done.
    # What that buys it is one refusal — `409 leased` at the job door for
    # anything that would take that thing off the card. Everything else about
    # this server is unchanged.

    @private.post("/models/{subject_id:path}/lease", status_code=201)
    async def open_lease(
        request: Request, subject_id: str, body: LeaseOpen
    ) -> dict[str, Any]:
        """Take the one lease this server holds at a time, on ANY resident kind.

        The order of the checks is their specificity, which is the job door's
        rule: a bad ttl and an unknown act are true of the request whatever this
        server is doing, so a client with a typo is told about the typo rather
        than about somebody else's lease. Residency comes next, because leasing a
        thing that is not here is a different mistake from being too late for
        one that is.

        **The id may name a model, a voice or an aligner** (PHASE7-LANES.md
        section 5.2, extended 2026-09-14). The route keeps its `/models/` path
        and its one route family, because the question it asks does not change
        with the kind: *is this the thing on the card?* The card holds ONE thing,
        so the kind is read off the residency rather than sent — and the
        namespaces being separate (a voice may be called `qwen3.5-9b`) cannot
        produce an ambiguity here, since only one of two colliding ids can be
        resident at a time and a lease is only ever on the resident one.

        Without this a book rendered chapter by chapter paid a narrator load per
        chapter and a book aligned chapter by chapter paid an aligner load per
        chapter, because the unload ruling clears the card the moment nothing
        holds it and the lease — the one thing that can hold it — could only name
        a model.
        """
        leases: Leases = request.app.state.leases
        _refuse_lease_on_an_upstream(subject_id)
        ttl_seconds = require_ttl(body.ttl_seconds)
        act = require_act_name(body.act.strip(), "a lease's `act`")
        # A CLEARANCE IS WAITED OUT, AND THE LEASE IS TAKEN ATOMICALLY AGAINST
        # ONE BEGINNING (2026-09-24, Briefcase). Briefcase's second run got a
        # 201 here on a model the settlement was already clearing, and its
        # first chat under that lease was `model_not_resident`. Now the door
        # waits the clearance out and answers from the settled card, and the
        # residency check and `leases.open` are made under the lock the
        # settlement's check-and-claim takes — so a lease granted here is one
        # the settlement will see, and never one on a thing already leaving.
        async with residency.settled_for(f"a lease on {subject_id!r}"):
            # Whatever is on the card, of any kind. A lease NEVER loads anything
            # — it is the promise not to move what is already there — so the
            # honest answer to an id that is not resident is the same one an
            # empty card gets, and it names what IS there so the client is not
            # left guessing which of the two mistakes it made.
            resident = residency.resident
            if resident is None or resident.id != subject_id:
                raise ApiError(
                    409,
                    "not_resident",
                    f"{subject_id!r} is not resident on this server; "
                    + (
                        f"the resident {KIND_NOUNS[resident.kind]} is "
                        f"{resident.id!r}. "
                        if resident is not None
                        else "nothing is. "
                    )
                    + "A lease promises not to move what is on the card; it "
                    "never loads anything — load it first (load-model, "
                    "load-voice, or an align job for an aligner), then lease "
                    "what that left resident.",
                    {
                        "requested": subject_id,
                        "resident": None if resident is None else resident.id,
                        "resident_kind": (
                            None if resident is None else resident.kind
                        ),
                    },
                )
            lease = leases.open(
                kind=resident.kind,
                subject=subject_id,
                act=act,
                client=_client_agent(request),
                ttl_seconds=ttl_seconds,
            )
        # The lease is now the thing holding the card, and its deadline is the
        # one moment a holder lets go that this server would otherwise never
        # see. Armed at the client's own `expires_at` (crucible/settle.py).
        request.app.state.settlement.arm_for_lease_expiry()
        return lease.receipt()

    @private.post("/leases/{lease_id}/heartbeat")
    async def heartbeat_lease(request: Request, lease_id: str) -> dict[str, Any]:
        """I am still here. Pushes the deadline out by the lease's own ttl.

        A 404 here is not an error to log and continue past: it means this
        client's run is no longer protected, and the card may move under it at
        any moment. The body says whether the lease was released or expired,
        which is the difference between "somebody took it from me" and "I stopped
        talking for too long".
        """
        leases: Leases = request.app.state.leases
        extended = leases.heartbeat(lease_id)
        # The deadline moved, so the one-shot that watches it moves with it.
        request.app.state.settlement.arm_for_lease_expiry()
        return {"expires_at": extended.expires_at.isoformat()}

    @private.delete("/leases/{lease_id}", status_code=204)
    async def release_lease(request: Request, lease_id: str) -> Response:
        """Give the card back before the ttl does it for you.

        The usual end of a lease, and the one that matters: expiry is the
        backstop for a client that died, not the way a finished run ends. A run
        that releases frees the next client immediately instead of after up to an
        hour of nothing happening.
        """
        leases: Leases = request.app.state.leases
        settlement: Settlement = request.app.state.settlement
        leases.release(lease_id)
        # The lease is gone, so the deadline it was watched by is too.
        settlement.arm_for_lease_expiry()
        # OWEN'S RULING, 2026-09-14: a released lease is a holder letting go, so
        # if the lane, the claim and the chats are also clear the card is cleared
        # before this 204 is written. That is Foundry's *"down when the queue
        # drains"* (FROM-FOUNDRY-WSL-VLLM.md section 3) with the drain stated by
        # the client instead of guessed at. Off the loop, because stopping an
        # engine waits on a process.
        await asyncio.to_thread(settlement.settle_quietly, "the lease was released")
        return Response(status_code=204)

    # ---------------------------------------------------------------- voices

    @private.get("/voices")
    async def voices(request: Request) -> list[dict[str, Any]]:
        """Every voice this build has a manifest for, and where it stands here."""
        if not config.enable_tts:
            raise disabled_error("tts", config)
        return voice_rows(
            config, backend, residency,
            leases=request.app.state.leases, store=request.app.state.store,
        )

    def _pinned_backends(live: Config, block: dict[str, Any]) -> dict[str, Any]:
        """Fill in a missing `revision` per backend from the repo's head sha.

        ── Why a caller is allowed to leave it out ────────────────────────────

        `voices.py` requires a full 40-character sha and refuses a branch name,
        because a pull has to be reproducible. That is right and it makes the
        field impossible for a person to supply from a link: nobody knows their
        own repo's head sha, and looking it up needs Hub access and, for a
        private repo, a token. The engine has both.

        ── ABSENT AND NULL BOTH MEAN "PIN IT FOR ME", and a STRING IS OBEYED ──

        A caller that names a revision gets exactly that revision, unresolved
        and unchecked here — asking for an older commit is a real thing to want
        and this is not the door that second-guesses it. Only the absence is
        filled, so nothing a caller wrote is ever replaced.

        A block that is not a table, or names no `hf_repo`, is left ALONE rather
        than repaired: the validator has a sentence for each of those and it is
        better than anything invented here.

        THAT IS ALSO WHAT KEEPS A LOCAL BLOCK OUT OF THIS
        (PHASE18-UNCERTIFIED.md section 3). One that names `path` names no
        `hf_repo`, so there is nothing to resolve a revision from and nothing
        here touches it — which is right rather than lucky: a directory has no
        commit to look up, and a `revision` beside a `path` is refused by the
        validator anyway.
        """
        backends = block.get("backends")
        if not isinstance(backends, dict):
            return block
        pinned: dict[str, Any] = {}
        for name, spec in backends.items():
            if not isinstance(spec, dict):
                pinned[name] = spec
                continue
            repo = spec.get("hf_repo")
            if spec.get("revision") is None and isinstance(repo, str) and repo != "":
                pinned[name] = {**spec, "revision": resolve_revision(live, repo)}
                continue
            # `None` with no repo to resolve from would reach the validator as a
            # type error about `revision`, which is the wrong sentence. Drop the
            # key so it is reported as missing, which is what it is.
            pinned[name] = {k: v for k, v in spec.items()
                            if not (k == "revision" and v is None)}
        return {**block, "backends": pinned}

    # ------------------------------------------------- voices this host owns
    #
    # PHASE3-TTS.md section 2 gave `voices/*.toml` one home: the package. That
    # made deploying a voice mean cutting a release, which Owen ended on
    # 2026-09-16 — *"we dont have to cut a new release every time we deploy a
    # model do we? ... i train models all the time. nearly every night"* — with
    # the `<CRUCIBLE_HOME>/voices` overlay that `voice_dirs()` reads after the
    # packaged set.
    #
    # THESE TWO DOORS ARE WHAT MAKE THAT OVERLAY REACHABLE. Until now the only
    # way to fill it was to put a file on the machine by hand, which works for
    # the engine you are sitting at and not at all for one across the room —
    # and "across the room" is the ordinary case, because the Mac is a backend
    # and nobody wants to ssh to add a voice they just trained.
    #
    # THE BODY IS THE FILE. A request carries the same `{"voice": {...}}`
    # document the TOML holds, and `crucible/voices.py` validates it with the
    # same `_parse` every packaged manifest goes through. A pydantic mirror of
    # the manifest schema would be a second author of that format, and the two
    # would disagree the first time a field grew a type — silently, because the
    # file would still parse.

    def _repin(live: Config, voice_id: str, body: Any) -> dict[str, Any]:
        """`{"pin": {...}}` — write this machine's pins row. Section 2.4.

        THE PIN IS LOADED BEFORE IT IS WRITTEN. `voice_for_pin` fetches the
        `crucible-voice.toml` at that revision, parses it and merges it with this
        box's `[tts.<engine>]` table, so a revision that carries no manifest, a
        manifest this build's schema cannot read, and a server that has never
        been told what this engine costs are each refused HERE, by name, with
        nothing written. A door that wrote first would leave a server holding a
        pin nothing can load and would report that by breaking the catalog.

        `revision: null` and an ABSENT revision both mean "pin the head for me",
        exactly as `_pinned_backends` reads them on the other body: nobody knows
        their own repo's head sha, and the engine is what holds the HuggingFace
        credential and does the fetching.
        """
        if not isinstance(body, dict):
            raise ApiError(
                400,
                "voice_invalid",
                "`pin` must be a table of hf_repo and revision, got "
                f"{type(body).__name__}",
            )
        unknown = sorted(set(body) - {"hf_repo", "revision"})
        if unknown:
            raise ApiError(
                400,
                "voice_invalid",
                f"`pin` carries unknown key(s) {unknown}; a pin is exactly "
                "hf_repo and revision. Everything else about a voice lives in "
                "its own repo's crucible-voice.toml at that revision",
            )
        repo = body.get("hf_repo")
        if not isinstance(repo, str) or repo.strip() == "":
            raise ApiError(
                400, "voice_invalid", "`pin` names no hf_repo"
            )
        revision = body.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise ApiError(
                400,
                "voice_invalid",
                "`pin.revision` must be a 40-character commit sha, or null to "
                f"resolve this repo's head, got {type(revision).__name__}",
            )
        if revision is None or revision.strip() == "":
            try:
                revision = resolve_revision(live, repo)
            except WeightsError as exc:
                raise ApiError(400, "revision_unresolved", str(exc)) from exc

        candidate = VoicePin(
            id=voice_id,
            hf_repo=repo,
            revision=revision,
            path=home_pins_path(),
        )
        try:
            voice_for_pin(candidate)
            pin = write_home_pin(voice_id, repo, revision)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc

        rows = [row for row in voice_rows(live, backend, residency,
                                          leases=app.state.leases,
                                          store=app.state.store)
                if row.get("id") == voice_id]
        return {"voice": rows[0] if rows else None, "path": str(pin.path)}

    @private.put("/voices/{voice_id}")
    async def voice_write(request: Request, voice_id: str) -> dict[str, Any]:
        """Add or replace a voice this machine owns. Returns its `/v1/voices` row.

        `revision` MAY BE OMITTED per backend, and that is the whole reason a
        person can paste a repo id: a manifest needs a full commit sha so a pull
        is reproducible, and resolving one is the engine's job because the
        engine is what fetches and what holds the HuggingFace credential.

        REPLACING A PACKAGED VOICE IS ALLOWED and is not an accident: the
        overlay is documented to win on a shared id, and deleting the overlay
        brings the packaged voice back, which is what makes trying an override
        safe. A door that refused would make the safe thing impossible and the
        unsafe thing (editing the install) the only way.
        """
        if not config.enable_tts:
            raise disabled_error("tts", config)
        live: Config = request.app.state.config

        try:
            document = await request.json()
        except Exception as exc:
            raise ApiError(
                400, "voice_invalid", f"the request body is not JSON: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ApiError(
                400,
                "voice_invalid",
                "the body must be the manifest document — a table with a `voice` "
                "key, exactly as voices/<id>.toml holds it",
            )

        # A voice that is ON THE CARD is not re-described underneath itself: its
        # pace, cap and sampling are in use by a render that is already running,
        # and a manifest is read per job rather than held, so the change would
        # land mid-book. Refused by name, with what to do about it.
        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(
                409,
                "voice_in_use",
                f"voice {voice_id!r} is loaded on this server right now, so its "
                "manifest cannot be rewritten underneath it. Unload it and try "
                "again",
                {"id": voice_id},
            )

        # TWO BODIES, TOLD APART BY SHAPE (PHASE21 section 2.4), and the two are
        # different decisions rather than two spellings of one.
        #
        #   {"pin": {...}}    REPIN. The voice's facts live in its own repo at
        #                     the named sha, and this machine is choosing which
        #                     sha. This is what a deploy does, per machine, and
        #                     it is the door this phase exists to make ordinary.
        #
        #   {"voice": {...}}  A LOCAL OVERRIDE: a whole manifest, exactly as
        #                     `voices/<id>.toml` holds it, written into this
        #                     machine's overlay. The PHASE18 `path` + `identity`
        #                     arm lives here, and so does a person retuning a
        #                     shipped voice on their own machine.
        #
        # BOTH IS REFUSED and so is NEITHER. A body carrying both has two
        # answers to "what is this voice" and the loader would pick one
        # (`load_all_voices` prefers the override); a body carrying neither is
        # not a manifest at all.
        has_pin = "pin" in document
        has_voice = "voice" in document
        if has_pin and has_voice:
            raise ApiError(
                400,
                "voice_invalid",
                "the body carries both a `pin` and a `voice`. They are two "
                "different decisions — which published revision this machine "
                "serves, and a whole manifest written on this machine — and a "
                "request making both leaves which one wins to the loader. Send "
                "one",
            )
        if not has_pin and not has_voice:
            raise ApiError(
                400,
                "voice_invalid",
                "the body must be either {\"pin\": {hf_repo, revision}} — the "
                "repo and commit this machine serves this voice from — or "
                "{\"voice\": {...}}, the manifest document exactly as "
                "voices/<id>.toml holds it",
            )

        if has_pin:
            return _repin(live, voice_id, document["pin"])

        block = document.get("voice")
        if isinstance(block, dict):
            document = {**document, "voice": _pinned_backends(live, block)}

        try:
            manifest = write_home_voice(voice_id, document)
        except WeightsError as exc:
            raise ApiError(400, "revision_unresolved", str(exc)) from exc
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc

        rows = [row for row in voice_rows(live, backend, residency,
                                          leases=app.state.leases,
                                          store=app.state.store)
                if row.get("id") == manifest.id]
        # The row is drawn from the same reader every other caller uses, so what
        # comes back is what `GET /v1/voices` will say — not an echo of the
        # request, which would report a save the file may have shaped otherwise.
        return {"voice": rows[0] if rows else None, "path": str(manifest.path)}

    @private.delete("/voices/{voice_id}", status_code=204)
    async def voice_remove(request: Request, voice_id: str) -> Response:
        """Delete this machine's own manifest for a voice. The packaged set stands.

        A voice that only ever existed in the overlay goes entirely; one that was
        SHADOWING a packaged voice reverts to the packaged manifest, which is the
        undo for an override that did not work out.

        This deletes the MANIFEST, never the weights. They are a subject like any
        other and `DELETE /v1/catalog/voice/{id}` is what removes them — two
        doors because they are two decisions, and somebody re-describing a voice
        they have just downloaded 8 GB of should not lose the download.
        """
        if not config.enable_tts:
            raise disabled_error("tts", config)
        if residency.resident_kind == "tts" and residency.resident_id == voice_id:
            raise ApiError(
                409,
                "voice_in_use",
                f"voice {voice_id!r} is loaded on this server right now. Unload "
                "it before removing the manifest it is running from",
                {"id": voice_id},
            )
        # THE PIN ROW OR THE OVERRIDE, WHICHEVER THIS ID HAS (section 2.4), and
        # both are tried rather than one being guessed at from the request:
        # `PUT` writes one or the other and a person deleting a voice knows only
        # that they added it. Never the weights — those are a catalog subject and
        # `DELETE /v1/catalog/voice/{id}` is what removes them, because somebody
        # re-describing a voice they have just downloaded 8.5 GB of should not
        # lose the download.
        try:
            gone_pin = remove_home_pin(voice_id)
            gone_voice = remove_home_voice(voice_id)
        except VoiceError as exc:
            raise ApiError(400, "voice_invalid", str(exc)) from exc
        if not gone_pin and not gone_voice:
            # IDEMPOTENT ON THE STATE REACHED (the ladder's ask, 2026-09-21): a
            # DELETE whose answer was lost to a restart is sent again, and the
            # second one must not read as a failure — `ladder-screen` stayed
            # registered three hours because it did. So a voice this server
            # simply does not have answers 204: the state asked for is the
            # state there is. The 404 stays for a voice that IS here and is not
            # this server's own — packaged, engine-base or a shipped pin —
            # because that one cannot be removed by this door and saying 204
            # would be a lie about it.
            if voice_id in load_all_voices():
                raise ApiError(
                    404,
                    "voice_not_custom",
                    f"{voice_id!r} is a shipped voice here, not one this server "
                    "added, and this door removes only what was added. The "
                    "packaged set is the install and is restored by it",
                    {"id": voice_id},
                )
        return Response(status_code=204)

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
            raise disabled_error("tts", config)
        manifest = known_voice(voice)
        require_streamable(manifest, config.backend_kind)
        return manifest

    @private.post("/tts/stream", status_code=201)
    async def open_stream(request: Request, body: StreamOpen) -> dict[str, Any]:
        """Open the one streaming session this server will hold at a time."""
        streams: StreamManager = request.app.state.streams
        manifest = _streaming_voice(body.voice)
        # WAITED OUT, NOT REFUSED (2026-09-24, Briefcase). A session opening
        # while the settlement clears the card used to be refused
        # `engine_in_use` by `claim()`, naming the settlement. It now waits and
        # opens against the settled card — or says `voice_not_resident`, which
        # is then true. `streams.open` is synchronous and takes its claim
        # inside, under the lock the settlement's check-and-claim takes, so the
        # two cannot interleave.
        async with residency.settled_for(f"streaming {body.voice!r}"):
            session = streams.open(
                voice=body.voice,
                language=body.language,
                manifest=manifest,
                client=_client_agent(request),
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
            sampling = require_sayable(manifest, body.take)
            # NO LENGTH REFUSAL HERE SINCE 2026-09-19. `chunk_too_long` stood
            # between these two lines — the cap certificate, refused rather
            # than re-split — and it is retired with the render door's twin
            # (PHASE18-UNCERTIFIED.md section 4, `jobs/tts/render.py`). Owen:
            # *"I don't think it's crucible's place to refuse chunks outside
            # the band… especially if we add a different tts engine."*
            # Chunking is still the client's and the cap is still advertised on
            # `/v1/voices` for it to pack to; what this server no longer does
            # is act on a number that a screening checkpoint may not have and
            # that another engine's frame arithmetic would not be described by.
            return {"id": session.say(body.id, body.text, body.take, sampling)}
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
        """
        store: JobStore = request.app.state.store
        leases: Leases = request.app.state.leases
        plugin = resolve(store.registry, body.type, config)
        if body.model is not None:
            # An upstream model is never resident and never on the lane, so
            # `load-model` naming one is the same mistake a lease on one is,
            # and gets the same name (PHASE15-HOST.md section 3.4). Checked
            # before `resolve_model`, whose refusal would be
            # `unknown_model` — true of a local catalog and wrong about what
            # the caller actually did.
            _refuse_lease_on_an_upstream(body.model)
        model = resolve_model(plugin, body.model)

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
            plugin.preflight(model, body.params)

            job = store.create(
                body.type, model, body.params,
                client=_client_agent(request), client_ref=body.client_ref,
            )
            try:
                _materialise_inputs(config, store, job, body.inputs)
                # `enqueue` asks admission again and is the authority on it;
                # nothing awaits between here and the check above — the body of
                # `settled_for` must not — so the two are one atomic stretch on
                # the event loop. Inside the same `try` so that a refusal from
                # either leaves no half-built job behind.
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
        return store.hold(store.get(job_id), _client_agent(request))

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

    # ----------------------------------------------------------------- tasks
    #
    # PHASE13-OPERATOR.md section 3.3. Five routes with the shapes the job
    # routes have — a 202 with an id, a status read, an SSE stream, a DELETE
    # that cancels, a list — because a page that already knows how to watch a
    # job should not have to learn a second protocol to watch an install. What
    # they are NOT is `POST /v1/jobs`: a job is work a client wants done with
    # this server's card, a task is work done to the server itself, and
    # `crucible/tasks.py` is where that difference is written down.

    @private.post("/tasks", status_code=202)
    async def create_task(request: Request, body: TaskCreate) -> dict[str, str]:
        """Admit one operator task, or refuse by name.

        Every refusal is made here, before the 202, and in the order the job
        door uses: what is wrong with the REQUEST first (`unknown_subject`,
        `unknown_job_type`, `narrator_engine_required`, `invalid_module`), then
        what is already true (`already_installed`, `job_type_installed`), then
        what this server is doing (`task_busy`, and for anything that reloads
        the registry, `server_busy`). A client with a misspelled id told "busy"
        would come back in ten minutes to be told about the typo.
        """
        tasks: TaskStore = request.app.state.tasks
        return {"task_id": tasks.submit(body.request()).id}

    @private.get("/tasks")
    async def list_tasks(request: Request) -> dict[str, Any]:
        """The last few tasks, newest first. In memory; a restart forgets them."""
        tasks: TaskStore = request.app.state.tasks
        return {"tasks": [task.to_dict() for task in tasks.recent()]}

    @private.get("/tasks/{task_id}")
    async def get_task(request: Request, task_id: str) -> dict[str, Any]:
        tasks: TaskStore = request.app.state.tasks
        return tasks.get(task_id).to_dict()

    @private.delete("/tasks/{task_id}")
    async def cancel_task(request: Request, task_id: str) -> dict[str, str]:
        """Cancel. A pull stops at its next chunk and its partial bytes go.

        `cancelling` and not `cancelled`, exactly as the job door answers:
        the flag is set here and the runner ends when it sees it, which for a
        pull is the next progress callback and for an install is the SIGTERM
        landing. Watch the stream for the `cancelled` event — telling a caller
        "cancelled" before the download thread has stopped would be the
        ambiguous answer R3 forbids.
        """
        tasks: TaskStore = request.app.state.tasks
        task = tasks.get(task_id)
        return {"task_id": task.id, "status": tasks.cancel(task)}

    @private.get("/tasks/{task_id}/events")
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        tasks: TaskStore = request.app.state.tasks
        task = tasks.get(task_id)
        delivered = _last_event_id(request)
        return StreamingResponse(
            _task_event_stream(request, tasks, task, delivered),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # --------------------------------------------------------------- openai

    @private.get("/openai/models")
    @openai.get("/models")
    async def openai_models(request: Request) -> dict[str, Any]:
        """The resident model in OpenAI's list shape, plus every routed upstream one.

        `resident_model` rather than `resident`: with a voice on the card there
        is no model to list, and narrator answers no OpenAI route.

        **The upstream rows are the ROUTED ones and not a catalog**
        (PHASE15-HOST.md section 3.4). This route answers *"what may I send as
        `model`"*, and the answer is the resident thing plus whatever the
        operator routed to — the upstream's whole catalog is a different
        question with a different door,
        `POST /v1/settings/upstreams/{name}/test`, and putting it here would
        make a client believe this server had agreed to serve any of them.
        """
        live: Config = request.app.state.config
        resident = residency.resident_model
        data: list[dict[str, Any]] = _routed_upstream_rows(live)
        if resident is None:
            return {"object": "list", "data": data}
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
                    # What this engine will be sent for a knob the request
                    # leaves out (PHASE2-LLM.md section 9). Here for
                    # `max_model_len`'s reason: Foundry reads the OpenAI-shaped
                    # listing rather than `/v1/models`, and the defaults are a
                    # thing it has to be able to see before it decides what to
                    # send. `null` means the engine's own default.
                    "defaults": resident.defaults.to_dict(),
                },
                *data,
            ],
        }

    @private.post("/openai/chat/completions")
    @openai.post("/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """Proxied to the resident engine, or forwarded to an upstream.

        PHASE2-LLM.md section 5 is the local half and is unchanged in every
        respect. PHASE15-HOST.md section 3.4 is the other: a `model` of the
        form `<upstream>/<id>` goes to that upstream on the operator's account.

        **The slash is the whole of the test**, and it works because a local
        model id can never contain one — refused at manifest load,
        `manifest_model_id_slash`. One character, one owner, no table.
        """
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
        if upstreams.split_model(requested) is not None:
            return await _forward_to_upstream(
                request, requested, body, client_agent=_client_agent(request)
            )
        # A CLEARANCE IS WAITED OUT, AND THE CHAT'S RECORD IS OPENED ATOMICALLY
        # AGAINST ONE BEGINNING (2026-09-24, Briefcase). A chat that arrives
        # while the settlement is SIGTERMing the engine waits for it to finish
        # and is then answered from the settled card — `model_not_resident`,
        # which is true — instead of being proxied to an engine on its way
        # out. The resident check and `inflight.open` are made under the lock
        # the settlement's check-and-claim takes, so a chat that IS admitted
        # is an `InFlight` row the settlement will see and not clear under.
        async with residency.settled_for("a chat request"):
            # The resident MODEL: a voice on the card is not something a chat
            # request can be proxied to, so this door's honest answer is the
            # same `model_not_resident` it gives for an empty card.
            resident = residency.resident_model
            if resident is None or resident.model_id != requested:
                raise _model_not_resident(requested, resident, "a chat request")

            # The manifest's gaps, filled — and the audit of what filled them
            # (PHASE2-LLM.md section 9). `resident.defaults` is what the
            # manifest said at LOAD time, not what it says now, for the same
            # reason `max_model_len` comes off the engine's record.
            applied = apply_defaults(body, resident.defaults)
            sampling_headers = {SAMPLING_HEADER: applied.header()}
            forwarded = _forward_body(raw, applied, resident)
            url = f"{resident.base_url}/v1/chat/completions"
            client: httpx.AsyncClient = request.app.state.http
            inflight: InFlight = request.app.state.inflight
            # Read BEFORE the work starts, so an unknown act is a 400 instead of
            # a completion that ran and was then reported under a name nobody
            # knows.
            act = read_act(request.headers)

            settlement: Settlement = request.app.state.settlement
            chat_over = _chat_over(settlement)

            # A chat is the one piece of accelerator work that took no lane,
            # made no job row and left a record NOWHERE. Tracked so
            # `/v1/activity` can say what this machine is doing; it still gates
            # nothing — see crucible/inflight.py for why taking the lane would
            # have been the wrong fix for the right bug.
            # WHAT THIS ENGINE CAN ACTUALLY HAVE OPEN AT ONCE (2026-09-20).
            # Until today this door admitted everything and
            # `crucible/inflight.py` said, in so many words, that the record
            # gates nothing. For a BATCHING engine that is still exactly right
            # (2026-09-24: vLLM now STATES its batch, `--max-num-seqs`, so it is
            # bounded at that plus one and a client can size to it; the batch
            # is admitted whole, so nothing it could overlap is refused.)
            #
            # It was wrong for a SERIAL one. mlx-lm accepts every connection on
            # a ThreadingHTTPServer and then generates on ONE thread draining
            # ONE queue, so twelve accepted requests are one running and eleven
            # waiting with nothing on the wire saying so. Foundry's clean pass
            # died there on 2026-09-20: 12 in flight, a 300 s client deadline, a
            # request that had not started when it passed, the pass dead at
            # block 352 of 940.
            #
            # (CORRECTED 2026-09-24: mlx-lm is not serial — that one thread runs
            # a `BatchGenerator` `--decode-concurrency` wide. The door was right
            # to bound it and wrong about the width; the width is now read off
            # the resident engine's argv, `engines/mlx_lm.py`.)
            #
            # A refusal a client can act on beats a socket that goes quiet. The
            # limit is the engine's own measured concurrency plus one (see
            # `engines.chat_admission`), the wait is the median of what
            # completions on this engine have actually been taking, and a server
            # that has finished none states no `Retry-After` rather than
            # inventing one — `_rate_limited`'s rule, applied to a number of
            # our own.
            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                wait = inflight.retry_after()
                return _chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis, wait=wait
                )

            entry = inflight.open(
                act=act, model=resident.model_id, client=_client_agent(request)
            )
        try:
            if body.get("stream") is True:
                # THE RELAY OUTLIVES THIS HANDLER, so the record and the
                # settlement belong to the relay's end and not to this `return`.
                # Closing them here would count a streamed completion as finished
                # the moment its first byte was ready — and since 2026-09-14 that
                # would clear the card out from under a stream still producing
                # tokens.
                return await _proxy_stream(
                    client,
                    url,
                    forwarded,
                    resident,
                    sampling_headers,
                    when_relayed=_after_the_stream(inflight, entry, chat_over),
                )
            try:
                upstream = await _sent_across_the_wire(
                    lambda: _post_unless_the_caller_leaves(
                        client, url, forwarded, request, JSON_HEADERS
                    ),
                    where=f"the engine serving {resident.model_id!r}",
                )
            except httpx.HTTPError as exc:
                raise _engine_unreachable(resident, exc) from None
            if upstream is None:
                response = _caller_gone(resident)
            else:
                content = upstream.content
                if upstream.status_code == 200:
                    content = _restore_model_id(content, resident)
                response = Response(
                    content=content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get(
                        "content-type", "application/json"
                    ),
                    # On the engine's own refusal too: a 400 whose `max_tokens`
                    # came from the manifest is a 400 the manifest caused, and the
                    # reader needs that on the response that carries it.
                    headers=sampling_headers,
                )
            inflight.close(entry)
            # AFTER THE ANSWER IS WRITTEN, not before it. Starlette runs a
            # response's background task once the body has gone out, so a client
            # gets its completion at the speed the engine produced it and the
            # card is cleared behind it. A settlement can wait on a SIGTERM for
            # as long as three minutes, and no answer should be held for that.
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            # A refusal, a disconnect, an engine that died: the record must not
            # outlive the request, and the card is as free now as it would have
            # been had this succeeded.
            inflight.close(entry)
            await chat_over()
            raise

    # --------------------------------------------------------------- decide

    @private.post(
        "/decide",
        response_model=None,
        responses={200: {"model": DecideResponse}},
    )
    async def decide(request: Request, body: DecideRequest) -> Response:
        """One distribution per question, read off the resident model.

        PHASE22-DECIDE.md is the contract. A decision is the chat door's
        sibling and walks through the chat door's machinery — the act header,
        the resident check, `chat_admission`, the `InFlight` record,
        `_chat_over` — with a different body in and out. What is its own is the
        reading (`crucible/decide.py`): the frame, the letters, the parser.

        EVERY REFUSAL A CALLER CAN CAUSE IS MADE BEFORE ANYTHING IS SENT: the
        act, an upstream id, a model that is not resident, too many images,
        images on a text model, too many options, an engine that returns no
        top logprobs or too few of them, a full door. A decision that spent the
        card and then failed on a question the server could have read first
        would be a decision the client paid for twice.
        """
        # Read BEFORE the work starts, as on chat: an unknown act is a 400
        # rather than a decision reported under a name nobody knows.
        act = read_act(request.headers)
        if upstreams.split_model(body.model) is not None:
            raise ApiError(
                400,
                "decide_needs_logprobs",
                f"{body.model!r} names an upstream; a decision reads the next-token "
                "distribution at the resident model, and no upstream returns one. "
                "Name the resident Crucible model id",
                {"requested": body.model},
            )
        # WAITED OUT AND ATOMIC, exactly as on the chat door (2026-09-24,
        # Briefcase): a decision arriving mid-clearance is answered from the
        # settled card, and the one it proxies is an `InFlight` row the
        # settlement sees. Everything in this block is synchronous, which is
        # what `settled_for` requires of it.
        async with residency.settled_for("a decision"):
            resident = residency.resident_model
            if resident is None or resident.model_id != body.model:
                raise _model_not_resident(body.model, resident, "a decision")

            n_images = decide_core.check_image_count(body.images)
            if n_images:
                # Read at decision time and only with images in hand: the record
                # carries no modalities, and a manifest whose modalities changed
                # also changed its engine line (`--language-model-only`, an
                # `mmproj`), which is a reload either way.
                #
                # WHAT THIS BACKEND SERVES, not what the weights accept (PHASE22
                # section 2.9). `qwen3.5-4b` accepts images everywhere and is
                # served them on cuda-linux and llama-windows only: on the Mac its
                # engine is mlx-lm, which is never handed a picture. Reading the
                # model-wide list here would pass a page to an engine that drops it
                # and answer from the text alone.
                manifest = load_manifest(resident.model_id)
                served = manifest.serves(backend.kind)
                if "image" not in served:
                    raise ApiError(
                        400,
                        "model_text_only",
                        f"{resident.model_id!r} is served {list(served)} on "
                        f"{backend.kind} (its weights accept "
                        f"{list(manifest.modalities)}) and this decision carries "
                        f"{n_images} image(s). Whether a model answers images HERE is "
                        "its manifest's backend block (`serves`, PHASE22 section 2.9)",
                        {"model": resident.model_id, "backend": backend.kind,
                         "serves": list(served),
                         "modalities": list(manifest.modalities),
                         "images": n_images},
                    )
            plans = decide_core.plan_all(body)

            reading = decide_reading(resident.engine)
            if not reading.served:
                raise _decide_not_served(resident, reading.basis, None)
            widest = max(plans, key=lambda item: len(item.labels))
            if reading.max_logprobs is not None and len(widest.labels) > reading.max_logprobs:
                raise _decide_not_served(
                    resident,
                    f"{resident.engine} returns at most {reading.max_logprobs} top "
                    f"logprobs and question {widest.name!r} has {len(widest.labels)} "
                    f"options ({reading.basis})",
                    {"question": widest.name, "options": len(widest.labels),
                     "max_logprobs": reading.max_logprobs},
                )

            inflight: InFlight = request.app.state.inflight
            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                wait = inflight.retry_after()
                return _chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis, wait=wait
                )
            concurrency = (
                limit if limit is not None else decide_core.UNSTATED_ENGINE_CONCURRENCY
            )

            settlement: Settlement = request.app.state.settlement
            chat_over = _chat_over(settlement)
            client: httpx.AsyncClient = request.app.state.http
            entry = inflight.open(
                act=act, model=resident.model_id, client=_client_agent(request)
            )
        try:
            answered = await _unless_the_caller_leaves(
                _decide_on_engine(
                    client,
                    resident,
                    body,
                    plans,
                    max_logprobs=reading.max_logprobs,
                    concurrency=concurrency,
                ),
                request,
            )
            if answered is None:
                response: Response = _caller_gone(resident)
            else:
                response = JSONResponse(content=answered.model_dump(mode="json"))
            inflight.close(entry)
            # After the answer is written, as on chat: the card is cleared
            # behind the decision, never in front of it.
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            inflight.close(entry)
            await chat_over()
            raise

    app.include_router(public)
    app.include_router(private)
    app.include_router(openai)
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


def _forward_body(raw: bytes, applied: Applied, resident: Any) -> bytes:
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

    Since phase 2's section 9 there is a second reason to re-serialise: a
    manifest default the request left room for. It is held to the same rule —
    **the bytes pass through untouched unless something actually changed**, so a
    request that states every knob this server knows about reaches the engine
    exactly as it was written, grammar and all.
    """
    if not applied.changed and resident.engine_model_name == resident.model_id:
        return raw
    document = dict(applied.body)
    if resident.engine_model_name != resident.model_id:
        document["model"] = resident.engine_model_name
    return json.dumps(document).encode("utf-8")


async def _watch_for_disconnect(request: Request) -> None:
    """Return the moment the caller's connection has gone away.

    A WAIT ON `receive`, NOT A POLL OF `is_disconnected()` (2026-09-24). The
    poll asks `receive` inside an already-cancelled scope and takes whatever is
    there; how that answer survives the trip depends on every layer between the
    server and the route, and one layer (`BaseHTTPMiddleware`, from 1.0.18) lost
    it every time — a caller that had hung up read as present until the engine
    answered. A blocked `receive` is the server's own statement: uvicorn wakes
    it on `connection_lost` and it returns `http.disconnect`. That is the
    earliest anyone on this side can know, with no clock to choose.

    Only for a route that has read its whole body — both callers have (the
    chat door reads `request.body()`, the decision door's body is a parsed
    parameter) — so what `receive` has left to say is the disconnect. An
    `http.request` that arrives anyway is an empty tail and is passed over;
    anything else is a fault in the server and is raised, not waited past.
    Cancelling this while it waits consumes nothing: the next reader of
    `receive` finds the channel as it was.
    """
    while True:
        message = await request.receive()
        kind = message.get("type")
        if kind == "http.disconnect":
            return
        if kind != "http.request":
            raise RuntimeError(
                f"the caller's channel said {kind!r} while its request was being "
                "answered; only `http.request` or `http.disconnect` can come there"
            )


async def _post_unless_the_caller_leaves(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    request: Request,
    headers: dict[str, str],
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
    return await _unless_the_caller_leaves(
        client.post(url, content=body, headers=headers), request
    )


async def _sent_across_the_wire(
    attempt: Callable[[], Awaitable[Any]], *, where: str
) -> Any:
    """`attempt()`, sent once more on a fresh socket if the wire loses it.

    `LOST_ON_THE_WIRE` says which faults qualify and why; `WIRE_ATTEMPTS` is the
    entire budget. Every loss is said in the server log by name, first attempt
    or last, so a socket that keeps dying is visible there rather than folded
    into a success. A caller that hangs up during the first attempt is not a
    loss: `_post_unless_the_caller_leaves` returns None for that, and None is
    returned, never repeated.
    """
    attempt_number = 0
    while True:
        attempt_number += 1
        try:
            return await attempt()
        except LOST_ON_THE_WIRE as exc:
            detail = type(exc).__name__ + (f": {exc}" if str(exc) else "")
            if attempt_number >= WIRE_ATTEMPTS:
                print(
                    f"crucible: lost on the wire to {where} ({detail}), attempt "
                    f"{attempt_number} of {WIRE_ATTEMPTS}; giving up",
                    file=sys.stderr,
                )
                raise
            print(
                f"crucible: lost on the wire to {where} ({detail}), attempt "
                f"{attempt_number} of {WIRE_ATTEMPTS}; sending it again on a "
                "fresh socket",
                file=sys.stderr,
            )


def _attempts(exc: Exception) -> str:
    """How many times the wire was tried, for the 502 that ends it.

    A fault in `LOST_ON_THE_WIRE` only reaches a 502 after the whole budget;
    anything else (a timeout) was tried once, and saying "on 2 attempts" of it
    would be a lie.
    """
    if isinstance(exc, LOST_ON_THE_WIRE):
        return f" on {WIRE_ATTEMPTS} attempts"
    return ""


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


def _model_not_resident(requested: str, resident: Any, answering: str) -> ApiError:
    """409: the named model is not the one on the card, and none will be loaded.

    One body for the chat door and the decision door (PHASE22 section 2.1: "the
    same body the chat door gives"), with only the noun changed.
    """
    return ApiError(
        409,
        "model_not_resident",
        f"{requested!r} is not resident on this server; "
        + (f"{resident.model_id!r} is. " if resident is not None else "no model is. ")
        + f"Crucible never loads a model to answer {answering} — submit "
        'a {"type": "load-model"} job first.',
        {"requested": requested, "resident": None if resident is None
         else resident.model_id},
    )


def _decide_not_served(
    resident: Any, reason: str, extra: dict[str, Any] | None
) -> ApiError:
    """503: the resident engine cannot return what a decision reads."""
    return ApiError(
        503,
        "decide_not_served",
        f"the {resident.engine} engine serving {resident.model_id!r} cannot serve "
        f"this decision: {reason}. Nothing was sent to it",
        {"model": resident.model_id, "engine": resident.engine, "reason": reason,
         **(extra or {})},
    )


def _decide_engine_refused(resident: Any, response: httpx.Response) -> ApiError:
    """502: the engine answered a decision's request with something other than 200.

    Named `engine_error` and not relayed as it stood, unlike the chat door: a
    decision is several requests and one answer, so there is no single engine
    body to hand back — the one that failed is quoted instead.
    """
    text = response.text
    return ApiError(
        502,
        "engine_error",
        f"the {resident.engine} engine serving {resident.model_id!r} answered a "
        f"decision's request with {response.status_code}: {text[:500]}. Its log is "
        f"{resident.log_path}",
        {"engine": resident.engine, "status": response.status_code,
         "body": text[:2000]},
    )


async def _unless_the_caller_leaves(
    work: Awaitable[Any], request: Request
) -> Any:
    """`work`, raced against the caller hanging up; None if the caller went first.

    THE ONE OWNER of "the caller left, so the engine stops" for every door that
    answers once rather than streaming: the chat door's POST
    (`_post_unless_the_caller_leaves`) and the whole of a decision. A decision
    is a prime and up to sixteen questions, and ONE watcher cancelling the
    whole of it is what closes every in-flight socket AND takes back every
    question still waiting at its gate — so nothing more is sent to an engine
    nobody is listening to. Sixteen watchers on one ASGI `receive` would be
    sixteen readers of one channel.

    When the caller goes first, `work` is cancelled and AWAITED before this
    returns, so the cancellation has reached httpx and closed the sockets by
    the time the door closes its `InFlight` row and asks the settlement about
    the card. The door is cancelled itself (the server shutting down): the work
    goes with it.
    """
    task = asyncio.ensure_future(work)
    watch = asyncio.create_task(_watch_for_disconnect(request))
    try:
        done, _ = await asyncio.wait({task, watch}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        raise
    finally:
        watch.cancel()
    if task in done:
        return task.result()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # Ours: cancelled two lines above, and awaited so the cancellation
        # reaches httpx and closes the sockets. Letting it propagate would
        # report the caller's own departure as this request being cancelled.
        pass
    except Exception as exc:
        # The work failed in the same instant the caller left. There is nobody
        # to answer, so it is said where a person can find it and the caller's
        # departure is what this returns.
        print(
            f"crucible: work for a caller who left failed as it went: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    # The watcher finished, which is the caller leaving — or the watcher's own
    # fault, raised here now that the work is down rather than lost with it.
    watch.result()
    return None


async def _decide_on_engine(
    client: httpx.AsyncClient,
    resident: Any,
    body: DecideRequest,
    plans: list[decide_core.Plan],
    *,
    max_logprobs: int | None,
    concurrency: int,
) -> DecideResponse:
    """The prime, then every question, on the resident engine's chat route.

    PHASE22 section 2.5. With more than one question the shared prefix goes
    first and ALONE, so its KV is cached (vLLM) or its context checkpoint laid
    down (llama-server, hybrid Qwen3.5) before the questions ask for it; then
    the questions go out together, at most `concurrency` at once. Answers come
    back in the request's question order, and when questions fail the one
    reported is the first in THAT order, so a retry with the same body meets
    the same refusal first.
    """
    started = time.perf_counter()
    url = f"{resident.base_url}/v1/chat/completions"
    state_text = decide_core.render_state(body.state)
    images = list(body.images or [])
    engine = resident.engine

    async def forward(
        msgs: list[dict[str, Any]], k: int | None
    ) -> tuple[decide_core.Reading, float]:
        payload = json.dumps(
            decide_core.request_body(resident.engine_model_name, msgs, k)
        ).encode("utf-8")
        sent = time.perf_counter()
        try:
            response = await _sent_across_the_wire(
                lambda: client.post(url, content=payload, headers=JSON_HEADERS),
                where=f"the engine serving {resident.model_id!r}",
            )
        except httpx.HTTPError as exc:
            raise _engine_unreachable(resident, exc) from None
        wall_ms = (time.perf_counter() - sent) * 1000.0
        if response.status_code != 200:
            raise _decide_engine_refused(resident, response)
        try:
            data = response.json()
        except ValueError as exc:
            raise ApiError(
                502,
                "engine_error",
                f"the {engine} engine serving {resident.model_id!r} answered 200 "
                f"with a body that is not JSON: {exc}. Its log is {resident.log_path}",
                {"engine": engine},
            ) from None
        return decide_core.read_reply(data, engine, want_probs=k is not None), wall_ms

    def timing(reading: decide_core.Reading, wall_ms: float) -> decide_core.ForwardTiming:
        return decide_core.ForwardTiming(
            wall_ms=round(wall_ms, 1),
            prompt_tokens=reading.prompt_tokens,
            cached_tokens=reading.cached_tokens,
        )

    prime: decide_core.ForwardTiming | None = None
    if len(plans) > 1:
        # A PRIME IS NOT AN ANSWER (snap `3509bc5`): it asks for no logprobs
        # and its token is never read, so a reply without them is no fault.
        reading, wall_ms = await forward(
            decide_core.prime_messages(state_text, images), None
        )
        prime = timing(reading, wall_ms)

    gate = asyncio.Semaphore(concurrency)

    async def ask(
        item: decide_core.Plan,
    ) -> tuple[Any, decide_core.ForwardTiming, int]:
        k = decide_core.top_k(len(item.labels), max_logprobs)
        async with gate:
            reading, wall_ms = await forward(
                decide_core.question_messages(state_text, images, item), k
            )
        assert reading.top is not None  # want_probs=True always reads them
        dist = decide_core.label_distribution(
            reading.top, item, engine, missing=body.missing
        )
        return (
            decide_core.answer(item, dist, body.missing),
            timing(reading, wall_ms),
            reading.prompt_tokens,
        )

    tasks = [asyncio.create_task(ask(item)) for item in plans]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if task.cancelled():
                continue
            failure = task.exception()
            if failure is not None:
                raise failure from None
        raise

    answers = {}
    per_question = {}
    tokens = {}
    for item, (answered, timed, prompt_tokens) in zip(plans, results):
        answers[item.name] = answered
        per_question[item.name] = timed
        tokens[item.name] = prompt_tokens
    return DecideResponse(
        model=decide_core.ModelProvenance(
            id=resident.model_id,
            revision=resident.revision,
            fingerprint=resident.fingerprint,
        ),
        engine=engine,
        answers=answers,
        timing_ms=decide_core.DecideTiming(
            total=round((time.perf_counter() - started) * 1000.0, 1),
            per_question=per_question,
            prime=prime,
        ),
        tokens=decide_core.DecideTokens(per_question=tokens, images=len(images)),
    )


def _engine_unreachable(resident: Any, exc: Exception) -> ApiError:
    return ApiError(
        502,
        "engine_unreachable",
        f"the engine serving {resident.model_id!r} at {resident.base_url} did not "
        f"answer{_attempts(exc)}: {type(exc).__name__}: {exc}. Its log is "
        f"{resident.log_path}",
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
    """The same substitution inside one SSE frame of a streamed completion."""
    return _set_model_in_frame(frame, resident.model_id)


def _set_model_in_frame(frame: bytes, model_id: str) -> bytes:
    """Name `model_id` in every JSON `data:` chunk of one SSE frame.

    Only `data:` lines carrying a JSON chunk are touched, and only their `model`
    field. `data: [DONE]`, comments, and any line the engine frames some other
    way are passed through as they arrived: mid-stream there is no way to raise,
    and a frame Crucible does not recognise is the engine's to explain.

    Two callers, one substitution: the local proxy puts Crucible's id back where
    it wrote the engine's, and the upstream proxy puts `<upstream>/<model>` back
    where it wrote the bare id. Same rule — **the id the caller asked for is the
    id the answer names** — so it is one function rather than two that could
    come to frame SSE differently.
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
        chunk["model"] = model_id
        lines[index] = b"data: " + json.dumps(chunk).encode("utf-8")
        changed = True
    return b"\n".join(lines) if changed else frame


def _chat_over(settlement: Settlement) -> Callable[[], Awaitable[None]]:
    """OWEN'S RULING, 2026-09-14: the chat is over, so who still has the card?

    A chat holds nothing and reserves nothing, which is right while it runs and
    is exactly why its END is worth asking at: a client that did not lease has
    now said everything it is going to say, and if the lane, the lease and the
    claim are all clear the card goes. **A run of chats with no lease therefore
    reloads its model between requests**, which is the bill for not stating an
    intention rather than a bug in the rule (crucible/settle.py).

    A CLEANUP FAILURE IS NOT AN OPERATION FAILURE. An engine that will not stop
    is said, loudly, in the server log — it does not turn a completion that
    arrived into a 500, and it does not break a stream that had already been
    delivered.
    """

    async def over() -> None:
        await asyncio.to_thread(
            settlement.settle_quietly, "the last chat completion finished"
        )

    return over


def _after_the_stream(
    inflight: InFlight, entry: Entry, chat_over: Callable[[], Awaitable[None]]
) -> Callable[[], Awaitable[None]]:
    """Close the record and ask about the card, once the relay is really done."""

    async def done() -> None:
        inflight.close(entry)
        await chat_over()

    return done


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

    def __init__(
        self,
        *args: Any,
        upstream: httpx.Response,
        when_relayed: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._upstream = upstream
        # THE END OF A STREAMED COMPLETION, for the same reason the upstream's
        # close lives here rather than in the generator: this `finally` is the
        # one place that runs on every path Starlette can take. It is where the
        # chat stops being in flight and where the card is asked about.
        self._when_relayed = when_relayed

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._upstream.aclose()
            await self._when_relayed()


async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    body: bytes,
    resident: Any,
    extra_headers: dict[str, str],
    when_relayed: Callable[[], Awaitable[None]],
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
        # Opening the stream is the only moment a retry is honest here: nothing
        # has been relayed yet. A socket that dies mid-stream is the client's
        # to notice, because half of an answer has already gone out.
        upstream = await _sent_across_the_wire(
            lambda: client.send(request, stream=True),
            where=f"the engine serving {resident.model_id!r}",
        )
    except httpx.HTTPError as exc:
        raise ApiError(
            502,
            "engine_unreachable",
            f"the resident engine at {url} did not answer{_attempts(exc)}: "
            f"{type(exc).__name__}: {exc}. Its log is {log_path}",
        ) from None

    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=payload,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers=dict(extra_headers),
            # There is no relay on this path, so the end of the completion is
            # here: the record closes and the card is asked about, after the
            # engine's own refusal has been written out.
            background=BackgroundTask(when_relayed),
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
        when_relayed=when_relayed,
        status_code=200,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        # The sampling audit rides on the response headers, which is the one
        # place a streamed completion HAS to put it: there is nowhere in an SSE
        # body to add a field a client would not have to learn to skip.
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            **extra_headers,
        },
    )


# --------------------------------------------------- forwarding to an upstream
#
# PHASE15-HOST.md section 3.4. Owen, 2026-09-14: *"they dont have ollama
# fallbacks or cloud anything at all … one contract, one SDK, one API, one
# communication method."* The provider code left BookForge and Foundry; this is
# where it landed, and `crucible/upstreams.py` holds everything that differs
# between the three.
#
# WHAT THIS PATH DELIBERATELY DOES NOT DO. No lease, no lane, no settlement:
# nothing was on the card, so there is nothing to ask about when the answer is
# written. It DOES open an `inflight` record, because `/v1/activity` must be
# able to say "translating on anthropic" — a chat that is invisible is the
# defect `crucible/inflight.py` exists to have fixed, and where it runs does
# not change that.
#
# AND IT NEVER RETRIES. A `429` comes back to the caller with the upstream's own
# `Retry-After` and the CALLER waits. A request that reached the upstream may
# already be billed, and a server that quietly sent it twice would be spending
# somebody's money to smooth a graph.


def _refuse_lease_on_an_upstream(subject_id: str) -> None:
    """`lease_not_needed` — there is nothing on the card to hold in place.

    PHASE15-HOST.md section 3.4. A lease is the promise not to MOVE what is
    resident, and an upstream model is never resident: no load, no eviction,
    nothing another job could take away. A server that accepted the lease would
    be issuing a promise about a card the work never touches, and the client
    that took it would hold the one lease this server has, refusing everybody
    else's `load-model` for a run that is happening in somebody else's
    datacentre.

    Both doors that can name a model call this — `POST /v1/models/{id}/lease`
    and a `load-model` job — because both mistakes come from the same wrong
    belief and a client owed one sentence should not get two.
    """
    if upstreams.split_model(subject_id) is None:
        return
    raise ApiError(
        409,
        "lease_not_needed",
        "an upstream model is never resident; send the chat",
        {"model": subject_id},
    )


def _routed_upstream_rows(config: Config) -> list[dict[str, Any]]:
    """Every distinct upstream model a route names, in `UPSTREAM_NAMES` order.

    DISTINCT, because `translate` and `simplify` routed to the same Anthropic
    model are one thing a client may send and two classes that send it;
    `routed_for` is where the second fact goes. Two identical rows would make a
    client's model picker show the same entry twice.

    No `created`, no `max_model_len`, no `defaults`: this server did not load
    it, cannot see its context window and has no manifest for it. Absent is the
    honest value, and a client that sizes `max_tokens` against `max_model_len`
    already skips the clamp when the field is missing (CLIENT-SURFACES.md 6.1).
    """
    routed_for: dict[str, list[str]] = {}
    for entry in config.routes:
        routed_for.setdefault(entry.model, []).append(entry.capability)
    rows: list[dict[str, Any]] = []
    for name in upstreams.UPSTREAM_NAMES:
        for model, classes in routed_for.items():
            if model.partition("/")[0] != name:
                continue
            rows.append(
                {
                    "id": model,
                    "object": "model",
                    "owned_by": name,
                    "upstream": name,
                    "routed_for": classes,
                }
            )
    return rows


def _chat_limit_of(residency: Residency) -> tuple[int | None, str | None]:
    """The chat door's admission limit for whatever model is resident, and why.

    `(None, None)` when no model is resident: the limit is a property of the
    ENGINE, and with nothing loaded there is no engine to ask. That is not the
    same as "unlimited", and `/v1/activity` reports it as null rather than as a
    number, so a client reading the field cannot mistake an empty card for a
    door that will take anything.
    """
    resident = residency.resident_model
    if resident is None:
        return (None, None)
    return chat_admission(resident.engine, resident.engine_args)


def _chat_queue_full(
    *,
    resident: Any,
    limit: int,
    basis: str | None,
    wait: int | None,
) -> Response:
    """503: this engine already has everything it can run, and it will not queue.

    A REFUSAL RATHER THAN A HELD SOCKET, which is the whole point. The failure
    this replaces looked like a healthy server: the request was accepted, the
    connection stayed open, nothing was generated, and the client found out at
    its own deadline. A caller that is told "full, try in 12 seconds" can pace
    itself; a caller holding an accepted socket cannot.

    503 and not 429: nothing here is a rate limit or a quota. The engine is
    genuinely at capacity for a moment, which is what 503 means, and it is the
    status a client is most likely to already treat as "wait and retry".

    A `JSONResponse` rather than a raised `ApiError` for one reason: `Retry-After`
    is a HEADER, and `ApiError` carries a body. `_rate_limited` below does the
    same for the same reason, and states the rule both follow — a `Retry-After`
    is real or it is absent, never invented. Here it is the median of what
    completions on this engine have recently taken, and a server that has
    finished none omits the header.
    """
    error = ApiError(
        503,
        "chat_queue_full",
        f"this server already has {limit} chat completion(s) open on "
        f"{resident.model_id!r} and its {resident.engine} engine will not queue "
        "another: "
        + (basis or "no basis stated")
        + ". Nothing was sent to the engine, so this request cost nothing and "
        "can be made again"
        + ("" if wait is None else f"; about {wait}s is what completions on this "
           "engine have recently been taking"),
        {
            "model": resident.model_id,
            "engine": resident.engine,
            "max_in_flight": limit,
            "max_in_flight_basis": basis,
            "retry_after": wait,
        },
    )
    headers = {} if wait is None else {"Retry-After": str(wait)}
    return JSONResponse(status_code=503, headers=headers, content=error.body())


def _upstream_unreachable(name: str, url: str, exc: Exception) -> ApiError:
    return ApiError(
        502,
        "upstream_unreachable",
        f"{name} did not answer at {url}{_attempts(exc)}: {type(exc).__name__}: "
        f"{exc}",
        {"upstream": name},
    )


def _rate_limited(name: str, response: httpx.Response, body: bytes) -> Response:
    """The upstream's own 429, passed through with its own `Retry-After`.

    Passed through rather than translated, because the only correct response to
    a rate limit is for the thing that decided to make the request to decide
    when to make it again. `Retry-After` is the upstream's number and this
    server has no better one; it is copied when it is there and absent when it
    is not, never invented.
    """
    headers: dict[str, str] = {}
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return JSONResponse(
        status_code=429,
        headers=headers,
        content=ApiError(
            429,
            "upstream_rate_limited",
            f"{name} rate-limited this request: "
            f"{upstreams.upstream_message(body)}. This server never retries a "
            "billed request — the caller waits",
            {
                "upstream": name,
                "retry_after": retry_after,
                "upstream_status": response.status_code,
            },
        ).body(),
    )


def _upstream_rejected(name: str, status_code: int, body: bytes) -> ApiError:
    """Anything the upstream refused, with the upstream's own words.

    ONE name for every non-200 that is not a 429, and a `502` for all of them.
    Section 3.4 spells out the `401` case; the rest — a model id the account
    cannot reach, an overloaded region, a body the provider did not like — are
    the same event from this server's side: *the hop failed and the upstream
    said why*. `details.upstream_status` carries the real number, so nothing is
    lost by not multiplying the names, and a client is never left reading a
    Crucible refusal that sounds like Crucible's own fault without the
    provider's sentence beside it.
    """
    return ApiError(
        502,
        "upstream_rejected",
        f"{name} refused this request with {status_code}: "
        f"{upstreams.upstream_message(body)}",
        {"upstream": name, "upstream_status": status_code},
    )


def _set_model_in_response(raw: bytes, model_id: str) -> bytes:
    """Name the id the CALLER asked for in a completion the upstream named its own.

    Anthropic and OpenAI both echo the bare model name they were sent, and the
    caller sent `<upstream>/<model>`. Crucible's id is the contract in both
    directions (`_restore_model_id`'s rule, one hop further out).
    """
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw
    if not isinstance(body, dict):
        return raw
    body["model"] = model_id
    return json.dumps(body).encode("utf-8")


async def _forward_to_upstream(
    request: Request,
    requested: str,
    body: dict[str, Any],
    *,
    client_agent: str | None,
) -> Response:
    """One chat, sent to the upstream the operator configured."""
    live: Config = request.app.state.config
    name, model_id = upstreams.require_upstream_model(requested)
    record = live.upstream(name)
    if record is None:
        # 409 and not 404: the model id is well formed and this server knows
        # the upstream — what is missing is the operator's key, which is a
        # state of the server rather than a mistake in the request. The app
        # that reads this shows its settings window, which is 5.2's whole
        # point.
        raise ApiError(
            409,
            "upstream_unconfigured",
            f"{requested!r} names the {name} upstream and this server has no "
            f"{upstreams.UPSTREAM_FIELD[name]} for it. Configure it with "
            f"`PUT /v1/settings` — nothing here falls back to a local model, "
            "because a route is a decision somebody made and not a guess this "
            "server gets to improvise",
            {"upstream": name, "model": requested},
        )

    # Before the work, so an unknown act is a 400 rather than a BILLED
    # completion reported under a name nobody knows — and before Ollama's
    # context lookup, which is a round trip a refused request should not cost.
    act = read_act(request.headers)
    client: httpx.AsyncClient = request.app.state.http
    if name == "ollama":
        # Section 3.4a: Ollama is spoken natively and its body cannot be built
        # without the `num_ctx` it runs at, which may have to be asked for.
        forwarded = await upstreams.forward_ollama(
            client, record, model_id, body, request.app.state.ollama_contexts
        )
    else:
        forwarded = upstreams.forward_body(name, model_id, body)
    sampling_headers = {
        SAMPLING_HEADER: json.dumps(
            forwarded.sources, separators=(",", ":"), sort_keys=True
        ),
        upstreams.CONTEXT_HEADER: json.dumps(
            forwarded.context, separators=(",", ":"), sort_keys=True
        ),
    }
    url = upstreams.chat_url(record)
    headers = upstreams.chat_headers(record)
    inflight: InFlight = request.app.state.inflight
    entry = inflight.open(act=act, model=requested, client=client_agent)
    try:
        if body.get("stream") is True:
            return await _stream_from_upstream(
                client,
                record,
                url,
                headers,
                forwarded,
                requested,
                sampling_headers,
                when_relayed=_close_inflight(inflight, entry),
            )
        try:
            upstream = await _sent_across_the_wire(
                lambda: _post_unless_the_caller_leaves(
                    client, url, forwarded.body, request, headers
                ),
                where=f"{name} at {url}",
            )
        except httpx.HTTPError as exc:
            raise _upstream_unreachable(name, url, exc) from None
        if upstream is None:
            # The caller hung up. Nothing reads this; it exists so the handler
            # returns something that is not shaped like a completion.
            return JSONResponse(
                status_code=499,
                content=ApiError(
                    499,
                    "client_disconnected",
                    f"the caller closed the connection before {name} answered; "
                    "the upstream request was cancelled with it",
                ).body(),
            )
        payload = upstream.content
        if upstream.status_code == 429:
            return _rate_limited(name, upstream, payload)
        if upstream.status_code != 200:
            raise _upstream_rejected(name, upstream.status_code, payload)
        if forwarded.dialect != upstreams.DIALECT_OPENAI:
            try:
                document = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ApiError(
                    502,
                    "upstream_rejected",
                    f"{name} answered 200 with a body that is not JSON: {exc}",
                    {"upstream": name, "upstream_status": 200},
                ) from None
            translated = (
                upstreams.anthropic_to_openai(document, requested)
                if forwarded.dialect == upstreams.DIALECT_ANTHROPIC
                else upstreams.ollama_to_openai(document, requested)
            )
            content = json.dumps(translated).encode("utf-8")
        else:
            content = _set_model_in_response(payload, requested)
        return Response(
            content=content,
            status_code=200,
            media_type="application/json",
            headers=sampling_headers,
        )
    except BaseException:
        inflight.close(entry)
        raise
    finally:
        # Only the non-streamed paths reach here with the record still open;
        # the streamed one hands its close to the relay's end and returns
        # above. `close` is idempotent (crucible/inflight.py), so the two
        # cannot double-count and neither can leave a row behind.
        if body.get("stream") is not True:
            inflight.close(entry)


def _close_inflight(inflight: InFlight, entry: Entry) -> Callable[[], Awaitable[None]]:
    """The end of a streamed upstream completion. No settlement: no card moved."""

    async def done() -> None:
        inflight.close(entry)

    return done


async def _stream_from_upstream(
    client: httpx.AsyncClient,
    record: upstreams.UpstreamRecord,
    url: str,
    headers: dict[str, str],
    forwarded: upstreams.Forwarded,
    requested: str,
    extra_headers: dict[str, str],
    when_relayed: Callable[[], Awaitable[None]],
) -> Response:
    """A streamed completion from an upstream, in OpenAI chunk shape.

    The upstream response is opened before anything is returned, so a refusal
    comes back with its own status and body rather than as a 200 whose stream
    turns out to be an error — the same rule `_proxy_stream` follows for the
    local engine.

    Anthropic's SSE and Ollama's NDJSON are different protocols and are
    TRANSLATED (`upstreams.AnthropicStreamTranslator`,
    `upstreams.OllamaStreamTranslator`); OpenAI's is relayed with one
    substitution, the `model` the caller asked for.
    """
    name = record.name
    upstream_request = client.build_request(
        "POST",
        url,
        content=forwarded.body,
        headers=headers,
        # No read deadline, for `_proxy_stream`'s reason: a completion emits a
        # token at a time and may think for a long while before the first one.
        timeout=httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT, read=None, write=60.0, pool=10.0
        ),
    )
    try:
        # `_proxy_stream`'s reason: before the first byte, a retry costs nothing.
        upstream = await _sent_across_the_wire(
            lambda: client.send(upstream_request, stream=True),
            where=f"{name} at {url}",
        )
    except httpx.HTTPError as exc:
        await when_relayed()
        raise _upstream_unreachable(name, url, exc) from None

    if upstream.status_code != 200:
        payload = await upstream.aread()
        await upstream.aclose()
        if upstream.status_code == 429:
            response = _rate_limited(name, upstream, payload)
            response.background = BackgroundTask(when_relayed)
            return response
        await when_relayed()
        raise _upstream_rejected(name, upstream.status_code, payload)

    if forwarded.dialect != upstreams.DIALECT_OPENAI:
        translator: (
            upstreams.AnthropicStreamTranslator | upstreams.OllamaStreamTranslator
        ) = (
            upstreams.AnthropicStreamTranslator(requested)
            if forwarded.dialect == upstreams.DIALECT_ANTHROPIC
            else upstreams.OllamaStreamTranslator(
                requested, include_usage=forwarded.include_usage
            )
        )

        async def relay() -> AsyncIterator[bytes]:
            async for chunk in upstream.aiter_bytes():
                for frame in translator.feed(chunk):
                    yield frame
            for frame in translator.finish():
                yield frame

    else:

        async def relay() -> AsyncIterator[bytes]:
            # SSE frames end at a blank line, so the relay holds a partial
            # frame until it has one. Whatever is left when the upstream stops
            # is forwarded as it stands rather than swallowed.
            buffer = b""
            async for chunk in upstream.aiter_bytes():
                buffer += chunk
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    yield _set_model_in_frame(frame, requested) + b"\n\n"
            if buffer:
                yield _set_model_in_frame(buffer, requested)

    return _RelayResponse(
        relay(),
        upstream=upstream,
        when_relayed=when_relayed,
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            **extra_headers,
        },
    )


# ------------------------------------------------------------------- helpers


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


def _referenced_artifact(store: JobStore, name: str, ref: ArtifactRef) -> Path:
    """The file an artifact input names, or `409 artifact_expired` by name.

    EXPIRED IS THE ONE ANSWER for every way the bytes are not here: the job was
    reaped (released, fetched, or past the seven-day collector), this server
    never ran it (a different server, or ids from before a wipe), or the job
    exists and published no such file. In every case the client's correct next
    move is the same: upload the bytes. It is refused before the job exists.
    """
    try:
        validate_member_name(ref.name)
    except ValueError as exc:
        raise ApiError(400, "invalid_artifact_name", str(exc)) from None
    try:
        source_job = store.get(ref.job_id)
    except ApiError as exc:
        raise ApiError(
            409,
            "artifact_expired",
            f"input {name!r} names artifact {ref.name!r} of job {ref.job_id!r}, and "
            f"this server no longer holds that job ({exc.code}: {exc.message}). "
            "Upload the bytes instead",
            {"input": name, "job_id": ref.job_id, "artifact": ref.name, "why": exc.code},
        ) from None
    source = source_job.artifacts_dir / ref.name
    if ref.name not in source_job.artifacts or not source.is_file():
        raise ApiError(
            409,
            "artifact_expired",
            f"input {name!r} names artifact {ref.name!r} of job {ref.job_id!r}, "
            f"which that job does not hold (it holds {len(source_job.artifacts)} "
            "artifact(s)). Upload the bytes instead",
            {"input": name, "job_id": ref.job_id, "artifact": ref.name, "why": "no_such_artifact"},
        )
    return source


def _materialise_inputs(
    config: Config, store: JobStore, job: Job, inputs: dict[str, JobInput]
) -> None:
    """Write every declared input into the job's scratch dir before it is queued.

    **AN UPLOAD IS MOVED, NOT COPIED** (Owen's ruling, 2026-09-18). It used to
    be `copyfile`d with the original left in `uploads/`, so every byte a client
    sent was on this disk twice for ever: nothing deleted an upload, and the PC
    was measured at 2.4 GB in 2,935 of them. A blob exists to become a job's
    input, and `os.replace` is the same operation with one copy — and, on one
    filesystem, without reading the bytes at all.

    **EVERYTHING IS CHECKED BEFORE ANYTHING MOVES**, which the copy did not
    have to care about. A refusal on the third input used to leave the first
    two copied and the uploads untouched; now it would leave the first two
    MOVED, and a job that is then discarded takes them with it. So the two
    passes below: nothing is written until every name, every blob and every
    inline payload has been read and accepted.
    """
    planned: list[tuple[str, Path, str | None, bytes | None]] = []
    linked: list[tuple[Path, Path]] = []
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
            # Asked BEFORE the file check, because the two are different
            # answers about the same missing file: a blob some job already
            # took is not one this server never had, and telling a client the
            # second when the first is true sends it looking for a bug in its
            # own upload.
            store.refuse_if_blob_consumed(declared.blob_id)
            source = Path(config.uploads_dir) / declared.blob_id
            if not source.is_file():
                raise ApiError(
                    400,
                    "unknown_blob",
                    f"input {name!r} names blob {declared.blob_id!r}, which this "
                    "server does not hold",
                )
            planned.append((name, target, declared.blob_id, None))
        elif declared.inline_base64 is not None:
            try:
                payload = base64.b64decode(declared.inline_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ApiError(
                    400,
                    "invalid_inline_input",
                    f"input {name!r} is not valid base64: {exc}",
                ) from None
            planned.append((name, target, None, payload))
        elif declared.artifact is not None:
            linked.append((target, _referenced_artifact(store, name, declared.artifact)))
        else:  # unreachable: JobInput's validator requires exactly one source
            raise ApiError(
                400,
                "invalid_input",
                f"input {name!r} names neither a blob nor inline bytes",
            )

    # A REFERENCE IS A HARD LINK, not a copy: the same bytes under the new
    # job's name, on the one filesystem both directories live in. Reaping the
    # referenced job afterwards unlinks its name and leaves this one intact.
    for target, source in linked:
        try:
            os.link(source, target)
        except OSError as exc:
            raise ApiError(
                500,
                "artifact_link_failed",
                f"could not link {source} as input {target.name}: "
                f"{type(exc).__name__}: {exc}",
            ) from None
    for name, target, blob_id, payload in planned:
        if blob_id is None:
            assert payload is not None  # one of the two, by the loop above
            target.write_bytes(payload)
            continue
        # The record first, so that the one line that makes the bytes
        # unreachable from `uploads/` is the one line that says who has them.
        # Two inputs of one request naming the same blob land here, and are
        # refused naming this job — which is what happened: the first of them
        # took it.
        store.consume_blob(blob_id, job)
        source = Path(config.uploads_dir) / blob_id
        os.replace(source, target)
        # The upload's metadata sidecar describes a blob that is no longer in
        # `uploads/`. It is written by `POST /v1/uploads` and read by nothing,
        # so it goes with the blob rather than being moved beside it. A
        # CLEANUP FAILURE IS NOT AN OPERATION FAILURE: the input is in place
        # and the job is going to run, so a sidecar that will not unlink is
        # said out loud and left.
        meta = Path(config.uploads_dir) / f"{blob_id}.json"
        try:
            meta.unlink(missing_ok=True)
        except OSError as exc:
            print(
                f"crucible: blob {blob_id} moved into job {job.id} but its "
                f"metadata {meta} would not delete: {type(exc).__name__}: {exc}",
                file=sys.stderr,
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


async def _task_event_stream(
    request: Request, tasks: TaskStore, task: Task, last_event_id: int
) -> AsyncIterator[str]:
    """A task's events, in the job stream's shape and with its own terminal set.

    A third copy of this loop rather than a parameterised one, which is the
    call `_session_event_stream` already made and for the same kind of reason:
    the three logs have three lifetimes. A job's log lives as long as its
    directory, a session's is pruned behind its readers, and a TASK's can be
    dropped whole when it ages past `HISTORY` — so this one has to survive its
    subject disappearing between two iterations, which the others never do.
    """
    waiter = tasks.subscribe(task)
    index = last_event_id
    try:
        while True:
            while index < len(task.events):
                event = task.events[index]
                index += 1
                yield _format_event(event)
                if event["event"] in TERMINAL_EVENTS:
                    return
            waiter.clear()
            if index < len(task.events):
                continue
            try:
                await asyncio.wait_for(waiter.wait(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    return
                yield ": keepalive\n\n"
    finally:
        tasks.unsubscribe(task, waiter)


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
