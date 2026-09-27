from __future__ import annotations

import secrets
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .. import API_HEADER, API_VERSION
from .. import peer as peer_module
from ..config import Config
from ..errors import ApiError


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


def _error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.body())


def _bearer_problem(request: Request) -> str | None:
    """What is wrong with the request's bearer token, or None when it is this server's."""
    config: Config = request.app.state.config
    header = request.headers.get("authorization")
    if header is None:
        return "missing Authorization header; send `Authorization: Bearer <token>`"
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or presented == "":
        return "Authorization header must be `Bearer <token>`"
    if not secrets.compare_digest(presented.strip(), config.token):
        return "bearer token is not this server's token"
    return None


def _presented_version(request: Request) -> tuple[str | None, int | None]:
    """The version header as sent, and its major number, or None where it has none."""
    presented = request.headers.get(API_HEADER)
    if presented is None:
        return None, None
    try:
        return presented, int(presented.strip().split(".")[0])
    except ValueError:
        return presented, None


def require_auth(request: Request) -> None:
    problem = _bearer_problem(request)
    if problem is not None:
        raise ApiError(401, "unauthorized", problem)


def require_api_version(request: Request) -> None:
    presented, major = _presented_version(request)
    if presented is None:
        raise ApiError(
            426,
            "api_version_required",
            f"send `{API_HEADER}: {API_VERSION}`; this server speaks API version "
            f"{API_VERSION}",
            {"server_api_version": API_VERSION, "client_api_version": None},
        )
    if major is None:
        raise ApiError(
            426,
            "api_version_unreadable",
            f"{API_HEADER}: {presented!r} is not a version; this server speaks API "
            f"version {API_VERSION}",
            {"server_api_version": API_VERSION, "client_api_version": presented},
        )
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
    if _bearer_problem(request) is not None:
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
    presented, major = _presented_version(request)
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
