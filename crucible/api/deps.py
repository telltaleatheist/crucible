from __future__ import annotations

import secrets
from typing import Any, Callable

from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .. import peer as peer_module
from ..config import Config
from ..errors import ApiError
from ..jobs import disabled_error
from ..protocol import API_HEADER, API_VERSION


class BeforeEveryRequest:
    def __init__(self, app: ASGIApp, *, step: Callable[[], None]) -> None:
        self.app = app
        self.step = step

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            self.step()
        await self.app(scope, receive, send)


def error_response(error: ApiError, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, headers=headers, content=error.body())


def tts_enabled(config: Config) -> Any:
    def refuse_unless_tts_is_on() -> None:
        if not config.enable_tts:
            raise disabled_error("tts", config)

    return Depends(refuse_unless_tts_is_on)


def _bearer_problem(request: Request) -> str | None:
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
    if _bearer_problem(request) is not None:
        raise ApiError(
            401,
            peer_module.PEER_TOKEN_MISMATCH,
            "the peer door takes THIS engine's bearer token, which is the one "
            "the orchestrator already holds: it reads it from the guest's "
            "pairing line, or it is the token in the config it wrote itself. "
            "There is no second credential for the relation "
            "(docs/internals/host-and-platform.md, \"Orchestrator/engine relation\")",
        )


def require_peer_api_version(request: Request) -> None:
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
