from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors import CrucibleError
from ..protocol import API_VERSION

if TYPE_CHECKING:
    from .connection import Connection

CONNECT_NOTE = (
    "Crucible answers /v1/ping without a token; try that first if this is the "
    "wrong address"
)


class ClientRefusal(CrucibleError):
    ...


@dataclass(frozen=True)
class ServerError:

    code: str
    message: str
    details: dict[str, Any]


def error_in(raw: bytes) -> tuple[Any, ServerError | None]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return body, None
    code, message = error.get("code"), error.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return body, None
    details = error.get("details")
    return body, ServerError(
        code=code, message=message,
        details=details if isinstance(details, dict) else {},
    )


def host_of(connection: Connection | None) -> str:
    if connection is None:
        return "the server"
    if connection.name:
        return connection.name
    return urllib.parse.urlsplit(connection.url).hostname or connection.url


def next_step(error: ServerError, connection: Connection | None) -> str | None:
    host = host_of(connection)
    if error.code == "unauthorized":
        if connection is not None and connection.source == "local":
            return (
                f"{error.code}: this machine's own token is not accepted by "
                f"{host} ({error.message}); the engine and the config disagree. "
                "Run `crucible doctor` here"
            )
        return (
            f"{error.code}: this pairing is not accepted by {host} "
            f"({error.message}); re-run `crucible pair <address>` for it"
        )
    if error.code.startswith("api_version_"):
        theirs = error.details.get("server_api_version")
        ours = error.details.get("client_api_version")
        if isinstance(theirs, int) and isinstance(ours, int) and theirs < ours:
            return (
                f"{error.code}: {host} speaks API version {theirs} and this "
                f"computer speaks {ours}; {host} is older. Update Crucible on "
                f"{host}"
            )
        return (
            f"{error.code}: {host} speaks API version {theirs} and this computer "
            f"speaks {API_VERSION}; this computer is older. Update Crucible here"
        )
    return None


def unreachable(connection: Connection, exc: BaseException) -> str:
    host = host_of(connection)
    if connection.source == "local":
        step = (
            "This machine's engine is not answering: `crucible local status` "
            "says whether it is running and `crucible doctor` whether it is "
            "installed"
        )
    else:
        step = (
            f"On {host}, run `crucible doctor` to see whether Crucible is up; "
            "if it is up but not reachable from this network, `crucible lan "
            f"enable` there (Windows) opens it. {CONNECT_NOTE}"
        )
    return (
        f"server_unreachable: {host} at {connection.url} (from "
        f"{connection.source}) did not answer: {exc}. {step}"
    )
