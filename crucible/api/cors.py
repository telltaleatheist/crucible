"""Cross-origin requests from the web pages this server is told to trust.

B-Side's iPhone app (Owen, 2026-10-04) runs its pages from `capacitor://localhost`
and pairs with a Crucible by address, then follows jobs with fetch-streamed SSE. A
WebView asks first (an `OPTIONS` preflight) and reads an answer only when it carries
`Access-Control-Allow-Origin` for its origin; until now this server sent no such
header, so every such call failed before it began.

`[server] cors_origins` in config.toml is the allow-list: exact origins
(`scheme://host[:port]`), empty by default, so a server nobody configured answers as it
always has. The list is the server's live config, read on every request, so editing the
file takes effect without a restart (the config follower re-reads it).

The origin list is not the security boundary - the bearer token is. A page in a listed
origin gets nothing it does not already hold the token for; listing it only lets the
browser hand the page the answer. That is why `*` is refused (config.py): a wildcard
would make every page on the internet a candidate holder of whatever token it can find.
"""
from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config import Config
from ..protocol import (
    ACT_HEADER,
    API_HEADER,
    CLIENT_HEADER,
    QUEUE_TICKET_HEADER,
    SESSION_HEADER,
    USER_AGENT_HEADER,
)

ALLOW_METHODS = "GET, POST, PUT, DELETE"
# Every header @crucible/client sets (tests/test_cors.py reads them out of the SDK's own
# source, so a header the SDK starts sending cannot be forgotten here). User-Agent is in it
# on purpose: the SDK sets it on every request, and WebKit treats a script-set User-Agent
# as a header the preflight must allow - without it every authenticated call from B-Side's
# iPhone app failed with "Load failed" (1.0.101).
ALLOW_HEADERS = ", ".join(
    [
        "Authorization",
        "Content-Type",
        "Range",
        "Last-Event-ID",
        USER_AGENT_HEADER,
        API_HEADER,
        CLIENT_HEADER,
        ACT_HEADER,
        QUEUE_TICKET_HEADER,
        SESSION_HEADER,
    ]
)
EXPOSE_HEADERS = "Content-Range, Accept-Ranges, Content-Length"
# How long a browser may reuse a preflight answer: short, so a removed origin stops
# working within minutes.
PREFLIGHT_MAX_AGE_S = 600


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


class AllowListedOrigins:
    """Answer preflights from, and label responses for, the origins in `[server] cors_origins`."""

    def __init__(self, app: ASGIApp, *, config: Config) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        origin = _header(scope, b"origin")
        if origin is None or origin not in self.config.cors_origins:
            # Same-origin, a native client, or an origin nobody listed: unchanged.
            await self.app(scope, receive, send)
            return

        if scope["method"] == "OPTIONS" and _header(scope, b"access-control-request-method") is not None:
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", origin.encode("latin-1")),
                        (b"access-control-allow-methods", ALLOW_METHODS.encode()),
                        (b"access-control-allow-headers", ALLOW_HEADERS.encode()),
                        (b"access-control-max-age", str(PREFLIGHT_MAX_AGE_S).encode()),
                        (b"vary", b"Origin"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def labelled(message: Message) -> None:
            # Added at the start of the response, so a streamed one (SSE, an artifact)
            # carries it like any other.
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"access-control-allow-origin", origin.encode("latin-1")))
                headers.append((b"access-control-expose-headers", EXPOSE_HEADERS.encode()))
                headers.append((b"vary", b"Origin"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, labelled)


__all__ = ["ALLOW_HEADERS", "ALLOW_METHODS", "AllowListedOrigins", "EXPOSE_HEADERS"]
