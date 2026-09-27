"""API v1 — exactly the surface in DESIGN.md section 4.

Base path is `/v1`. Protected routes need `Authorization: Bearer <token>` and
`X-Crucible-Api: 1`, checked in that order. Public discovery (`GET /v1/ping`)
and the limited pairing start/poll exchange do not require an existing token.
Pairing approval always requires authentication; start/poll require the version header.
Errors are always `{"error": {"code", "message", "details"?}}`.
"""

from .app import UI_DIR, create_app
from .deps import require_api_version, require_auth
from .proxy import (
    LOST_ON_THE_WIRE,
    PROXY_KEEPALIVE_EXPIRY,
    WIRE_ATTEMPTS,
    _chat_limit_of,
    _chat_queue_full,
)

__all__ = [
    "LOST_ON_THE_WIRE",
    "PROXY_KEEPALIVE_EXPIRY",
    "UI_DIR",
    "WIRE_ATTEMPTS",
    "_chat_limit_of",
    "_chat_queue_full",
    "create_app",
    "require_api_version",
    "require_auth",
]
