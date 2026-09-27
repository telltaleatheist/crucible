from __future__ import annotations

API_VERSION = 1
API_HEADER = "X-Crucible-Api"
ACT_HEADER = "X-Crucible-Act"
CLIENT_HEADER = "X-Crucible-Client"
HANDOVER_HEADER = "X-Crucible-Handover"

LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 7100
DOOR_PORT = 7101


def user_agent(role: str) -> str:
    from . import VERSION

    return f"crucible-{role}/{VERSION}"


def api_headers(token: str | None = None) -> dict[str, str]:
    headers = {API_HEADER: str(API_VERSION)}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers
