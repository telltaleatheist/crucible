from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi import Request

from ..protocol import CLIENT_HEADER, SESSION_HEADER, USER_AGENT_HEADER

if TYPE_CHECKING:
    from ..queuesessions import QueueSession, QueueSessions

_CLIENT_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")


def client_agent(request: Request) -> str | None:
    stated = (request.headers.get(CLIENT_HEADER) or "").strip()
    if stated and _CLIENT_NAME.fullmatch(stated):
        return stated
    return (request.headers.get(USER_AGENT_HEADER) or "").strip()[:200] or None


def queue_session(request: Request, sessions: "QueueSessions") -> "QueueSession | None":
    """The open queue session this request is an item of: the one its
    session header names (refused by name when that session is not open or not
    the caller's), else the open session when the caller is the client holding it."""
    named = (request.headers.get(SESSION_HEADER) or "").strip() or None
    return sessions.member(named, client_agent(request))
