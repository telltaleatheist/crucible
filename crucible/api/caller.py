from __future__ import annotations

import re

from fastapi import Request

from .. import CLIENT_NAME_HEADER

_CLIENT_NAME = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")


def client_agent(request: Request) -> str | None:
    stated = (request.headers.get(CLIENT_NAME_HEADER) or "").strip()
    if stated and _CLIENT_NAME.fullmatch(stated):
        return stated
    return (request.headers.get("user-agent") or "").strip()[:200] or None
