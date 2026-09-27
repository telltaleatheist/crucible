from __future__ import annotations

import platform as platform_module
import sys
from typing import Any, Callable

from .. import API_VERSION, VERSION
from .. import peer as peer_module
from .state import Owner

ORCHESTRATOR_GPU = {"vendor": "none", "name": "", "vram_bytes": 0}

LOCAL_LIFECYCLE_VERSION = 1

OWNER_ON_THE_WIRE = {
    Owner.WSL_UNIT: peer_module.OWNER_WSL_UNIT,
    Owner.HOST_CHILD: peer_module.OWNER_CHILD,
    Owner.FOUND: peer_module.OWNER_FOUND,
}


def _read_engine(url: str, token: str, log: Callable[[str], None]) -> dict[str, Any] | None:
    try:
        return peer_module.read_info(url, token, api_version=API_VERSION)
    except peer_module.PeerCallFailed as exc:
        log(f"info: the engine did not answer: {exc.code}")
        return None


def _engine_row(owner: Owner, url: str, read: dict[str, Any] | None) -> dict[str, Any]:
    server = read.get("server") if read is not None else None
    host = read.get("host") if read is not None else None
    return {
        "name": server.get("name") if isinstance(server, dict) else None,
        "url": url,
        "backend": host.get("backend") if isinstance(host, dict) else None,
        "owner": OWNER_ON_THE_WIRE[owner],
    }


def _capabilities(read: dict[str, Any] | None) -> list[Any]:
    rows = read.get("capabilities") if read is not None else None
    return rows if isinstance(rows, list) else []


def controller_info(
    name: str,
    owner: Owner,
    *,
    engine_url: str,
    token: Callable[[], str | None],
    log: Callable[[str], None],
) -> dict[str, Any]:
    engine: dict[str, Any] | None = None
    capabilities: list[Any] = []
    if owner in OWNER_ON_THE_WIRE:
        bearer = token()
        read = None if bearer is None else _read_engine(engine_url, bearer, log)
        engine = _engine_row(owner, engine_url, read)
        capabilities = _capabilities(read)
    return {
        "server": {"name": name, "version": VERSION, "api_version": API_VERSION},
        "host": {
            "platform": sys.platform,
            "arch": platform_module.machine(),
            "backend": peer_module.BACKEND_ORCHESTRATOR,
            "gpu": dict(ORCHESTRATOR_GPU),
        },
        "role": peer_module.ROLE_ORCHESTRATOR,
        "local_lifecycle_version": LOCAL_LIFECYCLE_VERSION,
        "job_types": [],
        "engine": engine,
        "capabilities": capabilities,
    }
