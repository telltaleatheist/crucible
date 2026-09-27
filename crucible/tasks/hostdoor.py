from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable

from ..backend import LLAMA_WINDOWS, Backend
from ..errors import ApiError
from ..platform.paths import HOST_DOOR_ENV
from .states import Task

ENGINE_TARGETS: tuple[str, ...] = ("wsl",)

HOST_DOOR_PATH = "/install"

HOST_DOOR_RESTART_PATH = "/restart"

ENGINE_RESTART_NEEDS_ORCHESTRATOR = "engine_restart_needs_orchestrator"

HOST_UNREACHABLE = "host_unreachable"
HOST_INSTALL_FAILED = "host_install_failed"

HOST_DOOR_CONNECT_SECONDS = 30.0

MOVE_REFUSED_BY_HOST = "engine_move_needs_host"

TERMINAL_EVENTS = frozenset({"done", "failed", "cancelled"})

Emit = Callable[[str, dict[str, Any]], None]


def host_refusal_code(body: str) -> str:
    try:
        payload = json.loads(body)
        code = payload["error"]["code"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return MOVE_REFUSED_BY_HOST
    return str(code) if isinstance(code, str) and code else MOVE_REFUSED_BY_HOST


def door_from_environment() -> str:
    return os.environ.get(HOST_DOOR_ENV, "").strip()


def door_for_move(backend: Backend, target: str) -> str:
    if target not in ENGINE_TARGETS:
        raise ApiError(
            400,
            "engine_target_unknown",
            f"{target!r} is not an engine this build moves to; the targets are "
            f"{list(ENGINE_TARGETS)}. Moving BACK to Windows is an explicit "
            "operator act (PHASE15-HOST.md section 6) and is refused rather "
            "than half-done",
            {"target": target, "targets": list(ENGINE_TARGETS)},
        )
    if backend.kind != LLAMA_WINDOWS:
        raise ApiError(
            409,
            "engine_move_not_here",
            f"this server runs the {backend.kind} backend on "
            f"{backend.platform}, and the engine move is a Windows machine "
            "swapping llama.cpp for the WSL2 guest. There is nothing here to "
            "move from",
            {"backend": backend.kind, "platform": backend.platform},
        )
    door = door_from_environment()
    if door == "":
        raise ApiError(
            409,
            MOVE_REFUSED_BY_HOST,
            "this server was not started by `crucible orchestrator`, so there is "
            f"nothing to hand the move to (${HOST_DOOR_ENV} is not set). Only "
            "the host can run wsl.exe, prompt for administrator and survive "
            "the reboot the move may need — a server doing it itself would "
            "stop halfway through and take its own event stream with it. "
            "Start the host and press it again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def door_for_restart() -> str:
    door = door_from_environment()
    if door == "":
        raise ApiError(
            409,
            ENGINE_RESTART_NEEDS_ORCHESTRATOR,
            "this server was not started by an orchestrator "
            f"(${HOST_DOOR_ENV} is not set), so there is nothing here that "
            "can restart it. Only the orchestrator can run wsl.exe, name the "
            "guest's unit or respawn a child — a server restarting itself "
            "would take its own event stream with it and leave nobody to say "
            "whether it came back. Start `crucible orchestrator` and press it "
            "again from the page",
            {"env": HOST_DOOR_ENV},
        )
    return door.rstrip("/")


def relay_lines(task: Task, lines: Iterable[bytes], emit: Emit) -> str | None:
    terminal: str | None = None
    for raw in lines:
        task.raise_if_cancelled()
        line = raw.decode("utf-8", "replace").strip()
        if line == "":
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            emit("progress", {"line": line})
            continue
        name = str(event.get("event") or "progress")
        data = event.get("data")
        emit(name, data if isinstance(data, dict) else {})
        if name in TERMINAL_EVENTS:
            terminal = name
    return terminal


def relay(
    task: Task,
    door: str,
    path: str,
    body_fields: dict[str, Any],
    *,
    token: str,
    emit: Emit,
) -> str | None:
    request = urllib.request.Request(
        f"{door}{path}",
        data=json.dumps(body_fields).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=HOST_DOOR_CONNECT_SECONDS) as stream:
            return relay_lines(task, stream, emit)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ApiError(
            502 if exc.code >= 500 else 409,
            host_refusal_code(detail),
            f"the host refused the move with HTTP {exc.code}: {detail}",
            {"door": door, "host_status": exc.code},
        ) from None
    except (urllib.error.URLError, OSError) as exc:
        raise ApiError(
            502,
            HOST_UNREACHABLE,
            f"the orchestrator's door at {door}{path} did not answer: "
            f"{type(exc).__name__}: {exc}. A host started this server "
            f"(${HOST_DOOR_ENV} is set) and its door is not answering "
            "now. Start it from the Startup item, or run `crucible orchestrator` "
            "from the host runtime, and press it again",
            {"door": door},
        ) from None


def silent_move(door: str) -> ApiError:
    return ApiError(
        502,
        HOST_INSTALL_FAILED,
        f"the host's door at {door}{HOST_DOOR_PATH} closed its stream "
        "without saying whether the move finished. Nothing here can "
        "tell a completed install from an abandoned one, so it is "
        "reported as a failure; the host's log says what happened",
        {"door": door},
    )


def silent_restart(door: str) -> ApiError:
    return ApiError(
        502,
        HOST_INSTALL_FAILED,
        f"the orchestrator's door at {door}{HOST_DOOR_RESTART_PATH} "
        "closed its stream without saying whether the engine came "
        "back. That is the EXPECTED shape when the engine being "
        "restarted is the one relaying: read `GET /v1/info` for the "
        "answer, which is the only place it is reliably true",
        {"door": door},
    )
