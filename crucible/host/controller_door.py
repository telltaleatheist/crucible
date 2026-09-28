from __future__ import annotations

import hmac
import json
import queue
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from .. import API_VERSION
from ..peer import ROLE_ORCHESTRATOR
from ..platform.errors import HostError
from ..platform.paths import DOOR_HOST, DOOR_PORT
from ..protocol import HANDOVER_HEADER
from .installer import ENGINE_TARGET_WSL, Event
from .log import HostLog

INSTALL_PATH = "/install"
INSTALL_EVENTS_PATH = "/install/events"
RESTART_PATH = "/restart"
QUIT_PATH = "/quit"
INFO_PATH = "/v1/info"
PING_PATH = "/v1/ping"


class OrchestratorPort(Protocol):
    @property
    def name(self) -> str:
        ...

    def info(self) -> dict[str, Any]:
        ...

    def check_restartable(self) -> None:
        ...

    def restart_engine(self, emit: Callable[[Event], None]) -> None:
        ...

    def local_status(self) -> dict[str, object]:
        ...

    def local_start(self) -> dict[str, object]:
        ...

    def local_stop(self) -> dict[str, object]:
        ...

    def quit(self, *, handover: bool = False) -> None:
        ...

    def presence(self) -> dict[str, object]:
        ...

    def install_outcome(self) -> dict[str, object] | None:
        ...

TARGETS = (ENGINE_TARGET_WSL,)

INSTALL_FIELDS = frozenset({"target", "release", "job_types", "home", "bind"})

CONTENT_TYPE = "application/x-ndjson"

MAX_BODY_BYTES = 4096

Sequence_ = Callable[[Callable[[Event], None]], None]

MAX_RING_EVENTS = 200

WATCH_POLL_SECONDS = 1.0

_END = None


class OrchestratorDoor:
    def __init__(
        self,
        log: HostLog,
        run_sequence: Sequence_,
        *,
        token: Callable[[], str | None],
        orchestrator: OrchestratorPort,
        token_detail: Callable[[], str] | None = None,
    ) -> None:
        self._log = log
        self._run_sequence = run_sequence
        self._orchestrator = orchestrator
        self._token = token
        self._token_detail = token_detail
        self._lock = threading.Lock()
        self._running = False
        self._events = threading.Lock()
        self._ring: list[dict[str, object]] = []
        self._emitted = 0
        self._watchers: list["queue.Queue[dict[str, object] | None]"] = []
        self._move_open = False

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def authorised(self, header: str | None) -> bool:
        expected = self._token()
        if expected is None:
            if self._token_detail is not None:
                raise HostError("host_no_token", self._token_detail())
            raise HostError(
                "host_no_token",
                "this host has no config yet, so the install door has no token to "
                "check against. It gets one the first time the Windows server is "
                "initialised, which is seconds after the host first starts.",
            )
        if header is None or not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[len("Bearer ") :].strip(), expected)

    def claim(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            return True

    def release(self) -> None:
        with self._lock:
            self._running = False


    def _begin_move(self) -> None:
        with self._events:
            self._ring = []
            self._emitted = 0
            self._move_open = True

    def _record(self, event: Event) -> dict[str, object]:
        with self._events:
            self._emitted += 1
            envelope: dict[str, object] = {
                "id": self._emitted,
                "event": event.event,
                "data": event.data,
            }
            self._ring.append(envelope)
            if len(self._ring) > MAX_RING_EVENTS:
                del self._ring[: len(self._ring) - MAX_RING_EVENTS]
            watchers = list(self._watchers)
        for watcher in watchers:
            watcher.put(envelope)
        return envelope

    def attach(self) -> tuple[list[dict[str, object]], "queue.Queue[dict[str, object] | None]"]:
        watcher: "queue.Queue[dict[str, object] | None]" = queue.Queue()
        with self._events:
            backlog = list(self._ring)
            if self._move_open:
                self._watchers.append(watcher)
            else:
                watcher.put(_END)
        return backlog, watcher

    def detach(self, watcher: "queue.Queue[dict[str, object] | None]") -> None:
        with self._events:
            if watcher in self._watchers:
                self._watchers.remove(watcher)

    def has_events(self) -> bool:
        with self._events:
            return len(self._ring) > 0

    def run_recorded(self, sink: Callable[[dict[str, object]], None] | None = None) -> None:
        self._begin_move()
        terminal = False

        def emit(event: Event) -> None:
            nonlocal terminal
            terminal = terminal or event.event in ("done", "failed")
            envelope = self._record(event)
            if sink is not None:
                sink(envelope)

        try:
            self._run_sequence(emit)
        except BaseException as exc:
            if not terminal:
                code = exc.code if isinstance(exc, HostError) else "task_failed"
                message = (
                    exc.message
                    if isinstance(exc, HostError)
                    else f"{type(exc).__name__}: {exc}"
                )
                emit(Event("failed", {"code": code, "message": message}))
            raise
        finally:
            with self._events:
                self._move_open = False
                watchers, self._watchers = list(self._watchers), []
            for watcher in watchers:
                watcher.put(_END)

    def run(self, emit: Callable[[Event], None]) -> None:
        self._run_sequence(emit)

    def restart(self, emit: Callable[[Event], None]) -> None:
        self._orchestrator.restart_engine(emit)

    def check_restartable(self) -> None:
        self._orchestrator.check_restartable()

    def quit(self, *, handover: bool = False) -> None:
        if handover:
            self._orchestrator.quit(handover=True)
        else:
            self._orchestrator.quit()

    def info(self) -> dict[str, Any]:
        return self._orchestrator.info()

    def local_status(self) -> dict[str, object]:
        return self._orchestrator.local_status()

    def local_start(self) -> dict[str, object]:
        return self._orchestrator.local_start()

    def local_stop(self) -> dict[str, object]:
        return self._orchestrator.local_stop()

    def install_outcome(self) -> dict[str, object] | None:
        return self._orchestrator.install_outcome()

    def presence(self) -> dict[str, object]:
        return self._orchestrator.presence()

    def note(self, line: str) -> None:
        self._log.write(line)

    @property
    def name(self) -> str:
        return self._orchestrator.name


class DoorRequest(BaseHTTPRequestHandler):
    door: OrchestratorDoor

    def log_message(self, fmt: str, *args: object) -> None:
        self.door.note(f"door: {fmt % args}")

    def route_path(self) -> str:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def refuse(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def answer(self, body: dict[str, object]) -> None:
        self._send_json(200, body)

    def _send_json(self, status: int, body: dict[str, object]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def open_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def send_event(self, envelope: dict[str, object]) -> None:
        self.wfile.write(json.dumps(envelope).encode("utf-8") + b"\n")
        self.wfile.flush()

    def authorised(self) -> bool:
        try:
            ok = self.door.authorised(self.headers.get("Authorization"))
        except HostError as exc:
            self.refuse(503, exc.code, exc.message)
            return False
        if not ok:
            self.refuse(
                401,
                "host_unauthorized",
                "this door takes the ENGINE's bearer token. An app reads "
                "it from the pairing file (3.6); the server relaying a "
                "task already holds it.",
            )
        return ok

    def not_a_door(self) -> None:
        self.refuse(
            404,
            "not_found",
            f"this orchestrator serves {INSTALL_PATH}, {INSTALL_EVENTS_PATH}, "
            f"{RESTART_PATH}, "
            f"{QUIT_PATH}, {INFO_PATH} and {PING_PATH}, and nothing else "
            f"(docs/internals/host-and-platform.md, \"The door\"); {self.path} is not a door. An "
            f"app wanting anything else reads {INFO_PATH}'s `engine.url` "
            "and goes there.",
        )

    def do_GET(self) -> None:
        self._dispatch(GET_ROUTES)

    def do_POST(self) -> None:
        self._dispatch(POST_ROUTES)

    def _dispatch(self, routes: dict[str, "Route"]) -> None:
        route = routes.get(self.route_path())
        if route is None:
            self.not_a_door()
            return
        route(self)


Route = Callable[[DoorRequest], None]


def _ping(request: DoorRequest) -> None:
    request.answer(
        {
            "crucible": True,
            "name": request.door.name,
            "api_version": API_VERSION,
            "role": ROLE_ORCHESTRATOR,
        }
    )


def _info(request: DoorRequest) -> None:
    if request.authorised():
        request.answer(request.door.info())


def _local_status(request: DoorRequest) -> None:
    if request.authorised():
        request.answer(request.door.local_status())


def _install_status(request: DoorRequest) -> None:
    if not request.authorised():
        return
    try:
        recorded = request.door.install_outcome()
    except HostError as exc:
        request.refuse(503, exc.code, exc.message)
        return
    request.answer({"running": request.door.running, "outcome": recorded, "presence": request.door.presence()})


def _follow(request: DoorRequest, watcher: "queue.Queue[dict[str, object] | None]") -> None:
    while True:
        try:
            envelope = watcher.get(timeout=WATCH_POLL_SECONDS)
        except queue.Empty:
            if not request.door.running:
                return
            continue
        if envelope is _END:
            return
        request.send_event(envelope)


def _watch_install(request: DoorRequest) -> None:
    door = request.door
    if not request.authorised():
        return
    if not door.running and not door.has_events():
        request.refuse(
            404,
            "no_install_running",
            "no engine move is running on this machine and none has run "
            f"since this orchestrator started. {INSTALL_PATH} says what "
            "the last one ENDED as; a POST to it starts one.",
        )
        return
    backlog, watcher = door.attach()
    request.open_stream()
    try:
        for envelope in backlog:
            request.send_event(envelope)
        _follow(request, watcher)
    except OSError:
        return
    finally:
        door.detach(watcher)


def _local_operation(operation: Callable[[OrchestratorDoor], dict[str, object]]) -> Route:
    def route(request: DoorRequest) -> None:
        if not request.authorised():
            return
        if not request.door.claim():
            request.refuse(409, "host_install_running", "An engine operation is already running")
            return
        try:
            request.answer(operation(request.door))
        except HostError as exc:
            request.refuse(409, exc.code, exc.message)
        finally:
            request.door.release()

    return route


@dataclass(frozen=True)
class BodyProblem:
    status: int
    message: str


def _install_body_problem(request: DoorRequest) -> BodyProblem | None:
    length = int(request.headers.get("Content-Length") or 0)
    if length > MAX_BODY_BYTES:
        return BodyProblem(
            413,
            f"the body of {INSTALL_PATH} is {{\"target\": \"wsl\"}} and "
            f"nothing larger; {length} bytes is not that.",
        )
    raw = request.rfile.read(length) if length else b"{}"
    try:
        body = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return BodyProblem(400, f"the body is not JSON: {exc}")
    return _install_fields_problem(body)


def _install_fields_problem(body: object) -> BodyProblem | None:
    if not isinstance(body, dict):
        return BodyProblem(
            400,
            "the body of this door is an object; a bare "
            f"{type(body).__name__} says nothing about what to move.",
        )
    target = body.get("target")
    if target not in TARGETS:
        return BodyProblem(
            400,
            f"target {target!r} is not one this build moves to; the targets "
            f"are {list(TARGETS)}. Moving BACK to Windows is not in this "
            "phase (4.7) and is refused rather than half-done.",
        )
    unknown = sorted(set(body) - INSTALL_FIELDS)
    if unknown:
        return BodyProblem(
            400,
            f"the body of {INSTALL_PATH} carries {unknown}, which this "
            f"door does not take; its fields are {sorted(INSTALL_FIELDS)}.",
        )
    return None


def _start_install(request: DoorRequest) -> None:
    if not request.authorised():
        return
    problem = _install_body_problem(request)
    if problem is not None:
        request.refuse(problem.status, "engine_target_unknown", problem.message)
        return
    if not request.door.claim():
        request.refuse(
            409,
            "host_install_running",
            "an engine move is already running on this machine. There is "
            f"one install on a machine; attach to {INSTALL_EVENTS_PATH} "
            "and watch the one in flight rather than starting a second. "
            "On a fresh install the runner is usually this machine's own "
            "tray, which starts the move at every start (docs/internals/host-and-platform.md, \"The Windows to WSL move\").",
        )
        return
    _run_install(request)


def _run_install(request: DoorRequest) -> None:
    door = request.door
    request.open_stream()
    dead = False

    def sink(envelope: dict[str, object]) -> None:
        nonlocal dead
        if dead:
            return
        try:
            request.send_event(envelope)
        except OSError:
            dead = True
            door.note("door: the install's caller hung up; the move continues")

    try:
        door.run_recorded(sink)
    except HostError as exc:
        door.note(f"door: install failed: {exc.code}: {exc.message}")
    except Exception as exc:
        door.note(f"door: install crashed: {type(exc).__name__}: {exc}")
    finally:
        door.release()


def _restart(request: DoorRequest) -> None:
    door = request.door
    if not request.authorised():
        return
    try:
        door.check_restartable()
    except HostError as exc:
        request.refuse(409, exc.code, exc.message)
        return
    if not door.claim():
        request.refuse(
            409,
            "host_install_running",
            "an engine move is already running on this machine, and a "
            "restart in the middle of one would restart a server the "
            "move is in the act of replacing. Watch the move.",
        )
        return
    _stream(request, door.restart)


def _quit(request: DoorRequest) -> None:
    door = request.door
    handover = request.headers.get(HANDOVER_HEADER) == "1"
    if not request.authorised():
        return
    door.note(
        f"door: POST {QUIT_PATH} — running the menu's Quit"
        + (" for an upgrade, handing the distro hold over" if handover else "")
    )
    request.answer({"quit": True, "name": door.name})
    try:
        request.wfile.flush()
    except OSError as exc:
        door.note(f"door: the quit answer did not land ({exc}); stopping anyway")
    door.quit(handover=handover)


def _failed_last_words(emit: Callable[[Event], None], code: str, message: str) -> None:
    try:
        emit(Event("failed", {"code": code, "message": message}))
    except OSError:
        pass


def _stream(request: DoorRequest, sequence: Callable[[Callable[[Event], None]], None]) -> None:
    door = request.door
    request.open_stream()
    index = 0
    terminal = False

    def emit(event: Event) -> None:
        nonlocal index, terminal
        index += 1
        terminal = terminal or event.event in ("done", "failed")
        request.send_event({"id": index, "event": event.event, "data": event.data})

    try:
        sequence(emit)
    except HostError as exc:
        door.note(f"door: install failed: {exc.code}: {exc.message}")
        if not terminal:
            _failed_last_words(emit, exc.code, exc.message)
    except Exception as exc:
        door.note(f"door: install crashed: {type(exc).__name__}: {exc}")
        _failed_last_words(emit, "task_failed", f"{type(exc).__name__}: {exc}")
    finally:
        door.release()


GET_ROUTES: dict[str, Route] = {
    "/local/status": _local_status,
    INSTALL_EVENTS_PATH: _watch_install,
    INSTALL_PATH: _install_status,
    PING_PATH: _ping,
    INFO_PATH: _info,
}

POST_ROUTES: dict[str, Route] = {
    "/local/start": _local_operation(lambda door: door.local_start()),
    "/local/stop": _local_operation(lambda door: door.local_stop()),
    RESTART_PATH: _restart,
    QUIT_PATH: _quit,
    INSTALL_PATH: _start_install,
}


def make_handler(door: OrchestratorDoor) -> type[BaseHTTPRequestHandler]:
    return type("DoorHandler", (DoorRequest,), {"door": door})


def serve(door: OrchestratorDoor, *, host: str = DOOR_HOST, port: int = DOOR_PORT) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HostError(
            "host_unauthorized",
            f"the install door binds loopback only; {host!r} is not loopback. "
            "It installs software on this machine and is not a thing to expose.",
        )
    server = ThreadingHTTPServer((host, port), make_handler(door))
    thread = threading.Thread(target=server.serve_forever, name="crucible-door", daemon=True)
    thread.start()
    return server
