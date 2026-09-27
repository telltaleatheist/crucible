from __future__ import annotations

import hmac
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from .. import API_VERSION
from ..peer import ROLE_ORCHESTRATOR
from .errors import HostError
from .installer import ENGINE_TARGET_WSL, Event
from .log import HostLog
from ..platform.paths import DOOR_HOST, DOOR_PORT

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

    @property
    def name(self) -> str:
        return self._orchestrator.name


def make_handler(door: OrchestratorDoor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            door._log.write(f"door: {fmt % args}")

        def _refuse(self, status: int, code: str, message: str) -> None:
            body = json.dumps({"error": {"code": code, "message": message}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _answer(self, body: dict[str, object]) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/local/status":
                if self._authorised():
                    self._answer(door._orchestrator.local_status())
                return
            if path == INSTALL_EVENTS_PATH:
                self._watch_install()
                return
            if path == INSTALL_PATH:
                self._install_status()
                return
            if path == PING_PATH:
                self._answer(
                    {
                        "crucible": True,
                        "name": door.name,
                        "api_version": API_VERSION,
                        "role": ROLE_ORCHESTRATOR,
                    }
                )
                return
            if path != INFO_PATH:
                self._refuse(404, "not_found", self._what_this_door_is())
                return
            if not self._authorised():
                return
            self._answer(door.info())

        def _install_status(self) -> None:
            if not self._authorised():
                return
            try:
                recorded = door._orchestrator.install_outcome()
            except HostError as exc:
                self._refuse(503, exc.code, exc.message)
                return
            self._answer(
                {
                    "running": door.running,
                    "outcome": recorded,
                    "presence": door._orchestrator.presence(),
                }
            )

        def _watch_install(self) -> None:
            if not self._authorised():
                return
            if not door.running and not door.has_events():
                self._refuse(
                    404,
                    "no_install_running",
                    "no engine move is running on this machine and none has run "
                    f"since this orchestrator started. {INSTALL_PATH} says what "
                    "the last one ENDED as; a POST to it starts one.",
                )
                return
            backlog, watcher = door.attach()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                for envelope in backlog:
                    self._send_event(envelope)
                while True:
                    try:
                        envelope = watcher.get(timeout=WATCH_POLL_SECONDS)
                    except queue.Empty:
                        if not door.running:
                            return
                        continue
                    if envelope is _END:
                        return
                    self._send_event(envelope)
            except OSError:
                return
            finally:
                door.detach(watcher)

        def _send_event(self, envelope: dict[str, object]) -> None:
            self.wfile.write(json.dumps(envelope).encode("utf-8") + b"\n")
            self.wfile.flush()

        def _what_this_door_is(self) -> str:
            return (
                f"this orchestrator serves {INSTALL_PATH}, {INSTALL_EVENTS_PATH}, "
                f"{RESTART_PATH}, "
                f"{QUIT_PATH}, {INFO_PATH} and {PING_PATH}, and nothing else "
                f"(PHASE17-ORCHESTRATOR.md 3.2); {self.path} is not a door. An "
                f"app wanting anything else reads {INFO_PATH}'s `engine.url` "
                "and goes there."
            )

        def _authorised(self) -> bool:
            try:
                ok = door.authorised(self.headers.get("Authorization"))
            except HostError as exc:
                self._refuse(503, exc.code, exc.message)
                return False
            if not ok:
                self._refuse(
                    401,
                    "host_unauthorized",
                    "this door takes the ENGINE's bearer token. An app reads "
                    "it from the pairing file (3.6); the server relaying a "
                    "task already holds it.",
                )
                return False
            return True

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/local/start", "/local/stop"):
                if not self._authorised():
                    return
                if not door.claim():
                    self._refuse(409, "host_install_running", "An engine operation is already running")
                    return
                try:
                    operation = door._orchestrator.local_start if path.endswith("/start") else door._orchestrator.local_stop
                    self._answer(operation())
                except HostError as exc:
                    self._refuse(409, exc.code, exc.message)
                finally:
                    door.release()
                return
            if path == RESTART_PATH:
                self._restart()
                return
            if path == QUIT_PATH:
                from ..protocol import HANDOVER_HEADER

                self._quit(handover=self.headers.get(HANDOVER_HEADER) == "1")
                return
            if path != INSTALL_PATH:
                self._refuse(404, "not_found", self._what_this_door_is())
                return
            if not self._authorised():
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._refuse(
                    413,
                    "engine_target_unknown",
                    f"the body of {INSTALL_PATH} is {{\"target\": \"wsl\"}} and "
                    f"nothing larger; {length} bytes is not that.",
                )
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._refuse(400, "engine_target_unknown", f"the body is not JSON: {exc}")
                return
            if not isinstance(request, dict):
                self._refuse(
                    400,
                    "engine_target_unknown",
                    "the body of this door is an object; a bare "
                    f"{type(request).__name__} says nothing about what to move.",
                )
                return
            target = request.get("target")
            if target not in TARGETS:
                self._refuse(
                    400,
                    "engine_target_unknown",
                    f"target {target!r} is not one this build moves to; the targets "
                    f"are {list(TARGETS)}. Moving BACK to Windows is not in this "
                    "phase (4.7) and is refused rather than half-done.",
                )
                return
            unknown = sorted(set(request) - INSTALL_FIELDS)
            if unknown:
                self._refuse(
                    400,
                    "engine_target_unknown",
                    f"the body of {INSTALL_PATH} carries {unknown}, which this "
                    f"door does not take; its fields are {sorted(INSTALL_FIELDS)}.",
                )
                return
            if not door.claim():
                self._refuse(
                    409,
                    "host_install_running",
                    "an engine move is already running on this machine. There is "
                    f"one install on a machine; attach to {INSTALL_EVENTS_PATH} "
                    "and watch the one in flight rather than starting a second. "
                    "On a fresh install the runner is usually this machine's own "
                    "tray, which starts the move at every start (PHASE19 2.3).",
                )
                return
            self._install()

        def _install(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            dead = False

            def sink(envelope: dict[str, object]) -> None:
                nonlocal dead
                if dead:
                    return
                try:
                    self._send_event(envelope)
                except OSError:
                    dead = True
                    door._log.write("door: the install's caller hung up; the move continues")

            try:
                door.run_recorded(sink)
            except HostError as exc:
                door._log.write(f"door: install failed: {exc.code}: {exc.message}")
            except Exception as exc:
                door._log.write(f"door: install crashed: {type(exc).__name__}: {exc}")
            finally:
                door.release()

        def _restart(self) -> None:
            if not self._authorised():
                return
            try:
                door.check_restartable()
            except HostError as exc:
                self._refuse(409, exc.code, exc.message)
                return
            if not door.claim():
                self._refuse(
                    409,
                    "host_install_running",
                    "an engine move is already running on this machine, and a "
                    "restart in the middle of one would restart a server the "
                    "move is in the act of replacing. Watch the move.",
                )
                return
            self._stream(door.restart)

        def _quit(self, *, handover: bool = False) -> None:
            if not self._authorised():
                return
            door._log.write(
                f"door: POST {QUIT_PATH} — running the menu's Quit"
                + (" for an upgrade, handing the distro hold over" if handover else "")
            )
            self._answer({"quit": True, "name": door.name})
            try:
                self.wfile.flush()
            except OSError as exc:
                door._log.write(f"door: the quit answer did not land ({exc}); stopping anyway")
            door.quit(handover=handover)

        def _stream(self, sequence: Callable[[Callable[[Event], None]], None]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            index = 0
            terminal = False

            def emit(event: Event) -> None:
                nonlocal index, terminal
                index += 1
                terminal = terminal or event.event in ("done", "failed")
                line = json.dumps(
                    {"id": index, "event": event.event, "data": event.data}
                )
                self.wfile.write(line.encode("utf-8") + b"\n")
                self.wfile.flush()

            try:
                sequence(emit)
            except HostError as exc:
                door._log.write(f"door: install failed: {exc.code}: {exc.message}")
                if not terminal:
                    try:
                        emit(Event("failed", {"code": exc.code, "message": exc.message}))
                    except OSError:
                        pass
            except Exception as exc:
                door._log.write(f"door: install crashed: {type(exc).__name__}: {exc}")
                try:
                    emit(
                        Event(
                            "failed",
                            {
                                "code": "task_failed",
                                "message": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    )
                except OSError:
                    pass
            finally:
                door.release()

    return Handler


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
