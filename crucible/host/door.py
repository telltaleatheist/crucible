"""The orchestrator's door on 127.0.0.1:7101. PHASE15-HOST.md 4.3, PHASE17 3.2/4.2.

FIVE ROUTES, AND THEY ARE THE WHOLE OF WHAT AN ORCHESTRATOR SERVES
-------------------------------------------------------------------
    POST /install    the engine move (PHASE15 4.7)              — unchanged
    POST /restart    restart this orchestrator's engine (4.2)   — new
    POST /quit       stop THIS orchestrator (4.4)               — new
    GET  /v1/info    who this process is, and its engine (3.2)  — new
    GET  /v1/ping    "is this a Crucible"                       — new

This is NOT a second API surface. An orchestrator serves ZERO job types and
never carries a byte of anybody's data (PHASE17 section 0: control is
Windows's, data is the card's). An app that reaches this door reads
`engine.url` out of `/v1/info` and goes there for everything else, once.

TWO CALLERS AND NO THIRD
-------------------------
1. **The Windows server**, when the operator page posts
   `POST /v1/tasks {"type": "engine", "target": "wsl"}` (4.7). The server
   relays these events under its own task id, which is why they are shaped
   exactly like `crucible/tasks.py`'s and not merely similarly: a relay that
   reshapes is a second owner of the shape.
2. **`@crucible/bootstrap`'s `install()`**, directly, on a machine that has no
   server yet — the very first install, before there is a page to open.

WHY A LOOPBACK PORT AND NOT A PIPE
-----------------------------------
A named pipe would need a second transport to keep working, with a second
authentication story, for one route. Loopback plus a bearer is what every
other Crucible door already is, and the bearer is the ENGINE's token, so a
caller that can reach the engine can reach this and nothing else can.

WHY `http.server` AND NOT FastAPI
----------------------------------
This runs inside the tray process, which must start in well under a second at
login and must not hold a card, an event loop or a uvicorn. `http.server` is
in the standard library, it is one thread per request, and this route is
called at most a handful of times in a machine's life. Importing the server
stack into the tray would also make the tray fail to start for reasons that
belong to the server.
"""

from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from .. import API_VERSION
from ..peer import ROLE_ORCHESTRATOR
from .errors import HostError
from .installer import ENGINE_TARGET_WSL, Event
from .log import HostLog
from .paths import DOOR_HOST, DOOR_PORT

#: The routes. Anything else is 404 with a named body, so a caller that
#: guessed a path is told what the door is rather than nothing.
INSTALL_PATH = "/install"
RESTART_PATH = "/restart"
QUIT_PATH = "/quit"
INFO_PATH = "/v1/info"
PING_PATH = "/v1/ping"


class OrchestratorPort(Protocol):
    """What the door needs from the tray, and nothing else.

    A protocol rather than an import of `app.Host`, for `menu.py`'s reason one
    level out: the door is a transport, every DECISION it serves is made
    elsewhere, and a test must be able to drive the transport without building
    a tray. `app.py` is the one implementation.
    """

    @property
    def name(self) -> str:
        """This orchestrator's name, for `/v1/ping`. e.g. `crucible-orchestrator@owens-pc`."""

    def info(self) -> dict[str, Any]:
        """PHASE17 3.2's document, with the engine's capabilities READ THROUGH."""

    def check_restartable(self) -> None:
        """Raises `HostError('engine_not_ours')` when the engine is a `found` one.

        Asked BEFORE the stream opens, because 4.1a's refusal is a refusal of
        the REQUEST and not a failure of a sequence — the caller reads it from
        a status code, not from the last line of an ndjson body.
        """

    def restart_engine(self, emit: Callable[[Event], None]) -> None:
        """PHASE17 4.2's sequence, by the owner-appropriate means."""

    def quit(self) -> None:
        """PHASE17 4.4's stop — THE SAME ONE the tray menu's Quit runs.

        Release the claim (2.2), let the held distro go (PHASE15 7b.4c), take
        a child engine down when this process is the one that started it
        (`owner == child`), and end the process. `app.Host.quit` is the one
        implementation; the menu item and this door are its two callers.
        """

#: What the body may contain. 4.7: the reverse move is not in this phase.
TARGETS = (ENGINE_TARGET_WSL,)

CONTENT_TYPE = "application/x-ndjson"

#: A body bigger than this is refused before it is read: the request is
#: `{"target": "wsl"}` and nothing on this door has a reason to be larger.
MAX_BODY_BYTES = 4096

#: Runs the sequence, emitting events. Injected so the test drives a fake one.
Sequence_ = Callable[[Callable[[Event], None]], None]


class OrchestratorDoor:
    """The door's state: who may call it, and whether one is already running.

    Renamed from `InstallDoor` by PHASE17, because it no longer serves only
    an install. The class named after one of its four routes would be the same
    kind of stale label the phase exists to remove.
    """

    def __init__(
        self,
        log: HostLog,
        run_sequence: Sequence_,
        *,
        token: Callable[[], str | None],
        orchestrator: OrchestratorPort,
    ) -> None:
        self._log = log
        self._run_sequence = run_sequence
        self._orchestrator = orchestrator
        # A CALLABLE and not a string: the token changes under this door during
        # the very sequence it runs (the Windows config's becomes the guest's,
        # 3.5), and a door holding a copy would start refusing its own caller
        # halfway through.
        self._token = token
        self._lock = threading.Lock()
        self._running = False

    def authorised(self, header: str | None) -> bool:
        """Constant-time, and `host_no_token` is NOT an authorisation failure."""
        expected = self._token()
        if expected is None:
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
        """One install on a machine. The second caller waits, it does not queue."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            return True

    def release(self) -> None:
        with self._lock:
            self._running = False

    def run(self, emit: Callable[[Event], None]) -> None:
        self._run_sequence(emit)

    def restart(self, emit: Callable[[Event], None]) -> None:
        """PHASE17 4.2's sequence — the orchestrator's, by the owner's means."""
        self._orchestrator.restart_engine(emit)

    def check_restartable(self) -> None:
        self._orchestrator.check_restartable()

    def quit(self) -> None:
        """PHASE17 4.4 — the menu's Quit, reached by the transport instead."""
        self._orchestrator.quit()

    def info(self) -> dict[str, Any]:
        return self._orchestrator.info()

    @property
    def name(self) -> str:
        return self._orchestrator.name


def make_handler(door: OrchestratorDoor) -> type[BaseHTTPRequestHandler]:
    """The handler class, bound to one door. A closure, so nothing is global."""

    class Handler(BaseHTTPRequestHandler):
        # The default logs every request to stderr, which in a `pythonw`
        # process goes nowhere at all. The host's log is the one place.
        def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
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

        def do_GET(self) -> None:  # noqa: N802
            """`/v1/ping` and `/v1/info`. PHASE17 3.2.

            **`/v1/ping` is UNAUTHENTICATED**, exactly as the engine's is and
            for the same reason: it is what lets a client tell "wrong token"
            from "not a Crucible". `/v1/info` takes the bearer, because it
            names the engine's address and reads the engine's capabilities
            through, and neither is a thing to hand an anonymous caller.
            """
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
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

        def _what_this_door_is(self) -> str:
            return (
                f"this orchestrator serves {INSTALL_PATH}, {RESTART_PATH}, "
                f"{QUIT_PATH}, {INFO_PATH} and {PING_PATH}, and nothing else "
                f"(PHASE17-ORCHESTRATOR.md 3.2); {self.path} is not a door. An "
                f"app wanting anything else reads {INFO_PATH}'s `engine.url` "
                "and goes there."
            )

        def _authorised(self) -> bool:
            """Check the bearer, having already written the refusal if not."""
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

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == RESTART_PATH:
                self._restart()
                return
            if path == QUIT_PATH:
                self._quit()
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
            # EVERY OTHER FIELD IS ACCEPTED AND NOT VALIDATED HERE.
            # `@crucible/bootstrap` sends `release`, `job_types`, and
            # optionally `home` and `bind`, because on the first install of a
            # machine there is no server and therefore no coordinate record to
            # read the job types out of (4.7's "the coordinate records say
            # which" is about the PAGE-driven caller). Refusing an unknown
            # field would make the older of the two clients fail against the
            # newer door for carrying something it was told to carry;
            # `engine_target_unknown` is reserved for a `target` that is not
            # `wsl`, which is the one field whose wrong value would DO the
            # wrong thing.
            if not door.claim():
                self._refuse(
                    409,
                    "host_install_running",
                    "an engine move is already running on this machine. There is "
                    "one install on a machine; watch the one in flight rather than "
                    "starting a second.",
                )
                return
            self._stream(door.run)

        def _restart(self) -> None:
            """`POST /restart` — PHASE17 4.2.

            **`engine_not_ours` is refused BEFORE the stream opens**, because
            it is a refusal of the request and not a failure of a sequence:
            4.1a's rule is that an engine this orchestrator did not start is
            watched and never acted on, and the caller must be able to read
            that from a status code rather than from the last line of an
            ndjson body. Everything that goes wrong AFTER the restart begins
            is an event, like the move's.
            """
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

        def _quit(self) -> None:
            """`POST /quit` — PHASE17 4.4. The only non-interactive stop.

            **IT ANSWERS BEFORE IT STOPS, AND THE ANSWER IS THE LAST EVENT.**
            4.3's rule holds — a door served BY the orchestrator cannot
            survive the act of stopping the orchestrator — so this is not a
            task and there is no stream: the 200 and its body are written and
            flushed onto the socket FIRST, and only then does the shutdown
            run. A caller gets a response, never a dropped connection, and
            there is nothing left afterwards for it to re-read.

            **NOT REFUSABLE WHILE AN INSTALL IS RUNNING**, unlike `/restart`.
            The measurement in 4.4 is that `taskkill` without `/F` is a no-op
            against a console-less `pythonw` and `/F` runs none of `quit()`,
            so this route is the ONLY orderly stop this process has. A stop
            that a wedged sequence could refuse would send the operator
            straight back to `/F`, which is the thing 4.4 exists to remove.

            The bearer is the one every other route on this door takes — the
            ENGINE's token — and the refusals are `host_unauthorized` (401)
            and `host_no_token` (503), by the same `_authorised()`.
            """
            if not self._authorised():
                return
            door._log.write(f"door: POST {QUIT_PATH} — running the menu's Quit")
            self._answer({"quit": True, "name": door.name})
            try:
                self.wfile.flush()
            except OSError as exc:
                # The caller hung up between the request and the answer. The
                # stop still runs: it was asked for, and an orchestrator that
                # stayed up because nobody was listening to its goodbye would
                # be the no-op 4.4 was written about.
                door._log.write(f"door: the quit answer did not land ({exc}); stopping anyway")
            door.quit()

        def _stream(self, sequence: Callable[[Callable[[Event], None]], None]) -> None:
            """The ndjson. Flushed per line: a progress bar that arrives at the
            end is not a progress bar."""
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            index = 0

            def emit(event: Event) -> None:
                nonlocal index
                index += 1
                line = json.dumps(
                    {"id": index, "event": event.event, "data": event.data}
                )
                self.wfile.write(line.encode("utf-8") + b"\n")
                self.wfile.flush()

            try:
                sequence(emit)
            except HostError as exc:
                # The sequence has already emitted its own `failed` event with
                # this code; this is the case where something threw before it
                # could. A stream that stops without a terminal event is what
                # the client reports as truncated, so one is always sent.
                door._log.write(f"door: install failed: {exc.code}: {exc.message}")
            except Exception as exc:  # noqa: BLE001 - the stream must terminate
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
    """Start the door on a daemon thread and return the server.

    LOOPBACK, always: `DOOR_HOST` is `127.0.0.1` and this function does not
    take a wildcard. A door that installs software and takes a bearer must not
    be one an argument can put on the LAN.
    """
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
