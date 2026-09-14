"""The install door — `POST /install` on 127.0.0.1:7101. PHASE15-HOST.md 4.3.

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
from typing import Callable

from .errors import HostError
from .installer import ENGINE_TARGET_WSL, Event
from .log import HostLog
from .paths import DOOR_HOST, DOOR_PORT

#: The one route. Anything else is 404 with a named body, so a caller that
#: guessed a path is told what the door is rather than nothing.
INSTALL_PATH = "/install"

#: What the body may contain. 4.7: the reverse move is not in this phase.
TARGETS = (ENGINE_TARGET_WSL,)

CONTENT_TYPE = "application/x-ndjson"

#: A body bigger than this is refused before it is read: the request is
#: `{"target": "wsl"}` and nothing on this door has a reason to be larger.
MAX_BODY_BYTES = 4096

#: Runs the sequence, emitting events. Injected so the test drives a fake one.
Sequence_ = Callable[[Callable[[Event], None]], None]


class InstallDoor:
    """The door's state: who may call it, and whether one is already running."""

    def __init__(
        self,
        log: HostLog,
        run_sequence: Sequence_,
        *,
        token: Callable[[], str | None],
    ) -> None:
        self._log = log
        self._run_sequence = run_sequence
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


def make_handler(door: InstallDoor) -> type[BaseHTTPRequestHandler]:
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

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != INSTALL_PATH:
                self._refuse(
                    404,
                    "not_found",
                    f"this host serves {INSTALL_PATH} and nothing else "
                    f"(PHASE15-HOST.md 4.3); {self.path} is not a door.",
                )
                return
            try:
                ok = door.authorised(self.headers.get("Authorization"))
            except HostError as exc:
                self._refuse(503, exc.code, exc.message)
                return
            if not ok:
                self._refuse(
                    401,
                    "host_unauthorized",
                    "the install door takes the engine's bearer token. An app "
                    "reads it from the pairing file (3.6); the server relaying a "
                    "task already holds it.",
                )
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
            self._stream()

        def _stream(self) -> None:
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
                door.run(emit)
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


def serve(door: InstallDoor, *, host: str = DOOR_HOST, port: int = DOOR_PORT) -> ThreadingHTTPServer:
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
