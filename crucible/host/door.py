"""The orchestrator's door on 127.0.0.1:7101. PHASE15-HOST.md 4.3, PHASE17 3.2/4.2.

SEVEN ROUTES, AND THEY ARE THE WHOLE OF WHAT AN ORCHESTRATOR SERVES
--------------------------------------------------------------------
    POST /install         the engine move (PHASE15 4.7)
    GET  /install         running, outcome, presence (PHASE19 2.6)  — new
    GET  /install/events  attach to a move in flight (PHASE19 2.6)  — new
    POST /restart         restart this orchestrator's engine (4.2)
    POST /quit            stop THIS orchestrator (4.4)
    GET  /v1/info         who this process is, and its engine (3.2)
    GET  /v1/ping         "is this a Crucible"

**THE DOOR CAN BE WATCHED, NOT ONLY DRIVEN** (PHASE19 2.6). The tray now starts
the move itself at every start (2.3), so by the time an app asks, the thing it
wanted to start is usually already running — which is why `POST /install`'s 409
is a normal answer rather than an error, and why the two GETs exist: the client
reads the status, attaches to the stream from its current step, and never posts
a second walk over the same distro.

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
import queue
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
INSTALL_EVENTS_PATH = "/install/events"
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

    def local_status(self) -> dict[str, object]:
        """Report controller-observed engine state and explicit stop intent."""

    def local_start(self) -> dict[str, object]:
        """Start the managed engine and clear explicit stop intent."""

    def local_stop(self) -> dict[str, object]:
        """Stop the managed engine and preserve that intent across login."""

    def quit(self) -> None:
        """PHASE17 4.4's stop — THE SAME ONE the tray menu's Quit runs.

        Release the claim (2.2), let the held distro go (PHASE15 7b.4c), take
        a child engine down when this process is the one that started it
        (`owner == child`), and end the process. `app.Host.quit` is the one
        implementation; the menu item and this door are its two callers.
        """

    def presence(self) -> dict[str, object]:
        """The tray's presence, for `GET /install` (PHASE19 2.6).

        The four words `presence.Presence` already holds, and not a fifth
        composed here: an app reading this and the tray's own log must be
        reading the same measurement.
        """

    def install_outcome(self) -> dict[str, object] | None:
        """`wsl-outcome.json` as 2.2 shapes it, or None when there is none.

        Asked of the orchestrator rather than read here, because the file lives
        in CRUCIBLE_HOME and this door does not know where that is — the tray
        does, and `crucible/host/outcome.py` is the reader.
        """

#: What the body may contain. 4.7: the reverse move is not in this phase.
TARGETS = (ENGINE_TARGET_WSL,)

CONTENT_TYPE = "application/x-ndjson"

#: A body bigger than this is refused before it is read: the request is
#: `{"target": "wsl"}` and nothing on this door has a reason to be larger.
MAX_BODY_BYTES = 4096

#: Runs the sequence, emitting events. Injected so the test drives a fake one.
Sequence_ = Callable[[Callable[[Event], None]], None]

#: PHASE19 2.6: how many events a move keeps so that a LATE attacher sees the
#: step it joined at. The number is the plan's — "a ring of the last 200 events
#: is kept" — and it is a ring rather than the whole stream because
#: `install.sh` inside the guest prints thousands of pip lines and this is the
#: tray, which must not grow a transcript of one in memory.
MAX_RING_EVENTS = 200

#: How long a watcher waits on its queue before asking the door whether the
#: move is still running. Not a heartbeat — nothing is written on a timeout —
#: it is the interval at which a handler thread notices that the move it was
#: following is gone. One second, because that is a tray thread doing nothing.
WATCH_POLL_SECONDS = 1.0

#: The sentinel a watcher's queue gets when the move ends. `None` and not a
#: fabricated `done`: the real terminal event has already been sent through the
#: same queue, and a second one invented here would be an event no step emitted.
_END = None


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
        token_detail: Callable[[], str] | None = None,
    ) -> None:
        self._log = log
        self._run_sequence = run_sequence
        self._orchestrator = orchestrator
        # A CALLABLE and not a string: the token changes under this door during
        # the very sequence it runs (the Windows config's becomes the guest's,
        # 3.5), and a door holding a copy would start refusing its own caller
        # halfway through.
        self._token = token
        #: Why the token is missing, asked of the host rather than guessed
        #: at here. There is more than one way to have no bearer and the
        #: door cannot tell them apart; naming the wrong one sent a reader
        #: to the wrong file (2026-09-17).
        self._token_detail = token_detail
        self._lock = threading.Lock()
        self._running = False
        #: PHASE19 2.6's ring and its watchers. One lock over both, because an
        #: attacher that read the backlog and subscribed in two steps would
        #: miss every event that landed between them — the one defect a
        #: "replay then follow" stream has.
        self._events = threading.Lock()
        self._ring: list[dict[str, object]] = []
        self._emitted = 0
        self._watchers: list["queue.Queue[dict[str, object] | None]"] = []
        #: Whether MORE EVENTS CAN STILL ARRIVE. Not the same fact as the claim:
        #: the claim is released by the caller after `run_recorded` returns, and
        #: an attacher that subscribed in that gap would wait for an `_END`
        #: nobody is left to send. Set and cleared under `_events`, which is the
        #: lock `attach` decides under.
        self._move_open = False

    @property
    def running(self) -> bool:
        """Is a move in flight? The claim IS the answer; there is no second flag."""
        with self._lock:
            return self._running

    def authorised(self, header: str | None) -> bool:
        """Constant-time, and `host_no_token` is NOT an authorisation failure."""
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
        """One install on a machine. The second caller waits, it does not queue."""
        with self._lock:
            if self._running:
                return False
            self._running = True
            return True

    def release(self) -> None:
        with self._lock:
            self._running = False

    # -------------------------------------------- PHASE19 2.6 the watchers

    def _begin_move(self) -> None:
        """A new move starts a new ring. The last one's events are not this one's."""
        with self._events:
            self._ring = []
            self._emitted = 0
            self._move_open = True

    def _record(self, event: Event) -> dict[str, object]:
        """One event into the ring and out to every watcher. Returns the envelope."""
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
        """The ring as it stands, and a queue of everything after it.

        Both taken under ONE lock, so the join is seamless: an attacher gets
        every event exactly once, in order, whether it arrived before or after
        it asked.
        """
        watcher: "queue.Queue[dict[str, object] | None]" = queue.Queue()
        with self._events:
            backlog = list(self._ring)
            if self._move_open:
                self._watchers.append(watcher)
            else:
                # Nothing is running, so nothing more will arrive: the queue is
                # closed before it is handed over rather than left to wait for
                # an event that cannot come.
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
        """The move, with every event recorded and fanned out.

        THE ONE PATH BOTH CALLERS TAKE (PHASE19 2.3): the tray's own start-time
        decision and `POST /install` run this, so the ring an attacher reads is
        the same stream the poster is reading, and there is one place a move's
        events are numbered.

        It RAISES what the sequence raised, but only after a terminal event has
        been recorded — a watcher whose stream simply stopped cannot tell a
        failure from a socket that died.
        """
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
            """`GET /install` — PHASE19 2.6.

            Three facts and no fourth: whether a move is in flight, what the
            last one ENDED as (`wsl-outcome.json`, 2.2), and the tray's
            presence. An app that has just pressed Install reads this to know
            whether to attach; one that has been away reads it to know how the
            machine ended up.
            """
            if not self._authorised():
                return
            try:
                recorded = door._orchestrator.install_outcome()
            except HostError as exc:
                # A present-and-unreadable outcome is refused by name rather
                # than answered as `null`, which an app would read as "nothing
                # has happened here yet" (2.2).
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
            """`GET /install/events` — attach to the move, from where it is.

            The ring is replayed first, so a client that arrives twenty minutes
            into a guest install sees the step it joined at instead of silence
            until the next line. Then it follows, and the stream ends when the
            move does.

            **A machine with no move to watch is a 404 by name**, not an empty
            200: "there is nothing running and nothing has run" is a fact an app
            acts on (it posts one), and an empty success is that fact spelled as
            an absence.
            """
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
                        # The `_END` can only be lost if the move's thread died
                        # between clearing `_move_open` and posting it. Asking
                        # again is how this handler stops being a thread waiting
                        # for a sentinel nobody will send.
                        if not door.running:
                            return
                        continue
                    if envelope is _END:
                        return
                    self._send_event(envelope)
            except OSError:
                # The watcher hung up. That is not a failure of the move, and
                # the move is not told about it.
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
                    f"one install on a machine; attach to {INSTALL_EVENTS_PATH} "
                    "and watch the one in flight rather than starting a second. "
                    "On a fresh install the runner is usually this machine's own "
                    "tray, which starts the move at every start (PHASE19 2.3).",
                )
                return
            self._install()

        def _install(self) -> None:
            """The move's own stream, and the ring every other watcher reads.

            PHASE19 2.6: this and the tray's start-time decision go through
            `run_recorded`, so there is ONE numbering of a move's events and one
            ring, whoever started it.
            """
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
                    # The POSTER hung up. The move goes on — it is installing
                    # software on this machine and stopping halfway because
                    # nobody is reading would be worse than finishing unwatched.
                    dead = True
                    door._log.write("door: the install's caller hung up; the move continues")

            try:
                door.run_recorded(sink)
            except HostError as exc:
                door._log.write(f"door: install failed: {exc.code}: {exc.message}")
            except Exception as exc:  # noqa: BLE001 - the stream must terminate
                door._log.write(f"door: install crashed: {type(exc).__name__}: {exc}")
            finally:
                door.release()

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
                # The sequence has already emitted its own `failed` event with
                # this code; this is the case where something threw before it
                # could. A stream that stops without a terminal event is what
                # the client reports as truncated, so one is always sent.
                door._log.write(f"door: install failed: {exc.code}: {exc.message}")
                if not terminal:
                    try:
                        emit(Event("failed", {"code": exc.code, "message": exc.message}))
                    except OSError:
                        pass
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
