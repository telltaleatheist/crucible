"""An `Engine` that serves a trivial OpenAI-shaped endpoint in-process.

The load/unload state machine, the `warming` stream and the proxy are all
testable without a GPU because the only thing the server needs from vLLM or
mlx-lm is: a process that can be started and SIGTERMed, and an HTTP surface with
`/v1/models` and `/v1/chat/completions`. This provides exactly that, in a thread,
on a real loopback port — so the proxy's socket path is the real one.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from crucible.engines import find_free_port

#: What the fake completion answers with, so tests can assert on exact bytes.
ANSWER = "Crucible is a server."
DELTAS = ["Crucible ", "is ", "a ", "server."]


class _Handler(BaseHTTPRequestHandler):
    served_name: str = "unset"
    last_request: dict[str, Any] | None = None
    #: Every completion body this engine was sent, in arrival order. `ThreadingHTTPServer`
    #: serves each connection on its own thread, so `last_request` is whichever
    #: one finished last — no use at all to a test about concurrency.
    requests: list[dict[str, Any]] | None = None
    #: How many bytes of body arrived for each of those, which is the only way to
    #: say "nothing was truncated" about an 11 MB data URI without trusting the
    #: JSON to have parsed.
    request_bytes: list[int] | None = None
    #: Called once per completion, on the serving thread, before anything is
    #: answered. A test that needs several requests to be genuinely in flight at
    #: the same moment puts a `threading.Barrier.wait` here; nothing else can
    #: tell "twelve at once" from "twelve quickly".
    on_post: Callable[[], None] | None = None
    lock: threading.Lock | None = None

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self._json(404, {"error": "not found"})
            return
        self._json(
            200,
            {
                "object": "list",
                "data": [{"id": type(self).served_name, "object": "model"}],
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) or b"{}"
        body = json.loads(raw)
        type(self).last_request = body
        with type(self).lock:
            type(self).requests.append(body)
            type(self).request_bytes.append(len(raw))
        if type(self).on_post is not None:
            type(self).on_post()

        if body.get("model") != type(self).served_name:
            # What a real engine does with a name it is not serving. Crucible's
            # proxy must never let a request get this far.
            self._json(
                404,
                {"error": {"message": f"model {body.get('model')!r} not found"}},
            )
            return

        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for index, delta in enumerate(DELTAS):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": type(self).served_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {"role": "assistant", "content": delta}
                                if index == 0
                                else {"content": delta}
                            ),
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(
                    f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": type(self).served_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
            },
        )


class FakeEngine:
    """Implements the Engine protocol against a threaded HTTP server."""

    name = "fake"

    #: Set by a test to make `ready()` fail the way a real engine fails.
    def __init__(
        self,
        python: Path,
        log_path: Path,
        *,
        warmings: int = 3,
        fail_ready: str | None = None,
        hold: threading.Event | None = None,
        on_post: Callable[[], None] | None = None,
    ) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None
        self._warmings = warmings
        self._fail_ready = fail_ready
        #: When given, `ready()` blocks on it, so a test can look at the server
        #: while a load is genuinely in flight.
        self._hold = hold
        self._on_post = on_post
        #: Set once `ready()` has been entered, so a test knows the lane has
        #: reached the engine without polling on a sleep.
        self.warming_started = threading.Event()
        self.stopped = False
        #: Exactly the argument list `start()` was handed — the manifest's
        #: `engine_args` plus what Crucible always adds. A test that cares what
        #: the engine was told reads this rather than guessing.
        self.args: list[str] = []

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def base_url(self) -> str:
        if self._port is None:
            raise RuntimeError("fake engine has not been started")
        return f"http://127.0.0.1:{self._port}"

    @property
    def pids(self) -> frozenset[int]:
        return frozenset()

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        handler = type(
            "BoundHandler",
            (_Handler,),
            {
                "served_name": served_name,
                # Per engine, not per class: two engines in one test (a load that
                # evicts another) must not share a request log.
                "requests": [],
                "request_bytes": [],
                "lock": threading.Lock(),
                "on_post": self._on_post,
            },
        )
        self._handler = handler
        self.args = list(args)
        # The port the caller found may have been taken; the fake binds its own
        # and reports it, which is all the proxy reads.
        self._port = port if port else find_free_port()
        self._server = ThreadingHTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fake-engine", daemon=True
        )
        self._thread.start()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text(
            f"fake engine for {served_name} on {self._port}\n", encoding="utf-8"
        )

    def ready(
        self, timeout: float, on_progress: Callable[[str], None] | None = None
    ) -> None:
        from crucible.engines import EngineError

        self.warming_started.set()
        if self._fail_ready is not None:
            raise EngineError(self._fail_ready)
        for step in range(self._warmings):
            if on_progress is not None:
                on_progress(f"fake engine warming, step {step + 1}/{self._warmings}")
        if self._hold is not None and not self._hold.wait(timeout=30):
            raise EngineError("the test never released the hold on ready()")

    def stop(self) -> None:
        self.stopped = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._port = None

    @property
    def last_request(self) -> dict[str, Any] | None:
        return getattr(self, "_handler", _Handler).last_request

    @property
    def requests(self) -> list[dict[str, Any]]:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.requests

    @property
    def request_bytes(self) -> list[int]:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.request_bytes
