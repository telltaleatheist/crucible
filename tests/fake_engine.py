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

#: The call a `finish_reason: "tool_calls"` completion says the model wants. Its
#: `content` is null, which is the part worth proxying correctly: a body whose
#: answer is not text at all still has to come back as the engine wrote it.
TOOL_CALL = {
    "id": "call_fake",
    "type": "function",
    "function": {"name": "light_the_forge", "arguments": '{"heat":"white"}'},
}


class _Handler(BaseHTTPRequestHandler):
    served_name: str = "unset"
    last_request: dict[str, Any] | None = None
    #: The request body exactly as it arrived on the wire. `last_request` says
    #: what the engine understood; this says what Crucible actually sent, which
    #: is the only way to ask whether the proxy is verbatim.
    last_request_bytes: bytes | None = None
    #: What the completion stops for. A real engine's own word, which the proxy
    #: must hand back untouched: Foundry turns `length` into a degradation rather
    #: than a wrong answer (CLIENT-SURFACES.md section 6.2).
    finish_reason: str = "stop"
    #: Answer any body carrying `response_format` with a 400 in the engine's own
    #: shape — what vLLM does with a schema it cannot compile.
    reject_response_format: bool = False

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
        raw = self.rfile.read(length)
        body = json.loads(raw or b"{}")
        type(self).last_request_bytes = raw
        type(self).last_request = body

        if type(self).reject_response_format and "response_format" in body:
            # vLLM's own refusal shape for a schema it will not compile. The
            # proxy has to relay this as it stands: rewritten into a Crucible
            # error, the client would be told the server refused when the engine
            # did, and the schema it must fix would be gone.
            self._json(
                400,
                {
                    "object": "error",
                    "message": "unsupported json_schema: 'prefixItems' is not "
                    "supported by the guided-decoding backend",
                    "type": "BadRequestError",
                    "code": 400,
                },
            )
            return

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
                self._frame(
                    {
                        "index": 0,
                        "delta": (
                            {"role": "assistant", "content": delta}
                            if index == 0
                            else {"content": delta}
                        ),
                        "finish_reason": None,
                    }
                )
            # The closing frame, which is where a streamed completion says why it
            # stopped. Every OpenAI engine sends one; the fake used to skip it,
            # which left "the proxy never touches finish_reason" untestable on
            # the streaming half.
            self._frame({"index": 0, "delta": {}, "finish_reason": type(self).finish_reason})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        reason = type(self).finish_reason
        message: dict[str, Any] = (
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
            if reason == "tool_calls"
            else {"role": "assistant", "content": ANSWER}
        )
        self._json(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": type(self).served_name,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 5,
                    "total_tokens": 12,
                },
            },
        )

    def _frame(self, choice: dict[str, Any]) -> None:
        """One `chat.completion.chunk` SSE frame, flushed as a real engine does."""
        chunk = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": type(self).served_name,
            "choices": [choice],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()


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
        finish_reason: str = "stop",
        reject_response_format: bool = False,
    ) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._finish_reason = finish_reason
        self._reject_response_format = reject_response_format
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None
        self._warmings = warmings
        self._fail_ready = fail_ready
        #: When given, `ready()` blocks on it, so a test can look at the server
        #: while a load is genuinely in flight.
        self._hold = hold
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
                "finish_reason": self._finish_reason,
                "reject_response_format": self._reject_response_format,
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
    def last_request_bytes(self) -> bytes | None:
        """The last chat body as it arrived, before anything parsed it."""
        return getattr(self, "_handler", _Handler).last_request_bytes
