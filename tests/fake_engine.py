from __future__ import annotations

import json
import math
import select
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Literal

from crucible.engines import find_free_port

ANSWER = "Crucible is a server."
DELTAS = ["Crucible ", "is ", "a ", "server."]

ProbsFor = Callable[[list[dict[str, Any]]], dict[str, float]]

FAKE_MAX_LOGPROBS = 20

FAKE_KV_BLOCK = 16

PrefixCache = Literal["blocks", "segments"]

FILLER_TOKENS: tuple[tuple[str, float], ...] = (("The", 0.6), (" A", 0.4))

TOOL_CALL = {
    "id": "call_fake",
    "type": "function",
    "function": {"name": "light_the_forge", "arguments": '{"heat":"white"}'},
}


class _Handler(BaseHTTPRequestHandler):
    served_name: str = "unset"
    last_request: dict[str, Any] | None = None
    last_request_bytes: bytes | None = None
    finish_reason: str = "stop"
    reject_response_format: bool = False
    answer_delay: float = 0.0
    stream_forever: bool = False
    delay_for: Callable[[dict[str, Any]], float] | None = None
    aborts: list[int] | None = None
    aborted: threading.Event = threading.Event()
    requests: list[dict[str, Any]] | None = None
    request_bytes: list[int] | None = None
    on_post: Callable[[], None] | None = None
    lock: threading.Lock | None = None
    drops_left: list[int] | None = None
    probs_for: ProbsFor | None = None
    max_logprobs: int = FAKE_MAX_LOGPROBS
    report_cached: bool = True
    prompts: list[str] | None = None
    prefix_cache: str = "blocks"
    saved: list[str] | None = None
    events: list[tuple[str, int]] | None = None
    in_flight: list[int] | None = None
    max_in_flight: list[int] | None = None

    def log_message(self, *args: Any) -> None:
        return

    def _peer_gone(self, timeout: float) -> bool:
        try:
            ready, _, _ = select.select([self.connection], [], [], timeout)
            if not ready:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

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
        with type(self).lock:
            index = len(type(self).requests)
            type(self).requests.append(body)
            type(self).request_bytes.append(len(raw))
            type(self).events.append(("start", index))
            type(self).in_flight[0] += 1
            type(self).max_in_flight[0] = max(
                type(self).max_in_flight[0], type(self).in_flight[0]
            )
        try:
            self._serve_completion(body)
        finally:
            with type(self).lock:
                type(self).in_flight[0] -= 1
                type(self).events.append(("end", index))

    def _serve_completion(self, body: dict[str, Any]) -> None:
        if type(self).on_post is not None:
            type(self).on_post()

        with type(self).lock:
            drop = type(self).drops_left[0] > 0
            if drop:
                type(self).drops_left[0] -= 1
        if drop:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            self.close_connection = True
            return

        if type(self).reject_response_format and "response_format" in body:
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
            if type(self).stream_forever:
                self._stream_until_hung_up()
                return
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
            self._frame({"index": 0, "delta": {}, "finish_reason": type(self).finish_reason})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        wants_logprobs = body.get("logprobs") is True
        if wants_logprobs and type(self).probs_for is not None:
            asked = body.get("top_logprobs")
            if isinstance(asked, int) and asked > type(self).max_logprobs:
                self._json(
                    400,
                    {
                        "error": {
                            "message": f"Requested sample logprobs of {asked}, "
                            "which is greater than max allowed: "
                            f"{type(self).max_logprobs}",
                            "type": "BadRequestError",
                            "param": "logprobs",
                            "code": 400,
                        }
                    },
                )
                return

        delay = (
            type(self).answer_delay
            if type(self).delay_for is None
            else type(self).delay_for(body)
        )
        if delay > 0.0 and self._wait_out_the_answer(delay):
            return

        if type(self).probs_for is not None:
            self._json(200, self._decision_reply(body, wants_logprobs))
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

    def _decision_reply(self, body: dict[str, Any], wants_logprobs: bool) -> dict[str, Any]:
        msgs = body.get("messages", [])
        text = _rendered(msgs)
        n_prompt = max(1, len(text) // 4)
        with type(self).lock:
            if type(self).prefix_cache == "segments":
                saved = type(self).saved
                hit = max((len(k) for k in saved if text.startswith(k)), default=0)
                cached = hit // 4
                key = system_segment(msgs)
                if key is not None and key not in saved:
                    saved.append(key)
                saved.append(text + END_OF_TURN)
            else:
                common = max(
                    (_common_prefix(text, seen) for seen in type(self).prompts),
                    default=0,
                )
                cached = (common // 4) // FAKE_KV_BLOCK * FAKE_KV_BLOCK
            type(self).prompts.append(text)

        letters = type(self).probs_for(body.get("messages", []))
        entries = list(letters.items())
        rest = max(0.0, 1.0 - sum(letters.values()))
        entries += [(token, rest * share) for token, share in FILLER_TOKENS]
        entries.sort(key=lambda entry: -entry[1])
        top_n = body.get("top_logprobs") or 0
        tops = [
            {
                "token": token,
                "logprob": math.log(p) if p > 0 else -9999.0,
                "bytes": list(token.encode("utf-8")),
            }
            for token, p in entries[:top_n]
        ]
        sampled = entries[0][0]
        logprobs = (
            {"content": [{**tops[0], "top_logprobs": tops}]}
            if wants_logprobs and tops
            else None
        )
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": type(self).served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": sampled},
                    "logprobs": logprobs,
                    "finish_reason": "length",
                    "stop_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt,
                "total_tokens": n_prompt + 1,
                "completion_tokens": 1,
                "prompt_tokens_details": (
                    {"cached_tokens": cached} if type(self).report_cached else None
                ),
            },
        }

    def _wait_out_the_answer(self, delay: float) -> bool:
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if self._peer_gone(0.02):
                with type(self).lock:
                    type(self).aborts[0] += 1
                type(self).aborted.set()
                return True
        return False

    def _stream_until_hung_up(self) -> None:
        while True:
            if self._peer_gone(type(self).answer_delay):
                type(self).aborted.set()
                return
            try:
                self._frame(
                    {"index": 0, "delta": {"content": "on "}, "finish_reason": None}
                )
            except OSError:
                type(self).aborted.set()
                return

    def _frame(self, choice: dict[str, Any]) -> None:
        chunk = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "model": type(self).served_name,
            "choices": [choice],
        }
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()


def _rendered(messages: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            pieces = [content]
        else:
            pieces = [
                part.get("text") if part.get("type") == "text"
                else part.get("image_url", {}).get("url", "")
                for part in content or []
            ]
        out.append(f"<|{message.get('role')}|>" + "\n".join(pieces))
    return "".join(out)


END_OF_TURN = "<|end|>"


def system_segment(messages: list[dict[str, Any]]) -> str | None:
    if not messages or messages[-1].get("role") != "user":
        return None
    leading = []
    for message in messages:
        if message.get("role") != "system":
            break
        leading.append(message)
    if not leading:
        return None
    return _rendered(leading) + "<|user|>"


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class FakeEngine:

    name = "fake"

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
        answer_delay: float = 0.0,
        stream_forever: bool = False,
        delay_for: Callable[[dict[str, Any]], float] | None = None,
        on_post: Callable[[], None] | None = None,
        drop_requests: int = 0,
        probs_for: ProbsFor | None = None,
        max_logprobs: int = FAKE_MAX_LOGPROBS,
        report_cached: bool = True,
        prefix_cache: PrefixCache = "blocks",
    ) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._finish_reason = finish_reason
        self._reject_response_format = reject_response_format
        self._answer_delay = answer_delay
        self._stream_forever = stream_forever
        self._delay_for = delay_for
        self.aborted = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None
        self._warmings = warmings
        self._fail_ready = fail_ready
        self._hold = hold
        self._on_post = on_post
        self._drop_requests = drop_requests
        self._probs_for = probs_for
        self._max_logprobs = max_logprobs
        self._report_cached = report_cached
        self._prefix_cache = prefix_cache
        self.warming_started = threading.Event()
        self.stopped = False
        self.exited_with: int | None = None
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

    @property
    def exit_code(self) -> int | None:
        return self.exited_with

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
                "answer_delay": self._answer_delay,
                "stream_forever": self._stream_forever,
                "delay_for": (
                    None if self._delay_for is None else staticmethod(self._delay_for)
                ),
                "aborts": [0],
                "aborted": self.aborted,
                "requests": [],
                "request_bytes": [],
                "lock": threading.Lock(),
                "on_post": self._on_post,
                "drops_left": [self._drop_requests],
                "probs_for": (
                    None if self._probs_for is None else staticmethod(self._probs_for)
                ),
                "max_logprobs": self._max_logprobs,
                "report_cached": self._report_cached,
                "prompts": [],
                "prefix_cache": self._prefix_cache,
                "saved": [],
                "events": [],
                "in_flight": [0],
                "max_in_flight": [0],
            },
        )
        self._handler = handler
        self.args = list(args)
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
        self,
        timeout: float,
        on_progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        from crucible.engines import EngineError
        from crucible.errors import JobCancelled

        self.warming_started.set()
        if self._fail_ready is not None:
            raise EngineError(self._fail_ready)
        for step in range(self._warmings):
            if on_progress is not None:
                on_progress(f"fake engine warming, step {step + 1}/{self._warmings}")
        if self._hold is None:
            return
        # Held like a real engine still warming: the real ready() polls its cancel flag,
        # so this does too.
        deadline = time.monotonic() + 30
        while not self._hold.wait(timeout=0.05):
            if cancelled is not None and cancelled():
                raise JobCancelled("fake engine was cancelled while it was starting")
            if time.monotonic() >= deadline:
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
        return getattr(self, "_handler", _Handler).last_request_bytes

    @property
    def requests(self) -> list[dict[str, Any]]:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.requests

    @property
    def events(self) -> list[tuple[str, int]]:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.events

    @property
    def aborts(self) -> int:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.aborts[0]

    @property
    def max_in_flight(self) -> int:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.max_in_flight[0]

    @property
    def request_bytes(self) -> list[int]:
        handler = getattr(self, "_handler", None)
        if handler is None:
            raise RuntimeError("fake engine has not been started")
        return handler.request_bytes
