"""One HTTP server that answers as any of the three upstreams, on a real socket.

PHASE15-HOST.md section 3.9 asks for chat forwarding tested "against a fake
upstream". This is it, and it is a REAL server on a real loopback port for
`tests/fake_engine.py`'s reason: the thing under test is a proxy, and a proxy
mocked at the httpx layer is a proxy whose socket path — the streaming relay,
the disconnect, the header the provider actually received — is never exercised.

It speaks all three dialects because the three differ in exactly the places
this phase translates: Anthropic's `/v1/messages` with `x-api-key` and a
message envelope, OpenAI's `/v1/chat/completions` with a bearer, Ollama's
native `/api/tags` (with digests), `/api/show` and `/api/chat` (NDJSON when
streamed) — section 3.4a. A test points `crucible.upstreams.ANTHROPIC_BASE` (or `OPENAI_BASE`,
or an ollama `url`) at this server's address and the rest of the server is
untouched.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from crucible.engines import find_free_port

#: What every fake completion says, so a test can assert exact bytes.
ANSWER = "Routed, and answered."
DELTAS = ["Routed, ", "and ", "answered."]

#: The model ids each fake upstream claims to serve, for `test`.
ANTHROPIC_MODELS = ["claude-sonnet-5", "claude-haiku-5"]
OPENAI_MODELS = ["gpt-5", "gpt-5-mini"]
OLLAMA_MODELS = ["qwen3.5:9b", "llama3:8b"]

#: What `/api/show` says about each fake Ollama tag. `qwen3.5:9b` states no
#: `num_ctx`, so its context is the trained maximum in `model_info`;
#: `llama3:8b` states one in its Modelfile, which is what a tag like Owen's
#: `qwen3.8:27b-24g` (98304) looks like.
OLLAMA_TRAINED_CONTEXT = 262144
OLLAMA_MODELFILE_CONTEXT = 98304
OLLAMA_SHOW: dict[str, dict[str, Any]] = {
    "qwen3.5:9b": {
        "parameters": 'stop                           "<|im_end|>"',
        "model_info": {
            "general.architecture": "qwen35",
            "qwen35.context_length": OLLAMA_TRAINED_CONTEXT,
        },
    },
    "llama3:8b": {
        "parameters": (
            f"num_ctx                        {OLLAMA_MODELFILE_CONTEXT}\n"
            'stop                           "<|eot_id|>"'
        ),
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 8192,
        },
    },
}

#: What the fake Ollama "thinks" when a chat asks it to.
THINKING = "Pondering the question."


class _Handler(BaseHTTPRequestHandler):
    #: Set per server instance in `FakeUpstream.start`.
    owner: "FakeUpstream" = None  # type: ignore[assignment]

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        return

    def _json(self, status: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _record(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else {}
        owner = type(self).owner
        with owner.lock:
            owner.requests.append(body)
            # Lower-cased: HTTP header names are case-insensitive and httpx
            # picks its own casing, so a test asserting on `authorization`
            # must not be testing which case a client library chose.
            owner.headers_seen.append(
                {key.lower(): value for key, value in self.headers.items()}
            )
        return body

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:
        owner = type(self).owner
        with owner.lock:
            # Lower-cased: HTTP header names are case-insensitive and httpx
            # picks its own casing, so a test asserting on `authorization`
            # must not be testing which case a client library chose.
            owner.headers_seen.append(
                {key.lower(): value for key, value in self.headers.items()}
            )
        if owner.status != 200:
            self._json(owner.status, owner.error_body)
            return
        if self.path == "/v1/models":
            # Anthropic and OpenAI list the same shape under the same path;
            # which one this is depends only on which header arrived, and the
            # test that cares asserts on `headers_seen`.
            ids = (
                ANTHROPIC_MODELS
                if self.headers.get("x-api-key") is not None
                else OPENAI_MODELS
            )
            self._json(200, {"data": [{"id": name} for name in ids]})
            return
        if self.path == "/api/tags":
            with owner.lock:
                owner.tags_calls += 1
                digests = dict(owner.digests)
            self._json(
                200,
                {
                    "models": [
                        {"name": n, "model": n, "digest": digests[n]}
                        for n in OLLAMA_MODELS
                    ]
                },
            )
            return
        self._json(404, {"error": {"message": "not found"}})

    # ----------------------------------------------------------------- POST

    def do_POST(self) -> None:
        owner = type(self).owner
        body = self._record()
        if owner.status != 200:
            self._json(owner.status, owner.error_body, owner.extra_headers)
            return
        if self.path == "/v1/messages":
            self._anthropic(body)
            return
        if self.path == "/v1/chat/completions":
            self._openai(body)
            return
        if self.path == "/api/show":
            self._ollama_show(body)
            return
        if self.path == "/api/chat":
            self._ollama_chat(body)
            return
        self._json(404, {"error": {"message": "not found"}})

    def _ollama_show(self, body: dict[str, Any]) -> None:
        owner = type(self).owner
        with owner.lock:
            owner.show_calls += 1
            failing = owner.show_failures > 0
            if failing:
                owner.show_failures -= 1
        if failing:
            self._json(owner.show_status, {"error": "model runner is loading"})
            return
        name = body.get("model")
        shown = OLLAMA_SHOW.get(name) or OLLAMA_SHOW.get(f"{name}:latest")
        if shown is None:
            # Ollama's own words for a tag that is not pulled.
            self._json(404, {"error": f"model '{name}' not found"})
            return
        self._json(200, {"modelfile": "", "template": "", **shown})

    def _ollama_line(self, payload: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(payload).encode("utf-8") + b"\n")
        self.wfile.flush()

    def _ollama_chat(self, body: dict[str, Any]) -> None:
        owner = type(self).owner
        thinks = body.get("think") is True
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            if thinks:
                self._ollama_line(
                    {"model": body.get("model"), "done": False,
                     "message": {"role": "assistant", "content": "",
                                 "thinking": THINKING}}
                )
            for delta in DELTAS:
                self._ollama_line(
                    {"model": body.get("model"), "done": False,
                     "message": {"role": "assistant", "content": delta}}
                )
            if owner.truncate_stream:
                return
            self._ollama_line(
                {"model": body.get("model"), "done": True, "done_reason": "stop",
                 "message": {"role": "assistant", "content": ""},
                 "prompt_eval_count": 11, "eval_count": 7}
            )
            return
        message: dict[str, Any] = {"role": "assistant", "content": ANSWER}
        if thinks:
            message["thinking"] = THINKING
        self._json(
            200,
            {
                "model": body.get("model"),
                "created_at": "2026-09-23T00:00:00Z",
                "message": message,
                "done": True,
                "done_reason": owner.done_reason,
                "prompt_eval_count": 11,
                "eval_count": 7,
            },
        )

    # ------------------------------------------------------------- dialects

    def _anthropic(self, body: dict[str, Any]) -> None:
        tool = (body.get("tools") or [None])[0]
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._event("message_start", {"message": {"id": "msg_fake"}})
            self._event("content_block_start", {"index": 0})
            for delta in DELTAS:
                self._event(
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "text_delta", "text": delta}},
                )
            self._event("content_block_stop", {"index": 0})
            self._event("message_delta", {"delta": {"stop_reason": "end_turn"}})
            self._event("message_stop", {})
            return
        if tool is not None:
            content = [
                {
                    "type": "tool_use",
                    "id": "toolu_fake",
                    "name": tool["name"],
                    "input": {"verdict": "routed"},
                }
            ]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": ANSWER}]
            stop_reason = "end_turn"
        self._json(
            200,
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": body.get("model"),
                "content": content,
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        frame = f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def _openai(self, body: dict[str, Any]) -> None:
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, delta in enumerate(DELTAS):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": body.get("model"),
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
                    b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
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
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )


class FakeUpstream:
    """A running fake provider. `url` is what a test points Crucible at."""

    def __init__(self) -> None:
        self.port = find_free_port()
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.headers_seen: list[dict[str, str]] = []
        #: What every route answers with instead of 200. A test sets this to
        #: 401, 429 or 500 and asserts the refusal Crucible produces.
        self.status = 200
        self.error_body: Any = {"error": {"message": "nope"}}
        self.extra_headers: dict[str, str] = {}
        #: Ollama's per-tag digests, as `/api/tags` reports them. A test that
        #: changes one is `ollama create` over the same name.
        self.digests: dict[str, str] = {n: f"sha256:{i:064x}" for i, n in enumerate(OLLAMA_MODELS)}
        self.tags_calls = 0
        #: `/api/show`: how many calls, and how many of the next ones fail with
        #: `show_status` before it answers.
        self.show_calls = 0
        self.show_failures = 0
        self.show_status = 500
        #: The non-streamed `/api/chat`'s `done_reason`.
        self.done_reason = "stop"
        #: A streamed `/api/chat` that stops before its `done` line.
        self.truncate_stream = False
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeUpstream":
        handler = type("_Bound", (_Handler,), {"owner": self})
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "FakeUpstream":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
