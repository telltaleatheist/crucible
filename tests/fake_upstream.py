from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from crucible.engines import find_free_port

ANSWER = "Routed, and answered."
DELTAS = ["Routed, ", "and ", "answered."]

ANTHROPIC_MODELS = ["claude-sonnet-5", "claude-haiku-5"]
OPENAI_MODELS = ["gpt-5", "gpt-5-mini"]
OLLAMA_MODELS = ["qwen3.5:9b", "llama3:8b"]

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

THINKING = "Pondering the question."

ANTHROPIC_JSON = '{"verdict":"routed"}'

# The Messages API's request members (platform.claude.com/docs/en/api/messages/create),
# plus the sampling members the older models still take.
ANTHROPIC_MEMBERS = frozenset(
    {
        "model", "messages", "max_tokens", "system", "stop_sequences", "metadata",
        "output_config", "cache_control", "container", "diagnostics", "inference_geo",
        "service_tier", "stream", "thinking", "tool_choice", "tools",
        "temperature", "top_p", "top_k",
    }
)
# Kept apart from crucible.upstreams' own table on purpose, so a row wrong there is
# caught here.
ANTHROPIC_NO_FORCED_TOOL = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"}
)
ANTHROPIC_NO_SAMPLING = frozenset(
    {
        "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
        "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5-5", "claude-sonnet-5",
        "claude-haiku-5-5",
    }
)


class _Handler(BaseHTTPRequestHandler):
    owner: "FakeUpstream" = None

    def log_message(self, *args: Any) -> None:
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
            owner.headers_seen.append(
                {key.lower(): value for key, value in self.headers.items()}
            )
        return body


    def do_GET(self) -> None:
        owner = type(self).owner
        with owner.lock:
            owner.headers_seen.append(
                {key.lower(): value for key, value in self.headers.items()}
            )
        if owner.status != 200:
            self._json(owner.status, owner.error_body)
            return
        if self.path == "/v1/models":
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


    def do_POST(self) -> None:
        owner = type(self).owner
        body = self._record()
        if owner.drop_connection:
            # The request arrived; the answer never leaves. To the caller it is a
            # connection lost after sending.
            self.close_connection = True
            return
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


    def _anthropic_refusal(self, body: dict[str, Any]) -> str | None:
        """What the Messages API answers 400 for, as its documentation says it does for
        the current models: a member it does not have, a forced tool_choice on the
        models that removed it, a non-default sampling value on the models that removed
        those, a schema-less structured output."""
        extra = sorted(set(body) - ANTHROPIC_MEMBERS)
        if extra:
            return f"{extra[0]}: Extra inputs are not permitted"
        model = body.get("model")
        if not isinstance(body.get("max_tokens"), int):
            return "max_tokens: Field required"
        choice = body.get("tool_choice")
        if (
            model in ANTHROPIC_NO_FORCED_TOOL
            and isinstance(choice, dict)
            and choice.get("type") in ("any", "tool")
        ):
            return 'tool_choice: type "tool" and "any" are not supported for this model.'
        if model in ANTHROPIC_NO_SAMPLING:
            if "top_p" in body or "top_k" in body or body.get("temperature", 1) != 1:
                return "temperature, top_p and top_k are not supported for this model"
        for message in body.get("messages") or []:
            if message.get("role") not in ("user", "assistant"):
                return f"messages: role {message.get('role')!r} is not supported"
        output_config = body.get("output_config")
        if isinstance(output_config, dict):
            form = output_config.get("format")
            if form is not None and (
                form.get("type") != "json_schema" or not isinstance(form.get("schema"), dict)
            ):
                return "output_config.format: a json_schema format needs a schema"
        return None

    def _anthropic(self, body: dict[str, Any]) -> None:
        owner = type(self).owner
        refused = self._anthropic_refusal(body)
        if refused is not None:
            self._json(
                400,
                {"type": "error",
                 "error": {"type": "invalid_request_error", "message": refused}},
            )
            return
        output_config = body.get("output_config") or {}
        structured = output_config.get("format") is not None
        text = ANTHROPIC_JSON if structured else ANSWER
        usage = {
            "input_tokens": 11,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 7,
        }
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._event(
                "message_start",
                {"message": {"id": "msg_fake", "type": "message", "role": "assistant",
                             "content": [], "model": body.get("model"),
                             "stop_reason": None,
                             "usage": {**usage, "output_tokens": 1}}},
            )
            self._event(
                "content_block_start",
                {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            )
            self._event("ping", {})
            self._event(
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            )
            self._event("content_block_stop", {"index": 0})
            self._event(
                "content_block_start",
                {"index": 1, "content_block": {"type": "text", "text": ""}},
            )
            for delta in [text] if structured else DELTAS:
                self._event(
                    "content_block_delta",
                    {"index": 1, "delta": {"type": "text_delta", "text": delta}},
                )
            if owner.stream_error is not None:
                self._event("error", {"error": owner.stream_error})
                return
            if owner.truncate_stream:
                return
            self._event("content_block_stop", {"index": 1})
            self._event(
                "message_delta",
                {"delta": {"stop_reason": owner.stop_reason, "stop_sequence": None},
                 "usage": {"output_tokens": 7}},
            )
            self._event("message_stop", {})
            return
        self._json(
            200,
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": body.get("model"),
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "text", "text": text},
                ],
                "stop_reason": owner.stop_reason,
                "usage": usage,
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

    def __init__(self) -> None:
        self.port = find_free_port()
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.headers_seen: list[dict[str, str]] = []
        self.status = 200
        self.error_body: Any = {"error": {"message": "nope"}}
        self.extra_headers: dict[str, str] = {}
        self.digests: dict[str, str] = {n: f"sha256:{i:064x}" for i, n in enumerate(OLLAMA_MODELS)}
        self.tags_calls = 0
        self.show_calls = 0
        self.show_failures = 0
        self.show_status = 500
        self.done_reason = "stop"
        self.truncate_stream = False
        self.stream_error: dict[str, Any] | None = None
        self.stop_reason = "end_turn"
        self.drop_connection = False
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
