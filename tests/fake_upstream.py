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

# What the Anthropic fake's thinking block says when it is asked for summaries
# (thinking.display "summarized"); otherwise its thinking text is empty, as the
# Messages API's default display ("omitted") leaves it. Two thinking_delta pieces.
CLAUDE_SUMMARY = ["Weighing the ", "question first."]
CLAUDE_SIGNATURE = "EqQBCkYIBxgCKkDsig=="

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


def _blocks(message: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == kind]


def _anthropic_tool_refusal(body: dict[str, Any]) -> str | None:
    """The Messages API's tool rules, as its documentation states them: a tool has a
    name and an input_schema; tool_choice is auto, any, tool or none; every tool_use
    of an assistant turn is answered by a tool_result in the very next (user) turn,
    and a tool_result answers a tool_use of the turn just before it."""
    for tool in body.get("tools") or []:
        if set(tool) - {"name", "description", "input_schema", "strict"}:
            return f"tools: unexpected members {sorted(set(tool))}"
        if not isinstance(tool.get("name"), str) or not isinstance(
            tool.get("input_schema"), dict
        ):
            return "tools: a tool needs a name and an input_schema"
    choice = body.get("tool_choice")
    if choice is not None:
        if choice.get("type") not in ("auto", "any", "tool", "none"):
            return f"tool_choice: unknown type {choice.get('type')!r}"
        if choice.get("type") == "tool" and not isinstance(choice.get("name"), str):
            return "tool_choice: type tool needs a name"
    messages = body.get("messages") or []
    for number, message in enumerate(messages):
        uses = [b["id"] for b in _blocks(message, "tool_use")]
        results = [b["tool_use_id"] for b in _blocks(message, "tool_result")]
        if uses:
            if message.get("role") != "assistant":
                return f"messages.{number}: tool_use outside an assistant turn"
            following = messages[number + 1] if number + 1 < len(messages) else None
            answered = (
                {b["tool_use_id"] for b in _blocks(following, "tool_result")}
                if following is not None and following.get("role") == "user" else set()
            )
            if following is not None and set(uses) - answered:
                return (f"messages.{number + 1}: `tool_use` ids were found without "
                        f"`tool_result` blocks immediately after: "
                        f"{sorted(set(uses) - answered)}")
        if results:
            before = messages[number - 1] if number > 0 else {}
            made = {b["id"] for b in _blocks(before, "tool_use")}
            if set(results) - made:
                return (f"messages.{number}: unexpected `tool_use_id` found in "
                        f"`tool_result` blocks: {sorted(set(results) - made)}")
    return None


def _anthropic_thinking_refusal(body: dict[str, Any]) -> str | None:
    """The Messages API's thinking rules the translation must keep: `disabled` is a 400
    on the models that removed it, and a thinking block sent back carries its
    signature (a redacted one its data)."""
    thinking = body.get("thinking")
    if thinking is not None:
        if thinking.get("type") == "disabled" and body.get("model") in (
            "claude-opus-5-5", "claude-sonnet-5-5"
        ):
            return "thinking: type disabled is not supported for this model"
        if set(thinking) - {"type", "display", "budget_tokens"}:
            return f"thinking: unexpected members {sorted(thinking)}"
    for number, message in enumerate(body.get("messages") or []):
        for block in _blocks(message, "thinking"):
            if not isinstance(block.get("signature"), str) or block["signature"] == "":
                return f"messages.{number}: a thinking block needs its signature"
        for block in _blocks(message, "redacted_thinking"):
            if not isinstance(block.get("data"), str):
                return f"messages.{number}: a redacted_thinking block needs its data"
    return None


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
        if owner.ollama_tool_calls is not None:
            calls = owner.ollama_tool_calls
            final = {"model": body.get("model"), "done": True, "done_reason": "stop",
                     "prompt_eval_count": 11, "eval_count": 7}
            if body.get("stream") is True:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                # Ollama streams each call whole, one per line.
                for call in calls:
                    self._ollama_line(
                        {"model": body.get("model"), "done": False,
                         "message": {"role": "assistant", "content": "",
                                     "tool_calls": [call]}}
                    )
                self._ollama_line({**final, "message": {"role": "assistant",
                                                        "content": ""}})
                return
            self._json(200, {**final, "message": {"role": "assistant", "content": "",
                                                  "tool_calls": calls}})
            return
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
        refused = _anthropic_tool_refusal(body)
        if refused is not None:
            return refused
        refused = _anthropic_thinking_refusal(body)
        if refused is not None:
            return refused
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
        if owner.anthropic_tool_uses is not None:
            self._anthropic_tool_answer(body, owner.anthropic_tool_uses)
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
        summarized = (body.get("thinking") or {}).get("display") == "summarized"
        summary = CLAUDE_SUMMARY if summarized else []
        signature = CLAUDE_SIGNATURE if summarized else "sig"
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
                {"index": 0, "content_block": {"type": "thinking", "thinking": "",
                                               "signature": ""}},
            )
            self._event("ping", {})
            for piece in summary:
                self._event(
                    "content_block_delta",
                    {"index": 0, "delta": {"type": "thinking_delta", "thinking": piece}},
                )
            self._event(
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": signature}},
            )
            self._event("content_block_stop", {"index": 0})
            index = 1
            if owner.redacted_thinking is not None:
                self._event(
                    "content_block_start",
                    {"index": index, "content_block": {"type": "redacted_thinking",
                                                       "data": owner.redacted_thinking}},
                )
                self._event("content_block_stop", {"index": index})
                index += 1
            self._event(
                "content_block_start",
                {"index": index, "content_block": {"type": "text", "text": ""}},
            )
            for delta in [text] if structured else DELTAS:
                self._event(
                    "content_block_delta",
                    {"index": index, "delta": {"type": "text_delta", "text": delta}},
                )
            if owner.stream_error is not None:
                self._event("error", {"error": owner.stream_error})
                return
            if owner.truncate_stream:
                return
            self._event("content_block_stop", {"index": index})
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
                    {"type": "thinking", "thinking": "".join(summary),
                     "signature": signature},
                    *(
                        [{"type": "redacted_thinking", "data": owner.redacted_thinking}]
                        if owner.redacted_thinking is not None else []
                    ),
                    {"type": "text", "text": text},
                ],
                "stop_reason": owner.stop_reason,
                "usage": usage,
            },
        )

    def _anthropic_tool_answer(
        self, body: dict[str, Any], uses: list[dict[str, Any]]
    ) -> None:
        """An answer that says TOOL_PREAMBLE (when set) and then calls each of `uses`
        ({id, name, input}), stopping for `tool_use`."""
        owner = type(self).owner
        preamble = owner.tool_preamble
        usage = {"input_tokens": 11, "output_tokens": 7}
        if owner.interleaved_thinking:
            self._anthropic_interleaved(body, uses, usage)
            return
        if body.get("stream") is not True:
            content: list[dict[str, Any]] = [
                {"type": "thinking", "thinking": "", "signature": "sig"}
            ]
            if preamble is not None:
                content.append({"type": "text", "text": preamble})
            content.extend({"type": "tool_use", **use} for use in uses)
            self._json(200, {"id": "msg_tools", "type": "message", "role": "assistant",
                             "model": body.get("model"), "content": content,
                             "stop_reason": "tool_use", "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._event("message_start", {"message": {"id": "msg_tools", "usage": usage}})
        index = 0
        if preamble is not None:
            self._event("content_block_start",
                        {"index": index, "content_block": {"type": "text", "text": ""}})
            self._event("content_block_delta",
                        {"index": index, "delta": {"type": "text_delta", "text": preamble}})
            self._event("content_block_stop", {"index": index})
            index += 1
        for use in uses:
            self._event(
                "content_block_start",
                {"index": index, "content_block": {"type": "tool_use", "id": use["id"],
                                                   "name": use["name"], "input": {}}},
            )
            encoded = json.dumps(use["input"]) if use["input"] else ""
            # Split the arguments mid-token, as Anthropic's input_json_delta does.
            middle = len(encoded) // 2
            for piece in (encoded[:middle], encoded[middle:]):
                if piece:
                    self._event("content_block_delta",
                                {"index": index,
                                 "delta": {"type": "input_json_delta",
                                           "partial_json": piece}})
            self._event("content_block_stop", {"index": index})
            index += 1
        self._event("message_delta", {"delta": {"stop_reason": "tool_use"},
                                      "usage": {"output_tokens": 7}})
        self._event("message_stop", {})

    def _anthropic_interleaved(
        self, body: dict[str, Any], uses: list[dict[str, Any]], usage: dict[str, Any]
    ) -> None:
        """Opus 5.5's shape: a thinking block BEFORE EACH call, then the call, and a
        closing text: thinking, tool_use, thinking, tool_use, text."""
        content: list[dict[str, Any]] = []
        for number, use in enumerate(uses, start=1):
            content.append({"type": "thinking", "thinking": f"Progress {number}.",
                            "signature": f"sig-{number}"})
            content.append({"type": "tool_use", **use})
        content.append({"type": "text", "text": "Checking both."})
        if body.get("stream") is not True:
            self._json(200, {"id": "msg_interleaved", "type": "message", "role": "assistant",
                             "model": body.get("model"), "content": content,
                             "stop_reason": "tool_use", "usage": usage})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self._event("message_start", {"message": {"id": "msg_interleaved", "usage": usage}})
        for index, block in enumerate(content):
            if block["type"] == "thinking":
                self._event("content_block_start", {"index": index, "content_block": {
                    "type": "thinking", "thinking": "", "signature": ""}})
                half = len(block["thinking"]) // 2
                for piece in (block["thinking"][:half], block["thinking"][half:]):
                    self._event("content_block_delta", {"index": index, "delta": {
                        "type": "thinking_delta", "thinking": piece}})
                self._event("content_block_delta", {"index": index, "delta": {
                    "type": "signature_delta", "signature": block["signature"]}})
            elif block["type"] == "tool_use":
                self._event("content_block_start", {"index": index, "content_block": {
                    "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}})
                encoded = json.dumps(block["input"]) if block["input"] else ""
                middle = len(encoded) // 2
                for piece in (encoded[:middle], encoded[middle:]):
                    if piece:
                        self._event("content_block_delta", {"index": index, "delta": {
                            "type": "input_json_delta", "partial_json": piece}})
            else:
                self._event("content_block_start", {"index": index, "content_block": {
                    "type": "text", "text": ""}})
                self._event("content_block_delta", {"index": index, "delta": {
                    "type": "text_delta", "text": block["text"]}})
            self._event("content_block_stop", {"index": index})
        self._event("message_delta", {"delta": {"stop_reason": "tool_use"},
                                      "usage": {"output_tokens": 7}})
        self._event("message_stop", {})

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
        # Tool calling: what the Anthropic fake calls ({id, name, input} each) and the
        # text it says first; what the Ollama fake calls (Ollama's own call objects).
        self.anthropic_tool_uses: list[dict[str, Any]] | None = None
        self.tool_preamble: str | None = None
        # A redacted_thinking block's data the Anthropic fake answers with, after its
        # thinking block; None, it answers with none.
        self.redacted_thinking: str | None = None
        # The tool answer in Opus 5.5's interleaved shape (_anthropic_interleaved).
        self.interleaved_thinking = False
        self.ollama_tool_calls: list[dict[str, Any]] | None = None
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
