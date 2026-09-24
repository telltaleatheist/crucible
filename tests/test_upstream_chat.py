"""Chat completions forwarded to an upstream, in all three dialects.

PHASE15-HOST.md section 3.4. The local half of this door is
`tests/test_llm_api.py` and is untouched by any of it.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import capability, upstreams
from crucible.sampling import SAMPLING_HEADER

from . import fake_upstream
from .fake_upstream import ANSWER, FakeUpstream
from .test_settings_api import ANTHROPIC_KEY, OPENAI_KEY, decided


@pytest.fixture
def upstream():
    with FakeUpstream() as running:
        yield running


@pytest.fixture(autouse=True)
def no_lookup_waits(monkeypatch):
    """The Ollama context lookup's budget, without its sleeps.

    The ATTEMPTS are the budget under test; the waits between them are
    seconds that would only make the suite slower.
    """
    monkeypatch.setattr(upstreams, "OLLAMA_LOOKUP_BACKOFF_SECONDS", (0.0, 0.0))


@pytest.fixture
def routed(make_client: Callable[..., TestClient], auth, upstream, monkeypatch):
    """A server with all three upstreams configured, pointed at the fake.

    Configured through the real `PUT /v1/settings`, not by writing a config
    behind the door's back: the thing being tested is a server an app set up.
    """
    monkeypatch.setattr(upstreams, "ANTHROPIC_BASE", upstream.url)
    monkeypatch.setattr(upstreams, "OPENAI_BASE", upstream.url)
    with make_client(enable_llm=True, capability=decided()) as instance:
        written = instance.put(
            "/v1/settings",
            headers=auth,
            json={
                "upstreams": {
                    "anthropic": {"key": ANTHROPIC_KEY},
                    "openai": {"key": OPENAI_KEY},
                    "ollama": {"url": upstream.url},
                },
                "routes": {
                    "translate": "anthropic/claude-sonnet-5",
                    "simplify": "openai/gpt-5",
                    "clean": "ollama/qwen3.5:9b",
                },
            },
        )
        assert written.status_code == 200, written.text
        yield instance


def chat(client: TestClient, auth: dict[str, str], body: dict[str, Any], **kw):
    return client.post("/v1/openai/chat/completions", headers=auth, json=body, **kw)


# ------------------------------------------------------------ the happy path


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-5", "openai/gpt-5", "ollama/qwen3.5:9b"],
)
def test_a_completion_comes_back_in_openai_shape_naming_the_id_asked_for(
    routed, auth, model
) -> None:
    """Whatever the hop, the caller reads an OpenAI completion naming ITS model.

    Crucible's id is the contract in both directions — the same rule
    `_restore_model_id` follows for the local engine, one hop further out.
    """
    body = chat(routed, auth, {"model": model, "messages": [{"role": "user", "content": "hi"}]})
    assert body.status_code == 200, body.text
    document = body.json()
    assert document["object"] == "chat.completion"
    assert document["model"] == model
    assert document["choices"][0]["message"]["content"] == ANSWER
    assert document["choices"][0]["finish_reason"] == "stop"


def test_anthropic_gets_a_system_field_and_a_max_tokens_and_says_so(
    routed, auth, upstream
) -> None:
    """The three translations 3.4 names, on one request.

    A leading `system` message becomes Anthropic's top-level `system`;
    `max_tokens` is supplied because Anthropic requires it; and the audit
    header says which of them this server filled.
    """
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.2,
        },
    )
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["system"] == "You are terse."
    assert [m["role"] for m in sent["messages"]] == ["user"]
    assert sent["max_tokens"] == upstreams.ANTHROPIC_MAX_TOKENS_DEFAULT
    assert sent["temperature"] == 0.2
    sources = json.loads(body.headers[SAMPLING_HEADER])
    assert sources["max_tokens"] == "upstream default 4096"
    assert sources["temperature"] == "request"
    assert sources["top_p"] == "engine"


def test_a_stated_max_tokens_wins_and_the_audit_says_request(
    routed, auth, upstream
) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
        },
    )
    assert upstream.requests[-1]["max_tokens"] == 128
    assert json.loads(body.headers[SAMPLING_HEADER])["max_tokens"] == "request"


def test_thinking_is_dropped_and_the_audit_says_dropped(
    routed, auth, upstream
) -> None:
    """BookForge sends `thinking: false` on every cleanup call.

    Neither hosted upstream reads `chat_template_kwargs`, so the table is
    dropped rather than forwarded blind — and the header says `dropped`, which
    is neither "the model got it" nor "nobody asked". Ollama DOES take it, as
    `think` (section 3.4a; `test_thinking_reaches_ollama_as_think`).
    """
    for model in ("anthropic/claude-sonnet-5", "openai/gpt-5"):
        body = chat(
            routed,
            auth,
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        assert body.status_code == 200, body.text
        assert "chat_template_kwargs" not in upstream.requests[-1], model
        assert json.loads(body.headers[SAMPLING_HEADER])["thinking"] == "dropped"


def test_a_request_that_said_nothing_about_thinking_says_engine(
    routed, auth
) -> None:
    body = chat(
        routed,
        auth,
        {"model": "openai/gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert json.loads(body.headers[SAMPLING_HEADER])["thinking"] == "engine"


def test_a_json_schema_becomes_a_forced_tool_and_its_input_is_the_content(
    routed, auth, upstream
) -> None:
    """How `analysis` gets guided decoding out of a cloud model.

    Anthropic has no `response_format`; a tool with a forced choice IS guided
    decoding, and the tool's argument object is what a caller that sent a
    schema reads out of `content`.
    """
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
    }
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "judge"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema},
            },
        },
    )
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["tools"][0]["name"] == upstreams.ANTHROPIC_JSON_TOOL
    assert sent["tools"][0]["input_schema"] == schema
    assert sent["tool_choice"] == {
        "type": "tool",
        "name": upstreams.ANTHROPIC_JSON_TOOL,
    }
    content = body.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"verdict": "routed"}
    # `tool_use` is not told to the caller as a tool call: this server forced
    # the tool to carry JSON, and the caller asked for JSON.
    assert body.json()["choices"][0]["finish_reason"] == "stop"


def test_a_bare_json_object_response_format_forces_no_tool(
    routed, auth, upstream
) -> None:
    """Nothing to force a tool WITH, so nothing is invented."""
    chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert "tools" not in upstream.requests[-1]


def test_the_provider_sees_its_own_credentials(routed, auth, upstream) -> None:
    chat(
        routed,
        auth,
        {"model": "anthropic/claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert upstream.headers_seen[-1]["x-api-key"] == ANTHROPIC_KEY
    assert upstream.headers_seen[-1]["anthropic-version"] == upstreams.ANTHROPIC_VERSION
    chat(
        routed,
        auth,
        {"model": "openai/gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert upstream.headers_seen[-1]["authorization"] == f"Bearer {OPENAI_KEY}"
    chat(
        routed,
        auth,
        {"model": "ollama/qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Ollama is reached by address and has no account, so there is no
    # credential header at all. That is what Ollama is, not a hole this
    # server opened.
    assert "authorization" not in {k.lower() for k in upstream.headers_seen[-1]}
    # …and the prefix is this server's, so the upstream is sent the bare id.
    assert upstream.requests[-1]["model"] == "qwen3.5:9b"


# ------------------------------------------------------------------ streaming


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-5", "openai/gpt-5", "ollama/qwen3.5:9b"],
)
def test_a_streamed_completion_arrives_as_openai_chunks(routed, auth, model) -> None:
    """Anthropic's envelope is translated; the other two are relayed.

    Either way the caller reads OpenAI chunks naming its own model id and
    ending in `[DONE]`, because that is the protocol it spoke to this door.
    """
    with routed.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        frames = list(response.iter_lines())
    payloads = [
        json.loads(line[len("data: "):])
        for line in frames
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert frames[-1] == "data: [DONE]" or "data: [DONE]" in frames
    assert payloads, frames
    assert {chunk["model"] for chunk in payloads} == {model}
    text = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in payloads
    )
    assert text == ANSWER
    assert payloads[0]["object"] == "chat.completion.chunk"


def test_the_streamed_audit_header_is_on_the_response(routed, auth) -> None:
    with routed.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        sources = json.loads(response.headers[SAMPLING_HEADER])
        response.read()
    assert sources["max_tokens"] == "upstream default 4096"


# ------------------------------------------------------------------ refusals


def test_an_unconfigured_upstream_is_refused_409_and_never_falls_back(
    make_client, auth
) -> None:
    """*"Nothing decides to go to the cloud because something local failed"* —
    and nothing decides to go local because the cloud is not configured."""
    with make_client(enable_llm=True, capability=decided()) as instance:
        body = chat(
            instance,
            auth,
            {
                "model": "anthropic/claude-sonnet-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert body.status_code == 409
    error = body.json()["error"]
    assert error["code"] == "upstream_unconfigured"
    assert error["details"]["upstream"] == "anthropic"


def test_a_model_whose_prefix_is_not_an_upstream_is_route_bad_model(
    routed, auth
) -> None:
    body = chat(
        routed,
        auth,
        {"model": "claud/sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 400
    assert body.json()["error"]["code"] == "route_bad_model"
    assert body.json()["error"]["details"]["field"] == "model"


def test_a_rejection_is_502_with_the_upstreams_own_words(
    routed, auth, upstream
) -> None:
    upstream.status = 401
    upstream.error_body = {"error": {"message": "invalid x-api-key"}}
    body = chat(
        routed,
        auth,
        {"model": "anthropic/claude-sonnet-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 502
    error = body.json()["error"]
    assert error["code"] == "upstream_rejected"
    assert "invalid x-api-key" in error["message"]
    assert error["details"]["upstream_status"] == 401


def test_a_rate_limit_is_passed_through_with_retry_after_and_never_retried(
    routed, auth, upstream
) -> None:
    """The caller waits. This server never sends a billed request twice."""
    upstream.status = 429
    upstream.error_body = {"error": {"message": "slow down"}}
    upstream.extra_headers = {"Retry-After": "42"}
    before = len(upstream.requests)
    body = chat(
        routed,
        auth,
        {"model": "openai/gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 429
    assert body.headers["Retry-After"] == "42"
    error = body.json()["error"]
    assert error["code"] == "upstream_rate_limited"
    assert error["details"]["retry_after"] == "42"
    assert len(upstream.requests) == before + 1


def test_an_unreachable_upstream_is_502_upstream_unreachable(
    make_client, auth, upstream
) -> None:
    dead = upstream.url
    upstream.stop()
    with make_client(enable_llm=True, capability=decided()) as instance:
        instance.put(
            "/v1/settings",
            headers=auth,
            json={
                "upstreams": {"ollama": {"url": dead}},
                "routes": {"clean": "ollama/qwen3.5:9b"},
            },
        )
        body = chat(
            instance,
            auth,
            {"model": "ollama/qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert body.status_code == 502
    assert body.json()["error"]["code"] == "upstream_unreachable"


def test_a_streamed_rejection_comes_back_before_any_frames(
    routed, auth, upstream
) -> None:
    upstream.status = 500
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert body.status_code == 502
    assert body.json()["error"]["code"] == "upstream_rejected"


# ------------------------------------------------------- the other two doors


def test_a_lease_on_an_upstream_model_is_refused_lease_not_needed(
    routed, auth
) -> None:
    body = routed.post(
        "/v1/models/anthropic/claude-sonnet-5/lease",
        headers=auth,
        json={"ttl_seconds": 60, "act": "translate"},
    )
    assert body.status_code == 409
    error = body.json()["error"]
    assert error["code"] == "lease_not_needed"
    assert error["message"] == "an upstream model is never resident; send the chat"


def test_load_model_naming_an_upstream_model_is_refused_the_same_way(
    routed, auth
) -> None:
    body = routed.post(
        "/v1/jobs",
        headers=auth,
        json={"type": "load-model", "model": "anthropic/claude-sonnet-5"},
    )
    assert body.status_code == 409
    assert body.json()["error"]["code"] == "lease_not_needed"


def test_openai_models_lists_the_routed_upstream_models_and_no_catalog(
    routed, auth
) -> None:
    """3.4: the ROUTED models, not the upstream's whole catalog."""
    rows = routed.get("/v1/openai/models", headers=auth).json()["data"]
    by_id = {row["id"]: row for row in rows}
    assert set(by_id) == {
        "anthropic/claude-sonnet-5",
        "openai/gpt-5",
        "ollama/qwen3.5:9b",
    }
    row = by_id["anthropic/claude-sonnet-5"]
    assert row["upstream"] == "anthropic"
    assert row["owned_by"] == "anthropic"
    assert row["routed_for"] == ["translate"]
    # This server did not load it and has no manifest for it.
    assert "max_model_len" not in row
    assert "defaults" not in row
    assert "created" not in row
    # …and the fake's other models are NOT here; that is `test`'s job.
    assert "anthropic/claude-haiku-5" not in by_id


def test_two_classes_on_one_model_are_one_row(routed, auth) -> None:
    routed.put(
        "/v1/settings",
        headers=auth,
        json={"routes": {"simplify": "anthropic/claude-sonnet-5"}},
    )
    rows = routed.get("/v1/openai/models", headers=auth).json()["data"]
    sonnet = [r for r in rows if r["id"] == "anthropic/claude-sonnet-5"]
    assert len(sonnet) == 1
    assert sorted(sonnet[0]["routed_for"]) == ["simplify", "translate"]


def test_a_forwarded_chat_is_recorded_in_flight_with_its_act_and_model(
    routed, auth, monkeypatch
) -> None:
    """*"so `/v1/activity` says 'translating on anthropic'"* — 3.4.

    A completion is over in the time the test client takes to return, so the
    row cannot be READ mid-flight without a barrier that would tell us nothing
    extra. What matters is that the record is opened at all, and with which
    two facts: the act the client named, and the id it asked for — never the
    bare upstream model, which would report a name no app sent.
    """
    from crucible.inflight import InFlight

    opened: list[dict[str, Any]] = []
    original = InFlight.open

    def spy(self, *, act, model, client):
        opened.append({"act": act, "model": model, "client": client})
        return original(self, act=act, model=model, client=client)

    monkeypatch.setattr(InFlight, "open", spy)
    response = routed.post(
        "/v1/openai/chat/completions",
        headers={**auth, "X-Crucible-Act": "translate", "User-Agent": "foundry/9"},
        json={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    assert opened == [
        {
            "act": "translate",
            "model": "anthropic/claude-sonnet-5",
            "client": "foundry/9",
        }
    ]
    # And it is closed: a record that outlived its request would make this
    # server look permanently busy with work that stopped.
    assert routed.get("/v1/activity", headers=auth).json()["chat"]["in_flight"] == 0


def test_a_forwarded_chat_takes_no_lane_and_settles_nothing(routed, auth) -> None:
    """No lease, no lane, the settlement untouched — nothing was on the card."""
    from crucible.settle import Settlement

    settled: list[str] = []

    original = Settlement.settle_quietly

    def spy(self, why):
        settled.append(why)
        return original(self, why)

    Settlement.settle_quietly, saved = spy, Settlement.settle_quietly
    try:
        before = routed.get("/v1/activity", headers=auth).json()
        chat(
            routed,
            auth,
            {
                "model": "openai/gpt-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        after = routed.get("/v1/activity", headers=auth).json()
    finally:
        Settlement.settle_quietly = saved
    assert settled == []
    assert after["slots"] == before["slots"]
    assert after["lease"] is None


# ------------------------------------------------ ollama, natively (3.4a)
#
# The bug these pin: the `ollama` upstream used to go through Ollama's OpenAI
# shim, which has no `num_ctx`, so every chat ran at Ollama's default 4096 and a
# longer prompt was silently cut from the front. Now it is Ollama's own
# `/api/chat`, and `options.num_ctx` is on every body.


def _chats(upstream: FakeUpstream) -> list[dict[str, Any]]:
    """The `/api/chat` bodies the fake received (`/api/show`'s carry no messages)."""
    return [body for body in upstream.requests if "messages" in body]


def _context(response) -> dict[str, Any]:
    return json.loads(response.headers[upstreams.CONTEXT_HEADER])


def test_ollama_gets_its_native_body_with_the_requests_context(
    routed, auth, upstream
) -> None:
    """Every OpenAI knob lands where `/api/chat` reads it, `num_ctx` first.

    A stated `context_tokens` is sent as it stands, and nothing is looked up:
    the caller knows its prompt and this server has no tokenizer for it.
    """
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "hi"},
            ],
            "context_tokens": 16384,
            "max_tokens": 256,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": 20,
            "seed": 7,
            "stop": "END",
            "repetition_penalty": 1.05,
        },
    )
    assert body.status_code == 200, body.text
    assert _chats(upstream)[-1] == {
        "model": "qwen3.5:9b",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ],
        # Stated even when false: Ollama's own default is to stream.
        "stream": False,
        "options": {
            "num_ctx": 16384,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": 20,
            "seed": 7,
            "repeat_penalty": 1.05,
            "num_predict": 256,
            "stop": ["END"],
        },
    }
    assert list(_chats(upstream)[-1]["options"])[0] == "num_ctx"
    assert upstream.show_calls == 0
    assert upstream.tags_calls == 0
    assert _context(body) == {"num_ctx": 16384, "source": "request"}
    sources = json.loads(body.headers[SAMPLING_HEADER])
    assert sources["max_tokens"] == "request"
    assert sources["top_p"] == "request"


def test_ollama_with_no_stated_context_is_sent_the_trained_maximum(
    routed, auth, upstream
) -> None:
    """A tag that states no `num_ctx` runs at `model_info.<arch>.context_length`.

    NEVER nothing: nothing is Ollama's 4096, which is the bug.
    """
    body = chat(
        routed,
        auth,
        {"model": "ollama/qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 200, body.text
    assert _chats(upstream)[-1]["options"] == {
        "num_ctx": fake_upstream.OLLAMA_TRAINED_CONTEXT
    }
    assert _context(body) == {
        "num_ctx": fake_upstream.OLLAMA_TRAINED_CONTEXT,
        "source": "model",
    }


def test_a_tags_own_modelfile_num_ctx_wins_over_the_trained_maximum(
    routed, auth, upstream
) -> None:
    """`PARAMETER num_ctx` is the tag author's statement of what fits the card.

    Owen's `qwen3.8:27b-24g` carries 98304 because 262144 does not fit 24 GB;
    overriding it with the trained maximum would push the KV cache off the card.
    """
    body = chat(
        routed,
        auth,
        {"model": "ollama/llama3:8b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 200, body.text
    assert (
        _chats(upstream)[-1]["options"]["num_ctx"]
        == fake_upstream.OLLAMA_MODELFILE_CONTEXT
    )
    assert _context(body) == {
        "num_ctx": fake_upstream.OLLAMA_MODELFILE_CONTEXT,
        "source": "modelfile",
    }


def test_the_looked_up_context_is_remembered_per_digest(
    routed, auth, upstream
) -> None:
    """`/api/show` once per digest; `ollama create` over the same name asks again."""
    request = {
        "model": "ollama/qwen3.5:9b",
        "messages": [{"role": "user", "content": "hi"}],
    }
    for _ in range(3):
        assert chat(routed, auth, request).status_code == 200
    assert upstream.show_calls == 1
    assert upstream.tags_calls == 3
    upstream.digests["qwen3.5:9b"] = "sha256:" + "f" * 64
    assert chat(routed, auth, request).status_code == 200
    assert upstream.show_calls == 2


@pytest.mark.parametrize("thinking", [True, False])
def test_thinking_reaches_ollama_as_think(routed, auth, upstream, thinking) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": thinking},
        },
    )
    assert body.status_code == 200, body.text
    sent = _chats(upstream)[-1]
    assert sent["think"] is thinking
    assert "chat_template_kwargs" not in sent
    assert json.loads(body.headers[SAMPLING_HEADER])["thinking"] == "request"
    message = body.json()["choices"][0]["message"]
    # Ollama's `message.thinking` is the door's `reasoning`, the field the
    # local engines answer in and the SDK reads.
    if thinking:
        assert message["reasoning"] == fake_upstream.THINKING
    else:
        assert "reasoning" not in message
    assert message["content"] == ANSWER


def test_a_request_silent_on_thinking_sends_no_think(routed, auth, upstream) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens": 4096,
        },
    )
    assert "think" not in _chats(upstream)[-1]
    assert json.loads(body.headers[SAMPLING_HEADER])["thinking"] == "engine"


def test_a_json_schema_becomes_ollamas_format(routed, auth, upstream) -> None:
    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
    }
    chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "judge"}],
            "context_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema, "strict": True},
            },
        },
    )
    assert _chats(upstream)[-1]["format"] == schema
    chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "judge"}],
            "context_tokens": 4096,
            "response_format": {"type": "json_object"},
        },
    )
    assert _chats(upstream)[-1]["format"] == "json"


def test_an_ollama_answer_is_an_openai_completion_with_usage(
    routed, auth, upstream
) -> None:
    request = {
        "model": "ollama/qwen3.5:9b",
        "messages": [{"role": "user", "content": "hi"}],
        "context_tokens": 4096,
    }
    document = chat(routed, auth, request).json()
    assert document["object"] == "chat.completion"
    assert document["model"] == "ollama/qwen3.5:9b"
    assert document["choices"][0]["finish_reason"] == "stop"
    assert document["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    upstream.done_reason = "length"
    document = chat(routed, auth, request).json()
    assert document["choices"][0]["finish_reason"] == "length"


def _stream(client: TestClient, auth: dict[str, str], body: dict[str, Any]):
    with client.stream(
        "POST", "/v1/openai/chat/completions", headers=auth, json=body
    ) as response:
        assert response.status_code == 200
        context = _context(response)
        frames = [line for line in response.iter_lines() if line]
    return frames, context


def _payloads(frames: list[str]) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: "):])
        for line in frames
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_ollamas_ndjson_stream_becomes_openai_sse(routed, auth, upstream) -> None:
    """Thinking as `reasoning` deltas, content as `content`, a finish, usage, `[DONE]`."""
    frames, context = _stream(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert _chats(upstream)[-1]["stream"] is True
    assert context["source"] == "model"
    assert frames[-1] == "data: [DONE]"
    payloads = _payloads(frames)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    with_choices = [p for p in payloads if p["choices"]]
    deltas = [p["choices"][0]["delta"] for p in with_choices]
    assert "".join(d.get("reasoning", "") for d in deltas) == fake_upstream.THINKING
    assert "".join(d.get("content", "") for d in deltas) == ANSWER
    assert with_choices[-1]["choices"][0]["finish_reason"] == "stop"
    usage = [p for p in payloads if not p["choices"]]
    assert len(usage) == 1
    assert usage[0]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert {p["model"] for p in payloads} == {"ollama/qwen3.5:9b"}


def test_an_ollama_stream_cut_before_done_gets_no_done_terminator(
    routed, auth, upstream
) -> None:
    """A `[DONE]` here would make a truncated answer read as a finished one."""
    upstream.truncate_stream = True
    frames, _ = _stream(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens": 4096,
            "stream": True,
        },
    )
    assert "data: [DONE]" not in frames
    error = json.loads(frames[-1][len("data: "):])["error"]
    assert "truncated" in error["message"]


def test_a_show_that_keeps_failing_is_refused_by_name_after_the_budget(
    routed, auth, upstream
) -> None:
    """Weather gets `OLLAMA_LOOKUP_ATTEMPTS`, then a name — never a guessed 4096."""
    upstream.show_failures = upstreams.OLLAMA_LOOKUP_ATTEMPTS
    body = chat(
        routed,
        auth,
        {"model": "ollama/qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 502
    error = body.json()["error"]
    assert error["code"] == "upstream_context_unknown"
    assert "context_tokens" in error["message"]
    assert upstream.show_calls == upstreams.OLLAMA_LOOKUP_ATTEMPTS
    assert _chats(upstream) == []


def test_a_show_that_recovers_inside_the_budget_is_answered(
    routed, auth, upstream
) -> None:
    upstream.show_failures = upstreams.OLLAMA_LOOKUP_ATTEMPTS - 1
    body = chat(
        routed,
        auth,
        {"model": "ollama/qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 200, body.text
    assert upstream.show_calls == upstreams.OLLAMA_LOOKUP_ATTEMPTS
    assert (
        _chats(upstream)[-1]["options"]["num_ctx"]
        == fake_upstream.OLLAMA_TRAINED_CONTEXT
    )


def test_a_tag_ollama_does_not_have_is_its_own_404_asked_once(
    routed, auth, upstream
) -> None:
    """Misconfiguration, not weather: Ollama's own words, and no retry."""
    body = chat(
        routed,
        auth,
        {"model": "ollama/nope:1b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert body.status_code == 502
    error = body.json()["error"]
    assert error["code"] == "upstream_rejected"
    assert error["details"]["upstream_status"] == 404
    assert "model 'nope:1b' not found" in error["message"]
    assert upstream.show_calls == 1
    assert _chats(upstream) == []


@pytest.mark.parametrize(
    "extra",
    [{"num_ctx": 8192}, {"options": {"num_ctx": 8192}}, {"tools": []}, {"n": 2}],
)
def test_a_field_the_ollama_translation_would_drop_is_refused_by_name(
    routed, auth, upstream, extra
) -> None:
    """A field left behind silently is how `num_ctx` was lost; now it is a 400.

    Refused before the context lookup: a request this translation cannot carry
    costs no round trip.
    """
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            **extra,
        },
    )
    assert body.status_code == 400
    error = body.json()["error"]
    assert error["code"] == "upstream_field_unsupported"
    assert error["details"]["fields"] == list(extra)
    assert upstream.tags_calls == 0
    assert upstream.show_calls == 0


@pytest.mark.parametrize("value", [0, -1, "8k", 8192.0, True])
def test_a_malformed_context_tokens_is_refused(routed, auth, upstream, value) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens": value,
        },
    )
    assert body.status_code == 400
    assert body.json()["error"]["details"]["field"] == "context_tokens"
    assert _chats(upstream) == []


def test_an_inline_image_becomes_ollamas_images(routed, auth, upstream) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "context_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
        },
    )
    assert body.status_code == 200, body.text
    assert _chats(upstream)[-1]["messages"] == [
        {"role": "user", "content": "What is this?", "images": ["iVBORw0KGgo="]}
    ]


def test_an_image_by_address_is_refused(routed, auth, upstream) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "ollama/qwen3.5:9b",
            "context_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                    ],
                }
            ],
        },
    )
    assert body.status_code == 400
    assert body.json()["error"]["details"]["field"] == "messages[0].content[0]"


def test_the_hosted_upstreams_drop_context_tokens_and_say_so(
    routed, auth, upstream
) -> None:
    """A hosted model's window is its provider's; OpenAI would refuse the field."""
    body = chat(
        routed,
        auth,
        {
            "model": "openai/gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "context_tokens": 8192,
        },
    )
    assert body.status_code == 200, body.text
    assert "context_tokens" not in upstream.requests[-1]
    assert _context(body) == {"num_ctx": None, "source": "dropped"}
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert "context_tokens" not in upstream.requests[-1]
    assert _context(body) == {"num_ctx": None, "source": "upstream"}
