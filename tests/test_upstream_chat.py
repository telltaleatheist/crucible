from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import upstreams
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
    monkeypatch.setattr(upstreams, "OLLAMA_LOOKUP_BACKOFF_SECONDS", (0.0, 0.0))


@pytest.fixture
def routed(make_client: Callable[..., TestClient], auth, upstream, monkeypatch):
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


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-5", "openai/gpt-5", "ollama/qwen3.5:9b"],
)
def test_a_completion_comes_back_in_openai_shape_naming_the_id_asked_for(
    routed, auth, model
) -> None:
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
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-4-6",
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


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "model", ["claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1", "claude-future-9"]
)
def test_a_json_schema_is_the_messages_apis_own_structured_output(
    routed, auth, upstream, model
) -> None:
    # Opus 5.5 and Fable 5.1 refuse a forced tool_choice (the fake does too), which is
    # what a json_schema used to become.
    body = chat(
        routed,
        auth,
        {
            "model": f"anthropic/{model}",
            "messages": [{"role": "user", "content": "judge"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": VERDICT_SCHEMA},
            },
        },
    )
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["output_config"] == {
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}
    }
    assert "tools" not in sent
    assert "tool_choice" not in sent
    content = body.json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"verdict": "routed"}
    assert body.json()["choices"][0]["finish_reason"] == "stop"


def test_a_streamed_json_schema_arrives_as_its_text(routed, auth, upstream) -> None:
    frames, _ = _stream(
        routed,
        auth,
        {
            "model": "anthropic/claude-opus-5-5",
            "messages": [{"role": "user", "content": "judge"}],
            "stream": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA},
            },
        },
    )
    text = "".join(
        p["choices"][0]["delta"].get("content", "") for p in _payloads(frames) if p["choices"]
    )
    assert json.loads(text) == {"verdict": "routed"}
    assert frames[-1] == "data: [DONE]"


def test_a_bare_json_object_response_format_is_refused_by_name(
    routed, auth, upstream
) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert body.status_code == 400
    error = body.json()["error"]
    assert error["code"] == "upstream_param_unsupported"
    assert error["details"]["param"] == "response_format.type json_object"
    assert upstream.requests == []


def test_a_json_schema_description_is_refused_rather_than_dropped(
    routed, auth, upstream
) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "v", "description": "a verdict",
                                "schema": VERDICT_SCHEMA},
            },
        },
    )
    assert body.status_code == 400
    assert body.json()["error"]["details"]["fields"] == [
        "response_format.json_schema.description"
    ]


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
    assert "authorization" not in {k.lower() for k in upstream.headers_seen[-1]}
    assert upstream.requests[-1]["model"] == "qwen3.5:9b"


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-5", "openai/gpt-5", "ollama/qwen3.5:9b"],
)
def test_a_streamed_completion_arrives_as_openai_chunks(routed, auth, model) -> None:
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


def test_an_unconfigured_upstream_is_refused_409_and_never_falls_back(
    make_client, auth
) -> None:
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


def test_a_queue_session_on_an_upstream_model_is_refused_upstream_never_resident(
    routed, auth
) -> None:
    body = routed.post(
        "/v1/queue/sessions",
        headers=auth,
        json={"act": "translate", "model": "anthropic/claude-sonnet-5"},
    )
    assert body.status_code == 409
    error = body.json()["error"]
    assert error["code"] == "upstream_never_resident"
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
    assert body.json()["error"]["code"] == "upstream_never_resident"


def test_openai_models_lists_every_model_each_upstream_serves(routed, auth) -> None:
    listed = routed.get("/v1/openai/models", headers=auth).json()
    by_id = {row["id"]: row for row in listed["data"]}
    assert set(by_id) == {
        *(f"anthropic/{m}" for m in fake_upstream.ANTHROPIC_MODELS),
        *(f"openai/{m}" for m in fake_upstream.OPENAI_MODELS),
        *(f"ollama/{m}" for m in fake_upstream.OLLAMA_MODELS),
    }
    row = by_id["anthropic/claude-sonnet-5"]
    assert (row["upstream"], row["owned_by"], row["routed_for"], row["listed"]) == (
        "anthropic", "anthropic", ["translate"], True)
    assert by_id["anthropic/claude-haiku-5"]["routed_for"] == [], (
        "a model no route names is listed all the same: an app picks it per job"
    )
    assert row["facts"] == {
        "tool_choice": ["auto", "none", "required", "function"],
        "reasoning_efforts": list(upstreams.ANTHROPIC_MODEL_RULES["claude-sonnet-5"].efforts),
        "sampling": False, "structured_output": True,
    }
    assert by_id["ollama/llama3:8b"]["facts"]["tool_choice"] == ["auto"]
    assert by_id["openai/gpt-5"]["facts"] is None
    for key in ("max_model_len", "defaults", "created"):
        assert key not in row
    assert set(listed["upstreams"]) == {"anthropic", "openai", "ollama"}
    assert all(s["error"] is None and not s["stale"] and s["listed_at"]
               for s in listed["upstreams"].values())


def test_the_vendor_lists_are_kept_and_a_vendor_that_stops_answering_is_served_stale(
    routed, auth, upstream, monkeypatch
) -> None:
    from crucible.api import upstream as upstream_api

    def vendor_reads() -> int:
        return sum(1 for h in upstream.headers_seen if "x-api-key" in h)

    routed.get("/v1/openai/models", headers=auth)
    first = vendor_reads()
    routed.get("/v1/openai/models", headers=auth)
    assert vendor_reads() == first, "a second read inside the window asks no vendor"

    monkeypatch.setattr(upstream_api, "UPSTREAM_LIST_SECONDS", 0.0)
    upstream.status = 503
    listed = routed.get("/v1/openai/models", headers=auth).json()
    assert vendor_reads() > first
    anthropic = listed["upstreams"]["anthropic"]
    assert anthropic["stale"] is True and anthropic["error"]["code"] == "upstream_rejected"
    assert "anthropic/claude-haiku-5" in {r["id"] for r in listed["data"]}, (
        "weather: the last list the vendor gave is served, and says it is stale"
    )


def test_an_upstream_that_never_answered_lists_only_what_a_route_names(
    make_client, auth, upstream, monkeypatch
) -> None:
    monkeypatch.setattr(upstreams, "ANTHROPIC_BASE", upstream.url)
    upstream.status = 503
    with make_client(enable_llm=True, capability=decided()) as client:
        client.put("/v1/settings", headers=auth, json={
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"translate": "anthropic/claude-sonnet-5"},
        })
        listed = client.get("/v1/openai/models", headers=auth).json()
    assert [(r["id"], r["listed"]) for r in listed["data"]] == [
        ("anthropic/claude-sonnet-5", False)]
    assert listed["upstreams"]["anthropic"]["listed_at"] is None
    assert listed["upstreams"]["anthropic"]["error"]["code"] == "upstream_rejected"


def test_a_new_key_reads_the_vendor_list_again_at_once(routed, auth, upstream) -> None:
    routed.get("/v1/openai/models", headers=auth)
    before = sum(1 for h in upstream.headers_seen if "x-api-key" in h)
    routed.put("/v1/settings", headers=auth,
               json={"upstreams": {"anthropic": {"key": ANTHROPIC_KEY + "x"}}})
    routed.get("/v1/openai/models", headers=auth)
    assert sum(1 for h in upstream.headers_seen if "x-api-key" in h) == before + 1

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
    assert routed.get("/v1/activity", headers=auth).json()["chat"]["in_flight"] == 0


def test_a_forwarded_chat_takes_no_lane_and_settles_nothing(routed, auth) -> None:
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
    assert after["session"] is None


def _chats(upstream: FakeUpstream) -> list[dict[str, Any]]:
    return [body for body in upstream.requests if "messages" in body]


def _context(response) -> dict[str, Any]:
    return json.loads(response.headers[upstreams.CONTEXT_HEADER])


def test_ollama_gets_its_native_body_with_the_requests_context(
    routed, auth, upstream
) -> None:
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
    [{"num_ctx": 8192}, {"options": {"num_ctx": 8192}}, {"parallel_tool_calls": False},
     {"n": 2}],
)
def test_a_field_the_ollama_translation_would_drop_is_refused_by_name(
    routed, auth, upstream, extra
) -> None:
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


def _anthropic(routed, auth, model: str = "claude-sonnet-5", **extra: Any):
    return chat(
        routed,
        auth,
        {
            "model": f"anthropic/{model}",
            "messages": [{"role": "user", "content": "hi"}],
            **extra,
        },
    )


def _refused(response, code: str) -> dict[str, Any]:
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == code, error
    return error


@pytest.mark.parametrize(
    "model, param, value",
    [
        ("claude-opus-5-5", "temperature", 0.2),
        ("claude-sonnet-5", "temperature", 0),
        ("claude-sonnet-5-5", "top_p", 0.9),
        ("claude-haiku-5-5", "top_k", 40),
        ("claude-fable-5-1", "top_p", 1),
        ("claude-opus-4-7", "temperature", 0.7),
    ],
)
def test_a_sampling_value_the_model_rejects_is_refused_naming_param_and_model(
    routed, auth, upstream, model, param, value
) -> None:
    error = _refused(
        _anthropic(routed, auth, model, **{param: value}), "upstream_param_unsupported"
    )
    assert error["details"]["param"] == param
    assert error["details"]["model"] == model
    assert error["details"]["rule"] == model
    assert error["details"]["value"] == value
    assert param in error["message"] and model in error["message"]
    assert upstream.requests == []


@pytest.mark.parametrize(
    "model, extra",
    [
        ("claude-opus-5-5", {"temperature": 1}),
        ("claude-sonnet-4-6", {"temperature": 0.2, "top_k": 40}),
        ("claude-haiku-4-5-20251001", {"temperature": 0.0}),
        ("claude-opus-4-5-20251101", {"top_p": 0.9}),
        ("claude-future-9", {"temperature": 0.3}),
    ],
)
def test_a_sampling_value_the_model_takes_is_forwarded(
    routed, auth, upstream, model, extra
) -> None:
    body = _anthropic(routed, auth, model, **extra)
    assert body.status_code == 200, body.text
    for key, value in extra.items():
        assert upstream.requests[-1][key] == value


def test_every_model_the_key_door_listed_has_a_row() -> None:
    # The models the PC's key listed on 2026-10-12.
    listed = [
        "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1",
        "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-8",
        "claude-opus-4-7", "claude-sonnet-4-6", "claude-opus-4-6",
        "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
    ]
    for model in listed:
        assert upstreams.anthropic_rule(model) is not None, model
    assert upstreams.anthropic_rule("claude-fable-5-1")[0] == "claude-fable-5-1"
    assert upstreams.anthropic_rule("claude-fable-5")[0] == "claude-fable-5"
    assert upstreams.anthropic_rule("claude-sonnet-5-5")[0] == "claude-sonnet-5-5"
    assert upstreams.anthropic_rule("claude-opus-4-5-20251101")[0] == "claude-opus-4-5"
    assert upstreams.anthropic_rule("claude-opus-50") is None
    assert upstreams.anthropic_rule("claude-future-9") is None


@pytest.mark.parametrize(
    "extra",
    [
        {"seed": 7},
        {"frequency_penalty": 0.5},
        {"presence_penalty": 0.5},
        {"logit_bias": {"50256": -100}},
        {"logprobs": True},
        {"parallel_tool_calls": False},
        {"functions": [{"name": "f"}]},
        {"repetition_penalty": 1.1},
        {"num_ctx": 8192},
    ],
)
def test_a_field_the_anthropic_translation_would_drop_is_refused_by_name(
    routed, auth, upstream, extra
) -> None:
    error = _refused(_anthropic(routed, auth, **extra), "upstream_field_unsupported")
    assert error["details"]["upstream"] == "anthropic"
    assert error["details"]["fields"] == list(extra)
    assert upstream.requests == []


def test_n_of_one_is_read_and_more_is_refused(routed, auth, upstream) -> None:
    assert _anthropic(routed, auth, n=1).status_code == 200
    assert "n" not in upstream.requests[-1]
    error = _refused(_anthropic(routed, auth, n=2), "upstream_field_unsupported")
    assert error["details"]["fields"] == ["n"]


def test_max_completion_tokens_is_max_tokens_and_the_audit_says_request(
    routed, auth, upstream
) -> None:
    body = _anthropic(routed, auth, max_completion_tokens=64)
    assert body.status_code == 200, body.text
    assert upstream.requests[-1]["max_tokens"] == 64
    assert "max_completion_tokens" not in upstream.requests[-1]
    assert json.loads(body.headers[SAMPLING_HEADER])["max_tokens"] == "request"


@pytest.mark.parametrize(
    "extra",
    [
        {"max_tokens": 10, "max_completion_tokens": 10},
        {"max_tokens": 0},
        {"max_tokens": "64"},
        {"max_tokens": 64.0},
        {"max_completion_tokens": True},
    ],
)
def test_a_max_tokens_this_server_cannot_send_is_refused_not_replaced(
    routed, auth, upstream, extra
) -> None:
    _refused(_anthropic(routed, auth, **extra), "invalid_request")
    assert upstream.requests == []


def test_a_system_message_mid_conversation_is_refused(routed, auth, upstream) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Now be verbose."},
                {"role": "user", "content": "again"},
            ],
        },
    )
    error = _refused(body, "upstream_field_unsupported")
    assert error["details"]["fields"] == ["messages[3].role"]
    assert upstream.requests == []


def test_leading_developer_and_system_messages_are_the_system_field(
    routed, auth, upstream
) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [
                {"role": "developer", "content": "Be terse."},
                {"role": "system", "content": [{"type": "text", "text": "Use French."}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
    )
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["system"] == "Be terse.\n\nUse French."
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]


@pytest.mark.parametrize(
    "message, field",
    [
        ({"role": "function", "content": "42", "name": "f"}, "messages[0].role"),
        ({"role": "user", "content": "hi", "name": "owen"}, "messages[0].name"),
        (
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]},
            "messages[0].content[0]",
        ),
    ],
)
def test_a_message_with_no_messages_api_form_is_refused(
    routed, auth, upstream, message, field
) -> None:
    body = chat(
        routed, auth, {"model": "anthropic/claude-sonnet-5", "messages": [message]}
    )
    error = _refused(body, "upstream_field_unsupported")
    assert error["details"]["fields"] == [field]


def test_images_become_image_blocks(routed, auth, upstream) -> None:
    body = chat(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these."},
                        {"type": "image_url",
                         "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                        {"type": "image_url",
                         "image_url": {"url": "https://example.com/a.jpg", "detail": "auto"}},
                    ],
                }
            ],
        },
    )
    assert body.status_code == 200, body.text
    assert upstream.requests[-1]["messages"][0]["content"] == [
        {"type": "text", "text": "Compare these."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": "iVBORw0KGgo="}},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.jpg"}},
    ]


def test_an_image_detail_is_refused_and_a_strange_address_is_invalid(
    routed, auth, upstream
) -> None:
    def ask(image: dict[str, Any]):
        return chat(
            routed,
            auth,
            {
                "model": "anthropic/claude-sonnet-5",
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": image}]}
                ],
            },
        )

    error = _refused(
        ask({"url": "https://example.com/a.jpg", "detail": "high"}),
        "upstream_field_unsupported",
    )
    assert error["details"]["fields"] == ["messages[0].content[0].image_url.detail"]
    error = _refused(ask({"url": "ftp://example.com/a.jpg"}), "invalid_request")
    assert error["details"]["field"] == "messages[0].content[0]"
    assert upstream.requests == []


@pytest.mark.parametrize(
    "model, effort",
    [
        ("claude-opus-5-5", "low"),
        ("claude-fable-5-1", "max"),
        ("claude-sonnet-5", "xhigh"),
        ("claude-opus-4-6", "max"),
        ("claude-future-9", "xhigh"),
    ],
)
def test_reasoning_effort_is_output_config_effort(
    routed, auth, upstream, model, effort
) -> None:
    body = _anthropic(routed, auth, model, reasoning_effort=effort)
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["output_config"] == {"effort": effort}
    assert "reasoning_effort" not in sent


@pytest.mark.parametrize(
    "model, effort, takes",
    [
        ("claude-opus-4-6", "xhigh", ["low", "medium", "high", "max"]),
        ("claude-sonnet-4-6", "xhigh", ["low", "medium", "high", "max"]),
        ("claude-opus-4-5-20251101", "max", ["low", "medium", "high"]),
        ("claude-haiku-4-5", "low", []),
        ("claude-sonnet-4-5-20250929", "high", []),
    ],
)
def test_an_effort_the_model_does_not_take_is_refused_with_what_it_takes(
    routed, auth, upstream, model, effort, takes
) -> None:
    error = _refused(
        _anthropic(routed, auth, model, reasoning_effort=effort),
        "upstream_param_unsupported",
    )
    assert error["details"]["param"] == f"reasoning_effort {effort!r}"
    assert error["details"]["takes"] == takes
    assert upstream.requests == []


def test_an_effort_with_no_anthropic_equivalent_is_refused(routed, auth, upstream) -> None:
    for effort in ("none", "minimal"):
        error = _refused(
            _anthropic(routed, auth, reasoning_effort=effort), "upstream_param_unsupported"
        )
        assert "low" in error["message"]
    _refused(_anthropic(routed, auth, reasoning_effort="extreme"), "invalid_request")
    assert upstream.requests == []


def test_effort_and_a_schema_share_output_config(routed, auth, upstream) -> None:
    body = _anthropic(
        routed, auth, "claude-opus-5-5",
        reasoning_effort="medium",
        response_format={"type": "json_schema",
                         "json_schema": {"name": "v", "schema": VERDICT_SCHEMA}},
    )
    assert body.status_code == 200, body.text
    assert upstream.requests[-1]["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
    }


def test_user_is_metadata_user_id(routed, auth, upstream) -> None:
    assert _anthropic(routed, auth, user="storyboard-7").status_code == 200
    assert upstream.requests[-1]["metadata"] == {"user_id": "storyboard-7"}


def test_a_chat_template_argument_other_than_thinking_is_refused(
    routed, auth, upstream
) -> None:
    error = _refused(
        _anthropic(routed, auth, chat_template_kwargs={"enable_thinking": True, "x": 1}),
        "upstream_field_unsupported",
    )
    assert error["details"]["fields"] == ["chat_template_kwargs.x"]


def test_a_completion_counts_its_usage_with_the_cache(routed, auth, upstream) -> None:
    document = _anthropic(routed, auth).json()
    assert document["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert upstreams.anthropic_usage(
        {"input_tokens": 5, "cache_creation_input_tokens": 100,
         "cache_read_input_tokens": 900, "output_tokens": 3}
    ) == {
        "prompt_tokens": 1005,
        "completion_tokens": 3,
        "total_tokens": 1008,
        "prompt_tokens_details": {"cached_tokens": 900},
    }


@pytest.mark.parametrize(
    "reason, finish",
    [
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "content_filter"),
        ("stop_sequence", "stop"),
        ("pause_turn", "pause_turn"),
    ],
)
def test_a_stop_reason_is_openais_word_or_anthropics_own(
    routed, auth, upstream, reason, finish
) -> None:
    upstream.stop_reason = reason
    assert _anthropic(routed, auth).json()["choices"][0]["finish_reason"] == finish
    frames, _ = _stream(
        routed, auth,
        {"model": "anthropic/claude-sonnet-5",
         "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    finishes = [
        p["choices"][0]["finish_reason"] for p in _payloads(frames) if p["choices"]
    ]
    assert finishes[-1] == finish


def test_a_stream_asked_for_usage_ends_with_a_usage_chunk(routed, auth, upstream) -> None:
    frames, _ = _stream(
        routed,
        auth,
        {
            "model": "anthropic/claude-opus-5-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert frames[-1] == "data: [DONE]"
    payloads = _payloads(frames)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    with_choices = [p for p in payloads if p["choices"]]
    assert "".join(
        p["choices"][0]["delta"].get("content", "") for p in with_choices
    ) == ANSWER
    assert with_choices[-1]["choices"][0]["finish_reason"] == "stop"
    usage = [p for p in payloads if not p["choices"]]
    assert len(usage) == 1
    assert payloads[-1] is usage[0]
    assert usage[0]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert {p["model"] for p in payloads} == {"anthropic/claude-opus-5-5"}


def test_a_stream_not_asked_for_usage_carries_none(routed, auth, upstream) -> None:
    frames, _ = _stream(
        routed,
        auth,
        {
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert all(p["choices"] for p in _payloads(frames))
    assert "usage" not in "".join(frames)


def test_a_stream_option_this_server_does_not_read_is_refused(
    routed, auth, upstream
) -> None:
    error = _refused(
        _anthropic(routed, auth, stream=True,
                   stream_options={"include_usage": True, "include_obfuscation": False}),
        "upstream_field_unsupported",
    )
    assert error["details"]["fields"] == ["stream_options.include_obfuscation"]


def test_an_anthropic_stream_cut_before_message_stop_gets_no_done(
    routed, auth, upstream
) -> None:
    upstream.truncate_stream = True
    frames, _ = _stream(
        routed, auth,
        {"model": "anthropic/claude-sonnet-5",
         "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert "data: [DONE]" not in frames
    error = json.loads(frames[-1][len("data: "):])["error"]
    assert "truncated" in error["message"]
    assert error["upstream"] == "anthropic"


def test_an_anthropic_stream_error_is_relayed_and_gets_no_done(
    routed, auth, upstream
) -> None:
    upstream.stream_error = {"type": "overloaded_error", "message": "Overloaded"}
    frames, _ = _stream(
        routed, auth,
        {"model": "anthropic/claude-sonnet-5",
         "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert "data: [DONE]" not in frames
    error = json.loads(frames[-1][len("data: "):])["error"]
    assert error == {"type": "overloaded_error", "message": "Overloaded",
                     "upstream": "anthropic"}


def test_the_stream_translator_skips_pings_thinking_and_unknown_events() -> None:
    translator = upstreams.AnthropicStreamTranslator("anthropic/x", include_usage=False)
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3}}},
        {"type": "ping"},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}},
        {"type": "some_future_event", "x": 1},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    raw = b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )
    out = b"".join(translator.feed(raw[:37])) + b"".join(translator.feed(raw[37:]))
    out += b"".join(translator.finish())
    lines = [line for line in out.decode().split("\n") if line]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert [c["choices"][0]["delta"] for c in chunks] == [
        {"role": "assistant"}, {"content": "ok"}, {}
    ]


def test_a_billed_request_is_never_sent_twice_once_it_may_have_arrived(
    routed, auth, upstream, monkeypatch
) -> None:
    from crucible.api import proxy

    monkeypatch.setattr(proxy, "WIRE_BACKOFF_SECONDS", (0.0, 0.0, 0.0, 0.0))
    upstream.drop_connection = True
    for model in ("anthropic/claude-sonnet-5", "openai/gpt-5"):
        before = len(upstream.requests)
        body = chat(routed, auth, {"model": model,
                                   "messages": [{"role": "user", "content": "hi"}]})
        assert body.status_code == 502, body.text
        error = body.json()["error"]
        assert error["code"] == "upstream_unreachable"
        assert "did not send it again" in error["message"]
        assert len(upstream.requests) == before + 1, model


def test_a_billed_request_that_never_connected_is_tried_again_and_says_so(
    make_client, auth, upstream, monkeypatch
) -> None:
    from crucible.api import proxy

    monkeypatch.setattr(proxy, "WIRE_BACKOFF_SECONDS", (0.0, 0.0, 0.0, 0.0))
    dead = upstream.url
    upstream.stop()
    monkeypatch.setattr(upstreams, "ANTHROPIC_BASE", dead)
    attempts: list[str] = []
    real = proxy.sent_before_any_byte

    async def counted(attempt, *, where):
        async def once():
            attempts.append(where)
            return await attempt()

        return await real(once, where=where)

    monkeypatch.setattr("crucible.api.upstream.sent_before_any_byte", counted)
    with make_client(enable_llm=True, capability=decided()) as instance:
        instance.put("/v1/settings", headers=auth,
                     json={"upstreams": {"anthropic": {"key": ANTHROPIC_KEY}}})
        body = chat(instance, auth, {"model": "anthropic/claude-sonnet-5",
                                     "messages": [{"role": "user", "content": "hi"}]})
    assert body.status_code == 502
    error = body.json()["error"]
    assert error["code"] == "upstream_unreachable"
    assert "no byte of the request reached it" in error["message"]
    assert len(attempts) == proxy.WIRE_ATTEMPTS


def test_the_rate_limit_message_says_what_this_server_does(routed, auth, upstream) -> None:
    upstream.status = 429
    upstream.error_body = {"error": {"message": "slow down"}}
    body = _anthropic(routed, auth)
    assert body.status_code == 429
    assert "never resends a billed request once it may have reached" in (
        body.json()["error"]["message"]
    )


# ---- Tool calling through the `anthropic` and `ollama` upstreams: OpenAI's form
# translated both ways. Crucible stores no tools and keeps nothing between rounds.

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": WEATHER_SCHEMA,
        },
    },
    {"type": "function", "function": {"name": "get_time"}},
]

USES = [
    {"id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    {"id": "toolu_2", "name": "get_time", "input": {}},
]

# The two-round conversation an app sends back: its question, the assistant's two
# parallel calls (with what it said first), and both results.
ROUND_TWO = [
    {"role": "system", "content": "Use the tools."},
    {"role": "user", "content": "Weather and time in Paris?"},
    {
        "role": "assistant",
        "content": "Checking both.",
        "tool_calls": [
            {"id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"id": "toolu_2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ],
    },
    {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny, 21C"},
    {"role": "tool", "tool_call_id": "toolu_2",
     "content": [{"type": "text", "text": "14:05"}]},
]


def _anthropic_tools(routed, auth, model: str = "claude-sonnet-5", **extra: Any):
    return chat(
        routed, auth,
        {"model": f"anthropic/{model}",
         "messages": [{"role": "user", "content": "Weather in Paris?"}],
         "tools": TOOLS, **extra},
    )


def _ollama_tools(routed, auth, **extra: Any):
    return chat(
        routed, auth,
        {"model": "ollama/qwen3.5:9b",
         "messages": [{"role": "user", "content": "Weather in Paris?"}],
         "tools": TOOLS, **extra},
    )


def _assembled(frames: list[str]) -> tuple[dict[int, dict[str, Any]], str | None, str]:
    """OpenAI stream deltas put back together the way a client does: tool calls by
    index (arguments concatenated), the finish reason, the text."""
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    text = ""
    for payload in _payloads(frames):
        if not payload["choices"]:
            continue
        choice = payload["choices"][0]
        finish = choice["finish_reason"] or finish
        delta = choice["delta"]
        text += delta.get("content") or ""
        for part in delta.get("tool_calls", []):
            call = calls.setdefault(part["index"], {"arguments": ""})
            for key in ("id", "type"):
                if key in part:
                    call[key] = part[key]
            function = part.get("function", {})
            if "name" in function:
                call["name"] = function["name"]
            call["arguments"] += function.get("arguments", "")
    return calls, finish, text


# ---- anthropic: the request


def test_tools_become_messages_api_tools(routed, auth, upstream) -> None:
    strict = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "strict": True}}]
    body = _anthropic_tools(routed, auth, tools=strict + TOOLS[1:], tool_choice="auto")
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["tools"] == [
        {"name": "get_weather", "description": "The weather in a city.",
         "input_schema": WEATHER_SCHEMA, "strict": True},
        {"name": "get_time", "input_schema": {"type": "object", "properties": {}}},
    ]
    assert sent["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize(
    "choice, sent",
    [
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "get_time"}},
         {"type": "tool", "name": "get_time"}),
    ],
)
def test_tool_choice_is_the_messages_apis_own(routed, auth, upstream, choice, sent) -> None:
    body = _anthropic_tools(routed, auth, tool_choice=choice)
    assert body.status_code == 200, body.text
    assert upstream.requests[-1]["tool_choice"] == sent


def test_no_tool_choice_sends_none(routed, auth, upstream) -> None:
    assert _anthropic_tools(routed, auth).status_code == 200
    assert "tool_choice" not in upstream.requests[-1]


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5"]
)
@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "get_weather"}}]
)
def test_a_forced_tool_choice_on_a_model_without_one_is_refused_by_name(
    routed, auth, upstream, model, choice
) -> None:
    error = _refused(_anthropic_tools(routed, auth, model, tool_choice=choice),
                     "upstream_param_unsupported")
    assert error["details"]["model"] == model
    assert error["details"]["rule"] == model
    assert error["details"]["param"].startswith("tool_choice")
    assert upstream.requests == []
    # The same model takes `auto`, which the fake (as Anthropic) accepts.
    assert _anthropic_tools(routed, auth, model, tool_choice="auto").status_code == 200


def test_the_second_round_carries_the_calls_and_one_turn_of_results(
    routed, auth, upstream
) -> None:
    body = chat(routed, auth, {"model": "anthropic/claude-sonnet-5",
                               "messages": ROUND_TWO, "tools": TOOLS})
    # The fake refuses, as the Messages API does, a tool_use not answered in the very
    # next turn: results split over two turns would be a 502 here.
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]
    assert sent["system"] == "Use the tools."
    assert sent["messages"] == [
        {"role": "user", "content": "Weather and time in Paris?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking both."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Paris"}},
                {"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "Sunny, 21C"},
                {"type": "tool_result", "tool_use_id": "toolu_2",
                 "content": [{"type": "text", "text": "14:05"}]},
            ],
        },
    ]


def test_a_call_with_no_text_is_tool_use_blocks_alone(routed, auth, upstream) -> None:
    messages = [dict(m) for m in ROUND_TWO[1:4]]
    messages[1]["content"] = None
    messages[1]["tool_calls"] = messages[1]["tool_calls"][:1]
    messages.append({"role": "user", "content": "And tomorrow?"})
    body = chat(routed, auth, {"model": "anthropic/claude-sonnet-5",
                               "messages": messages, "tools": TOOLS})
    assert body.status_code == 200, body.text
    sent = upstream.requests[-1]["messages"]
    assert sent[1]["content"] == [{"type": "tool_use", "id": "toolu_1",
                                   "name": "get_weather", "input": {"city": "Paris"}}]
    # The user's next words stay their own turn after the results.
    assert sent[2]["content"][0]["type"] == "tool_result"
    assert sent[3] == {"role": "user", "content": "And tomorrow?"}


# ---- anthropic: the answer


@pytest.mark.parametrize("preamble", [None, "Let me look."])
def test_tool_use_blocks_come_back_as_tool_calls(routed, auth, upstream, preamble) -> None:
    upstream.anthropic_tool_uses = USES
    upstream.tool_preamble = preamble
    body = _anthropic_tools(routed, auth)
    assert body.status_code == 200, body.text
    choice = body.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == preamble
    assert message["tool_calls"] == [
        {"id": "toolu_1", "type": "function",
         "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}},
        {"id": "toolu_2", "type": "function",
         "function": {"name": "get_time", "arguments": "{}"}},
    ]


def test_streamed_tool_use_arrives_as_tool_call_deltas(routed, auth, upstream) -> None:
    upstream.anthropic_tool_uses = USES
    upstream.tool_preamble = "Let me look."
    frames, _ = _stream(routed, auth, {
        "model": "anthropic/claude-sonnet-5",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS, "stream": True,
    })
    assert frames[-1] == "data: [DONE]"
    calls, finish, text = _assembled(frames)
    assert finish == "tool_calls"
    assert text == "Let me look."
    assert sorted(calls) == [0, 1]
    assert [(c["id"], c["type"], c["name"]) for c in calls.values()] == [
        ("toolu_1", "function", "get_weather"), ("toolu_2", "function", "get_time")
    ]
    assert json.loads(calls[0]["arguments"]) == {"city": "Paris"}
    # A call with no arguments streams none; it still arrives as a JSON object.
    assert calls[1]["arguments"] == "{}"
    # The arguments arrived in pieces, as Anthropic sent them, not buffered whole.
    pieces = [
        part for p in _payloads(frames) if p["choices"]
        for part in p["choices"][0]["delta"].get("tool_calls", [])
        if part["index"] == 0 and part.get("function", {}).get("arguments")
    ]
    assert len(pieces) == 2


# ---- anthropic: what stays refused


@pytest.mark.parametrize(
    "messages, code, field",
    [
        (ROUND_TWO[:3] + [{**ROUND_TWO[3], "name": "get_weather"}] + ROUND_TWO[4:],
         "upstream_field_unsupported", "messages[3].name"),
        (ROUND_TWO[:3] + [{**ROUND_TWO[3], "tool_call_id": "toolu_9"}] + ROUND_TWO[4:],
         "invalid_request", "messages[3].tool_call_id"),
        ([ROUND_TWO[1], {"role": "tool", "tool_call_id": "toolu_1", "content": "x"}],
         "invalid_request", "messages[1].tool_call_id"),
        (ROUND_TWO[:2] + [{**ROUND_TWO[2], "tool_calls": [
            {"id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{city: Paris"}}]}],
         "invalid_request", "messages[2].tool_calls[0].function.arguments"),
        (ROUND_TWO[:2] + [{**ROUND_TWO[2], "tool_calls": [
            {"id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "[1]"}}]}],
         "invalid_request", "messages[2].tool_calls[0].function.arguments"),
        ([{"role": "user", "content": "hi", "tool_calls": ROUND_TWO[2]["tool_calls"]}],
         "invalid_request", "messages[0].tool_calls"),
    ],
)
def test_a_tool_message_this_server_cannot_carry_is_refused(
    routed, auth, upstream, messages, code, field
) -> None:
    body = chat(routed, auth, {"model": "anthropic/claude-sonnet-5",
                               "messages": messages, "tools": TOOLS})
    error = _refused(body, code)
    if code == "upstream_field_unsupported":
        assert error["details"]["fields"] == [field]
    else:
        assert error["details"]["field"] == field
    assert upstream.requests == []


@pytest.mark.parametrize(
    "extra, code, field",
    [
        ({"tools": [{"type": "custom", "custom": {"name": "f"}}]},
         "upstream_field_unsupported", "tools[0].type"),
        ({"tools": [{"type": "function", "function": {"name": "f", "examples": []}}]},
         "upstream_field_unsupported", "tools[0].function.examples"),
        ({"tools": [{"type": "function", "function": {"name": "no spaces"}}]},
         "invalid_request", "tools[0].function.name"),
        ({"tools": TOOLS, "tool_choice": "sometimes"}, "invalid_request", "tool_choice"),
        ({"tools": TOOLS, "tool_choice": {"type": "function",
                                          "function": {"name": "nope"}}},
         "invalid_request", "tool_choice.function.name"),
        ({"tools": [], "tool_choice": "required"}, "invalid_request", "tool_choice"),
    ],
)
def test_a_tool_or_tool_choice_this_server_cannot_carry_is_refused(
    routed, auth, upstream, extra, code, field
) -> None:
    body = chat(routed, auth, {"model": "anthropic/claude-sonnet-5",
                               "messages": [{"role": "user", "content": "hi"}], **extra})
    error = _refused(body, code)
    if code == "upstream_field_unsupported":
        assert error["details"]["fields"] == [field]
    else:
        assert error["details"]["field"] == field
    assert upstream.requests == []


# ---- ollama


@pytest.mark.parametrize("choice", [None, "auto"])
def test_ollama_gets_its_native_tools_and_tool_turns(routed, auth, upstream, choice) -> None:
    extra = {} if choice is None else {"tool_choice": choice}
    body = chat(routed, auth, {"model": "ollama/qwen3.5:9b", "messages": ROUND_TWO,
                               "tools": TOOLS, **extra})
    assert body.status_code == 200, body.text
    sent = _chats(upstream)[-1]
    assert "tool_choice" not in sent
    assert sent["tools"] == [
        {"type": "function", "function": {"name": "get_weather",
                                          "description": "The weather in a city.",
                                          "parameters": WEATHER_SCHEMA}},
        {"type": "function", "function": {"name": "get_time",
                                          "parameters": {"type": "object",
                                                         "properties": {}}}},
    ]
    assert sent["messages"] == [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Weather and time in Paris?"},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
            {"function": {"name": "get_time", "arguments": {}}},
        ]},
        {"role": "tool", "content": "Sunny, 21C", "tool_name": "get_weather"},
        {"role": "tool", "content": "14:05", "tool_name": "get_time"},
    ]


@pytest.mark.parametrize(
    "choice", ["none", "required", {"type": "function", "function": {"name": "get_time"}}]
)
def test_an_ollama_tool_choice_other_than_auto_is_refused_by_name(
    routed, auth, upstream, choice
) -> None:
    error = _refused(_ollama_tools(routed, auth, tool_choice=choice), "upstream_param_unsupported")
    assert error["details"]["upstream"] == "ollama"
    assert error["details"]["param"].startswith("tool_choice")
    assert _chats(upstream) == []


def test_a_strict_tool_is_refused_by_ollama(routed, auth, upstream) -> None:
    strict = [{**TOOLS[0], "function": {**TOOLS[0]["function"], "strict": True}}]
    error = _refused(_ollama_tools(routed, auth, tools=strict), "upstream_field_unsupported")
    assert error["details"]["fields"] == ["tools[0].function.strict"]
    assert _chats(upstream) == []


OLLAMA_CALLS = [
    {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
    {"id": "call_abc", "function": {"index": 1, "name": "get_time", "arguments": {}}},
]


def test_ollamas_tool_calls_come_back_as_openai_tool_calls(routed, auth, upstream) -> None:
    upstream.ollama_tool_calls = OLLAMA_CALLS
    body = _ollama_tools(routed, auth)
    assert body.status_code == 200, body.text
    choice = body.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"] == [
        {"id": "call_0", "type": "function",
         "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}},
        {"id": "call_abc", "type": "function",
         "function": {"name": "get_time", "arguments": "{}"}},
    ]


def test_ollamas_streamed_tool_calls_arrive_as_deltas(routed, auth, upstream) -> None:
    upstream.ollama_tool_calls = OLLAMA_CALLS
    frames, _ = _stream(routed, auth, {
        "model": "ollama/qwen3.5:9b",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS, "stream": True,
    })
    assert frames[-1] == "data: [DONE]"
    calls, finish, _ = _assembled(frames)
    assert finish == "tool_calls"
    assert calls == {
        0: {"id": "call_0", "type": "function", "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"})},
        1: {"id": "call_abc", "type": "function", "name": "get_time", "arguments": "{}"},
    }
