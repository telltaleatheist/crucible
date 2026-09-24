"""Per-model `[defaults]`: what a manifest may state, and who wins.

PHASE2-LLM.md section 9. Three sources and one rule — **the request wins, the
manifest fills gaps, and what neither states is the engine's** — checked here
both as arithmetic (`crucible/sampling.py` in isolation) and through the real
chat door with the fake engine, so the thing asserted is what the engine
actually received.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.manifests import (
    NO_DEFAULTS,
    DEFAULTS_KEYS,
    ManifestError,
    ModelDefaults,
    load_manifest,
    parse_manifest,
)
from crucible.errors import ApiError
from crucible.sampling import (
    SAMPLING_HEADER,
    SOURCE_ENGINE,
    SOURCE_MANIFEST,
    SOURCE_REQUEST,
    TEMPLATE_KWARGS,
    THINKING_KEY,
    apply_defaults,
)

from .conftest import FAKE_BACKEND
from .fake_engine import FakeEngine
from .test_llm_api import (  # noqa: F401 — fixtures used by name
    MODEL,
    engines,
    fake_env,
    fake_weights,
    idle_card,
    llm_client,
    run_job,
)

BASE = """
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text"]
"""

BACKEND = """
[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
"""


def parse(defaults: str = ""):
    text = BASE + defaults + BACKEND
    return parse_manifest(text, Path("demo-1b.toml"), "demo-1b")


# --------------------------------------------------- what the manifest takes


def test_a_manifest_with_no_defaults_table_states_none() -> None:
    """Absent is a statement, and it is not the same as an empty table."""
    assert parse().defaults == NO_DEFAULTS
    assert parse().defaults.stated() == {}


def test_every_key_is_read() -> None:
    manifest = parse(
        """
[defaults]
temperature = 0.6
top_p = 0.8
top_k = 20
max_tokens = 2048
repetition_penalty = 1.05
thinking = false
"""
    )
    assert manifest.defaults == ModelDefaults(
        temperature=0.6,
        top_p=0.8,
        top_k=20,
        max_tokens=2048,
        repetition_penalty=1.05,
        thinking=False,
    )


def test_an_integer_temperature_is_a_float_and_zero_is_allowed() -> None:
    """TOML reads `temperature = 0` as an int, and zero is what a page reader
    wants — refusing it on a type technicality would be the wrong answer."""
    defaults = parse("\n[defaults]\ntemperature = 0\n").defaults
    assert defaults.temperature == 0.0
    assert isinstance(defaults.temperature, float)


def test_an_unknown_key_is_refused_by_name() -> None:
    with pytest.raises(ManifestError) as caught:
        parse("\n[defaults]\ntemperture = 0.6\n")
    assert "temperture" in str(caught.value)


def test_an_empty_defaults_table_is_refused() -> None:
    """A model that states nothing says so by having no table at all."""
    with pytest.raises(ManifestError) as caught:
        parse("\n[defaults]\n")
    assert "empty" in str(caught.value)


@pytest.mark.parametrize(
    "line,fragment",
    [
        ("temperature = -0.1", "temperature"),
        ("top_p = 0", "top_p"),
        ("top_p = 1.5", "top_p"),
        ("top_k = 0", "top_k"),
        ("max_tokens = 0", "max_tokens"),
        ("repetition_penalty = 0", "repetition_penalty"),
    ],
)
def test_a_value_outside_the_engines_range_is_refused(
    line: str, fragment: str
) -> None:
    with pytest.raises(ManifestError) as caught:
        parse(f"\n[defaults]\n{line}\n")
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    "line,fragment",
    [
        ('temperature = "hot"', "must be a number"),
        ("top_k = 1.5", "must be int"),
        ("thinking = 0", "must be bool"),
        ("temperature = true", "must be a number"),
    ],
)
def test_a_wrong_type_is_refused(line: str, fragment: str) -> None:
    with pytest.raises(ManifestError) as caught:
        parse(f"\n[defaults]\n{line}\n")
    assert fragment in str(caught.value)


def test_the_row_carries_every_key_with_null_where_nothing_is_stated() -> None:
    """Keys that came and went would make "this model states none" and "this
    build predates the field" read the same."""
    row = parse("\n[defaults]\nthinking = false\n").defaults.to_dict()
    assert set(row) == set(DEFAULTS_KEYS)
    assert row["thinking"] is False
    assert row["temperature"] is None


def test_the_cleanup_model_ships_thinking_off() -> None:
    """`qwen3.5-9b` is what BookForge and Foundry both clean text with, and a
    bounded budget spent entirely on `reasoning` returns no content at all."""
    defaults = load_manifest("qwen3.5-9b").defaults
    assert defaults.thinking is False
    # And nothing else: temperature and the rest stay the client's for this
    # model, because both apps send their own per act.
    assert defaults.stated() == {"thinking": False}


# ------------------------------------------------------------ the precedence


def test_a_field_the_request_states_wins() -> None:
    applied = apply_defaults(
        {"temperature": 0.9}, ModelDefaults(temperature=0.2)
    )
    assert applied.body["temperature"] == 0.9
    assert applied.sources["temperature"] == SOURCE_REQUEST
    assert applied.changed is False


def test_a_field_the_request_omits_takes_the_manifests() -> None:
    applied = apply_defaults({}, ModelDefaults(temperature=0.2))
    assert applied.body["temperature"] == 0.2
    assert applied.sources["temperature"] == SOURCE_MANIFEST
    assert applied.changed is True


def test_a_field_neither_states_is_the_engines() -> None:
    applied = apply_defaults({}, NO_DEFAULTS)
    assert applied.body == {}
    assert set(applied.sources.values()) == {SOURCE_ENGINE}
    assert applied.changed is False


def test_an_explicit_null_is_a_stated_value() -> None:
    """A client that wrote the key made a decision. Treating null as an absence
    would answer a stated request with a different number than it asked for."""
    applied = apply_defaults(
        {"max_tokens": None}, ModelDefaults(max_tokens=2048)
    )
    assert applied.body["max_tokens"] is None
    assert applied.sources["max_tokens"] == SOURCE_REQUEST


def test_every_key_gets_a_source_every_time() -> None:
    applied = apply_defaults({"top_p": 0.5}, ModelDefaults(temperature=0.2))
    assert list(applied.sources) == list(DEFAULTS_KEYS)
    assert applied.sources == {
        "temperature": SOURCE_MANIFEST,
        "top_p": SOURCE_REQUEST,
        "top_k": SOURCE_ENGINE,
        "max_tokens": SOURCE_ENGINE,
        "repetition_penalty": SOURCE_ENGINE,
        "thinking": SOURCE_ENGINE,
    }


def test_thinking_becomes_chat_template_kwargs() -> None:
    applied = apply_defaults({}, ModelDefaults(thinking=False))
    assert applied.body[TEMPLATE_KWARGS] == {THINKING_KEY: False}
    assert applied.sources["thinking"] == SOURCE_MANIFEST


def test_thinking_is_merged_into_the_clients_own_template_kwargs() -> None:
    """The table is the client's; this server owns exactly one key inside it."""
    applied = apply_defaults(
        {TEMPLATE_KWARGS: {"add_generation_prompt": True}},
        ModelDefaults(thinking=False),
    )
    assert applied.body[TEMPLATE_KWARGS] == {
        "add_generation_prompt": True,
        THINKING_KEY: False,
    }


def test_a_request_that_states_thinking_keeps_it() -> None:
    applied = apply_defaults(
        {TEMPLATE_KWARGS: {THINKING_KEY: True}}, ModelDefaults(thinking=False)
    )
    assert applied.body[TEMPLATE_KWARGS] == {THINKING_KEY: True}
    assert applied.sources["thinking"] == SOURCE_REQUEST
    assert applied.changed is False


def test_template_kwargs_that_is_not_an_object_is_refused_by_name() -> None:
    with pytest.raises(ApiError) as caught:
        apply_defaults({TEMPLATE_KWARGS: "off"}, ModelDefaults(thinking=False))
    assert caught.value.code == "invalid_request"
    assert TEMPLATE_KWARGS in caught.value.message


def test_the_header_is_compact_json_in_a_stable_order() -> None:
    applied = apply_defaults({}, ModelDefaults(thinking=False))
    assert json.loads(applied.header())["thinking"] == SOURCE_MANIFEST
    assert " " not in applied.header()
    assert applied.from_manifest() == ["thinking"]


def test_the_body_object_is_untouched_when_nothing_applies() -> None:
    """`changed is False` is what lets the proxy forward the client's own bytes,
    which is what keeps a json_schema grammar a grammar nobody re-encoded."""
    body: dict[str, Any] = {"messages": []}
    applied = apply_defaults(body, NO_DEFAULTS)
    assert applied.body is body


# ---------------------------------------------------------- through the door


def load(client: TestClient, auth: dict[str, str]) -> None:
    run_job(client, auth, type="load-model", model=MODEL)


def test_the_manifest_default_reaches_the_engine(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    sent = engines[0].last_request
    assert sent[TEMPLATE_KWARGS] == {THINKING_KEY: False}
    assert json.loads(response.headers[SAMPLING_HEADER])["thinking"] == (
        SOURCE_MANIFEST
    )


def test_the_request_wins_over_the_manifest_at_the_door(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        TEMPLATE_KWARGS: {THINKING_KEY: True},
    }
    response = llm_client.post(
        "/v1/openai/chat/completions", headers=auth, json=body
    )
    assert response.status_code == 200
    assert engines[0].last_request[TEMPLATE_KWARGS] == {THINKING_KEY: True}
    assert json.loads(response.headers[SAMPLING_HEADER])["thinking"] == (
        SOURCE_REQUEST
    )
    # Nothing changed, so the client's own bytes went across untouched.
    assert json.loads(engines[0].last_request_bytes) == body


def test_a_knob_neither_states_is_reported_as_the_engines(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        },
    )
    sources = json.loads(response.headers[SAMPLING_HEADER])
    assert sources["temperature"] == SOURCE_REQUEST
    assert sources["top_k"] == SOURCE_ENGINE
    assert "top_k" not in engines[0].last_request


def test_a_streamed_completion_carries_the_same_header(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The door where a body field could not have gone."""
    fake_weights(MODEL)
    load(llm_client, auth)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        sources = json.loads(response.headers[SAMPLING_HEADER])
        text = "".join(response.iter_text())
    assert sources["thinking"] == SOURCE_MANIFEST
    assert text.endswith("data: [DONE]\n\n")
    assert engines[0].last_request[TEMPLATE_KWARGS] == {THINKING_KEY: False}


def test_the_models_row_advertises_the_defaults(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[MODEL]["defaults"]["thinking"] is False
    assert rows[MODEL]["defaults"]["temperature"] is None
    # A model that states none says so with every key null, never by omitting
    # the field.
    assert rows["qwen3.8-27b-4bit"]["defaults"] == {key: None for key in DEFAULTS_KEYS}


def test_the_openai_listing_advertises_them_too(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """Foundry reads this listing rather than `/v1/models`."""
    fake_weights(MODEL)
    load(llm_client, auth)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    assert entry["defaults"]["thinking"] is False


def test_a_manifest_edited_under_a_running_engine_does_not_change_the_answer(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`max_model_len`'s rule, applied to the defaults: the record the engine
    was loaded against is what the proxy applies."""
    fake_weights(MODEL)
    load(llm_client, auth)
    residency = llm_client.app.state.residency
    assert residency.resident_model.defaults == load_manifest(MODEL).defaults
    assert residency.resident_model.defaults.thinking is False
    assert residency.resident_model.to_dict()["defaults"]["thinking"] is False
