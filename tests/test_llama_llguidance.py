"""llama-server on cuda-linux is sent a JSON schema as an llguidance grammar: Crucible's
build has llguidance (scripts/build-llama-server-linux.sh), and a response_format sent
as it is would be compiled by llama.cpp's own GBNF converter inside the chat template's
parser (crucible/structured.py, with_llguidance_grammar; docs/internals/
engines-and-capability.md, "Structured output"). What llguidance does with the grammar
is the binary's, checked on a card; this is what the door sends and refuses."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import llamacpp, weights
from crucible.api.routes.openai import as_sent, refuse_an_unenforced_constraint
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.manifests import load_manifest
from crucible.structured import (
    LLGUIDANCE_PREFIX,
    llguidance_grammar,
    refuse_unbuilt_llguidance_grammar,
    with_llguidance_grammar,
)

from .fake_engine import FakeEngine
from .fake_hub import FakeHub
from .test_llama_cuda_linux import _placed
from .test_llm_api import (
    engines,  # noqa: F401 - a fixture this module uses
    fake_env,  # noqa: F401 - a fixture this module uses
    idle_card,  # noqa: F401 - a fixture this module uses
    llm_client,  # noqa: F401 - a fixture this module uses
    run_job,
)

LLAMA = "llama-server"
LINUX = "cuda-linux"
WINDOWS = "llama-windows"
MODEL = "qwen3.5-4b-bside"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"title": {"type": "string", "maxLength": 40}},
    "required": ["title"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "album", "schema": SCHEMA, "strict": True},
}
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
USER = [{"role": "user", "content": "[album title]\nname it"}]


def refused(call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value


def grammar_schema(grammar: str) -> Any:
    head = f"{LLGUIDANCE_PREFIX} {{}}\nstart: %json "
    assert grammar.startswith(head), grammar
    return json.loads(grammar[len(head):])


# --- the grammar --------------------------------------------------------------------


def test_the_grammar_is_the_form_llama_cpp_writes_itself() -> None:
    """b10970 common/json-schema-to-grammar.cpp L993-996 with llguidance built in:
    "%llguidance {}\\nstart: %json " + schema.dump(), which is compact JSON."""
    assert llguidance_grammar(SCHEMA) == (
        "%llguidance {}\nstart: %json " + json.dumps(SCHEMA, separators=(",", ":"))
    )
    assert " " not in llguidance_grammar(SCHEMA).split("%json ", 1)[1]


def test_a_json_schema_goes_as_the_grammar_and_response_format_is_taken_off() -> None:
    body = {"messages": USER, "response_format": JSON_SCHEMA, "temperature": 0.7}
    before = copy.deepcopy(body)
    sent = with_llguidance_grammar(body)
    assert body == before
    assert "response_format" not in sent
    assert grammar_schema(sent["grammar"]) == SCHEMA
    assert sent["temperature"] == 0.7 and sent["messages"] == USER


def test_a_json_object_goes_as_the_object_schema() -> None:
    sent = with_llguidance_grammar({"response_format": {"type": "json_object"}})
    assert grammar_schema(sent["grammar"]) == {"type": "object"}


def test_a_json_object_with_llama_server_s_own_schema_member_keeps_it() -> None:
    """server-common.cpp L1189-1192: json_object carries a schema on llama-server."""
    sent = with_llguidance_grammar({"response_format": {"type": "json_object", "schema": SCHEMA}})
    assert grammar_schema(sent["grammar"]) == SCHEMA


def test_llama_server_s_own_json_schema_field_goes_as_the_grammar_too() -> None:
    sent = with_llguidance_grammar({"json_schema": SCHEMA})
    assert "json_schema" not in sent
    assert grammar_schema(sent["grammar"]) == SCHEMA


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"response_format": {"type": "text"}},
        {"grammar": 'root ::= "a"'},
        {"grammar": "%llguidance {}\nstart: /a+/"},
    ],
)
def test_a_body_without_a_json_schema_is_sent_as_it_is(body: dict[str, Any]) -> None:
    assert with_llguidance_grammar(body) is body


def test_compact_json_reaches_the_grammar_as_llguidance_s_own_option() -> None:
    sent = as_sent({"response_format": JSON_SCHEMA}, "compact", True)
    assert grammar_schema(sent["grammar"]) == {
        **SCHEMA,
        "x-guidance": {"whitespace_flexible": False},
    }
    sent = as_sent({"response_format": {"type": "json_object"}}, "compact", True)
    assert grammar_schema(sent["grammar"]) == {
        "type": "object",
        "x-guidance": {"whitespace_flexible": False},
    }


def test_flexible_is_the_schema_as_it_came() -> None:
    """llguidance's JSON default is whitespace_flexible true, vLLM's too."""
    for mode in (None, "flexible"):
        sent = as_sent({"response_format": JSON_SCHEMA}, mode, True)
        assert grammar_schema(sent["grammar"]) == SCHEMA


def test_a_build_without_llguidance_is_sent_the_response_format() -> None:
    body = {"response_format": JSON_SCHEMA}
    assert as_sent(body, None, False) is body


@pytest.mark.parametrize(
    ("body", "fields"),
    [
        ({"response_format": JSON_SCHEMA, "json_schema": SCHEMA}, ["response_format", "json_schema"]),
        ({"response_format": JSON_SCHEMA, "grammar": 'root ::= "a"'}, ["grammar", "response_format"]),
        ({"json_schema": SCHEMA, "grammar": 'root ::= "a"'}, ["grammar", "json_schema"]),
    ],
)
def test_a_schema_stated_twice_or_beside_a_grammar_is_refused(
    body: dict[str, Any], fields: list[str]
) -> None:
    error = refused(lambda: with_llguidance_grammar(body))
    assert error.status_code == 400 and error.code == "invalid_request"
    assert error.details == {"fields": fields}


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_schema", "json_schema": {"name": "v"}},
        {"type": "json_schema", "json_schema": {"name": "v", "schema": True}},
        {"type": "json_schema", "json_schema": "nope"},
    ],
)
def test_a_schema_that_is_not_an_object_is_refused(response_format: dict[str, Any]) -> None:
    error = refused(lambda: with_llguidance_grammar({"response_format": response_format}))
    assert error.code == "invalid_response_format"


# --- what the door refuses -----------------------------------------------------------


def door(backend: str, body: dict[str, Any], json_whitespace: str | None = None) -> bool:
    return refuse_an_unenforced_constraint(LLAMA, backend, MODEL, body, json_whitespace)


def test_the_door_routes_cuda_linux_through_llguidance_and_windows_not() -> None:
    body = {"response_format": JSON_SCHEMA, **THINKING_OFF}
    assert door(LINUX, body) is True
    assert door(WINDOWS, body) is False


@pytest.mark.parametrize("thinking", [None, True])
def test_a_json_schema_with_thinking_not_stated_off_is_refused_on_the_grammar_route(
    thinking: bool | None,
) -> None:
    body: dict[str, Any] = {"response_format": JSON_SCHEMA}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    error = refused(lambda: door(LINUX, body, "compact"))
    assert error.status_code == 400 and error.code == "structured_output_with_thinking"
    assert error.details == {"model": MODEL, "engine": LLAMA, "thinking": thinking}
    assert door(WINDOWS, body) is False, "the GBNF route admits a reasoning block"


def test_a_grammar_without_a_schema_is_not_held_to_thinking() -> None:
    assert door(LINUX, {"grammar": "%llguidance {}\nstart: /a+/"}) is True


def test_the_door_refuses_what_the_grammar_cannot_be_written_from_before_sending() -> None:
    body = {"response_format": JSON_SCHEMA, "grammar": 'root ::= "a"', **THINKING_OFF}
    assert refused(lambda: door(LINUX, body)).code == "invalid_request"


def test_an_llguidance_grammar_is_refused_where_the_build_would_abort_on_it() -> None:
    body = {"grammar": "%llguidance {}\nstart: /a+/"}
    error = refused(lambda: door(WINDOWS, body))
    assert error.status_code == 400 and error.code == "structured_output_not_served"
    assert error.details == {"model": MODEL, "engine": LLAMA, "fields": ["grammar"]}
    assert "abort" in error.message
    refuse_unbuilt_llguidance_grammar(engine=LLAMA, model_id=MODEL, built=True, body=body)
    door(WINDOWS, {"grammar": 'root ::= "a"'})


# --- the door, end to end ---------------------------------------------------------------


def test_thinking_on_is_refused_before_anything_loads(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": USER,
            "response_format": JSON_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "structured_output_with_thinking"
    assert engines == []


@pytest.fixture
def bside_resident(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    home: Path,
    fake_env: Path,  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> FakeEngine:
    lib = fake_env / "lib" / "python3.11" / "site-packages" / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    for name in llamacpp.CUDA_LINUX_LIBRARIES:
        (lib / name).write_bytes(b"\x7fELF")
    _placed(home, monkeypatch)
    hub = FakeHub(chunks=1)
    monkeypatch.setattr("huggingface_hub.snapshot_download", hub.snapshot_download, raising=False)
    manifest = load_manifest(MODEL)
    for form in ("bf16", "q8_0"):
        weights.pull(load_config(home), manifest, manifest.spec(LINUX, form))
    run_job(llm_client, auth, type="load-model", model=MODEL)
    return engines[0]


def test_the_engine_is_sent_the_grammar_and_no_response_format(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    bside_resident: FakeEngine,
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": USER,
            "response_format": JSON_SCHEMA,
            "json_whitespace": "compact",
        },
    )
    assert response.status_code == 200, response.json()
    sent = bside_resident.last_request
    assert sent is not None
    assert "response_format" not in sent and "json_whitespace" not in sent
    assert grammar_schema(sent["grammar"]) == {
        **SCHEMA,
        "x-guidance": {"whitespace_flexible": False},
    }
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}, (
        "B-Sides' manifest states thinking off"
    )


def test_a_plain_chat_is_sent_as_it_came(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    bside_resident: FakeEngine,
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER},
    )
    assert response.status_code == 200, response.json()
    sent = bside_resident.last_request
    assert sent is not None and "grammar" not in sent
