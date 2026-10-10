"""`"json_whitespace": "compact"`: a JSON answer with no whitespace between tokens, kept
by the engines that compile JSON with llguidance and refused by name everywhere else
(crucible/structured.py; docs/internals/engines-and-capability.md, "Structured
output"). What llguidance does with the option is tests/test_structured_mlx.py's."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.engines import ENGINES, structured_output_reading
from crucible.errors import ApiError
from crucible.structured import (
    JSON_WHITESPACE,
    refuse_unenforced_constraint,
    refuse_unkept_json_whitespace,
    take_json_whitespace,
    with_compact_json,
)

from .conftest import FAKE_MAC_BACKEND
from .fake_engine import FakeEngine
from .test_llm_api import (
    MODEL,
    engines,  # noqa: F401 - a fixture this module uses
    fake_env,  # noqa: F401 - a fixture this module uses
    fake_weights,  # noqa: F401 - a fixture this module uses
    idle_card,  # noqa: F401 - a fixture this module uses
    llm_client,  # noqa: F401 - a fixture this module uses
    run_job,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "v", "schema": SCHEMA, "strict": True},
}
COMPACT_SCHEMA = {**SCHEMA, "x-guidance": {"whitespace_flexible": False}}
USER = [{"role": "user", "content": "is it?"}]
# B-Sides' model: llama-server on cuda-linux, mlx-lm on the Mac.
LLAMA_MODEL = "qwen3.5-4b-bside"


def refused(call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value


def taken(body: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    sent = copy.deepcopy(body)
    return take_json_whitespace(sent), sent


# --- what each engine states -----------------------------------------------------


BACKENDS = ("cuda-linux", "mlx-darwin", "llama-windows")


def test_every_engine_states_whether_it_keeps_compact_json_and_where_it_read_it() -> None:
    for name in ENGINES:
        for backend in BACKENDS:
            reading = structured_output_reading(name, backend)
            assert reading.compact_json_basis.strip(), (name, backend)


def test_the_llguidance_builds_keep_it_and_the_others_do_not() -> None:
    def kept(backend: str) -> set[str]:
        return {
            name for name in ENGINES if structured_output_reading(name, backend).compact_json
        }

    assert kept("cuda-linux") == {"vllm", "mlx-lm", "llama-server"}
    assert kept("llama-windows") == {"vllm", "mlx-lm"}
    linux = structured_output_reading("llama-server", "cuda-linux").compact_json_basis
    assert "LLAMA_LLGUIDANCE=ON" in linux and "1.7.6" in linux
    windows = structured_output_reading("llama-server", "llama-windows").compact_json_basis
    assert "ggml-org" in windows and "json-schema-to-grammar.cpp" in windows
    mlx_vlm = structured_output_reading("mlx-vlm", "mlx-darwin").compact_json_basis
    assert "no structured output" in mlx_vlm


def test_only_llama_server_on_cuda_linux_is_sent_an_llguidance_grammar() -> None:
    sent = {
        (name, backend)
        for name in ENGINES
        for backend in BACKENDS
        if structured_output_reading(name, backend).llguidance_grammar
    }
    assert sent == {("llama-server", "cuda-linux")}


# --- the body ----------------------------------------------------------------------


def test_a_body_without_the_member_is_left_alone() -> None:
    assert taken({"response_format": JSON_SCHEMA}) == (None, {"response_format": JSON_SCHEMA})


@pytest.mark.parametrize("mode", ["compact", "flexible"])
def test_the_member_is_taken_off_the_body(mode: str) -> None:
    got, sent = taken({"response_format": JSON_SCHEMA, JSON_WHITESPACE: mode})
    assert got == mode and sent == {"response_format": JSON_SCHEMA}


@pytest.mark.parametrize("value", ["tight", "", None, True, ["compact"]])
def test_a_value_other_than_compact_or_flexible_is_refused(value: Any) -> None:
    error = refused(lambda: taken({"response_format": JSON_SCHEMA, JSON_WHITESPACE: value}))
    assert error.code == "invalid_request"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"response_format": {"type": "text"}},
        {"structured_outputs": {"regex": "[ab]"}},
        {"structured_outputs": {"choice": ["a", "b"]}},
        {"grammar": "root ::= \"a\""},
        {"structured_outputs": {"json_object": False, "regex": "a"}},
    ],
)
@pytest.mark.parametrize("mode", ["compact", "flexible"])
def test_it_is_refused_without_a_json_constraint(body: dict[str, Any], mode: str) -> None:
    error = refused(lambda: taken({**body, JSON_WHITESPACE: mode}))
    assert error.status_code == 400 and error.code == "json_whitespace_without_json"


@pytest.mark.parametrize(
    "body",
    [
        {"response_format": JSON_SCHEMA},
        {"response_format": {"type": "json_object"}},
        {"structured_outputs": {"json": SCHEMA}},
        {"structured_outputs": {"json": json.dumps(SCHEMA)}},
        {"structured_outputs": {"json_object": True}},
    ],
)
def test_every_json_constraint_takes_it(body: dict[str, Any]) -> None:
    assert taken({**body, JSON_WHITESPACE: "compact"})[0] == "compact"


@pytest.mark.parametrize("key", ["whitespace_flexible", "whitespace_pattern"])
def test_a_schema_that_states_the_whitespace_itself_is_a_conflict(key: str) -> None:
    schema = {**SCHEMA, "x-guidance": {key: False if key == "whitespace_flexible" else " ?"}}
    wrapper = {"name": "v", "schema": schema}
    body = {"response_format": {"type": "json_schema", "json_schema": wrapper}}
    for mode in ("compact", "flexible"):
        error = refused(lambda: taken({**body, JSON_WHITESPACE: mode}))
        assert error.code == "json_whitespace_conflict"
        assert error.details == {"fields": [f"x-guidance.{key}"]}


@pytest.mark.parametrize(
    "body",
    [
        {"response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": True}}},
        {"response_format": {"type": "json_schema", "json_schema": "nope"}},
        {"structured_outputs": {"json": "{not json"}},
    ],
)
def test_a_schema_it_cannot_be_written_into_is_refused(body: dict[str, Any]) -> None:
    error = refused(lambda: taken({**body, JSON_WHITESPACE: "compact"}))
    assert error.code == "invalid_request"


# --- what the engine is sent ---------------------------------------------------------


def test_a_schema_carries_the_option_and_the_callers_body_is_untouched() -> None:
    body = {"response_format": JSON_SCHEMA, "temperature": 0}
    before = copy.deepcopy(body)
    sent = with_compact_json(body)
    assert body == before
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "v", "schema": COMPACT_SCHEMA, "strict": True},
    }
    assert sent["temperature"] == 0


def test_other_x_guidance_options_are_kept() -> None:
    schema = {**SCHEMA, "x-guidance": {"lenient": True}}
    sent = with_compact_json(
        {"response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": schema}}}
    )
    assert sent["response_format"]["json_schema"]["schema"]["x-guidance"] == {
        "lenient": True,
        "whitespace_flexible": False,
    }


def test_a_json_object_goes_as_the_object_schema() -> None:
    sent = with_compact_json({"response_format": {"type": "json_object"}})
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "json_object",
            "schema": {"type": "object", "x-guidance": {"whitespace_flexible": False}},
        },
    }


@pytest.mark.parametrize("schema", [SCHEMA, json.dumps(SCHEMA)])
def test_structured_outputs_json_carries_the_option(schema: Any) -> None:
    sent = with_compact_json({"structured_outputs": {"json": schema}})
    assert sent["structured_outputs"] == {"json": COMPACT_SCHEMA}


def test_structured_outputs_json_object_goes_as_the_object_schema() -> None:
    sent = with_compact_json({"structured_outputs": {"json_object": True}})
    assert sent["structured_outputs"] == {
        "json": {"type": "object", "x-guidance": {"whitespace_flexible": False}}
    }


# --- the engine's refusal --------------------------------------------------------------


def unkept(engine: str, mode: str | None, backend: str = "cuda-linux") -> None:
    reading = structured_output_reading(engine, backend)
    refuse_unkept_json_whitespace(
        engine=engine,
        model_id=MODEL,
        compact=reading.compact_json,
        basis=reading.compact_json_basis,
        mode=mode,
    )


@pytest.mark.parametrize(
    ("engine", "backend"),
    [("vllm", "cuda-linux"), ("mlx-lm", "mlx-darwin"), ("llama-server", "cuda-linux")],
)
def test_compact_passes_where_it_is_kept(engine: str, backend: str) -> None:
    unkept(engine, "compact", backend)


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_flexible_and_absent_pass_everywhere(engine: str, backend: str) -> None:
    unkept(engine, "flexible", backend)
    unkept(engine, None, backend)


@pytest.mark.parametrize(
    ("engine", "backend"), [("llama-server", "llama-windows"), ("mlx-vlm", "mlx-darwin")]
)
def test_compact_is_refused_by_name_where_it_is_not_kept(engine: str, backend: str) -> None:
    error = refused(lambda: unkept(engine, "compact", backend))
    assert error.status_code == 400 and error.code == "json_whitespace_not_served"
    assert error.details == {"model": MODEL, "engine": engine}


@pytest.mark.parametrize(
    "option", [{"disable_any_whitespace": True}, {"whitespace_pattern": " "},
               {"disable_additional_properties": True}]
)
def test_a_structured_outputs_option_vllm_does_not_read_per_request_is_refused(
    option: dict[str, Any],
) -> None:
    reading = structured_output_reading("vllm", "cuda-linux")
    body = {"structured_outputs": {"json": SCHEMA, **option}}
    error = refused(
        lambda: refuse_unenforced_constraint(
            engine="vllm", model_id=MODEL, formats=reading.formats, fields=reading.fields,
            basis=reading.basis, body=body,
        )
    )
    assert error.code == "structured_output_not_served"
    assert error.details == {"fields": [f"structured_outputs.{next(iter(option))}"]}
    refuse_unenforced_constraint(
        engine="vllm", model_id=MODEL, formats=reading.formats, fields=reading.fields,
        basis=reading.basis,
        body={"structured_outputs": {"json": SCHEMA, "disable_any_whitespace": False}},
    )


# --- the door ----------------------------------------------------------------------


def load(client: TestClient, auth: dict[str, str]) -> None:
    run_job(client, auth, type="load-model", model=MODEL)


def chat(client: TestClient, auth: dict[str, str], **extra: Any) -> Any:
    return client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER, **extra},
    )


def test_the_door_sends_vllm_the_schema_with_the_option(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = chat(llm_client, auth, response_format=JSON_SCHEMA, json_whitespace="compact")
    assert response.status_code == 200, response.json()
    sent = engines[0].last_request
    assert JSON_WHITESPACE not in sent
    assert sent["response_format"]["json_schema"]["schema"] == COMPACT_SCHEMA


def test_the_door_sends_flexible_unchanged(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = chat(llm_client, auth, response_format=JSON_SCHEMA, json_whitespace="flexible")
    assert response.status_code == 200, response.json()
    sent = engines[0].last_request
    assert JSON_WHITESPACE not in sent and sent["response_format"] == JSON_SCHEMA


def test_the_door_refuses_compact_without_json_and_sends_nothing(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    before = engines[0].last_request
    response = chat(llm_client, auth, json_whitespace="compact")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "json_whitespace_without_json"
    assert engines[0].last_request == before


def test_compact_on_llama_server_on_cuda_linux_passes_the_door(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    """Not refused for the engine: B-Sides' model takes thinking off from its manifest,
    so the chat goes on to the door's next question, which with "queue": false and
    nothing resident is the model's residency."""
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": LLAMA_MODEL,
            "messages": USER,
            "response_format": JSON_SCHEMA,
            JSON_WHITESPACE: "compact",
            "queue": False,
        },
    )
    assert response.status_code != 200
    assert response.json()["error"]["code"] not in (
        "json_whitespace_not_served",
        "json_whitespace_without_json",
        "structured_output_with_thinking",
        "structured_output_not_served",
    )
    assert engines == []


def test_compact_on_the_mac_passes_the_door_for_mlx_lm(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
) -> None:
    """Not refused for the engine: the chat goes on to the door's next question, which
    with "queue": false and nothing resident is the model's residency."""
    with make_client(backend=FAKE_MAC_BACKEND, enable_llm=True) as client:
        response = client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={
                "model": LLAMA_MODEL,
                "messages": USER,
                "response_format": JSON_SCHEMA,
                JSON_WHITESPACE: "compact",
                "queue": False,
            },
        )
    assert response.status_code != 200
    assert response.json()["error"]["code"] not in (
        "json_whitespace_not_served",
        "json_whitespace_without_json",
    )


def test_an_upstream_model_is_refused_it(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": "anthropic/claude-x",
            "messages": USER,
            "response_format": JSON_SCHEMA,
            JSON_WHITESPACE: "compact",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "json_whitespace_not_served"
