from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.engines import ENGINES, chat_prefill_reading
from crucible.errors import ApiError
from crucible.prefill import (
    GRAMMAR_FIELDS,
    PREFILL_KEY,
    refuse_unkeepable_prefill,
    take_prefill,
    with_prefill,
)
from crucible.sampling import TEMPLATE_KWARGS, THINKING_KEY

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

USER = [{"role": "system", "content": "the rules"}, {"role": "user", "content": "a sentence"}]
OPENING = '{"edits": ['


def body(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": list(USER), PREFILL_KEY: OPENING, **extra}


def refused(call: Callable[[], Any]) -> ApiError:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value


# --- the body ----------------------------------------------------------------


def test_a_body_without_a_prefill_is_left_alone() -> None:
    sent = {"model": MODEL, "messages": list(USER)}
    assert take_prefill(sent) is None
    assert sent == {"model": MODEL, "messages": USER}


def test_the_prefill_is_taken_out_of_the_body() -> None:
    sent = body()
    assert take_prefill(sent) == OPENING
    assert PREFILL_KEY not in sent


@pytest.mark.parametrize("value", ["", 3, None, ["{"]])
def test_a_prefill_that_is_not_a_non_empty_string_is_refused(value: Any) -> None:
    error = refused(lambda: take_prefill(body(**{PREFILL_KEY: value})))
    assert error.status_code == 400 and error.code == "invalid_request"


@pytest.mark.parametrize("value", [' {"a": 1', '{"answer": ', "Answer:\n"])
def test_a_prefill_the_template_would_trim_is_refused(value: str) -> None:
    error = refused(lambda: take_prefill(body(**{PREFILL_KEY: value})))
    assert error.code == "invalid_request"
    assert "whitespace" in error.message


def test_a_prefill_with_no_messages_is_refused() -> None:
    error = refused(lambda: take_prefill(body(messages=[])))
    assert error.code == "invalid_request"


def test_a_prefill_and_a_final_assistant_message_are_refused_together() -> None:
    messages = [*USER, {"role": "assistant", "content": "{"}]
    error = refused(lambda: take_prefill(body(messages=messages)))
    assert error.code == "prefill_conflict"


@pytest.mark.parametrize("field", ["continue_final_message", "add_generation_prompt"])
def test_a_prefill_with_the_engines_own_switches_is_refused(field: str) -> None:
    error = refused(lambda: take_prefill(body(**{field: False})))
    assert error.code == "prefill_conflict"
    assert error.details == {"fields": [field]}


@pytest.mark.parametrize("field", GRAMMAR_FIELDS)
def test_a_prefill_with_a_grammar_is_refused(field: str) -> None:
    error = refused(lambda: take_prefill(body(**{field: {"type": "object"}})))
    assert error.code == "prefill_with_grammar"
    assert error.details == {"fields": [field]}


@pytest.mark.parametrize("kind", ["json_schema", "json_object"])
def test_a_prefill_with_a_response_format_is_refused(kind: str) -> None:
    error = refused(lambda: take_prefill(body(response_format={"type": kind})))
    assert error.code == "prefill_with_grammar"
    assert error.details == {"fields": ["response_format"]}


def test_a_text_response_format_is_no_grammar() -> None:
    assert take_prefill(body(response_format={"type": "text"})) == OPENING


def test_the_engine_is_sent_an_open_assistant_message() -> None:
    sent = with_prefill({"model": MODEL, "messages": USER}, OPENING)
    assert sent["messages"] == [*USER, {"role": "assistant", "content": OPENING}]
    assert sent["continue_final_message"] is True
    assert sent["add_generation_prompt"] is False


# --- the engine and the thinking --------------------------------------------


def test_every_engine_states_whether_it_continues_a_message() -> None:
    for name in ENGINES:
        assert chat_prefill_reading(name).basis
    assert chat_prefill_reading("vllm").served is True
    assert chat_prefill_reading("llama-server").served is True
    assert chat_prefill_reading("mlx-lm").served is False
    assert chat_prefill_reading("mlx-vlm").served is False


def thinking_off() -> dict[str, Any]:
    return {TEMPLATE_KWARGS: {THINKING_KEY: False}}


def test_an_engine_that_cannot_continue_is_refused_by_name() -> None:
    reading = chat_prefill_reading("mlx-lm")
    error = refused(lambda: refuse_unkeepable_prefill(
        engine="mlx-lm", model_id=MODEL, served=reading.served, basis=reading.basis,
        resolved_body=thinking_off(),
    ))
    assert error.code == "prefill_not_served"
    assert "add_generation_prompt=True" in error.message


@pytest.mark.parametrize("thinking", [True, None])
def test_thinking_must_be_stated_off(thinking: bool | None) -> None:
    resolved = {} if thinking is None else {TEMPLATE_KWARGS: {THINKING_KEY: thinking}}
    error = refused(lambda: refuse_unkeepable_prefill(
        engine="vllm", model_id=MODEL, served=True, basis="read", resolved_body=resolved,
    ))
    assert error.code == "prefill_with_thinking"
    assert error.details == {"model": MODEL, "thinking": thinking}


def test_a_served_engine_with_thinking_off_keeps_the_prefill() -> None:
    refuse_unkeepable_prefill(
        engine="vllm", model_id=MODEL, served=True, basis="read", resolved_body=thinking_off(),
    )


# --- the door ----------------------------------------------------------------


def load(client: TestClient, auth: dict[str, str]) -> None:
    run_job(client, auth, type="load-model", model=MODEL)


def test_the_door_sends_the_prefill_as_an_open_answer(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER, PREFILL_KEY: OPENING},
    )
    assert response.status_code == 200, response.json()
    sent = engines[0].last_request
    assert PREFILL_KEY not in sent
    assert sent["messages"] == [*USER, {"role": "assistant", "content": OPENING}]
    assert sent["continue_final_message"] is True
    assert sent["add_generation_prompt"] is False
    # qwen3.5-9b's manifest states thinking off, which is what lets it through.
    assert sent[TEMPLATE_KWARGS] == {THINKING_KEY: False}


def test_the_door_sends_a_chat_without_a_prefill_unchanged(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": USER},
    )
    assert response.status_code == 200
    sent = engines[0].last_request
    assert sent["messages"] == USER
    assert "continue_final_message" not in sent and "add_generation_prompt" not in sent


def test_a_prefill_with_thinking_on_is_refused_before_anything_loads(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": USER,
            PREFILL_KEY: OPENING,
            TEMPLATE_KWARGS: {THINKING_KEY: True},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prefill_with_thinking"
    assert engines == []


def test_a_prefill_with_a_schema_is_refused_and_the_engine_is_sent_nothing(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    engines: list[FakeEngine],  # noqa: F811
) -> None:
    fake_weights(MODEL)
    load(llm_client, auth)
    before = engines[0].last_request
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": USER,
            PREFILL_KEY: OPENING,
            "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {}}},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prefill_with_grammar"
    assert engines[0].last_request == before


def test_an_upstream_model_is_refused_a_prefill(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": "anthropic/claude-x", "messages": USER, PREFILL_KEY: OPENING},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prefill_not_served"


def test_a_mac_model_on_mlx_lm_is_refused_before_it_waits(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
) -> None:
    with make_client(backend=FAKE_MAC_BACKEND, enable_llm=True) as client:
        response = client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": MODEL, "messages": USER, PREFILL_KEY: OPENING},
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "prefill_not_served"
    assert error["details"] == {"model": MODEL, "engine": "mlx-lm"}


def test_the_refusal_names_the_fields_on_the_wire(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        content=json.dumps(
            {"model": MODEL, "messages": USER, PREFILL_KEY: OPENING, "add_generation_prompt": False}
        ),
    )
    assert response.status_code == 400
    assert response.json()["error"]["details"] == {"fields": ["add_generation_prompt"]}
