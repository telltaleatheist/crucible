"""Every backend speaks one thinking field at the chat door: `reasoning`. llama-server
says `reasoning_content` both ways (b10970: common/chat.cpp writes and reads it), so its
replies are renamed on the way out, whole and streamed, and the history it is sent is
renamed on the way in. vLLM and mlx-lm already say `reasoning` and pass untouched."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crucible.api import proxy
from crucible.engines import engine_class


def resident(engine: str, served: str = "qwen3.5-9b") -> SimpleNamespace:
    return SimpleNamespace(
        engine=engine, model_id="qwen3.5-9b", engine_model_name=served, log_path="engine.log"
    )


def completion(message: dict) -> bytes:
    return json.dumps({
        "id": "c", "object": "chat.completion", "model": "qwen3.5-9b",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }).encode("utf-8")


def frame(delta: dict) -> bytes:
    return b"data: " + json.dumps({
        "id": "c", "object": "chat.completion.chunk", "model": "qwen3.5-9b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }).encode("utf-8")


def test_llama_server_states_its_own_name_and_the_others_reasoning() -> None:
    assert engine_class("llama-server").reasoning_field == "reasoning_content"
    assert engine_class("vllm").reasoning_field == "reasoning"
    assert engine_class("mlx-lm").reasoning_field == "reasoning"


def test_a_llama_server_answer_comes_back_with_reasoning() -> None:
    raw = completion({"role": "assistant", "content": "A", "reasoning_content": "Hm."})
    document = json.loads(proxy.restore_model_id(raw, resident("llama-server")))
    assert document["choices"][0]["message"] == {
        "role": "assistant", "content": "A", "reasoning": "Hm.",
    }


def test_a_llama_server_stream_comes_back_with_delta_reasoning() -> None:
    out = proxy._restore_model_id_in_frame(frame({"reasoning_content": "Hm"}), resident("llama-server"))
    chunk = json.loads(out[len(b"data: "):])
    assert chunk["choices"][0]["delta"] == {"reasoning": "Hm"}
    done = proxy._restore_model_id_in_frame(b"data: [DONE]", resident("llama-server"))
    assert done == b"data: [DONE]"


def test_a_vllm_answer_that_needs_no_rename_passes_byte_for_byte() -> None:
    raw = completion({"role": "assistant", "content": "A", "reasoning": "Hm."})
    assert proxy.restore_model_id(raw, resident("vllm")) is raw
    assert proxy._speaks_as_crucible(resident("vllm"))
    assert not proxy._speaks_as_crucible(resident("llama-server"))
    assert not proxy._speaks_as_crucible(resident("vllm", served="/weights/qwen"))


def test_the_model_id_is_still_restored_beside_the_rename() -> None:
    raw = completion({"role": "assistant", "content": "A", "reasoning_content": "Hm."})
    document = json.loads(proxy.restore_model_id(raw, resident("llama-server", "alias")))
    assert document["model"] == "qwen3.5-9b"


HISTORY = {"messages": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "A", "reasoning": "Hm."},
    {"role": "user", "content": "and?"},
]}


def test_llama_server_is_sent_the_history_under_its_own_name() -> None:
    sent = proxy.history_for_engine(HISTORY, resident("llama-server"))
    assert sent is not None
    assert sent["messages"][1] == {"role": "assistant", "content": "A", "reasoning_content": "Hm."}
    # The caller's body is not changed under it.
    assert HISTORY["messages"][1]["reasoning"] == "Hm."


@pytest.mark.parametrize("engine", ["vllm", "mlx-lm"])
def test_an_engine_that_reads_reasoning_is_sent_the_history_as_it_came(engine: str) -> None:
    assert proxy.history_for_engine(HISTORY, resident(engine)) is None


def test_a_history_with_no_thinking_is_not_rewritten() -> None:
    plain = {"messages": [{"role": "user", "content": "hi"}]}
    assert proxy.history_for_engine(plain, resident("llama-server")) is None
