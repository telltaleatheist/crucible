"""Tool calling on the local engines: a chat model's manifest declares the tool-call format
its chat template is trained on ([tools]); each engine is started to write and read that
format; and the door refuses `tools` a model cannot take, by name, before it waits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.api.routes.openai import refuse_tools_this_model_cannot_take
from crucible.engines.llama_server import LlamaServerEngine
from crucible.engines.vllm import VllmEngine
from crucible.errors import ApiError, EngineError
from crucible.manifests import ManifestError, _parse_tools, load_manifest

QWEN_TOOL_MODELS = (
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-4b-8bit", "qwen3.5-9b",
    "qwen3.5-9b-vl", "qwen3.8-27b-4bit", "qwen3.8-27b-4bit-vl", "qwen3.8-27b-8bit",
)

TOOL = {"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}


@pytest.mark.parametrize("model", QWEN_TOOL_MODELS)
def test_every_qwen_chat_model_declares_the_xml_form_its_template_writes(model: str) -> None:
    tools = load_manifest(model).tools
    assert tools is not None and tools.format == "qwen3-xml"
    assert "Qwen3-Coder" in tools.basis


def test_vllm_starts_the_parser_for_the_declared_format() -> None:
    args = VllmEngine.model_args(load_manifest("qwen3.5-9b"), ["--x"], Path("."), 4096)
    assert args == [
        "--x", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
    ]


def test_a_model_with_no_tools_table_starts_without_tool_flags() -> None:
    manifest = load_manifest("qwen3.5-4b-bside")
    assert manifest.tools is None
    assert LlamaServerEngine.model_args(manifest, ["-m", "f"], Path("."), 4096) == ["-m", "f"]


def test_llama_server_runs_the_ggufs_own_template_for_a_tool_model() -> None:
    args = LlamaServerEngine.model_args(load_manifest("qwen3.5-9b"), ["-m", "f"], Path("."), 4096)
    assert args == ["-m", "f", "--jinja"]


@pytest.mark.parametrize("engine, flag", [
    (VllmEngine, "--tool-call-parser"), (VllmEngine, "--enable-auto-tool-choice"),
    (LlamaServerEngine, "--jinja"),
])
def test_a_manifest_that_states_a_tool_flag_itself_is_refused(engine: Any, flag: str) -> None:
    with pytest.raises(EngineError, match="tool_flags_stated"):
        engine.model_args(load_manifest("qwen3.5-9b"), ["-m", "f", flag], Path("."), 4096)


@pytest.mark.parametrize("table, words", [
    ({"format": "hermes", "basis": "x"}, "not a tool-call format"),
    ({"format": "qwen3-xml"}, "basis"),
    ({"format": "qwen3-xml", "basis": "x", "parser": "y"}, "unknown key"),
])
def test_a_tools_table_is_held_to_its_two_keys(table: dict, words: str) -> None:
    with pytest.raises(ManifestError, match=words):
        _parse_tools(table, Path("m.toml"))


def test_tools_to_a_model_with_no_format_are_refused_by_name() -> None:
    with pytest.raises(ApiError) as caught:
        refuse_tools_this_model_cannot_take(
            {"tools": [TOOL]}, "qwen3.5-4b-bside", "cuda-linux")
    assert caught.value.code == "tools_unsupported"
    assert "qwen3.5-9b" in caught.value.details["models_with_tools"]


def test_tools_beside_a_response_format_are_refused_by_name() -> None:
    with pytest.raises(ApiError) as caught:
        refuse_tools_this_model_cannot_take(
            {"tools": [TOOL], "response_format": {"type": "json_object"}},
            "qwen3.5-9b", "cuda-linux")
    assert caught.value.code == "tools_with_response_format"


@pytest.mark.parametrize("body", [{}, {"tools": []}, {"tools": [TOOL]}])
def test_a_tool_model_or_no_tools_passes(body: dict) -> None:
    refuse_tools_this_model_cannot_take(body, "qwen3.5-9b", "cuda-linux")


def test_the_door_refuses_before_anything_loads(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=True) as client:
        response = client.post("/v1/openai/chat/completions", headers=auth, json={
            "model": "qwen3.5-4b-bside", "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL], "queue": False,
        })
    assert response.status_code == 400, response.json()
    assert response.json()["error"]["code"] == "tools_unsupported"
