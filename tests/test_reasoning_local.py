"""Reasoning on the local engines: a chat model's manifest declares how its thinking is set
apart from its answer ([reasoning]); vLLM is started with the parser for it, so the
thinking comes back as `reasoning` and `content` is the answer alone. llama-server and
mlx-lm split it without being told."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.engines.llama_server import LlamaServerEngine
from crucible.engines.vllm import VllmEngine
from crucible.errors import EngineError
from crucible.manifests import ManifestError, _parse_reasoning, load_manifest
from crucible.reasoningformats import REASONING_FORMATS

QWEN_THINKING_MODELS = (
    "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-4b-8bit", "qwen3.5-9b",
    "qwen3.5-9b-vl", "qwen3.8-27b-4bit", "qwen3.8-27b-4bit-vl", "qwen3.8-27b-8bit",
)

# Their templates do not think when enable_thinking is unstated (`is true`), while vLLM's
# qwen3 parser assumes they do: the manifest's thinking default is what keeps the two
# agreeing, because the door then always states it.
TEMPLATES_OFF_WHEN_UNSTATED = ("qwen3.5-0.8b", "qwen3.5-2b")


@pytest.mark.parametrize("model", QWEN_THINKING_MODELS)
def test_every_qwen_chat_model_declares_its_think_tags(model: str) -> None:
    manifest = load_manifest(model)
    assert manifest.reasoning is not None and manifest.reasoning.format == "qwen3-think"
    assert "<think>" in manifest.reasoning.basis
    assert manifest.to_dict()["reasoning"] == manifest.reasoning.to_dict()


@pytest.mark.parametrize("model", TEMPLATES_OFF_WHEN_UNSTATED)
def test_a_template_that_does_not_think_unstated_has_its_thinking_stated(model: str) -> None:
    assert load_manifest(model).defaults.thinking is not None


def test_the_bside_model_declares_no_reasoning() -> None:
    # No vLLM block (llama-server on cuda-linux, mlx-lm on the Mac), thinking off, and its
    # fine-tune answers JSON at once: a [reasoning] table would change nothing it serves.
    manifest = load_manifest("qwen3.5-4b-bside")
    assert manifest.reasoning is None
    assert "vllm" not in {spec.engine for spec in manifest.backends.values()}


def test_vllm_starts_the_reasoning_parser_for_the_declared_format() -> None:
    args = VllmEngine.model_args(load_manifest("qwen3.8-27b-4bit"), ["--x"], Path("."), 4096)
    assert args[-2:] == ["--reasoning-parser", REASONING_FORMATS["qwen3-think"]["vllm"]]
    assert args[-1] == "qwen3"


def test_a_model_with_no_reasoning_table_gets_no_parser() -> None:
    manifest = SimpleNamespace(path=Path("m.toml"), tools=None, reasoning=None)
    assert VllmEngine.model_args(manifest, ["--x"], Path("."), 4096) == ["--x"]
    dots = load_manifest("dots-ocr")
    assert dots.reasoning is None
    assert "--reasoning-parser" not in VllmEngine.model_args(dots, ["--x"], Path("."), 4096)


def test_reasoning_without_tools_is_derived_on_its_own() -> None:
    manifest = SimpleNamespace(
        path=Path("m.toml"), tools=None,
        reasoning=SimpleNamespace(format="qwen3-think", basis="x"),
    )
    assert VllmEngine.model_args(manifest, ["--x"], Path("."), 4096) == [
        "--x", "--reasoning-parser", "qwen3",
    ]


@pytest.mark.parametrize("stated", [
    ["--reasoning-parser", "qwen3"], ["--reasoning-parser=deepseek_r1"],
])
def test_a_manifest_that_states_the_reasoning_parser_itself_is_refused(stated: list[str]) -> None:
    with pytest.raises(EngineError, match="reasoning_flags_stated"):
        VllmEngine.model_args(load_manifest("qwen3.5-9b"), ["--x", *stated], Path("."), 4096)


def test_llama_server_is_told_nothing_about_reasoning() -> None:
    # b10970's --reasoning-format defaults to deepseek: --jinja alone splits the thinking.
    args = LlamaServerEngine.model_args(load_manifest("qwen3.5-9b"), ["-m", "f"], Path("."), 4096)
    assert args == ["-m", "f", "--jinja"]


@pytest.mark.parametrize("table, words", [
    ({"format": "deepseek_r1", "basis": "x"}, "not a reasoning format"),
    ({"format": "qwen3-think"}, "basis"),
    ({"format": "qwen3-think", "basis": "  "}, "basis"),
    ({"format": "qwen3-think", "basis": "x", "parser": "qwen3"}, "unknown key"),
    ("qwen3-think", "must be a table"),
])
def test_a_reasoning_table_is_held_to_its_two_keys(table: Any, words: str) -> None:
    with pytest.raises(ManifestError, match=words):
        _parse_reasoning(table, Path("m.toml"))


def test_a_reasoning_table_parses_to_its_format_and_basis() -> None:
    spec = _parse_reasoning({"format": "qwen3-think", "basis": "  read here  "}, Path("m.toml"))
    assert (spec.format, spec.basis) == ("qwen3-think", "read here")
