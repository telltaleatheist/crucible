"""Reasoning formats a chat model can be trained on (a manifest's [reasoning] `format`)."""

from __future__ import annotations

# The ways a chat model can write its thinking, and what each engine is told for it. A
# model's [reasoning] table names the one its own chat template and weights write; the
# engine hands the thinking back apart from the answer, as the reply's `reasoning`
# (`delta.reasoning` streamed), so `content` is the answer alone.
#
# vLLM: its reasoning parser for the format (`--reasoning-parser <name>`); without one,
# vLLM 0.29.0 hands back everything the model wrote as `content`. llama-server: nothing
# to pass; b10970's `--reasoning-format` defaults to deepseek (common/common.h
# `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK`), which, with `--jinja`, lifts the
# template's thinking into `message.reasoning_content` (common/chat.cpp). mlx-lm: nothing
# to pass; 0.31.3's server splits the tokenizer's think tags into `reasoning` itself
# (mlx_lm/server.py).
REASONING_FORMATS: dict[str, dict[str, str]] = {
    # <think>...</think> before the answer, on or off by the template's `enable_thinking`
    # (on when unstated). Qwen 3, 3.5 and 3.8's chat templates: thinking on, the
    # generation prompt opens `<think>\n`; off, it writes an empty `<think>\n\n</think>\n\n`
    # and the model answers at once. vLLM's qwen3 parser (vllm/parser/qwen3.py) reads the
    # mode from `chat_template_kwargs.enable_thinking` (default true) and ends reasoning at
    # `</think>` or at a `<tool_call>` written before it.
    "qwen3-think": {"vllm": "qwen3"},
}


__all__ = ["REASONING_FORMATS"]
