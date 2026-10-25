"""Tool-call formats a chat model can be trained on (a manifest's [tools] `format`)."""

from __future__ import annotations

# The tool-call formats a chat model can be trained on, and what each engine is told for
# it. A model's [tools] table names the one its own chat template writes; the engine
# renders the request's `tools` into the prompt through that template and parses the
# model's calls back into OpenAI `tool_calls`. vLLM: its tool-call parser for the format
# (`--enable-auto-tool-choice --tool-call-parser <name>`); llama-server: `--jinja`, which
# runs the GGUF's own template and detects the format from it; mlx-lm: the tokenizer's own
# tool parser, nothing to pass.
TOOL_FORMATS: dict[str, dict[str, str]] = {
    # Qwen3-Coder's XML calls: <tool_call><function=NAME><parameter=P>V</parameter>...
    # Qwen 3.5 and 3.8's chat templates write this form, not Qwen 2.5/3's JSON one.
    "qwen3-xml": {"vllm": "qwen3_coder"},
}


__all__ = ["TOOL_FORMATS"]
