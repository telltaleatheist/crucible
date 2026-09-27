from __future__ import annotations

KIND_LLM = "llm"
KIND_TTS = "tts"
KIND_ALIGN = "align"
KIND_DENOISE = "denoise"

KIND_NOUNS: dict[str, str] = {
    KIND_LLM: "model",
    KIND_TTS: "voice",
    KIND_ALIGN: "aligner",
    KIND_DENOISE: "separator",
}

__all__ = ["KIND_ALIGN", "KIND_DENOISE", "KIND_LLM", "KIND_NOUNS", "KIND_TTS"]
