from __future__ import annotations

KIND_LLM = "llm"
KIND_TTS = "tts"
KIND_ALIGN = "align"
KIND_DENOISE = "denoise"
KIND_IMAGE = "image"
KIND_AUDIO = "audio"
KIND_SEGMENT = "segment"

KIND_NOUNS: dict[str, str] = {
    KIND_LLM: "model",
    KIND_TTS: "voice",
    KIND_ALIGN: "aligner",
    KIND_DENOISE: "separator",
    KIND_IMAGE: "generator",
    KIND_AUDIO: "audio generator",
    KIND_SEGMENT: "segmenter",
}

__all__ = [
    "KIND_ALIGN",
    "KIND_AUDIO",
    "KIND_DENOISE",
    "KIND_IMAGE",
    "KIND_LLM",
    "KIND_NOUNS",
    "KIND_SEGMENT",
    "KIND_TTS",
]
