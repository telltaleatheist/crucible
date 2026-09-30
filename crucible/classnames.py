from __future__ import annotations

ROUTABLE_CLASSES: tuple[str, ...] = (
    "clean",
    "translate",
    "simplify",
    "analysis",
    "generate",
)

AUDIO_CLASSES: tuple[str, ...] = ("sfx", "music", "song")

SELECTABLE_CLASSES: tuple[str, ...] = (
    *ROUTABLE_CLASSES,
    "decide",
    "pages",
    "tts",
    "asr",
    "align",
    "rvc",
    "denoise",
    "image",
    *AUDIO_CLASSES,
    "video",
)

CLASS_NAMES: tuple[str, ...] = ("echo", *SELECTABLE_CLASSES)
