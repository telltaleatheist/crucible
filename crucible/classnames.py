from __future__ import annotations

ROUTABLE_CLASSES: tuple[str, ...] = (
    "clean",
    "translate",
    "simplify",
    "analysis",
    "generate",
)

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
)

CLASS_NAMES: tuple[str, ...] = ("echo", *SELECTABLE_CLASSES)
