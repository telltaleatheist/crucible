from __future__ import annotations

ROUTABLE_CLASSES: tuple[str, ...] = (
    "clean",
    "translate",
    "simplify",
    "analysis",
    "generate",
)

AUDIO_CLASSES: tuple[str, ...] = ("sfx", "music", "song")

SEGMENT_CLASSES: tuple[str, ...] = ("cutout", "select")

SELECTABLE_CLASSES: tuple[str, ...] = (
    *ROUTABLE_CLASSES,
    "decide",
    "embed",
    "rerank",
    "pages",
    "tts",
    "asr",
    "align",
    "rvc",
    "denoise",
    "image",
    *AUDIO_CLASSES,
    *SEGMENT_CLASSES,
    "video",
)

CLASS_NAMES: tuple[str, ...] = ("echo", *SELECTABLE_CLASSES)

# Optional packages: models a server holds only when it is installed (`crucible install
# <name>`, recorded as `[packages] <name> = true`). `retrieval` is the embed and rerank
# verbs' own models (Owen, 2026-10-10: "an OPTIONAL install package, like the voice/Higgs
# package"); a model is in it by having an [embed] or [rerank] table (crucible/verbspec.py).
RETRIEVAL_PACKAGE = "retrieval"

PACKAGE_NAMES: tuple[str, ...] = (RETRIEVAL_PACKAGE,)
