from __future__ import annotations

ROUTABLE_CLASSES: tuple[str, ...] = ("generate",)

# Capability classes this server once had and no longer does. A name here is not a class:
# every door that takes a class refuses it, saying it was retired and where its work went,
# and a config.toml that still names it ([local_models], [routes], a [capability] row) has
# it DROPPED on load, said once on the server log, and gone from the file on the next write
# (crucible/config.py). That is a migration for a removed feature, not a fallback: a name
# that was never a class is still refused as unknown.
RETIRED_CLASSES_DATE = "2026-10-25"

RETIRED_CLASSES_REASON = (
    "the apps (BookForge, Foundry) choose their own text models now: their cleanup, "
    "translation, simplification and analysis are plain chat (verb `generate`) or "
    "`decide`, and sending a job to an API model instead is the app's choice per job"
)

RETIRED_CLASSES: tuple[str, ...] = ("clean", "translate", "simplify", "analysis")


def retired_class_words(name: str) -> str:
    """The sentence a door says about a retired class name, by name."""
    return (
        f"{name!r} was a capability class until {RETIRED_CLASSES_DATE}, when it was "
        f"retired to the apps: {RETIRED_CLASSES_REASON}"
    )

# The verbs a chat may name instead of a model (POST /v1/openai/chat/completions `verb`,
# docs/VERB-SIZING.md rule 8): the text verbs whose work is a chat. A chat that names no
# verb is `generate`'s, the open-ended chat (rule 2's table: "`generate` (chat)").
CHAT_VERBS: tuple[str, ...] = ROUTABLE_CLASSES

DEFAULT_CHAT_VERB = "generate"

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
    "score",
)

CLASS_NAMES: tuple[str, ...] = ("echo", *SELECTABLE_CLASSES)

# Optional packages: models a server holds only when it is installed (`crucible install
# <name>`, recorded as `[packages] <name> = true`). `retrieval` is the embed and rerank
# verbs' own models (Owen, 2026-10-10: "an OPTIONAL install package, like the voice/Higgs
# package"); a model is in it by having an [embed] or [rerank] table (crucible/verbspec.py).
RETRIEVAL_PACKAGE = "retrieval"

PACKAGE_NAMES: tuple[str, ...] = (RETRIEVAL_PACKAGE,)
