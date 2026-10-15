"""How to prompt each model Crucible serves, as its trainers and users say to.

One markdown guide per model (or per family, where the sizes share one), researched from
the model card, the official code and prompts, and community reports, each claim cited
and labelled OFFICIAL or COMMUNITY (Owen, 2026-10-15: "official prompting guidance for
each of the models we use … whatever the trainers and/or community have said about how
to prompt correctly"). They describe what the MODEL expects; the prompts themselves stay
the app's. Served at `GET /v1/prompting` and `GET /v1/prompting/{name}.md`.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent

# Each guide's file stem and the Crucible model ids it covers. A guide named after an
# engine (higgs-v3) covers every voice that engine serves, so it lists none.
GUIDES: dict[str, tuple[str, ...]] = {
    "yue2-3b": ("yue2-3b",),
    "stable-audio-3-medium": ("stable-audio-3-medium",),
    "stable-audio-3-small-sfx": ("stable-audio-3-small-sfx",),
    "qwen-image-2.1": ("qwen-image-2.1",),
    "dreamshaper-xl-lightning": ("dreamshaper-xl-lightning",),
    "ltx-2.5-distilled": ("ltx-2.5-distilled",),
    "higgs-v3": (),
    "qwen3.5": (
        "qwen3.5-0.8b",
        "qwen3.5-2b",
        "qwen3.5-4b",
        "qwen3.5-4b-8bit",
        "qwen3.5-9b",
        "qwen3.5-9b-vl",
    ),
    "qwen3.8-27b": ("qwen3.8-27b-4bit", "qwen3.8-27b-4bit-vl", "qwen3.8-27b-8bit"),
    "qwen3-embedding-8b": ("qwen3-embedding-8b",),
    "qwen3-reranker-8b": ("qwen3-reranker-8b",),
    "dots-ocr": ("dots-ocr",),
    "qwen3-asr": (
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b-mlx",
        "qwen3-asr-1.7b",
        "qwen3-asr-1.7b-mlx",
    ),
    "whisper": ("whisper-large-v3-turbo", "whisper-tiny"),
    "sam2.1-hiera-large": ("sam2.1-hiera-large",),
}

_BY_MODEL: dict[str, str] = {model: guide for guide, models in GUIDES.items() for model in models}


def guide_for(name: str) -> str | None:
    """The guide a model id or a guide name reaches, or None."""
    if name in GUIDES:
        return name
    return _BY_MODEL.get(name)


def title(guide: str) -> str:
    """The guide's own first heading."""
    first = text(guide).splitlines()[0]
    return first.lstrip("# ").strip()


def text(guide: str) -> str:
    return (HERE / f"{guide}.md").read_text(encoding="utf-8")
