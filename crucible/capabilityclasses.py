from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .alignmodels import align_manifests_dir, load_all_align_manifests
from .asrmodels import asr_manifests_dir, load_all_asr_manifests
from .audiomodels import (
    audio_manifests_dir,
    load_music_manifests,
    load_sfx_manifests,
    load_song_manifests,
)
from .classnames import CLASS_NAMES, ROUTABLE_CLASSES, SELECTABLE_CLASSES
from .denoisemodels import denoise_manifests_dir, load_all_denoise_manifests
from .enginespec import UNSTATED_ENGINE_CONCURRENCY
from .fit import Candidate, CatalogCandidates, WorkingContext, cached_catalog
from .imagemodels import image_manifests_dir, load_all_image_manifests
from .manifests import BACKEND_ENGINES, load_all_manifests, manifests_dir
from .pages import PAGE_CONCURRENCY
from .rvcmodels import load_all_rvc_manifests, rvc_manifests_dir
from .segmentmodels import (
    load_cutout_manifests,
    load_select_manifests,
    segment_manifests_dir,
)
from .videomodels import load_all_video_manifests, video_manifests_dir
from .voicecatalog import load_all_voices

CATALOG_DIRECTORY: dict[Callable[..., dict[str, Any]], Callable[[], Path]] = {
    load_all_manifests: manifests_dir,
    load_all_asr_manifests: asr_manifests_dir,
    load_all_align_manifests: align_manifests_dir,
    load_all_rvc_manifests: rvc_manifests_dir,
    load_all_denoise_manifests: denoise_manifests_dir,
    load_all_image_manifests: image_manifests_dir,
    load_sfx_manifests: audio_manifests_dir,
    load_music_manifests: audio_manifests_dir,
    load_song_manifests: audio_manifests_dir,
    load_cutout_manifests: segment_manifests_dir,
    load_select_manifests: segment_manifests_dir,
    load_all_video_manifests: video_manifests_dir,
}


def _from_catalog(
    load: Callable[..., dict[str, Any]],
    *families: str,
    aliases: bool = False,
) -> CatalogCandidates:
    return CatalogCandidates(load, families or None, aliases, CATALOG_DIRECTORY.get(load))


@dataclass(frozen=True)
class Goal:
    """The size a verb is meant to run at (docs/VERB-SIZING.md rule 2). The automatic pick
    never goes above it; below it the verb is smaller, never off."""

    params_b: float
    source: str

    @property
    def words(self) -> str:
        return f"{self.params_b:g}B"

    def to_dict(self) -> dict[str, Any]:
        return {"params_b": self.params_b, "source": self.source}


@dataclass(frozen=True)
class CapabilityClass:
    name: str
    job_type: str
    purpose: str
    plainly: str
    noun: str
    candidates: Callable[[str], tuple[Candidate, ...]] | None
    routable: bool = False
    work: "WorkingContext | None" = None
    client_sized: bool = False
    goal: "Goal | None" = None

    def pick_order(self, found: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        """The candidates the automatic pick may take, best first (docs/VERB-SIZING.md
        rule 3): at or below the goal, the most parameters, then the highest precision,
        then a model's own form before a `weights_of` alias of it (the alias is the same
        weights with more to hold). The catalog's order, largest need first, settles the
        rest. A class with no goal keeps the catalog's order whole."""
        if self.goal is None:
            return found
        for candidate in found:
            if candidate.params_b is None or candidate.bits is None:
                missing = "params_b" if candidate.params_b is None else "bits"
                raise ValueError(
                    f"{candidate.id} is a candidate for {self.name}, whose goal is "
                    f"{self.goal.words}, and its manifest states no {missing} for this "
                    "backend: the pick ranks by both. State it in the manifest"
                )
        return tuple(
            sorted(
                (c for c in found if c.params_b <= self.goal.params_b),
                key=lambda c: (-c.params_b, -c.bits, c.alias),
            )
        )


DECIDE_STATE_TOKENS = 8192

GENERATE_DEFAULT_TOKENS = 8192

TEXT_FAMILIES = ("qwen3.8", "qwen3.5")

TEXT_FAMILIES_NOUN = "qwen3.8 and qwen3.5 variants"

BATCHED_BLOCKS_WORK = WorkingContext(
    tokens=4096,
    concurrency=4,
    source=(
        "Owen 2026-09-16: \"translate/simplify/etc dont actually need "
        "that much kv cache because it's batched with small blocks. it "
        "isnt sending in the entire book to be translated, its only "
        "sending it in one block (roughly a paragraph) at a time. and "
        "its batched, so each block doesnt depend on the context of the "
        "one that came before it\""
    ),
)

TEXT_MODELS = _from_catalog(load_all_manifests, *TEXT_FAMILIES)

CHAT_GOAL = Goal(
    params_b=27,
    source=(
        "Owen 2026-10-09: \"chat should shoot for 27b\"; translate, simplify and "
        "analysis share it (docs/VERB-SIZING.md rule 2)"
    ),
)

DECIDE_GOAL = Goal(
    params_b=9,
    source=(
        "Owen 2026-10-09: \"decide shoots for 9b\" and \"each job should have a goal "
        "- 9b 16 bit for decide, for example\" (docs/VERB-SIZING.md rule 2)"
    ),
)

CLEAN_GOAL = Goal(
    params_b=9,
    source=(
        "docs/VERB-SIZING.md rule 2: clean's goal is 9B; cleanup was always "
        "9B-class work (docs/MODEL-CHOICE.md section 1)"
    ),
)


CLASSES: tuple[CapabilityClass, ...] = (
    CapabilityClass(
        name="echo",
        job_type="echo",
        purpose="the test job type; it never touches the accelerator",
        plainly="run the test job",
        noun="engines",
        candidates=None,
    ),
    CapabilityClass(
        name="clean",
        job_type="llm",
        routable=True,
        work=WorkingContext(
            tokens=8192,
            concurrency=2,
            source=(
                "Foundry clean/runner.ts CTX_MAX 16384 spent as two in flight; "
                "PLACEHOLDER until a cleanup run is watched"
            ),
        ),
        purpose="cleanup and the other 9B-class text work",
        plainly="clean up text",
        noun="qwen3.5 variants",
        candidates=_from_catalog(load_all_manifests, "qwen3.5"),
        goal=CLEAN_GOAL,
    ),
    CapabilityClass(
        name="translate",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="translation, with a 27B as its goal",
        plainly="translate",
        noun=TEXT_FAMILIES_NOUN,
        candidates=TEXT_MODELS,
        goal=CHAT_GOAL,
    ),
    CapabilityClass(
        name="simplify",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="simplification, with translation's 27B goal",
        plainly="simplify text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=TEXT_MODELS,
        goal=CHAT_GOAL,
    ),
    CapabilityClass(
        name="analysis",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="structured analysis answers, with translation's 27B goal",
        plainly="analyse text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=TEXT_MODELS,
        goal=CHAT_GOAL,
    ),
    CapabilityClass(
        name="generate",
        job_type="llm",
        routable=True,
        client_sized=True,
        work=WorkingContext(
            tokens=GENERATE_DEFAULT_TOKENS,
            concurrency=1,
            source=(
                "Owen 2026-09-23: \"context limit can be set to 8k tokens by "
                "default, and it can request higher\"; one in flight, as its "
                "first measured user (ContentStudio, whose calls are serial) "
                "sends. A client states its own with ?context_tokens= and "
                "?concurrency= (ContentStudio asks for up to its "
                "LOCAL_FIELD_CTX_MAX, 40960)"
            ),
        ),
        purpose="open-ended text generation, with a 27B as its goal",
        plainly="generate text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=TEXT_MODELS,
        goal=CHAT_GOAL,
    ),
    CapabilityClass(
        name="decide",
        job_type="llm",
        routable=False,
        work=WorkingContext(
            tokens=DECIDE_STATE_TOKENS,
            concurrency=2,
            source=(
                f"one {DECIDE_STATE_TOKENS}-token state (Foundry's Categorize "
                "tile: ~24 blocks with 12 of context each side) shared through "
                f"the prefix cache by up to {UNSTATED_ENGINE_CONCURRENCY} "
                "questions (enginespec.UNSTATED_ENGINE_CONCURRENCY), whose tails at "
                "one 544-token vLLM block each (docs/internals/engines-and-capability.md, \"The decision door\") come to "
                "about one more state"
            ),
        ),
        purpose="one-forward-pass decisions (the decision door)",
        plainly="decide",
        noun=TEXT_FAMILIES_NOUN,
        candidates=_from_catalog(load_all_manifests, *TEXT_FAMILIES, aliases=True),
        goal=DECIDE_GOAL,
    ),
    CapabilityClass(
        name="pages",
        job_type="llm",
        work=WorkingContext(
            tokens=32768,
            concurrency=PAGE_CONCURRENCY,
            source=(
                "models/dots-ocr.toml: context_default 32768 on cuda-linux, and "
                "its own note that one page is ~3450 image tokens plus up to "
                f"8192 of answer, at the {PAGE_CONCURRENCY} pages in flight "
                "crucible/pages.py publishes to clients"
            ),
        ),
        purpose="reading page images (the VLM door)",
        plainly="read pages",
        noun="page readers",
        candidates=_from_catalog(load_all_manifests, "dots"),
    ),
    CapabilityClass(
        name="tts",
        job_type="tts",
        purpose="narration",
        plainly="narrate",
        noun="voices",
        candidates=_from_catalog(load_all_voices),
    ),
    CapabilityClass(
        name="asr",
        job_type="asr",
        purpose="transcription",
        plainly="transcribe",
        noun="transcribers",
        candidates=_from_catalog(load_all_asr_manifests, aliases=True),
    ),
    CapabilityClass(
        name="align",
        job_type="align",
        purpose="forced alignment",
        plainly="align audio to text",
        noun="aligners",
        candidates=_from_catalog(load_all_align_manifests),
    ),
    CapabilityClass(
        name="rvc",
        job_type="rvc",
        purpose="voice conversion",
        plainly="convert a voice",
        noun="RVC models",
        candidates=_from_catalog(load_all_rvc_manifests),
    ),
    CapabilityClass(
        name="denoise",
        job_type="denoise",
        purpose="noise removal and stem separation",
        plainly="remove noise or split stems",
        noun="separator models",
        candidates=_from_catalog(load_all_denoise_manifests),
    ),
    CapabilityClass(
        name="image",
        job_type="image",
        purpose="image generation from a text prompt",
        plainly="generate images",
        noun="image models",
        candidates=_from_catalog(load_all_image_manifests),
    ),
    CapabilityClass(
        name="sfx",
        job_type="audio",
        purpose="sound effects from a text prompt",
        plainly="make sound effects",
        noun="sound-effect models",
        candidates=_from_catalog(load_sfx_manifests),
    ),
    CapabilityClass(
        name="music",
        job_type="audio",
        purpose="instrumental music from a text prompt",
        plainly="make music",
        noun="music models",
        candidates=_from_catalog(load_music_manifests),
    ),
    CapabilityClass(
        name="song",
        job_type="audio",
        purpose="songs with sung vocals from lyrics and style tags",
        plainly="make songs with vocals",
        noun="song models",
        candidates=_from_catalog(load_song_manifests),
    ),
    CapabilityClass(
        name="cutout",
        job_type="segment",
        purpose="the main subject's mask and cutout from a picture (background removal)",
        plainly="cut out a picture's subject",
        noun="cutout models",
        candidates=_from_catalog(load_cutout_manifests),
    ),
    CapabilityClass(
        name="select",
        job_type="segment",
        purpose="the mask of the object a caller points at with points or a box",
        plainly="select what is pointed at in a picture",
        noun="selection models",
        candidates=_from_catalog(load_select_manifests),
    ),
    CapabilityClass(
        name="video",
        job_type="video",
        purpose="video clips with synchronized sound from a text prompt or a start image",
        plainly="make video",
        noun="video models",
        candidates=_from_catalog(load_all_video_manifests),
    ),
)

BY_NAME: dict[str, CapabilityClass] = {entry.name: entry for entry in CLASSES}


def _names_agree_with_classnames() -> None:
    stated = {
        "CLASS_NAMES": (CLASS_NAMES, tuple(entry.name for entry in CLASSES)),
        "ROUTABLE_CLASSES": (
            ROUTABLE_CLASSES,
            tuple(entry.name for entry in CLASSES if entry.routable),
        ),
        "SELECTABLE_CLASSES": (
            SELECTABLE_CLASSES,
            tuple(entry.name for entry in CLASSES if entry.candidates is not None),
        ),
    }
    for name, (named, built) in stated.items():
        if named != built:
            raise RuntimeError(
                f"crucible/classnames.py {name} is {named} but capability.CLASSES "
                f"builds {built}; change crucible/classnames.py to match"
            )


_names_agree_with_classnames()


def classes_for_job_type(job_type: str) -> tuple[CapabilityClass, ...]:
    return tuple(entry for entry in CLASSES if entry.job_type == job_type)


def models_by_class() -> dict[str, set[str]]:
    served: dict[str, set[str]] = {}
    for entry in CLASSES:
        source = entry.candidates
        if not isinstance(source, CatalogCandidates):
            continue
        if source.load is not load_all_manifests:
            continue
        served[entry.name] = {c.id for kind in BACKEND_ENGINES for c in source(kind)}
    return served


def classes_for_model(model_id: str) -> tuple[str, ...]:
    catalog = cached_catalog(load_all_manifests, manifests_dir)
    if model_id not in catalog:
        raise ValueError(
            f"{model_id!r} is not a model in this build's catalog; it ships "
            f"{sorted(catalog)}"
        )
    return tuple(
        name for name, served in models_by_class().items() if model_id in served
    )


__all__ = [
    "BATCHED_BLOCKS_WORK",
    "BY_NAME",
    "CATALOG_DIRECTORY",
    "CHAT_GOAL",
    "CLASSES",
    "CLEAN_GOAL",
    "CapabilityClass",
    "DECIDE_GOAL",
    "DECIDE_STATE_TOKENS",
    "GENERATE_DEFAULT_TOKENS",
    "Goal",
    "ROUTABLE_CLASSES",
    "SELECTABLE_CLASSES",
    "TEXT_FAMILIES",
    "TEXT_MODELS",
    "classes_for_job_type",
    "classes_for_model",
    "models_by_class",
]
