from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .alignmodels import align_manifests_dir, load_all_align_manifests
from .asrmodels import asr_manifests_dir, load_all_asr_manifests
from .classnames import CLASS_NAMES, ROUTABLE_CLASSES, SELECTABLE_CLASSES
from .denoisemodels import denoise_manifests_dir, load_all_denoise_manifests
from .enginespec import UNSTATED_ENGINE_CONCURRENCY
from .fit import Candidate, CatalogCandidates, WorkingContext, cached_catalog
from .manifests import BACKEND_ENGINES, load_all_manifests, manifests_dir
from .pages import PAGE_CONCURRENCY
from .rvcmodels import load_all_rvc_manifests, rvc_manifests_dir
from .voices import load_all_voices

CATALOG_DIRECTORY: dict[Callable[..., dict[str, Any]], Callable[[], Path]] = {
    load_all_manifests: manifests_dir,
    load_all_asr_manifests: asr_manifests_dir,
    load_all_align_manifests: align_manifests_dir,
    load_all_rvc_manifests: rvc_manifests_dir,
    load_all_denoise_manifests: denoise_manifests_dir,
}


def _from_catalog(
    load: Callable[..., dict[str, Any]],
    *families: str,
    min_params_b: float | None = None,
    aliases: bool = False,
) -> CatalogCandidates:
    return CatalogCandidates(
        load, families or None, min_params_b, aliases, CATALOG_DIRECTORY.get(load)
    )


@dataclass(frozen=True)
class CapabilityClass:
    name: str
    job_type: str
    purpose: str
    plainly: str
    noun: str
    candidates: Callable[[str], tuple[Candidate, ...]] | None
    binary_note: str = ""
    routable: bool = False
    work: "WorkingContext | None" = None
    client_sized: bool = False

    @property
    def min_params_b(self) -> float | None:
        if isinstance(self.candidates, CatalogCandidates):
            return self.candidates.min_params_b
        return None


NINE_B_FLOOR = 9

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

TRANSLATIONS_FLOOR_NOTE = (
    "The floor for {work} is the 9B, for translation's reason: a host that "
    "cannot hold a 9B cannot do this work at all."
)


NINE_B_TEXT_MODELS = _from_catalog(
    load_all_manifests, *TEXT_FAMILIES, min_params_b=NINE_B_FLOOR
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
        candidates=_from_catalog(load_all_manifests, "qwen3.5", min_params_b=NINE_B_FLOOR),
        binary_note=(
            "This build ships no 4-bit 9B, and the 4B and 0.8B it does ship are "
            "below cleanup's 9B floor, so there is nothing smaller to fall back "
            "to (docs/internals/engines-and-capability.md, \"Classes\")."
        ),
    ),
    CapabilityClass(
        name="translate",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="translation, which needs a 27B-class model",
        plainly="translate",
        noun=TEXT_FAMILIES_NOUN,
        candidates=NINE_B_TEXT_MODELS,
        binary_note=(
            "The floor for translation is the 9B, not the 27B — so a host that "
            "cannot translate cannot hold a 9B either, and nothing smaller is "
            "coming (docs/MODEL-CHOICE.md section 1)."
        ),
    ),
    CapabilityClass(
        name="simplify",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="simplification, which runs on the same 27B translation needs",
        plainly="simplify text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=NINE_B_TEXT_MODELS,
        binary_note=TRANSLATIONS_FLOOR_NOTE.format(work="simplification"),
    ),
    CapabilityClass(
        name="analysis",
        job_type="llm",
        routable=True,
        work=BATCHED_BLOCKS_WORK,
        purpose="structured analysis answers, on the same 27B",
        plainly="analyse text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=NINE_B_TEXT_MODELS,
        binary_note=TRANSLATIONS_FLOOR_NOTE.format(work="analysis"),
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
        purpose="open-ended text generation, on the 9B-and-up text models",
        plainly="generate text",
        noun=TEXT_FAMILIES_NOUN,
        candidates=NINE_B_TEXT_MODELS,
        binary_note=TRANSLATIONS_FLOOR_NOTE.format(work="generation"),
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
    "CLASSES",
    "CapabilityClass",
    "DECIDE_STATE_TOKENS",
    "GENERATE_DEFAULT_TOKENS",
    "NINE_B_FLOOR",
    "NINE_B_TEXT_MODELS",
    "ROUTABLE_CLASSES",
    "SELECTABLE_CLASSES",
    "TEXT_FAMILIES",
    "TRANSLATIONS_FLOOR_NOTE",
    "classes_for_job_type",
    "classes_for_model",
    "models_by_class",
]
