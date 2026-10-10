from __future__ import annotations

from dataclasses import dataclass

from .cardkinds import (
    KIND_ALIGN,
    KIND_AUDIO,
    KIND_DENOISE,
    KIND_IMAGE,
    KIND_LLM,
    KIND_SEGMENT,
    KIND_TTS,
    KIND_VIDEO,
)
from .classnames import AUDIO_CLASSES, CLASS_NAMES, ROUTABLE_CLASSES, SEGMENT_CLASSES


@dataclass(frozen=True)
class CardEffect:
    makes_resident: str | None = None
    takes_off: str | None = None


@dataclass(frozen=True)
class Env:
    name: str
    worker: bool


LLM_ENV = Env("llm", worker=False)
TTS_ENV = Env("tts", worker=False)
ALIGN_ENV = Env("align", worker=True)
ASR_ENV = Env("asr", worker=True)
RVC_ENV = Env("rvc", worker=True)
IMAGE_ENV = Env("image", worker=True)
AUDIO_ENV = Env("audio", worker=False)
SEGMENT_ENV = Env("segment", worker=True)
VIDEO_ENV = Env("video", worker=False)

ENVS: tuple[Env, ...] = (
    LLM_ENV,
    TTS_ENV,
    ALIGN_ENV,
    ASR_ENV,
    RVC_ENV,
    IMAGE_ENV,
    AUDIO_ENV,
    SEGMENT_ENV,
    VIDEO_ENV,
)


@dataclass(frozen=True)
class Family:
    name: str
    env: Env | None
    capability_classes: tuple[str, ...]
    pullable_refusals: frozenset[str] = frozenset()
    base_subject_kinds: tuple[str, ...] = ()
    catalog_is_complete: bool = False

    @property
    def flag(self) -> str:
        return f"enable_{self.name}"


ECHO = Family("echo", None, ("echo",))
LLM = Family(
    "llm",
    LLM_ENV,
    (*ROUTABLE_CLASSES, "decide", "embed", "rerank", "pages"),
    frozenset({"model_not_installed"}),
)
TTS = Family("tts", TTS_ENV, ("tts",), frozenset({"voice_not_installed"}))
ASR = Family(
    "asr", ASR_ENV, ("asr",), frozenset({"model_not_installed"}), catalog_is_complete=True
)
ALIGN = Family(
    "align",
    ALIGN_ENV,
    ("align",),
    frozenset({"model_not_installed"}),
    catalog_is_complete=True,
)
RVC = Family(
    "rvc",
    RVC_ENV,
    ("rvc",),
    frozenset({"model_not_installed", "rvc_base_models_missing"}),
    base_subject_kinds=("rvc-base",),
    catalog_is_complete=True,
)
DENOISE = Family(
    "denoise",
    RVC_ENV,
    ("denoise",),
    frozenset({"denoise_model_missing"}),
    catalog_is_complete=True,
)
IMAGE = Family(
    "image",
    IMAGE_ENV,
    ("image",),
    frozenset({"model_not_installed"}),
    catalog_is_complete=True,
)
AUDIO = Family(
    "audio",
    AUDIO_ENV,
    AUDIO_CLASSES,
    frozenset({"model_not_installed"}),
    catalog_is_complete=True,
)
SEGMENT = Family(
    "segment",
    SEGMENT_ENV,
    SEGMENT_CLASSES,
    frozenset({"model_not_installed"}),
    catalog_is_complete=True,
)
VIDEO = Family(
    "video",
    VIDEO_ENV,
    ("video",),
    frozenset({"model_not_installed"}),
    catalog_is_complete=True,
)


@dataclass(frozen=True)
class JobTypeSpec:
    name: str
    family: Family
    card: CardEffect = CardEffect()
    leaves_it_resident: bool = False
    journal_identity: bool = False

    @property
    def unloads(self) -> str | None:
        return self.card.takes_off

    @property
    def installable(self) -> bool:
        return self.family.env is not None and self.unloads is None


ECHO_JOB = JobTypeSpec("echo", ECHO)
LOAD_MODEL = JobTypeSpec(
    "load-model", LLM, CardEffect(makes_resident=KIND_LLM), leaves_it_resident=True
)
UNLOAD_MODEL = JobTypeSpec("unload-model", LLM, CardEffect(takes_off=KIND_LLM))
LOAD_VOICE = JobTypeSpec(
    "load-voice", TTS, CardEffect(makes_resident=KIND_TTS), leaves_it_resident=True
)
UNLOAD_VOICE = JobTypeSpec("unload-voice", TTS, CardEffect(takes_off=KIND_TTS))
TTS_JOB = JobTypeSpec(
    "tts", TTS, CardEffect(makes_resident=KIND_TTS)
)
ASR_JOB = JobTypeSpec("asr", ASR, journal_identity=True)
ALIGN_JOB = JobTypeSpec(
    "align", ALIGN, CardEffect(makes_resident=KIND_ALIGN)
)
UNLOAD_ALIGNER = JobTypeSpec("unload-aligner", ALIGN, CardEffect(takes_off=KIND_ALIGN))
ALIGN_LONGFORM = JobTypeSpec("align-longform", ALIGN)
RVC_JOB = JobTypeSpec("rvc", RVC)
DENOISE_JOB = JobTypeSpec(
    "denoise",
    DENOISE,
    CardEffect(makes_resident=KIND_DENOISE),
)
UNLOAD_DENOISER = JobTypeSpec(
    "unload-denoiser", DENOISE, CardEffect(takes_off=KIND_DENOISE)
)
IMAGE_JOB = JobTypeSpec(
    "image", IMAGE, CardEffect(makes_resident=KIND_IMAGE)
)
UNLOAD_IMAGE = JobTypeSpec("unload-image", IMAGE, CardEffect(takes_off=KIND_IMAGE))
LOAD_IMAGE = JobTypeSpec(
    "load-image",
    IMAGE,
    CardEffect(makes_resident=KIND_IMAGE),
    leaves_it_resident=True,
)
AUDIO_JOB = JobTypeSpec(
    "audio", AUDIO, CardEffect(makes_resident=KIND_AUDIO)
)
UNLOAD_AUDIO = JobTypeSpec("unload-audio", AUDIO, CardEffect(takes_off=KIND_AUDIO))
LOAD_AUDIO = JobTypeSpec(
    "load-audio",
    AUDIO,
    CardEffect(makes_resident=KIND_AUDIO),
    leaves_it_resident=True,
)
SEGMENT_JOB = JobTypeSpec(
    "segment", SEGMENT, CardEffect(makes_resident=KIND_SEGMENT)
)
UNLOAD_SEGMENT = JobTypeSpec(
    "unload-segment", SEGMENT, CardEffect(takes_off=KIND_SEGMENT)
)
LOAD_SEGMENT = JobTypeSpec(
    "load-segment",
    SEGMENT,
    CardEffect(makes_resident=KIND_SEGMENT),
    leaves_it_resident=True,
)
VIDEO_JOB = JobTypeSpec(
    "video", VIDEO, CardEffect(makes_resident=KIND_VIDEO)
)
UNLOAD_VIDEO = JobTypeSpec("unload-video", VIDEO, CardEffect(takes_off=KIND_VIDEO))
LOAD_VIDEO = JobTypeSpec(
    "load-video",
    VIDEO,
    CardEffect(makes_resident=KIND_VIDEO),
    leaves_it_resident=True,
)

JOB_TYPE_SPECS: tuple[JobTypeSpec, ...] = (
    ECHO_JOB,
    LOAD_MODEL,
    UNLOAD_MODEL,
    LOAD_VOICE,
    UNLOAD_VOICE,
    TTS_JOB,
    ASR_JOB,
    ALIGN_JOB,
    UNLOAD_ALIGNER,
    ALIGN_LONGFORM,
    RVC_JOB,
    DENOISE_JOB,
    UNLOAD_DENOISER,
    IMAGE_JOB,
    UNLOAD_IMAGE,
    LOAD_IMAGE,
    AUDIO_JOB,
    UNLOAD_AUDIO,
    LOAD_AUDIO,
    SEGMENT_JOB,
    UNLOAD_SEGMENT,
    LOAD_SEGMENT,
    VIDEO_JOB,
    UNLOAD_VIDEO,
    LOAD_VIDEO,
)

BY_NAME: dict[str, JobTypeSpec] = {spec.name: spec for spec in JOB_TYPE_SPECS}

CARD_EFFECTS: dict[str, CardEffect] = {spec.name: spec.card for spec in JOB_TYPE_SPECS}

FAMILIES: tuple[Family, ...] = tuple(dict.fromkeys(spec.family for spec in JOB_TYPE_SPECS))


def spec_of(job_type: str) -> JobTypeSpec | None:
    return BY_NAME.get(job_type)


if len(BY_NAME) != len(JOB_TYPE_SPECS):
    raise TypeError(
        "two JobTypeSpecs in crucible/jobtypes.py share a name: "
        f"{sorted(spec.name for spec in JOB_TYPE_SPECS)}"
    )

_UNNAMED_CLASSES = sorted(
    {name for family in FAMILIES for name in family.capability_classes} - set(CLASS_NAMES)
)
if _UNNAMED_CLASSES:
    raise TypeError(
        f"capability class(es) {_UNNAMED_CLASSES} are named by a job family in "
        "crucible/jobtypes.py and are not in crucible/classnames.py's CLASS_NAMES"
    )

__all__ = [
    "CARD_EFFECTS",
    "ALIGN",
    "ALIGN_ENV",
    "ALIGN_JOB",
    "ALIGN_LONGFORM",
    "ASR",
    "ASR_ENV",
    "ASR_JOB",
    "AUDIO",
    "AUDIO_ENV",
    "AUDIO_JOB",
    "BY_NAME",
    "CardEffect",
    "DENOISE",
    "DENOISE_JOB",
    "ECHO",
    "ECHO_JOB",
    "ENVS",
    "Env",
    "FAMILIES",
    "Family",
    "IMAGE",
    "IMAGE_ENV",
    "IMAGE_JOB",
    "JOB_TYPE_SPECS",
    "JobTypeSpec",
    "LLM",
    "LLM_ENV",
    "LOAD_AUDIO",
    "LOAD_IMAGE",
    "LOAD_MODEL",
    "LOAD_SEGMENT",
    "LOAD_VOICE",
    "RVC",
    "RVC_ENV",
    "RVC_JOB",
    "SEGMENT",
    "SEGMENT_ENV",
    "SEGMENT_JOB",
    "TTS",
    "TTS_ENV",
    "TTS_JOB",
    "UNLOAD_ALIGNER",
    "UNLOAD_AUDIO",
    "UNLOAD_DENOISER",
    "UNLOAD_IMAGE",
    "UNLOAD_MODEL",
    "UNLOAD_SEGMENT",
    "UNLOAD_VIDEO",
    "UNLOAD_VOICE",
    "VIDEO",
    "VIDEO_ENV",
    "VIDEO_JOB",
    "LOAD_VIDEO",
    "spec_of",
]
