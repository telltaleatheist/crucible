from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...jobtypes import ALIGN_LONGFORM
from ..align import QWEN3_LANGUAGES

JOB_TYPE_NAME = ALIGN_LONGFORM.name

ALIGNER_MODEL = "qwen3-aligner"

STAGES: tuple[str, ...] = ("transcribe", "coarse-align", "align", "write")

ARTIFACTS: tuple[str, ...] = ("alignment.vtt", "align-report.json")


class LongformSentence(BaseModel):
    """One sentence of the book in reading order; `kind` comes from the client."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    text: str
    kind: str = "prose"

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError(
                "a sentence with no text cannot be placed in audio. An empty row "
                "here is an extraction defect on the client's side, and aligning "
                "around it would shift every cue after it"
            )
        return value


class AlignLongformParams(BaseModel):
    """Everything the client states. Everything else is the server's."""

    model_config = ConfigDict(extra="forbid")

    language: str
    sentences: list[LongformSentence]
    rough_model: str = "small"
    chunk_s: float = Field(default=240.0, gt=0)
    hole_min_s: float = Field(default=2.0, ge=0)
    snap_silence_s: float = Field(default=0.35, ge=0)
    silence_source: str = "decoded"

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value in QWEN3_LANGUAGES:
            return value
        raise ValueError(
            f"{value!r} is not a language Qwen3-ForcedAligner supports; it takes "
            f"one of {sorted(QWEN3_LANGUAGES)}. It does not fall back to English "
            "for a language it was not trained on — it places words badly, and a "
            "silently mis-aligned book is worse than a refused one. Refused here, "
            "before the pool, rather than after a book's worth of bad cues"
        )

    @field_validator("sentences")
    @classmethod
    def has_sentences(cls, value: list[LongformSentence]) -> list[LongformSentence]:
        if not value:
            raise ValueError(
                "no sentences were given, so there is nothing to place in the "
                "audio. A transcript job with no text is a request that can only "
                "produce an empty VTT, which reads as a successful run that "
                "aligned nothing"
            )
        return value

    @field_validator("sentences")
    @classmethod
    def indexes_are_a_sequence(
        cls, value: list[LongformSentence]
    ) -> list[LongformSentence]:
        seen = [s.index for s in value]
        duplicates = sorted({i for i in seen if seen.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"sentence index {duplicates} appears more than once; an index "
                "names one sentence and the VTT is keyed by it"
            )
        if seen != sorted(seen):
            first = next(
                i for i in range(1, len(seen)) if seen[i] < seen[i - 1]
            )
            raise ValueError(
                f"sentence indexes are not in reading order (index {seen[first]} "
                f"follows {seen[first - 1]}). The coarse alignment walks the book "
                "against a rough transcript and assumes both move forward "
                "together; out of order it still produces cues, and they are wrong"
            )
        return value

    @field_validator("silence_source")
    @classmethod
    def known_silence_source(cls, value: str) -> str:
        if value != "decoded":
            raise ValueError(
                f"{value!r} is not a silence source this server offers; the only "
                "one is 'decoded', the audio this job was given. An unrecognised "
                "value is refused rather than ignored, because a run that placed "
                "edges with no silence map is the pre-2026-09-06 build under a "
                "new label"
            )
        return value

    @property
    def language_name(self) -> str:
        return QWEN3_LANGUAGES[self.language]


QWEN3_MAX_AUDIO_S = 300.0


def chunk_fits_the_aligner(params: AlignLongformParams) -> None:
    if params.chunk_s > QWEN3_MAX_AUDIO_S:
        raise ValueError(
            f"chunk_s is {params.chunk_s:g} s and Qwen3-ForcedAligner places "
            f"timestamps within up to {QWEN3_MAX_AUDIO_S:g} s. A longer window is "
            "refused rather than split here: splitting would silently change the "
            "alignment this job was asked for"
        )


def validate(params: dict[str, Any]) -> AlignLongformParams:
    parsed = AlignLongformParams.model_validate(params)
    chunk_fits_the_aligner(parsed)
    return parsed
