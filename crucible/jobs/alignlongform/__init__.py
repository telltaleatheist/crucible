"""The `align-longform` job type's CONTRACT — what a whole-audiobook align takes.

`docs/PLAN.md` / bookforge `docs/CRUCIBLE_ROLLOUT_PLAN.md` §B7, ruled by Owen on
2026-09-15: *"align longform, is that the generate-sentences logic? that should be
a gpu job."*

NOT REGISTERED AS A SERVABLE TYPE YET, and that is deliberate
-------------------------------------------------------------
This module is the contract and its refusals. It is **not** in
`crucible/jobs/__init__.py`'s registry, so nothing can queue it. A job type whose
`run()` does not work is a type a client can select and then discover mid-book —
the "offered then refused" shape this codebase spends a lot of comments
preventing — so the door stays shut until the worker behind it exists.

What lands first is the part that is pure logic and fully testable without a card:
the params, and the refusals that must happen BEFORE the pool rather than after a
book's worth of bad cues.

Why the existing `align` job is not this job
--------------------------------------------
`align` is `{chunks:[{index,text}], inputs:{"<index>.flac"}}` — a caller who
ALREADY KNOWS which seconds of audio hold which sentences. That is true of a
render (narrator wrote the chunks) and is exactly what this act must DISCOVER.
The stages are `transcribe` (faster-whisper over the whole m4b, CPU) →
`coarse-align` (a DTW of the ebook's sentences onto that rough transcript, which
is what produces the chunk spans) → `align` (Qwen3 per chunk, on the card) →
the whisper-authority gate, monotonic clamps, drift correction and silence snap →
`write`. Only the third stage has the shape `align` offers, and the two before it
are most of the wall clock and all of the knowledge.

The whole step travels, CPU stages included
--------------------------------------------
Owen's TTS ruling applies here by the same reasoning: *"the entire tts step goes
to the other system. That includes anything the step needs to do even if it's
cpu."* So `transcribe` and `coarse-align` run HERE, on the server, and the job
charges one slot for its whole duration. Slicing locally to send only the GPU
stage would move the cheap stage and keep the expensive ones.

One input, and it is the m4b
-----------------------------
The audiobook as it is, not a wav. A 16 h book is roughly 460 MB at 64 kbps where
the 16 kHz mono wav the script makes internally would be ~1.84 GB; the server does
that conversion. The EPUB never crosses — it is the client's book, and the
sentences it extracted are text.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..align import QWEN3_LANGUAGES

#: The job type's name on the wire. One place, so a test and the registry cannot
#: disagree about it the day this is registered.
JOB_TYPE_NAME = "align-longform"

#: The aligner these sentences are placed with. Same weights as `align`; what
#: differs is the orchestration around them, not the model.
ALIGNER_MODEL = "qwen3-aligner"

#: Stage names, in order, as they appear on `progress` events. BookForge's
#: generate-sentences row draws a stacked bar per stage and matches on these
#: exact strings (`GsProgressEvent.stages`), so they are part of the contract
#: rather than log prose.
STAGES: tuple[str, ...] = ("transcribe", "coarse-align", "align", "write")

#: What the job returns. The same VTT the local script writes, plus its report.
ARTIFACTS: tuple[str, ...] = ("alignment.vtt", "align-report.json")


class LongformSentence(BaseModel):
    """One sentence of the book, in reading order.

    `kind` is carried and never inferred. Headings are already stamped at
    extraction (bookforge `narrator-unspoken-glyphs-and-caps-headings`), and a
    classifier here would be a second opinion about a fact the client already
    holds — which is how two owners of one fact get created.
    """

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
    #: The faster-whisper size for the rough pass. STATED rather than defaulted:
    #: it trades wall clock against how good the coarse alignment's spans are,
    #: and the app has always chosen it.
    rough_model: str = "small"
    #: Seconds per Qwen3 window. The `align` job refuses a chunk past 300 s
    #: (the model card's own limit), so the spans this produces must stay under
    #: it — see `chunk_fits_the_aligner`.
    chunk_s: float = Field(default=240.0, gt=0)
    #: A gap in the cue sequence longer than this is a hole to be reported.
    hole_min_s: float = Field(default=2.0, ge=0)
    #: How near a silence an edge may be snapped.
    snap_silence_s: float = Field(default=0.35, ge=0)
    #: Where the silence map comes from. `decoded` reads the audio this job was
    #: given; nothing else is offered yet, and an unknown value is refused rather
    #: than ignored.
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
        """Indexes must be unique AND in order.

        Unique because an index NAMES a sentence and the VTT is keyed by it. In
        order because the whole coarse-align stage rests on the book's sentences
        being monotonic in time; a shuffled list would align, produce cues, and
        be wrong in a way no check downstream is looking for.
        """
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


#: The `align` job's own ceiling, restated here because this job PRODUCES the
#: spans that job's successor will honour. The model card says timestamps
#: "within up to 5 minutes"; a window past it is not split silently.
QWEN3_MAX_AUDIO_S = 300.0


def chunk_fits_the_aligner(params: AlignLongformParams) -> None:
    """Refuse a `chunk_s` the aligner could not place, before anything decodes.

    Separate from the field validator on purpose: `gt=0` is a fact about the
    number, and this is a fact about the MODEL. Keeping them apart means the
    message names the right thing when the model's ceiling changes.
    """
    if params.chunk_s > QWEN3_MAX_AUDIO_S:
        raise ValueError(
            f"chunk_s is {params.chunk_s:g} s and Qwen3-ForcedAligner places "
            f"timestamps within up to {QWEN3_MAX_AUDIO_S:g} s. A longer window is "
            "refused rather than split here: splitting would silently change the "
            "alignment this job was asked for"
        )


def validate(params: dict[str, Any]) -> AlignLongformParams:
    """Parse and check, raising `ValueError` with a sentence that names the cause.

    The one door, so the preflight and any test agree about what is legal.
    """
    parsed = AlignLongformParams.model_validate(params)
    chunk_fits_the_aligner(parsed)
    return parsed
