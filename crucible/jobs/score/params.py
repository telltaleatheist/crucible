from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ...errors import ApiError
from ...scoremodels import ScoreBackendSpec

# SheetSage2's prompt sets, as YuE's cover skill sends them (scripts/transcribe_cover.py at
# 18a07bb6): the timing, meter, sections and key always, then the melody alone, or the
# melody and the chords.
BASE_PROMPTS: tuple[str, ...] = ("timestamp", "downbeat_meter", "structure", "key")

MELODY_PROMPTS: tuple[str, ...] = ("melody_full",)

HARMONY_PROMPTS: tuple[str, ...] = ("chord_full", "melody_full")

# SheetSage2's own names for the two tasks (transcribe_cover.py `--task`).
MELODY_TASK = "melody-full"

HARMONY_TASK = "full"


class ScoreParams(BaseModel):
    """`params` for a score job. Unknown keys are refused; every key is optional."""

    model_config = ConfigDict(extra="forbid")

    keep_harmony: bool = Field(
        default=False,
        description="True keeps the chords: SheetSage2's `full` task, chord symbols in "
        "the score's Vocal voice. False (the default, as YuE's cover skill does it) is "
        "the `melody-full` task: both melodies, no chords, and YuE2 harmonises the cover "
        "itself.",
    )
    start_s: float = Field(
        default=0.0,
        ge=0,
        description="Seconds into the recording to start at; 0 is its start.",
    )
    max_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Transcribe at most this many seconds from `start_s`; null is to "
        "the end. At most the model's ceiling (360 s for sheetsage2), which also bounds "
        "a recording sent whole.",
    )

    @property
    def task(self) -> str:
        return HARMONY_TASK if self.keep_harmony else MELODY_TASK

    @property
    def prompts(self) -> list[str]:
        return [*BASE_PROMPTS, *(HARMONY_PROMPTS if self.keep_harmony else MELODY_PROMPTS)]

    @property
    def melody_only(self) -> bool:
        return not self.keep_harmony


def refuse_past_the_ceiling(params: ScoreParams, model_id: str, spec: ScoreBackendSpec) -> None:
    if params.max_seconds is not None and params.max_seconds > spec.max_audio_s:
        raise ApiError(
            400,
            "audio_too_long",
            f"max_seconds {params.max_seconds:g} is past {model_id}'s ceiling of "
            f"{spec.max_audio_s} s: {spec.max_audio_note}",
            {"max_seconds": params.max_seconds, "max_audio_s": spec.max_audio_s},
        )


__all__ = [
    "BASE_PROMPTS",
    "HARMONY_PROMPTS",
    "HARMONY_TASK",
    "MELODY_PROMPTS",
    "MELODY_TASK",
    "ScoreParams",
    "refuse_past_the_ceiling",
]
