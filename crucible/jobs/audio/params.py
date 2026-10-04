from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...audiomodels import KIND_WORDS, OPTIONAL_PARAMS, AudioBackendSpec, AudioManifest
from ...errors import ApiError

MAX_SEED = 2**32 - 1

# flac first: it is the default. mp3 is 192 kbps CBR (audiocore.MP3_CBR_192_LEVEL), for
# players and phones that want small files (B-Side, Owen 2026-10-04).
FORMATS: tuple[str, ...] = ("flac", "wav", "mp3")

TEXT_FIELDS: tuple[str, ...] = ("prompt", "tags", "lyrics", "negative_prompt")

# A lyrics line that is only a section tag ([Verse], [Chorus], ...): all an instrumental's
# lyrics may hold, since it sings nothing (a sung word there was silently dropped, 2026-10-03).
SECTION_TAG = re.compile(r"\[[^\[\]]+\]")


class AudioParams(BaseModel):
    """`params` for an audio job. Unknown keys are refused, not ignored; which of these a model takes is in its manifest."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = None
    tags: str | None = None
    lyrics: str | None = None
    negative_prompt: str | None = None
    duration_s: float | None = Field(default=None, gt=0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    steps: int | None = Field(default=None, ge=1)
    cfg: float | None = Field(default=None, ge=0)
    instrumental: bool | None = None
    format: Literal["flac", "wav", "mp3"] = "flac"

    @field_validator(*TEXT_FIELDS)
    @classmethod
    def says_something(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("is empty; send words, or leave the param out")
        return value


def _refusal(code: str, message: str, manifest: AudioManifest, spec: AudioBackendSpec, **extra: Any) -> ApiError:
    return ApiError(
        400,
        code,
        message,
        {"model": manifest.id, "kind": manifest.kind, "backend": spec.backend, **extra},
    )


def _takes(manifest: AudioManifest, spec: AudioBackendSpec) -> list[str]:
    text = [manifest.text_param] + (["lyrics"] if manifest.takes_lyrics else [])
    return [*text, *spec.takes, "seed", "format"]


def _not_taken(param: str, manifest: AudioManifest, spec: AudioBackendSpec) -> ApiError:
    why = spec.why_not(param) or (
        f"{manifest.display} makes {KIND_WORDS[manifest.kind]} and has no {param}"
    )
    return _refusal(
        "audio_param_unsupported",
        f"{manifest.id} does not take {param!r}: {why}. Drop {param!r}; this model "
        f"takes {_takes(manifest, spec)}",
        manifest,
        spec,
        param=param,
        takes=_takes(manifest, spec),
    )


def _refuse_text(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    wanted = manifest.text_param
    other = "tags" if wanted == "prompt" else "prompt"
    if getattr(params, other) is not None:
        raise _refusal(
            "audio_param_unsupported",
            f"{manifest.id} makes {KIND_WORDS[manifest.kind]} and reads its "
            f"description from {wanted!r}, not {other!r}. Send the same words as "
            f"{wanted!r}",
            manifest,
            spec,
            param=other,
            takes=_takes(manifest, spec),
        )
    if params.lyrics is not None and not manifest.takes_lyrics:
        raise _not_taken("lyrics", manifest, spec)
    if params.instrumental and params.lyrics is not None:
        sung = [line for line in params.lyrics.splitlines()
                if line.strip() and not SECTION_TAG.fullmatch(line.strip())]
        if sung:
            raise _refusal(
                "audio_param_conflict",
                f"an instrumental sings nothing, but these lyrics have words to sing "
                f"({sung[0].strip()!r} first). Untick instrumental to sing them, or send "
                "only section tags such as [Verse] and [Chorus] to shape the instrumental",
                manifest,
                spec,
                param="lyrics",
            )
    # An instrumental song sings nothing: lyrics, if sent, only shape the score YuE2 plans
    # (its sections), so they are not required.
    sung = manifest.takes_lyrics and not params.instrumental
    missing = [
        name
        for name in (wanted, *(["lyrics"] if sung else []))
        if getattr(params, name) is None
    ]
    if missing:
        raise _refusal(
            "audio_param_missing",
            f"{manifest.id} needs {' and '.join(repr(m) for m in missing)}; "
            "docs/AUDIO.md shows what to write in each",
            manifest,
            spec,
            missing=missing,
        )


def _refuse_ranges(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    if params.duration_s is not None and params.duration_s > spec.max_duration_s:
        raise _refusal(
            "audio_too_long",
            f"{params.duration_s:g} s is longer than {manifest.id} makes on "
            f"{spec.backend}: at most {spec.max_duration_s} s. Ask for "
            f"{spec.max_duration_s} s or less, and join pieces if you need more",
            manifest,
            spec,
            duration_s=params.duration_s,
            max_duration_s=spec.max_duration_s,
        )
    for name, ceiling in (("steps", spec.max_steps), ("cfg", spec.max_cfg)):
        value = getattr(params, name)
        if value is not None and ceiling is not None and value > ceiling:
            raise _refusal(
                "audio_param_out_of_range",
                f"{name} {value:g} is above {manifest.id}'s ceiling of {ceiling:g}; "
                f"send {ceiling:g} or less",
                manifest,
                spec,
                param=name,
                maximum=ceiling,
            )


def refuse_what_the_model_cannot_take(
    params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec
) -> None:
    _refuse_text(params, manifest, spec)
    for param in OPTIONAL_PARAMS:
        if getattr(params, param) is not None and param not in spec.takes:
            raise _not_taken(param, manifest, spec)
    _refuse_ranges(params, manifest, spec)


@dataclass(frozen=True)
class Settled:
    duration_s: float | None
    steps: int | None
    cfg: float | None
    instrumental: bool
    seed: int


def settle(params: AudioParams, spec: AudioBackendSpec, seed: int) -> Settled:
    def chosen(value: Any, default: Any) -> Any:
        return default if value is None else value

    return Settled(
        duration_s=(
            None
            if "duration_s" not in spec.takes
            else float(chosen(params.duration_s, spec.default_duration_s))
        ),
        steps=None if "steps" not in spec.takes else chosen(params.steps, spec.default_steps),
        cfg=None if "cfg" not in spec.takes else float(chosen(params.cfg, spec.default_cfg)),
        instrumental=bool(params.instrumental),
        seed=seed,
    )


__all__ = [
    "AudioParams",
    "FORMATS",
    "MAX_SEED",
    "Settled",
    "refuse_what_the_model_cannot_take",
    "settle",
]
