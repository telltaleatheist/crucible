from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...audiomodels import KIND_WORDS, OPTIONAL_PARAMS, AudioBackendSpec, AudioManifest
from ...errors import ApiError
from . import planning

MAX_SEED = 2**32 - 1

# The shortest song a length range may ask for. The shortest songs measured through
# Crucible ran 38 to 62 s (six on the 3090 Ti, 2026-10-10), and an instrumental's
# smallest structure is one section of a pool set with its intro and outro; below 30 s
# no planning structure can land, so asking is refused rather than run to its budget.
MIN_SONG_SECONDS = 30.0

# flac first: it is the default. mp3 is 192 kbps CBR (audiocore.MP3_CBR_192_LEVEL), for
# players and phones that want small files (B-Side, Owen 2026-10-04).
FORMATS: tuple[str, ...] = ("flac", "wav", "mp3")

TEXT_FIELDS: tuple[str, ...] = (
    "prompt", "tags", "lyrics", "planning_lyrics", "planning_set", "negative_prompt"
)

# A lyrics line that is only a section tag ([Verse], [Chorus], ...): all an instrumental's
# lyrics may hold, since it sings nothing (a sung word there was silently dropped, 2026-10-03).
SECTION_TAG = re.compile(r"\[[^\[\]]+\]")


class AudioParams(BaseModel):
    """`params` for an audio job. Unknown keys are refused, not ignored; which of these a model takes is in its manifest."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(
        default=None,
        description="Sound effects and music (Stable Audio): the sound described "
        "(docs/AUDIO.md); required there, refused by a song model.",
    )
    tags: str | None = Field(
        default=None,
        description="Songs (YuE2): the style as comma-separated genre, instruments, "
        "voice, language and tempo; required there, refused by Stable Audio.",
    )
    lyrics: str | None = Field(
        default=None,
        description="Songs (YuE2): sections tagged [Verse], [Chorus] and so on, "
        "separated by blank lines; required unless `instrumental`, where only "
        "section tags are allowed.",
    )
    planning_lyrics: str | None = Field(
        default=None,
        description="Songs (YuE2), with `instrumental` only: lyrics the score is planned "
        "from and never sung, so the melody has a sung song's bounded phrases; then it "
        "moves to the instrument. Sections tagged [Verse], [Chorus] and so on, at most "
        f"{planning.MAX_LINES} lines and {planning.MAX_CHARS} characters. Null picks a set "
        "from the server's pool by the seed.",
    )
    planning_set: str | None = Field(
        default=None,
        description="Songs (YuE2), with `instrumental` only and never beside "
        "`planning_lyrics` or `lyrics`: the id of the server's pool set to plan the "
        "score from (GET /v1/playground lists the ids), so an album can give each track "
        "its own structure. Null picks a set by the seed.",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Taken only by a model whose manifest lists it; the shipped "
        "models refuse it by name.",
    )
    duration_s: float | None = Field(
        default=None,
        gt=0,
        description="Seconds of sound, for models that take it (Stable Audio: at "
        "most 120 sfx, 380 music); null is the model's default. A song's length "
        "follows its lyrics; ask a range with `min_duration_s` and `max_duration_s`.",
    )
    min_duration_s: float | None = Field(
        default=None,
        gt=0,
        description="Songs (YuE2): the shortest the song may be, in seconds "
        f"({MIN_SONG_SECONDS:g} up to the model's longest). Checked against the score "
        "before anything is composed: an instrumental planned from the server's pool is "
        "re-planned to land in the range, a song from the client's words is refused "
        "`song_length_out_of_range` (docs/AUDIO.md \"Song length\"). Null: no minimum.",
    )
    max_duration_s: float | None = Field(
        default=None,
        gt=0,
        description="Songs (YuE2): the longest the song may be, in seconds, above "
        "`min_duration_s` when both are sent; as `min_duration_s`. Null: no maximum "
        "beyond the model's own.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=MAX_SEED,
        description="0 to 4294967295; null lets the server choose one and report it.",
    )
    steps: int | None = Field(
        default=None,
        ge=1,
        description="Denoising steps, for models that take them (Stable Audio, up to "
        "its ceiling); null is the model's default.",
    )
    cfg: float | None = Field(
        default=None,
        ge=0,
        description="Guidance toward the tags and lyrics, for models that take it "
        "(YuE2, up to its ceiling); above 1 runs the model twice per token. Null is "
        "the model's default.",
    )
    instrumental: bool | None = Field(
        default=None,
        description="Songs (YuE2): true renders the planned melody on an instrument, "
        "so nothing is sung. Null is false.",
    )
    format: Literal["flac", "wav", "mp3"] = Field(
        default="flac",
        description="The artifact: `flac` (24-bit), `wav` (24-bit PCM) or `mp3` "
        "(192 kbps CBR).",
    )

    @field_validator(*TEXT_FIELDS)
    @classmethod
    def says_something(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("is empty; send words, or leave the param out")
        return value

    @field_validator("planning_lyrics")
    @classmethod
    def plans_something(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                planning.check(value)
            except planning.PlanningLyricsError as exc:
                raise ValueError(str(exc)) from None
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
    if params.planning_lyrics is not None:
        _refuse_planning_lyrics(params, manifest, spec)
    if params.planning_set is not None:
        _refuse_planning_set(params, manifest, spec)
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


def _refuse_planning_lyrics(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    if "instrumental" not in spec.takes:
        raise _not_taken("planning_lyrics", manifest, spec)
    if not params.instrumental:
        raise _refusal(
            "audio_param_conflict",
            "planning_lyrics plan an instrumental and are never sung, but this song is not "
            "instrumental. Send `instrumental: true` with them, or send the words as "
            "`lyrics` to sing them",
            manifest,
            spec,
            param="planning_lyrics",
        )
    if params.lyrics is not None:
        raise _refusal(
            "audio_param_conflict",
            "an instrumental is planned from its planning_lyrics or from its lyrics' section "
            "tags, not both. Drop `lyrics` to plan from the planning lyrics, or drop "
            "`planning_lyrics` to plan from the section tags alone",
            manifest,
            spec,
            param="planning_lyrics",
        )


def _refuse_planning_set(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    if "instrumental" not in spec.takes:
        raise _not_taken("planning_set", manifest, spec)
    if not params.instrumental:
        raise _refusal(
            "audio_param_conflict",
            "planning_set names the pool set an instrumental's score is planned from, but "
            "this song is not instrumental. Send `instrumental: true` with it, or drop it",
            manifest,
            spec,
            param="planning_set",
        )
    for other in ("planning_lyrics", "lyrics"):
        if getattr(params, other) is not None:
            raise _refusal(
                "audio_param_conflict",
                f"an instrumental is planned from one thing: planning_set names a pool set, "
                f"and {other} would plan it instead. Drop one of `planning_set` and `{other}`",
                manifest,
                spec,
                param="planning_set",
            )
    pool = planning_pool(spec)
    try:
        planning.named(pool, params.planning_set)
    except planning.PlanningLyricsError as exc:
        raise _refusal(
            "planning_set_unknown",
            str(exc),
            manifest,
            spec,
            param="planning_set",
            planning_sets=[entry.id for entry in pool],
        ) from None


def planning_pool(spec: AudioBackendSpec) -> list[planning.PlanningSet]:
    """The engine's pool; an unreadable one is this build's fault, never the request's."""
    try:
        return planning.load_pool(spec.engine)
    except planning.PlanningLyricsError as exc:
        raise ApiError(500, "planning_lyrics_unavailable", str(exc), {"engine": spec.engine}) from None


def _refuse_length_range(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    for name in ("min_duration_s", "max_duration_s"):
        value = getattr(params, name)
        if value is not None and not MIN_SONG_SECONDS <= value <= spec.max_duration_s:
            raise _refusal(
                "audio_param_out_of_range",
                f"{name} {value:g} is outside what {manifest.id} can aim a song at: "
                f"{MIN_SONG_SECONDS:g} to {spec.max_duration_s} s",
                manifest,
                spec,
                param=name,
                minimum=MIN_SONG_SECONDS,
                maximum=spec.max_duration_s,
            )
    low, high = params.min_duration_s, params.max_duration_s
    if low is not None and high is not None and low >= high:
        raise _refusal(
            "audio_param_conflict",
            f"min_duration_s {low:g} is not below max_duration_s {high:g}; send a range "
            "with the shorter end first",
            manifest,
            spec,
            param="min_duration_s",
        )


def _refuse_ranges(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    _refuse_length_range(params, manifest, spec)
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
    # What an instrumental is planned from (planning.record): the client's
    # planning_lyrics, or the pool's set for this seed. None for a sung song, a sound
    # without a score, and an instrumental shaped by the section tags in its `lyrics`.
    planning_lyrics: dict[str, Any] | None
    # The range a song's length is checked against after its score, before composing
    # (planning.LengthRange); None when the client sent neither end.
    length_range: planning.LengthRange | None = None


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
        planning_lyrics=_planning_lyrics(params, spec, seed),
        length_range=(
            None
            if params.min_duration_s is None and params.max_duration_s is None
            else planning.LengthRange(
                params.min_duration_s, params.max_duration_s, float(spec.max_duration_s)
            )
        ),
    )


def _planning_lyrics(params: AudioParams, spec: AudioBackendSpec, seed: int) -> dict[str, Any] | None:
    if params.planning_lyrics is not None:
        return planning.record(planning.REQUEST, params.planning_lyrics, None)
    if not params.instrumental or params.lyrics is not None:
        return None
    pool = planning.load_pool(spec.engine)
    if params.planning_set is not None:
        chosen = planning.named(pool, params.planning_set)
    else:
        chosen = planning.pick(pool, seed)
    return planning.record(
        planning.POOL, chosen.lyrics, chosen.id, requested=params.planning_set is not None
    )


__all__ = [
    "AudioParams",
    "FORMATS",
    "MAX_SEED",
    "MIN_SONG_SECONDS",
    "Settled",
    "refuse_what_the_model_cannot_take",
    "settle",
]
