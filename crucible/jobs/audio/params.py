from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...audiomodels import (
    HEADER_PARAMS,
    KIND_WORDS,
    NEGATIVE_BRANCH_PARAMS,
    OPTIONAL_PARAMS,
    AudioBackendSpec,
    AudioManifest,
)
from ...errors import ApiError
from ..base import validate_member_name
from . import planning, scoreheader

MAX_SEED = 2**32 - 1

# The shortest song a length range may ask for. The shortest songs measured through
# Crucible ran 38 to 62 s (six on the 3090 Ti, 2026-10-10), and an instrumental's
# smallest structure is one section of a pool set with its intro and outro; below 30 s
# no planning structure can land, so asking is refused rather than run to its budget.
MIN_SONG_SECONDS = 30.0

# flac first: it is the default. mp3 is 192 kbps CBR (workerio.MP3_CBR_192_LEVEL), for
# players and phones that want small files (B-Side, Owen 2026-10-04).
FORMATS: tuple[str, ...] = ("flac", "wav", "mp3")

TEXT_FIELDS: tuple[str, ...] = (
    "prompt", "tags", "lyrics", "planning_lyrics", "planning_set", "negative_prompt",
    "negative_tags", "negative_lyrics", "score"
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
    negative_tags: str | None = Field(
        default=None,
        description="Songs (YuE2), with `cfg` above 1: the style of the song's CFG negative "
        "branch, the tags guidance steers AWAY from (e.g. \"lead vocals, singing, choir\"). "
        "The negative is the song's own request with these tags and `negative_lyrics` in "
        "place of its own: the same score, cot and seed. Null beside `negative_lyrics` is "
        "no tags; both null is YuE2's stock negative, its instruction and the score alone "
        "(docs/AUDIO.md \"The negative branch\").",
    )
    negative_lyrics: str | None = Field(
        default=None,
        description="Songs (YuE2), with `cfg` above 1: the lyrics of the song's CFG negative "
        "branch, words guidance steers AWAY from, written as YuE2 takes lyrics (sections "
        "tagged [Verse], [Chorus] and so on). Null beside `negative_tags` is no lyrics; "
        "both null is YuE2's stock negative.",
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
        description="Songs (YuE2): the short end of a length to aim at, in seconds "
        f"({MIN_SONG_SECONDS:g} up to the model's longest). A nudge, never a guarantee: an "
        "instrumental planned from the server's pool has its set sized toward the range; "
        "a song from the client's words is made as written and the range only recorded. "
        "The song is always made in one pass and `audio.length` says where its score "
        "landed (docs/AUDIO.md \"Song length\"). Null: no minimum.",
    )
    max_duration_s: float | None = Field(
        default=None,
        gt=0,
        description="Songs (YuE2): the long end of a length to aim at, in seconds, above "
        "`min_duration_s` when both are sent; a nudge as `min_duration_s` is. Null: no "
        "maximum beyond the model's own.",
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
    score: str | None = Field(
        default=None,
        description="Covers (YuE2): the name of the job's input that carries a score "
        "(YuE2's two-voice ABC, uploaded or another job's held artifact), rendered "
        "instead of a score YuE2 plans. With `instrumental` its Vocal melody moves to the "
        "instrument and nothing is sung; without it the melody is sung with `lyrics`, "
        "written to the score's phrases. Never beside planning_lyrics, planning_set or "
        "a length range (docs/AUDIO.md \"Covers\").",
    )
    chords: Literal["keep", "strip"] | None = Field(
        default=None,
        description="Covers (YuE2), only beside `score`: keep the score's chord symbols "
        "(YuE2 renders with cot `full`, the harmony fixed) or strip them (cot `melody`, "
        "the accompaniment free). Null keeps them for an instrumental cover and strips "
        "them for a sung one, as YuE2's own skill does.",
    )
    key: str | None = Field(
        default=None,
        description="Songs (YuE2), planned only (never beside `score`): the key the score "
        "is written in, fixed as its `K:` line before YuE2 plans the rest - one of ABC's 15 "
        "major keys (Cb Gb Db Ab Eb Bb F C G D A E B F# C#) or 15 minor keys (Abm Ebm Bbm "
        "Fm Cm Gm Dm Am Em Bm F#m C#m G#m D#m A#m); no modes. Null: YuE2 chooses "
        "(docs/AUDIO.md \"Key, meter and tempo\").",
    )
    meter: str | None = Field(
        default=None,
        description="Songs (YuE2), planned only: the meter, fixed as the score's `M:` line - "
        f"a fraction such as \"4/4\", \"3/4\" or \"6/8\", 1 to {scoreheader.MAX_BEATS} "
        "beats over 2, 4, 8 or 16. Null: YuE2 chooses.",
    )
    bpm: int | None = Field(
        default=None,
        ge=scoreheader.MIN_BPM,
        le=scoreheader.MAX_BPM,
        description="Songs (YuE2), planned only: the tempo in quarter notes a minute, "
        f"{scoreheader.MIN_BPM} to {scoreheader.MAX_BPM}, fixed as the score's `Q:1/4=` "
        "line. Null: YuE2 chooses.",
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

    @field_validator("score")
    @classmethod
    def names_an_input(cls, value: str | None) -> str | None:
        if value is not None:
            validate_member_name(value)
        return value

    @field_validator("key")
    @classmethod
    def is_a_key(cls, value: str | None) -> str | None:
        return None if value is None else scoreheader.check_key(value)

    @field_validator("meter")
    @classmethod
    def is_a_meter(cls, value: str | None) -> str | None:
        return None if value is None else scoreheader.check_meter(value)

    @field_validator("bpm", mode="before")
    @classmethod
    def is_whole(cls, value: Any) -> Any:
        # 76.0 or "76" would be read as 76 by pydantic's lax mode; a tempo is sent as the
        # integer the score writes, so anything else is refused rather than rounded.
        if value is not None and type(value) is not int:
            raise ValueError(f"must be an integer number of quarter notes a minute, not {value!r}")
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
    if params.score is not None or params.chords is not None:
        _refuse_score(params, manifest, spec)
    if params.instrumental and params.lyrics is not None:
        sung = sung_lines(params.lyrics)
        if sung:
            raise _refusal(
                "audio_param_conflict",
                f"an instrumental sings nothing, but these lyrics have words to sing "
                f"({sung[0]!r} first). Untick instrumental to sing them, or send "
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


# What a given score replaces: the words a score is planned from and the length a planned
# score is sized toward. Beside a score they would mean nothing, so they are refused.
PLANNING_ONLY: tuple[str, ...] = (
    "planning_lyrics", "planning_set", "min_duration_s", "max_duration_s"
)


def sung_lines(lyrics: str) -> list[str]:
    """The lines of `lyrics` that are words to sing, not section tags or blanks."""
    return [line.strip() for line in lyrics.splitlines()
            if line.strip() and not SECTION_TAG.fullmatch(line.strip())]


def _refuse_score(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    """A cover (docs/AUDIO.md "Covers"): the score is the plan, so nothing that plans or
    sizes one is taken beside it; an instrumental cover sings nothing and takes no lyrics;
    a sung cover needs words to sing."""
    for param in ("score", "chords"):
        if getattr(params, param) is not None and param not in spec.takes:
            raise _not_taken(param, manifest, spec)
    if params.score is None:
        raise _refusal(
            "audio_param_conflict",
            "chords keeps or strips the chord symbols of a given score, and this song has "
            "none. Send `score` naming the input that carries it, or drop `chords`",
            manifest,
            spec,
            param="chords",
        )
    for param in HEADER_PARAMS:
        if getattr(params, param) is not None:
            raise _refusal(
                "audio_param_conflict",
                f"a cover renders the score it is given, and that score's header already "
                f"states its key (K:), meter (M:) and tempo (Q:); {param} fixes the header of "
                f"a score YuE2 plans and means nothing beside `score`. Drop {param!r}, or "
                "write it into the score's own header",
                manifest,
                spec,
                param=param,
            )
    for param in PLANNING_ONLY:
        if getattr(params, param) is not None:
            raise _refusal(
                "audio_param_conflict",
                f"a cover renders the score it is given, so nothing is planned and nothing "
                f"is sized: {param} means nothing beside `score`. Drop {param!r}; the "
                "score's own bars and tempo are the song's length",
                manifest,
                spec,
                param=param,
            )
    if params.instrumental:
        if params.lyrics is not None:
            raise _refusal(
                "audio_param_conflict",
                "an instrumental cover sings nothing, and its section tags come from the "
                "score itself. Drop `lyrics`, or untick instrumental to sing them to the "
                "score's melody",
                manifest,
                spec,
                param="lyrics",
            )
        return
    if params.lyrics is None or not sung_lines(params.lyrics):
        raise _refusal(
            "audio_param_missing",
            "a sung cover sings new words to the score's Vocal melody, and these lyrics "
            f"{'are missing' if params.lyrics is None else 'have only section tags'}. Send "
            "`lyrics` written to the score's phrases, in sections tagged [Verse], [Chorus] "
            "and so on, or send `instrumental: true` to play the melody",
            manifest,
            spec,
            missing=["lyrics"],
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
    _refuse_negative(params, manifest, spec)


def _refuse_negative(params: AudioParams, manifest: AudioManifest, spec: AudioBackendSpec) -> None:
    """A written negative branch exists only under guidance: with cfg 1 YuE2 runs no
    negative branch at all, so the words would be silently ignored. Refused instead."""
    sent = [name for name in NEGATIVE_BRANCH_PARAMS if getattr(params, name) is not None]
    if not sent:
        return
    cfg = spec.default_cfg if params.cfg is None else params.cfg
    if not cfg > 1:
        whose = f"{manifest.id}'s default cfg" if params.cfg is None else "cfg"
        raise _refusal(
            "audio_param_conflict",
            f"{' and '.join(sent)} write the CFG negative branch, which runs only with cfg "
            f"above 1; {whose} is {cfg:g}, where no negative branch runs and the words would "
            "be ignored. Send cfg above 1 (2.0 is where it was measured), or drop "
            f"{' and '.join(repr(name) for name in sent)}",
            manifest,
            spec,
            param=sent[0],
            cfg=cfg,
        )


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
    # The song's CFG negative branch as the client wrote it ({tags, lyrics}; the one not
    # sent is ""); None for YuE2's stock negative (docs/AUDIO.md "The negative branch").
    negative: dict[str, str] | None = None
    # The planned score's header fields the client fixed ({key, meter, bpm}, null where
    # not sent); None when none was (docs/AUDIO.md "Key, meter and tempo").
    header: dict[str, Any] | None = None


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
        negative=_negative(params),
        header=(
            None
            if all(getattr(params, name) is None for name in HEADER_PARAMS)
            else {name: getattr(params, name) for name in HEADER_PARAMS}
        ),
    )


def _negative(params: AudioParams) -> dict[str, str] | None:
    """The negative branch's two text fields. One sent alone leaves the other empty, not
    copied from the song: the negative holds only what the client wrote against, which
    is the nearest it can be to the stock negative (no tags, no lyrics) and still differ."""
    if params.negative_tags is None and params.negative_lyrics is None:
        return None
    return {
        "tags": "" if params.negative_tags is None else params.negative_tags,
        "lyrics": "" if params.negative_lyrics is None else params.negative_lyrics,
    }


def _planning_lyrics(params: AudioParams, spec: AudioBackendSpec, seed: int) -> dict[str, Any] | None:
    if params.planning_lyrics is not None:
        return planning.record(planning.REQUEST, params.planning_lyrics, None)
    # A cover's score is given, not planned (docs/AUDIO.md "Covers").
    if not params.instrumental or params.lyrics is not None or params.score is not None:
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
