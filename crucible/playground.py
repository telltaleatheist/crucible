"""The playground's pages: one per image, video and audio model this build declares, each
with the few params that model takes, read from its manifest, and whether this server can
run it now. A model install-on-submit can fetch is offered as it is: its first job
downloads what it lacks. Only what that cannot fix is unavailable, with the reason."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import catalog, installonsubmit, weights
from .audiomodels import KIND_WORDS, MUSIC, SFX, SONG
from .jobenv import INSTALLER_FOR
from .jobs import audio as audio_job
from .jobs import disabled_error
from .jobs import image as image_job
from .jobs import video as video_job
from .jobs.audio.params import FORMATS, MIN_SONG_SECONDS, planning_pool
from .jobs.template import ManifestCatalog
from .jobtypes import spec_of
from .tasks import env_installed

MAX_SEED = 2**32 - 1

READY = "ready"
DOWNLOAD = "download"
UNAVAILABLE = "unavailable"

INSTALLS_A_DISABLED_TYPE = frozenset({"not_installed", "undecided"})

TEXT = "text"
INTEGER = "integer"
NUMBER = "number"
BOOLEAN = "boolean"
CHOICE = "choice"
TAGS = "tags"

# One-click style chips for a song model's `tags` (crucible/audio/tags/song.toml). Read on
# every page build so an edit to the file shows without a restart.
SONG_TAGS_FILE = Path(__file__).resolve().parent / "audio" / "tags" / "song.toml"


def _song_tags_document() -> dict[str, Any]:
    with SONG_TAGS_FILE.open("rb") as handle:
        return tomllib.load(handle)


def song_tag_conflicts() -> dict[str, list[dict[str, str]]]:
    """Each tag (lower-cased) -> the tags it contradicts, each with why, from the file's
    [[conflict]] tables: `exclusive` (any two clash) and `between` (each side against the
    other). Every tag a conflict names must be one of the file's suggestions."""
    document = _song_tags_document()
    known = {tag.casefold() for group in document.get("group", []) for tag in group.get("tags", [])}
    pairs: dict[str, dict[str, str]] = {}

    def clash(a: str, b: str, why: str) -> None:
        for tag in (a, b):
            if tag.casefold() not in known:
                raise ValueError(f"{SONG_TAGS_FILE}: conflict names {tag!r}, which no [[group]] offers")
        if a.casefold() != b.casefold():
            pairs.setdefault(a.casefold(), {})[b] = why
            pairs.setdefault(b.casefold(), {})[a] = why

    for rule in document.get("conflict", []):
        why = rule.get("why")
        if not isinstance(why, str) or not why:
            raise ValueError(f"{SONG_TAGS_FILE}: every [[conflict]] says why")
        if "exclusive" in rule:
            for index, a in enumerate(rule["exclusive"]):
                for b in rule["exclusive"][index + 1:]:
                    clash(a, b, why)
        elif "between" in rule and len(rule["between"]) == 2:
            for a in rule["between"][0]:
                for b in rule["between"][1]:
                    clash(a, b, why)
        else:
            raise ValueError(f"{SONG_TAGS_FILE}: a [[conflict]] has `exclusive` or a two-sided `between`")
    return {tag: [{"tag": other, "why": why} for other, why in sorted(found.items())]
            for tag, found in sorted(pairs.items())}


def song_tag_suggestions() -> list[dict[str, Any]]:
    document = _song_tags_document()
    groups = document.get("group")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{SONG_TAGS_FILE} has no [[group]] tables")
    suggestions = []
    for group in groups:
        name, tags = group.get("name"), group.get("tags")
        if not isinstance(name, str) or not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag.strip() and "," not in tag for tag in tags
        ):
            raise ValueError(
                f"{SONG_TAGS_FILE}: every [[group]] needs a name and a list of tags, "
                "each a non-empty phrase with no comma"
            )
        suggestions.append({"group": name, "tags": list(tags)})
    return suggestions

IMAGE_EXAMPLE = (
    "A lighthouse on a basalt cliff at dusk, warm light in the lantern room, long "
    "exposure sea, 35 mm photograph"
)
VIDEO_EXAMPLE = (
    "A slow dolly shot along a rain-soaked neon street at night, reflections rippling in "
    "the puddles, a tram bell rings in the distance and rain patters on awnings"
)
AUDIO_EXAMPLES: dict[str, str] = {
    SFX: "TrackType: SFX. A heavy oak door creaks open slowly in a stone hallway, close mic, dry",
    MUSIC: (
        "TrackType: Music, VocalType: Instrumental. Warm lo-fi hip hop, dusty Rhodes, "
        "soft vinyl crackle, laid-back boom bap drums, 84 BPM"
    ),
    SONG: "English, warm piano pop, expressive female voice, 88 BPM",
}
LYRICS_EXAMPLE = "[Verse]\nThe kettle sings the morning in\n\n[Chorus]\nStay, stay a while"
AUDIO_LABELS: dict[str, str] = {
    SFX: "Describe the sound",
    MUSIC: "Describe the music",
    SONG: "Style tags",
}


def field(name: str, label: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "label": label, "kind": kind, "required": False, "default": None,
            **extra}


def seed_field() -> dict[str, Any]:
    return field("seed", "Seed", INTEGER, min=0, max=MAX_SEED, step=1,
                 hint="leave it blank for a new one each time")


def side_fields(default_width: int, default_height: int, low: int, high: int,
                multiple: int) -> list[dict[str, Any]]:
    low = math.ceil(low / multiple) * multiple
    high = high // multiple * multiple
    hint = f"a multiple of {multiple}, {low} to {high}"
    return [
        field("width", "Width", INTEGER, default=default_width, min=low, max=high,
              step=multiple, hint=hint),
        field("height", "Height", INTEGER, default=default_height, min=low, max=high,
              step=multiple, hint=hint),
    ]


def image_fields(manifest: Any, spec: Any) -> list[dict[str, Any]]:
    return [
        field("prompt", "Prompt", TEXT, required=True, placeholder=IMAGE_EXAMPLE),
        *side_fields(image_job.DEFAULT_SIDE, image_job.DEFAULT_SIDE, image_job.MIN_SIDE,
                     min(image_job.MAX_SIDE, spec.max_side), spec.size_multiple),
        field("steps", "Steps", INTEGER, default=image_job.DEFAULT_STEPS, min=1,
              max=image_job.MAX_STEPS, step=1, hint="more is slower, and usually cleaner"),
        seed_field(),
    ]


def video_fields(manifest: Any, spec: Any) -> list[dict[str, Any]]:
    longest = math.floor((spec.max_frames - 1) / max(spec.fps) * 10) / 10
    fields = [
        field("prompt", "Prompt", TEXT, required=True, placeholder=VIDEO_EXAMPLE),
        *side_fields(spec.default_width, spec.default_height, spec.min_side, spec.max_side,
                     spec.size_multiple),
        field("duration_s", "Length (seconds)", NUMBER, default=spec.default_duration_s,
              min=1, max=longest, step=0.1, hint=f"up to {longest:g} s"),
    ]
    if len(spec.fps) > 1:
        fields.append(field("fps", "Frames a second", CHOICE, default=spec.default_fps,
                            options=list(spec.fps)))
    fields.append(field("audio", "With sound", BOOLEAN, default=True))
    fields.append(seed_field())
    return fields


def audio_fields(manifest: Any, spec: Any) -> list[dict[str, Any]]:
    if manifest.text_param == "tags":
        first = field("tags", AUDIO_LABELS[manifest.kind], TAGS, required=True,
                      placeholder=AUDIO_EXAMPLES[manifest.kind],
                      suggestions=song_tag_suggestions(),
                      conflicts=song_tag_conflicts(),
                      hint="type a phrase and a comma to add it, or click a suggestion")
    else:
        first = field(manifest.text_param, AUDIO_LABELS[manifest.kind], TEXT, required=True,
                      placeholder=AUDIO_EXAMPLES[manifest.kind])
    fields = [first]
    if manifest.takes_lyrics:
        unsung = "instrumental" in spec.takes
        fields.append(field("lyrics", "Lyrics", TEXT, required=not unsung,
                            placeholder=LYRICS_EXAMPLE,
                            hint=("the song is as long as its lyrics; for an instrumental they "
                                  "are optional and only shape its sections (never sung)")
                            if unsung else "the song is as long as its lyrics"))
    if "instrumental" in spec.takes:
        fields.append(field("instrumental", "Instrumental (no vocals)", BOOLEAN, default=False,
                            hint="YuE2 writes the melody, then plays it on an instrument instead of singing it"))
        # The pool's set ids: where a client finds what `planning_set` may name.
        fields.append(field("planning_set", "Planning set (instrumental)", CHOICE, default=None,
                            options=["", *(entry.id for entry in planning_pool(spec))],
                            hint="the structure an instrumental is planned from; blank lets the seed pick"))
    for name, label in (("min_duration_s", "Shortest (seconds)"),
                        ("max_duration_s", "Longest (seconds)")):
        if name in spec.takes:
            fields.append(field(name, label, NUMBER, min=MIN_SONG_SECONDS,
                                max=spec.max_duration_s, step=1,
                                hint="a range the song's score must land in before it is composed"))
    if "duration_s" in spec.takes:
        fields.append(field("duration_s", "Length (seconds)", NUMBER,
                            default=spec.default_duration_s, min=1,
                            max=spec.max_duration_s, step=1,
                            hint=f"up to {spec.max_duration_s} s"))
    if "steps" in spec.takes:
        fields.append(field("steps", "Steps", INTEGER, default=spec.default_steps, min=1,
                            max=spec.max_steps, step=1))
    if "cfg" in spec.takes:
        fields.append(field("cfg", "Guidance (cfg)", NUMBER, default=spec.default_cfg, min=0,
                            max=spec.max_cfg, step=0.1))
    if "negative_prompt" in spec.takes:
        fields.append(field("negative_prompt", "Leave out", TEXT))
    fields.append(field("format", "File", CHOICE, default=FORMATS[0], options=list(FORMATS)))
    fields.append(seed_field())
    return fields


@dataclass(frozen=True)
class Family:
    job_type: str
    media: str
    manifests: ManifestCatalog[Any]
    fields: Callable[[Any, Any], list[dict[str, Any]]]

    def kind(self, manifest: Any) -> str:
        return getattr(manifest, "kind", self.media)


FAMILIES: tuple[Family, ...] = (
    Family(image_job.JOB_TYPE, "image", image_job.MANIFESTS, image_fields),
    Family(video_job.JOB_TYPE, "video", video_job.MANIFESTS, video_fields),
    Family(audio_job.JOB_TYPE, "audio", audio_job.MANIFESTS, audio_fields),
)


def _no_build(manifest: Any, backend_kind: str) -> str:
    return (
        f"{manifest.display} does not run on this server's backend ({backend_kind}); "
        f"{manifest.path.name} declares {sorted(manifest.backends)}"
    )


def _downloads(job_type: str, env_missing: bool, weights_missing: bool) -> str:
    what = " and ".join(
        part for part, needed in (
            (f"the {job_type} engine", env_missing), ("its weights", weights_missing)
        ) if needed
    )
    return (
        f"Not on this server yet. The first Generate downloads {what}, then makes "
        "it; after that it is ready straight away"
    )


def _download_bytes(family: Family, backend_kind: str, env_missing: bool,
                    subject: Any) -> int | None:
    parts: list[int | None] = []
    if env_missing:
        installer = INSTALLER_FOR.get(family.job_type, family.job_type)
        parts.append(installonsubmit._env_bytes(installer, None, backend_kind))
    if subject is None or subject.installed() is None:
        parts.append(None if subject is None else subject.expected_bytes)
    if any(part is None for part in parts):
        return None
    return sum(part for part in parts if part is not None)


def _spec_here(family: Family, manifest: Any, here: "Here") -> Any:
    spec = manifest.spec(here.backend.kind)
    if family.job_type == video_job.JOB_TYPE:
        return video_job.machine_spec(here.config, spec)
    return spec


@dataclass(frozen=True)
class Here:
    config: Any
    backend: Any
    registry: dict[str, Any]
    subjects: dict[str, Any]

    def can_install(self, job_type: str) -> bool:
        spec = spec_of(job_type)
        return bool(self.config.install_on_submit) and spec is not None and spec.installable

    def env_missing(self, job_type: str) -> bool:
        installer = INSTALLER_FOR.get(job_type, job_type)
        return not env_installed(self.config, self.backend, installer, None)


def _standing(family: Family, manifest: Any, here: Here) -> tuple[str, str | None, int | None]:
    kind = here.backend.kind
    if not manifest.supports(kind):
        return UNAVAILABLE, _no_build(manifest, kind), None
    plugin = here.registry.get(family.job_type)
    subject = here.subjects.get(manifest.id)
    weights_missing = subject is None or subject.installed() is None
    env_missing = here.env_missing(family.job_type)
    if plugin is None:
        refusal = disabled_error(family.job_type, here.config)
        why = (refusal.details or {}).get("reason")
        if not (env_missing and why in INSTALLS_A_DISABLED_TYPE
                and here.can_install(family.job_type)):
            return UNAVAILABLE, refusal.message, None
    elif not env_missing and not weights_missing:
        status = plugin.check(here.backend)
        return (READY, None, None) if status.ready else (UNAVAILABLE, status.detail, None)
    elif not here.can_install(family.job_type):
        return UNAVAILABLE, (
            f"{manifest.display} is not installed on this server, and this server does "
            "not install on first use ([jobs] install_on_submit is off). Install it from "
            f"the console, or run `{manifest.pull_command}` on the server"
        ), None
    spec = manifest.spec(kind)
    if weights_missing and getattr(spec, "gated", False) and weights.hf_token(here.config) is None:
        return UNAVAILABLE, weights.gated_message(spec.hf_repo, here.config,
                                                  manifest.pull_command), None
    return (
        DOWNLOAD,
        _downloads(family.job_type, env_missing, weights_missing),
        _download_bytes(family, kind, env_missing, subject),
    )


def pages(config: Any, backend: Any, registry: dict[str, Any]) -> list[dict[str, Any]]:
    here = Here(
        config=config,
        backend=backend,
        registry=registry,
        subjects={
            subject.id: subject
            for subject in catalog.subjects(config, backend)
            if subject.kind == "model"
            and subject.job_type in {family.job_type for family in FAMILIES}
        },
    )
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for manifest in family.manifests.all().values():
            standing, reason, size = _standing(family, manifest, here)
            kind = family.kind(manifest)
            rows.append({
                "job_type": family.job_type,
                "id": manifest.id,
                "name": manifest.display,
                "media": family.media,
                "kind": kind,
                "makes": KIND_WORDS.get(kind, family.media + "s"),
                "standing": standing,
                "available": standing != UNAVAILABLE,
                "reason": reason,
                "download_bytes": size,
                "fields": (
                    family.fields(manifest, _spec_here(family, manifest, here))
                    if manifest.supports(backend.kind) else []
                ),
            })
    return rows


__all__ = ["DOWNLOAD", "FAMILIES", "READY", "UNAVAILABLE", "pages"]
