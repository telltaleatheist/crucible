"""The playground's pages: one per image, video and audio model this build declares, each
with the few params that model takes, read from its manifest, and whether this server can
run it now and, when it cannot, why not."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .audiomodels import KIND_WORDS, MUSIC, SFX, SONG
from .jobs import disabled_error
from .jobs import audio as audio_job
from .jobs import image as image_job
from .jobs import video as video_job
from .jobs.audio.params import FORMATS
from .jobs.template import ManifestCatalog

MAX_SEED = 2**32 - 1

TEXT = "text"
INTEGER = "integer"
NUMBER = "number"
BOOLEAN = "boolean"
CHOICE = "choice"

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
    fields = [
        field(manifest.text_param, AUDIO_LABELS[manifest.kind], TEXT, required=True,
              placeholder=AUDIO_EXAMPLES[manifest.kind]),
    ]
    if manifest.takes_lyrics:
        fields.append(field("lyrics", "Lyrics", TEXT, required=True,
                            placeholder=LYRICS_EXAMPLE,
                            hint="the song is as long as its lyrics"))
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


def _not_running(family: Family, registry: dict[str, Any], config: Any,
                 backend: Any) -> tuple[Any, str | None]:
    plugin = registry.get(family.job_type)
    if plugin is None:
        return None, disabled_error(family.job_type, config).message
    status = plugin.check(backend)
    return plugin, None if status.ready else status.detail


def _reason(manifest: Any, backend_kind: str, installed: bool, blocked: str | None) -> str | None:
    if not manifest.supports(backend_kind):
        return (
            f"{manifest.display} does not run on this server's backend ({backend_kind}); "
            f"{manifest.path.name} declares {sorted(manifest.backends)}"
        )
    if not installed:
        return (
            f"{manifest.display}'s weights are not on this server. Pull {manifest.id} "
            f"from the Catalog on the console, or run `{manifest.pull_command}` on the "
            "server"
        )
    return blocked


def pages(config: Any, backend: Any, registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        plugin, blocked = _not_running(family, registry, config, backend)
        installed = (
            {} if plugin is None
            else {model.id: model.installed for model in plugin.describe_models()}
        )
        for manifest in family.manifests.all().values():
            if plugin is None:
                reason = blocked
            else:
                reason = _reason(manifest, backend.kind, installed.get(manifest.id, False),
                                 blocked)
            kind = family.kind(manifest)
            rows.append({
                "job_type": family.job_type,
                "id": manifest.id,
                "name": manifest.display,
                "media": family.media,
                "kind": kind,
                "makes": KIND_WORDS.get(kind, family.media + "s"),
                "available": reason is None,
                "reason": reason,
                "fields": (
                    family.fields(manifest, manifest.spec(backend.kind))
                    if manifest.supports(backend.kind) else []
                ),
            })
    return rows


__all__ = ["FAMILIES", "pages"]
