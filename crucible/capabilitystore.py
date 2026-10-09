from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import lowvram, verdict
from .backend import Backend, CardFacts
from .capabilityrecord import CapabilityRecord
from .cardfacts import card_for
from .config import CAPABILITY_FLAGS, Config, write_config


def decide_on(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    gpu_vendor: str,
    card: CardFacts | None,
    chosen: Mapping[str, str],
    audio_low_vram: bool,
) -> tuple[verdict.Decision, ...]:
    return verdict.decide_all(
        backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        gpu_vendor=gpu_vendor,
        chosen=chosen,
        audio_low_vram=audio_low_vram,
        card=card,
    )


def low_vram_for(config: Config, backend: Backend) -> lowvram.LowVram:
    """`[audio] low_vram` as this card is decided with: a person's setting as it is,
    Crucible's own decided again from this card (crucible/lowvram.py)."""
    return lowvram.on_card(
        config,
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
    )


def decide_for(
    config: Config, backend: Backend, *, card: CardFacts | None = None
) -> tuple[verdict.Decision, ...]:
    return decide_on(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=backend.gpu.vendor,
        chosen={entry.capability: entry.model for entry in config.local_models},
        audio_low_vram=low_vram_for(config, backend).on,
        card=card if card is not None else card_for(config.home, backend.gpu),
    )


def record_of(
    backend_kind: str,
    *,
    total_bytes: int,
    desktop_allowance_bytes: int,
    decisions: tuple[verdict.Decision, ...],
    routes: Mapping[str, str],
) -> CapabilityRecord:
    return verdict.record(
        backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        decisions=decisions,
        routes=dict(routes),
    )


@dataclass(frozen=True)
class Recorded:
    """What write_capability wrote: the file, and the sentence for `[audio] low_vram`
    when Crucible changed it with the record (None: it did not)."""

    path: Path
    low_vram_change: str | None


def write_capability(
    config: Config,
    backend: Backend,
    decisions: tuple[verdict.Decision, ...],
    flags: dict[str, bool],
) -> Recorded:
    """Record `decisions` (from decide_for on this same config) and the `[audio] low_vram`
    they were decided with, in one write, so the record and the setting the audio job
    loads with never disagree."""
    values = {
        flag: flags.get(flag, getattr(config, flag)) for flag in CAPABILITY_FLAGS
    }
    low_vram = low_vram_for(config, backend)
    path = write_config(
        config.home,
        name=config.name,
        host=config.host,
        port=config.port,
        token=config.token,
        backend_kind=config.backend_kind,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        desktop_allowance_basis=config.desktop_allowance_basis,
        desktop_allowance_note=config.desktop_allowance_note,
        retention_days=config.retention_days,
        tts_engines=config.tts_engines,
        capability=record_of(
            backend.kind,
            total_bytes=backend.gpu.vram_bytes,
            desktop_allowance_bytes=config.desktop_allowance_bytes,
            decisions=decisions,
            routes={entry.capability: entry.model for entry in config.routes},
        ),
        routes=config.routes,
        upstreams=config.upstreams,
        local_models=config.local_models,
        open_pairing=config.open_pairing,
        advertise=config.advertise,
        tailscale_advertise=config.tailscale_advertise,
        lan_advertise=config.lan_advertise,
        audio_low_vram=low_vram.setting if low_vram.changed else None,
        **values,
    )
    return Recorded(path, low_vram.change_sentence if low_vram.changed else None)


@dataclass(frozen=True)
class Redecided:
    decisions: tuple[verdict.Decision, ...]
    flags: dict[str, bool]
    recorded: Recorded


def redecide(config: Config, backend: Backend, *job_types: str) -> Redecided:
    """Decide this card again and record it, with each named type's [jobs] flag set to
    whether anything behind it fits: the install's capability step, and what follows a
    change to `[audio] low_vram` from the CLI or Settings."""
    decisions = decide_for(config, backend)
    flags = {
        f"enable_{job_type}": verdict.job_type_enabled(job_type, decisions)
        for job_type in job_types
    }
    return Redecided(decisions, flags, write_capability(config, backend, decisions, flags))


@dataclass(frozen=True)
class LowVramSet:
    """What a person's `[audio] low_vram` change wrote, and the audio verdict after it."""

    path: Path
    low_vram: lowvram.LowVram
    redecided: Redecided


def low_vram_not_offered(backend_kind: str) -> str:
    return (
        f"no audio model on {backend_kind} declares a low-VRAM figure, so "
        "[audio] low_vram would change nothing here. Nothing was written"
    )


def set_low_vram(config: Config, backend: Backend, state: str) -> LowVramSet:
    """A person's `on`, `off` or `auto` (let Crucible decide), for `crucible audio
    low-vram` and Settings alike: written through rewrite_config, then the audio verdict
    and [jobs] enable_audio decided again so nothing is left stale. `on` where no model
    can be split is refused by the caller first (low_vram_not_offered)."""
    from .config import load_config, rewrite_config

    path = rewrite_config(config, audio_low_vram=lowvram.state_setting(state))
    fresh = load_config(config.home)
    redecided = redecide(fresh, backend, "audio")
    after = load_config(config.home)
    return LowVramSet(path, low_vram_for(after, backend), redecided)


__all__ = [
    "LowVramSet",
    "Recorded",
    "Redecided",
    "decide_for",
    "decide_on",
    "low_vram_for",
    "low_vram_not_offered",
    "record_of",
    "redecide",
    "set_low_vram",
    "write_capability",
]
