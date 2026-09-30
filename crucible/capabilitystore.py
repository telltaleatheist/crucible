from __future__ import annotations

from pathlib import Path
from typing import Mapping

from . import verdict
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
) -> tuple[verdict.Decision, ...]:
    return verdict.decide_all(
        backend_kind,
        total_bytes=total_bytes,
        desktop_allowance_bytes=desktop_allowance_bytes,
        gpu_vendor=gpu_vendor,
        chosen=chosen,
        card=card,
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


def write_capability(
    config: Config,
    backend: Backend,
    decisions: tuple[verdict.Decision, ...],
    flags: dict[str, bool],
) -> Path:
    values = {
        flag: flags.get(flag, getattr(config, flag)) for flag in CAPABILITY_FLAGS
    }
    return write_config(
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
        **values,
    )


__all__ = ["decide_for", "decide_on", "record_of", "write_capability"]
