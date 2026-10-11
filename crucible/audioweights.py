"""An audio model's weights: a main repo plus companions (crucible/companionweights.py,
the one owner of that shape), under the noun "audio model"."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import companionweights, weights
from .audiomodels import AudioBackendSpec, AudioManifest
from .companionweights import companion_dir, part_dirs
from .config import Config

NOUN = "audio model"


def missing_companions(
    config: Config, manifest: AudioManifest, spec: AudioBackendSpec
) -> list[str]:
    return companionweights.missing_companions(config, manifest, spec)


def installed(
    config: Config, manifest: AudioManifest, spec: AudioBackendSpec
) -> weights.InstalledWeights | None:
    return companionweights.installed(config, manifest, spec)


def require_installed(
    config: Config, manifest: AudioManifest, spec: AudioBackendSpec
) -> weights.InstalledWeights:
    return companionweights.require_installed(config, manifest, spec, NOUN)


def refuse_gated_without_token(
    config: Config, manifest: AudioManifest, spec: AudioBackendSpec
) -> None:
    companionweights.refuse_gated_without_token(config, manifest, spec)


def pull(
    config: Config,
    manifest: AudioManifest,
    spec: AudioBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return companionweights.pull(
        config, manifest, spec, NOUN, force=force, on_line=on_line, on_progress=on_progress
    )


def remove(config: Config, manifest: AudioManifest, spec: AudioBackendSpec) -> Path:
    return companionweights.remove(config, manifest, spec)


__all__ = [
    "companion_dir",
    "installed",
    "missing_companions",
    "part_dirs",
    "pull",
    "remove",
    "require_installed",
]
