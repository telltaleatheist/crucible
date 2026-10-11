"""A score model's weights: SheetSage2's repo plus its encoder as a companion
(crucible/companionweights.py, the one owner of that shape), under the noun "score model"."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import companionweights, weights
from .companionweights import companion_dir, part_dirs
from .config import Config
from .scoremodels import ScoreBackendSpec, ScoreManifest

NOUN = "score model"


def missing_companions(
    config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec
) -> list[str]:
    return companionweights.missing_companions(config, manifest, spec)


def installed(
    config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec
) -> weights.InstalledWeights | None:
    return companionweights.installed(config, manifest, spec)


def require_installed(
    config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec
) -> weights.InstalledWeights:
    return companionweights.require_installed(config, manifest, spec, NOUN)


def pull(
    config: Config,
    manifest: ScoreManifest,
    spec: ScoreBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return companionweights.pull(
        config, manifest, spec, NOUN, force=force, on_line=on_line, on_progress=on_progress
    )


def remove(config: Config, manifest: ScoreManifest, spec: ScoreBackendSpec) -> Path:
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
