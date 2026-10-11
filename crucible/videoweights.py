"""A video model's weights: the gated Lightricks snapshot plus its GGUF companion.

The shape is a main repo plus companions (crucible/companionweights.py, the one owner of
it): the main repo is pulled through `weights.pull` (only the files the manifest names),
the companion through `weights.pull_files` into `<model dir>/<companion name>/`,
hash-verified against the manifest. A model is installed only when both are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import companionweights, weights
from .companionweights import companion_dir
from .config import Config
from .videomodels import VideoBackendSpec, VideoManifest

NOUN = "video model"


def missing_companions(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> list[str]:
    return companionweights.missing_companions(config, manifest, spec)


def installed(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> weights.InstalledWeights | None:
    return companionweights.installed(config, manifest, spec)


def require_installed(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> weights.InstalledWeights:
    return companionweights.require_installed(config, manifest, spec, NOUN)


def refuse_gated_without_token(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> None:
    companionweights.refuse_gated_without_token(config, manifest, spec)


def pull(
    config: Config,
    manifest: VideoManifest,
    spec: VideoBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return companionweights.pull(
        config, manifest, spec, NOUN, force=force, on_line=on_line, on_progress=on_progress
    )


def remove(config: Config, manifest: VideoManifest, spec: VideoBackendSpec) -> Path:
    return companionweights.remove(config, manifest, spec)


__all__ = [
    "companion_dir",
    "installed",
    "missing_companions",
    "pull",
    "remove",
    "require_installed",
]
