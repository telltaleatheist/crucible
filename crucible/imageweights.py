"""An image model's weights: its repo (the files its block names, or the whole repo) plus
companions (a replacement VAE), through crucible/companionweights.py, the one owner of
that shape. A model is installed only when every part is. Qwen-Image 2.1 has no
companion, so for it this is exactly `weights.installed` / `weights.pull`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import companionweights, weights
from .config import Config
from .imagemodels import ImageBackendSpec, ImageManifest

NOUN = "image model"


def companion_dirs(model_dir: Path, spec: ImageBackendSpec) -> dict[str, str]:
    return companionweights.part_dirs(model_dir, spec)


def installed(
    config: Config, manifest: ImageManifest, spec: ImageBackendSpec
) -> weights.InstalledWeights | None:
    return companionweights.installed(config, manifest, spec)


def require_installed(
    config: Config, manifest: ImageManifest, spec: ImageBackendSpec
) -> weights.InstalledWeights:
    return companionweights.require_installed(config, manifest, spec, NOUN)


def pull(
    config: Config,
    manifest: ImageManifest,
    spec: ImageBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return companionweights.pull(
        config, manifest, spec, NOUN, force=force, on_line=on_line, on_progress=on_progress
    )


def remove(config: Config, manifest: ImageManifest, spec: ImageBackendSpec) -> Path:
    return companionweights.remove(config, manifest, spec)


__all__ = ["companion_dirs", "installed", "pull", "remove", "require_installed"]
