"""A video model's weights: the gated Lightricks snapshot plus its GGUF companion.

The shape is audioweights': the main repo is pulled through `weights.pull`
(only the files the manifest names), the companion through `weights.pull_files`
into `<model dir>/<companion name>/`, hash-verified against the manifest. A
model is installed only when both are.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import weights
from .audiomodels import Companion
from .config import Config
from .videomodels import VideoBackendSpec, VideoManifest


def companion_dir(model_dir: Path, companion: Companion) -> Path:
    return model_dir / companion.name


def _companion_installed(
    model_dir: Path, companion: Companion
) -> weights.InstalledWeights | None:
    return weights.files_installed(
        companion_dir(model_dir, companion), companion.hf_repo, companion.revision
    )


def missing_companions(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> list[str]:
    model_dir = weights.subject_dir(config, manifest, spec.backend)
    return [
        companion.name
        for companion in spec.companions
        if _companion_installed(model_dir, companion) is None
    ]


def installed(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> weights.InstalledWeights | None:
    main = weights.installed(config, manifest, spec)
    if main is None:
        return None
    parts = [_companion_installed(main.path, companion) for companion in spec.companions]
    if any(part is None for part in parts):
        return None
    return weights.InstalledWeights(
        path=main.path,
        hf_repo=main.hf_repo,
        revision=main.revision,
        bytes=main.bytes + sum(part.bytes for part in parts if part is not None),
        pulled=main.pulled,
        source=main.source,
    )


def require_installed(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> weights.InstalledWeights:
    found = installed(config, manifest, spec)
    if found is not None:
        return found
    main = weights.require_installed(config, manifest, spec)
    absent = missing_companions(config, manifest, spec)
    raise weights.WeightsError(
        f"video model {manifest.id!r} is only partly installed for {spec.backend}: "
        f"its {', '.join(absent)} part(s) from "
        f"{', '.join(c.hf_repo for c in spec.companions if c.name in absent)} are "
        f"not under {main.path}. Run `{manifest.pull_command}`; it fetches only "
        "what is missing"
    )


def refuse_gated_without_token(
    config: Config, manifest: VideoManifest, spec: VideoBackendSpec
) -> None:
    if spec.gated and weights.hf_token(config) is None:
        raise weights.WeightsError(
            weights.gated_message(spec.hf_repo, config, manifest.pull_command)
        )


def pull(
    config: Config,
    manifest: VideoManifest,
    spec: VideoBackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    if force or installed(config, manifest, spec) is None:
        refuse_gated_without_token(config, manifest, spec)
    main = weights.pull(
        config, manifest, spec, force=force, on_line=on_line, on_progress=on_progress
    )
    for companion in spec.companions:
        weights.pull_files(
            config,
            hf_repo=companion.hf_repo,
            revision=companion.revision,
            files=companion.files,
            target_root=companion_dir(main.path, companion),
            label=f"{manifest.path.name} companion {companion.name!r}",
            force=force,
            on_line=on_line,
            on_progress=on_progress,
        )
    return require_installed(config, manifest, spec)


def remove(config: Config, manifest: VideoManifest, spec: VideoBackendSpec) -> Path:
    return weights.remove(config, manifest, spec)


__all__ = [
    "companion_dir",
    "installed",
    "missing_companions",
    "pull",
    "remove",
    "require_installed",
]
