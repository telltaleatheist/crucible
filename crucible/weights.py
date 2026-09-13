"""Weights on disk — `~/.crucible/<family>/<id>/<backend>/`.

Pulled by `crucible models pull <id>` (or `crucible voices pull <id>`) with
`huggingface_hub` at the manifest's pinned revision, never from GitHub Releases
(PHASE2-LLM.md section 1). A pull that finishes writes `crucible-pull.json`
beside the weights; nothing downstream treats a directory without that stamp as
installed, so an interrupted 19 GB download can never be handed to an engine as
if it were a model.

`family` is `models` or `voices`, and it comes off the manifest
(`weights_family`) rather than being passed around. Model ids and voice ids are
separate namespaces — nothing stops a voice being called `qwen3.5-9b` — and one
directory holding both would let a `crucible voices pull` overwrite a 19 GB model
with a 8.5 GB checkpoint and leave a stamp that reads as installed to either.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .config import Config
from .errors import CrucibleError

HF_TOKEN_ENV = "HF_TOKEN"
STAMP_NAME = "crucible-pull.json"

#: What to call the thing, and which command pulls it, per weights family. A
#: refusal that says "run `crucible models pull deathstalker`" for a voice sends
#: its reader to a command that will tell them there is no such model.
_FAMILY_WORDS: dict[str, tuple[str, str]] = {
    "models": ("model", "crucible models pull"),
    "voices": ("voice", "crucible voices pull"),
}


@runtime_checkable
class WeightsSubject(Protocol):
    """What this module needs from a manifest, model or voice alike.

    Structural rather than a base class: `ModelManifest` and `VoiceManifest`
    describe different things and share no fields beyond these, and inventing a
    parent for them would put the id and the path somewhere neither schema's
    reader would look for them.
    """

    id: str
    path: Path
    weights_family: str


@runtime_checkable
class WeightsSource(Protocol):
    """What this module needs from a backend block."""

    backend: str
    hf_repo: str
    revision: str


class WeightsError(CrucibleError):
    """Weights are missing, half-pulled, or could not be fetched."""


@dataclass(frozen=True)
class InstalledWeights:
    path: Path
    hf_repo: str
    revision: str
    bytes: int
    pulled: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "bytes": self.bytes,
            "pulled": self.pulled,
        }


def weights_root(config: Config, family: str) -> Path:
    return config.home / family


def weights_dir(
    config: Config, family: str, subject_id: str, backend_kind: str
) -> Path:
    return weights_root(config, family) / subject_id / backend_kind


def stamp_path(
    config: Config, family: str, subject_id: str, backend_kind: str
) -> Path:
    return weights_dir(config, family, subject_id, backend_kind) / STAMP_NAME


def installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights | None:
    """The installed weights for this (model or voice, backend), or None.

    A stamp naming a different revision than the manifest pins is *not* installed:
    the manifest moved, and serving the old bytes under the new id would be a
    silent substitution.
    """
    family = manifest.weights_family
    stamp = stamp_path(config, family, manifest.id, spec.backend)
    if not stamp.is_file():
        return None
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["revision"] != spec.revision or record["hf_repo"] != spec.hf_repo:
        return None
    return InstalledWeights(
        path=weights_dir(config, family, manifest.id, spec.backend),
        hf_repo=record["hf_repo"],
        revision=record["revision"],
        bytes=record["bytes"],
        pulled=record["pulled"],
    )


def require_installed(
    config: Config, manifest: WeightsSubject, spec: WeightsSource
) -> InstalledWeights:
    """Installed weights, or `model_not_installed` / `voice_not_installed`."""
    found = installed(config, manifest, spec)
    if found is not None:
        return found
    family = manifest.weights_family
    noun, command = _FAMILY_WORDS[family]
    directory = weights_dir(config, family, manifest.id, spec.backend)
    stamp = stamp_path(config, family, manifest.id, spec.backend)
    if stamp.is_file():
        record = json.loads(stamp.read_text(encoding="utf-8"))
        raise WeightsError(
            f"{directory} holds {record['hf_repo']}@{record['revision'][:12]}, but "
            f"{manifest.path.name} now pins {spec.hf_repo}@{spec.revision[:12]} — "
            f"run `{command} {manifest.id}`"
        )
    raise WeightsError(
        f"{noun} {manifest.id!r} is not installed for {spec.backend}; there are no "
        f"weights at {directory} — run `{command} {manifest.id}`"
    )


def hf_token(config: Config) -> str | None:
    """`$HF_TOKEN`, else `[hf] token` in config.toml, else None."""
    from_env = os.environ.get(HF_TOKEN_ENV)
    if from_env is not None and from_env.strip() != "":
        return from_env.strip()
    try:
        with config.path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError:
        return None
    section = document.get("hf")
    if not isinstance(section, dict):
        return None
    token = section.get("token")
    if isinstance(token, str) and token.strip() != "":
        return token.strip()
    return None


def directory_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def pull(
    config: Config,
    manifest: WeightsSubject,
    spec: WeightsSource,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> InstalledWeights:
    """Fetch this model's or voice's weights for this backend at its pin."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - a dependency, not a condition
        raise WeightsError(
            f"huggingface_hub is not importable in {config.name}'s interpreter: {exc}"
        ) from exc

    target = weights_dir(config, manifest.weights_family, manifest.id, spec.backend)
    existing = installed(config, manifest, spec)
    if existing is not None and not force:
        return existing
    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    stamp = target / STAMP_NAME
    if stamp.exists():
        stamp.unlink()

    token = hf_token(config)
    if on_line is not None:
        on_line(
            f"pulling {spec.hf_repo}@{spec.revision[:12]} -> {target} "
            f"({'with' if token else 'without'} an HF token)"
        )
    started = time.monotonic()
    try:
        snapshot_download(
            repo_id=spec.hf_repo,
            revision=spec.revision,
            local_dir=str(target),
            token=token,
            max_workers=8,
        )
    except GatedRepoError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is gated and this server has no HF token that opens it "
            f"(set ${HF_TOKEN_ENV} or [hf] token in {config.path}): {exc}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} is private or does not exist; if it is private set "
            f"${HF_TOKEN_ENV} or [hf] token in {config.path}: {exc}"
        ) from exc
    except RevisionNotFoundError as exc:
        raise WeightsError(
            f"{spec.hf_repo} has no revision {spec.revision}; "
            f"{manifest.path.name} pins a commit that repo does not have: {exc}"
        ) from exc
    except Exception as exc:
        raise WeightsError(
            f"pulling {spec.hf_repo}@{spec.revision[:12]} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    elapsed = time.monotonic() - started
    size = directory_bytes(target)
    record = {
        "family": manifest.weights_family,
        "id": manifest.id,
        "backend": spec.backend,
        "hf_repo": spec.hf_repo,
        "revision": spec.revision,
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(
            f"pulled {size / 1e9:.2f} GB in {elapsed:.0f}s "
            f"({size / 1e6 / max(elapsed, 1e-6):.0f} MB/s)"
        )
    result = installed(config, manifest, spec)
    if result is None:  # pragma: no cover - the stamp was just written
        raise WeightsError(f"wrote {stamp} but it does not read back as installed")
    return result
