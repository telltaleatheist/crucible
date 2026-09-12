"""Model weights on disk — `~/.crucible/models/<id>/<backend>/`.

Pulled by `crucible models pull <id>` with `huggingface_hub` at the manifest's
pinned revision, never from GitHub Releases (PHASE2-LLM.md section 1). A pull
that finishes writes `crucible-pull.json` beside the weights; nothing downstream
treats a directory without that stamp as installed, so an interrupted 19 GB
download can never be handed to an engine as if it were a model.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .errors import CrucibleError
from .manifests import BackendSpec, ModelManifest

HF_TOKEN_ENV = "HF_TOKEN"
STAMP_NAME = "crucible-pull.json"


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


def models_root(config: Config) -> Path:
    return config.home / "models"


def model_dir(config: Config, model_id: str, backend_kind: str) -> Path:
    return models_root(config) / model_id / backend_kind


def stamp_path(config: Config, model_id: str, backend_kind: str) -> Path:
    return model_dir(config, model_id, backend_kind) / STAMP_NAME


def installed(
    config: Config, manifest: ModelManifest, spec: BackendSpec
) -> InstalledWeights | None:
    """The installed weights for this (model, backend), or None.

    A stamp naming a different revision than the manifest pins is *not* installed:
    the manifest moved, and serving the old bytes under the new id would be a
    silent substitution.
    """
    stamp = stamp_path(config, manifest.id, spec.backend)
    if not stamp.is_file():
        return None
    record = json.loads(stamp.read_text(encoding="utf-8"))
    if record["revision"] != spec.revision or record["hf_repo"] != spec.hf_repo:
        return None
    return InstalledWeights(
        path=model_dir(config, manifest.id, spec.backend),
        hf_repo=record["hf_repo"],
        revision=record["revision"],
        bytes=record["bytes"],
        pulled=record["pulled"],
    )


def require_installed(
    config: Config, manifest: ModelManifest, spec: BackendSpec
) -> InstalledWeights:
    """Installed weights, or `model_not_installed` by name."""
    found = installed(config, manifest, spec)
    if found is not None:
        return found
    directory = model_dir(config, manifest.id, spec.backend)
    stamp = stamp_path(config, manifest.id, spec.backend)
    if stamp.is_file():
        record = json.loads(stamp.read_text(encoding="utf-8"))
        raise WeightsError(
            f"{directory} holds {record['hf_repo']}@{record['revision'][:12]}, but "
            f"{manifest.path.name} now pins {spec.hf_repo}@{spec.revision[:12]} — "
            f"run `crucible models pull {manifest.id}`"
        )
    raise WeightsError(
        f"model {manifest.id!r} is not installed for {spec.backend}; there are no "
        f"weights at {directory} — run `crucible models pull {manifest.id}`"
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
    manifest: ModelManifest,
    spec: BackendSpec,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> InstalledWeights:
    """Fetch this model's weights for this backend at the pinned revision."""
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

    target = model_dir(config, manifest.id, spec.backend)
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
        "model": manifest.id,
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
