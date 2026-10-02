from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import weights
from .config import Config
from .errors import CrucibleError
from .tomltable import (
    HF_REPO_PATTERN,
    MODEL_ID_PATTERN,
    REVISION_PATTERN,
    SHA256_PATTERN,
    check_table,
)

RVC_BASE_DIR_ENV = "CRUCIBLE_RVC_BASE_DIR"

ULTIMATE_RVC = "ultimate-rvc"

PULL_COMMAND = "crucible rvc pull-base"


_ENGINE_REQUIRED: dict[str, type] = {
    "id": str,
    "hf_repo": str,
    "revision": str,
}
_FILE_REQUIRED: dict[str, type] = {
    "source": str,
    "target": str,
    "sha256": str,
    "bytes": int,
    "why": str,
}


class RvcBaseError(CrucibleError):
    ...


@dataclass(frozen=True)
class BaseFile:
    source: str
    target: str
    sha256: str
    bytes: int
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "why": self.why,
        }


@dataclass(frozen=True)
class RvcBaseAssets:
    id: str
    hf_repo: str
    revision: str
    files: tuple[BaseFile, ...]
    path: Path

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(entry.target for entry in self.files)

    @property
    def total_bytes(self) -> int:
        return sum(entry.bytes for entry in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "total_bytes": self.total_bytes,
            "files": [entry.to_dict() for entry in self.files],
        }


def rvc_base_declarations_dir() -> Path:
    override = os.environ.get(RVC_BASE_DIR_ENV)
    if override is not None and override != "":
        path = Path(override).expanduser()
        if not path.is_dir():
            raise RvcBaseError(f"{RVC_BASE_DIR_ENV}={override!r} is not a directory")
        return path
    path = Path(__file__).resolve().parent / "rvcbase"
    if not path.is_dir():
        raise RvcBaseError(
            f"no base-asset declaration at {path}; it is package data and this "
            f"install has lost it, or ${RVC_BASE_DIR_ENV} must point at it"
        )
    return path


def base_root(config: Config) -> Path:
    return config.home / "rvc-base"


def _parse(document: dict[str, Any], path: Path, expected_id: str) -> RvcBaseAssets:
    unknown = sorted(set(document) - {"engine", "files"})
    if unknown:
        raise RvcBaseError(
            f"{path.name}: unknown top-level table(s) {unknown}; a base-asset "
            "declaration has exactly [engine] and [[files]]"
        )
    if "engine" not in document:
        raise RvcBaseError(f"{path.name}: missing the [engine] table")
    engine = document["engine"]
    if not isinstance(engine, dict):
        raise RvcBaseError(f"{path.name}: [engine] must be a table")
    check_table(f"{path.name} [engine]", engine, _ENGINE_REQUIRED, error=RvcBaseError)

    engine_id = engine["id"]
    if not MODEL_ID_PATTERN.match(engine_id):
        raise RvcBaseError(
            f"{path.name}: engine.id {engine_id!r} must be lower-case and start "
            "with a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if engine_id != expected_id:
        raise RvcBaseError(
            f"{path.name}: engine.id is {engine_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if not HF_REPO_PATTERN.match(engine["hf_repo"]):
        raise RvcBaseError(
            f"{path.name}: hf_repo {engine['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not REVISION_PATTERN.match(engine["revision"]):
        raise RvcBaseError(
            f"{path.name}: revision {engine['revision']!r} must be a full "
            "40-character commit sha, so a pull is reproducible; `main` is what "
            "the engine's own downloader uses and it is not a pin"
        )

    raw_files = document.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise RvcBaseError(
            f"{path.name}: missing [[files]]; a base-asset set with no files in "
            "it would be a job type that refuses for no reason"
        )
    files: list[BaseFile] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        where = f"{path.name} [[files]][{index}]"
        if not isinstance(raw, dict):
            raise RvcBaseError(f"{where}: must be a table")
        check_table(where, raw, _FILE_REQUIRED, error=RvcBaseError)
        if not SHA256_PATTERN.match(raw["sha256"]):
            raise RvcBaseError(
                f"{where}: sha256 {raw['sha256']!r} is not a 64-character digest. "
                "A file fetched by path gets the assurance a snapshot download "
                "gets from its revision, or it gets none"
            )
        if raw["bytes"] <= 0:
            raise RvcBaseError(f"{where}: bytes must be positive, got {raw['bytes']}")
        target = raw["target"]
        if target.startswith("/") or ".." in Path(target).parts:
            raise RvcBaseError(
                f"{where}: target {target!r} is absolute or climbs out of the "
                "base directory. A target is a path inside the tree the engine "
                "reads, never a way to write somewhere else"
            )
        if target in seen:
            raise RvcBaseError(
                f"{where}: target {target!r} is declared twice; two sources for "
                "one path is a set whose contents depend on the order it ran in"
            )
        seen.add(target)
        files.append(
            BaseFile(
                source=raw["source"],
                target=target,
                sha256=raw["sha256"],
                bytes=raw["bytes"],
                why=raw["why"],
            )
        )

    return RvcBaseAssets(
        id=engine_id,
        hf_repo=engine["hf_repo"],
        revision=engine["revision"],
        files=tuple(files),
        path=path,
    )


def parse_rvc_base(text: str, path: Path, expected_id: str) -> RvcBaseAssets:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RvcBaseError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_rvc_base(
    engine_id: str = ULTIMATE_RVC, directory: Path | None = None
) -> RvcBaseAssets:
    root = directory if directory is not None else rvc_base_declarations_dir()
    path = root / f"{engine_id}.toml"
    if not path.is_file():
        known = sorted(p.stem for p in root.glob("*.toml"))
        raise RvcBaseError(
            f"no base-asset declaration for {engine_id!r} at {path}; this build "
            f"ships {known}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RvcBaseError(f"could not read {path}: {exc}") from exc
    return parse_rvc_base(text, path, engine_id)


def missing(config: Config, assets: RvcBaseAssets) -> list[str]:
    root = base_root(config)
    return [target for target in assets.targets if not (root / target).is_file()]


def installed(config: Config, assets: RvcBaseAssets) -> weights.InstalledWeights | None:
    return weights.files_installed(
        base_root(config), assets.hf_repo, assets.revision
    )


def pull(
    config: Config,
    assets: RvcBaseAssets,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: weights.ProgressHook | None = None,
) -> weights.InstalledWeights:
    return weights.pull_files(
        config,
        hf_repo=assets.hf_repo,
        revision=assets.revision,
        files=assets.files,
        target_root=base_root(config),
        label=f"{assets.id}'s base assets",
        force=force,
        on_line=on_line,
        on_progress=on_progress,
    )


__all__ = [
    "PULL_COMMAND",
    "RVC_BASE_DIR_ENV",
    "ULTIMATE_RVC",
    "BaseFile",
    "RvcBaseAssets",
    "RvcBaseError",
    "base_root",
    "installed",
    "load_rvc_base",
    "missing",
    "parse_rvc_base",
    "pull",
    "rvc_base_declarations_dir",
]
