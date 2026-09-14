"""ultimate-rvc's shared base assets — `rvcbase/<engine>.toml`.

PHASE4-AUDIO.md section 4.1, and the discharge of PLAN.md's owed ruling 3.

urvc needs a content embedder and a pitch predictor before it can convert a
single file, and they are **the engine's rather than any model's**: every RVC
voice in the catalog uses the same ones. Until this module existed, Crucible
refused a job by name when they were absent and could tell the operator nothing
except "put a urvc models tree here", because the only source anybody had
written down was a GitHub release in BookForge — which DESIGN.md section 5
refuses as a source of weights.

The engine's own downloader names a HuggingFace repo
----------------------------------------------------
Read out of the installed fork rather than guessed
(`ultimate_rvc/rvc/lib/tools/prerequisites_download.py`, 2026-09-13):

    url_base = "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/main/Resources"

That file *is* the first-run downloader `URVC_SKIP_INIT=1` turns off, so what
this pulls are exactly the bytes the engine would have fetched for itself, from
the repo it would have fetched them from — at a pinned revision instead of
`main`, with a digest per file.

Why this is not a manifest in `rvc/`
------------------------------------
`crucible/rvcmodels.py` loads `rvc/*.toml` as voice-conversion MODELS, and these
are not one: they have no id a client can ask for, no per-backend block (a torch
checkpoint is the same file on both), and no memory estimate of their own — the
rvc model's estimate already counts them, which is what its manifest comment
about "540 MB of base weights" is. Dropping a non-model into that directory
would make every loader there have to know about the exception.

So it is its own directory with its own loader, exactly as `denoise/` is, and
`crucible rvc pull-base` is its one door. One set, one command, one owner.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import weights
from .config import Config
from .errors import CrucibleError

RVC_BASE_DIR_ENV = "CRUCIBLE_RVC_BASE_DIR"

#: The engine whose assets these are. One today, and the id is in the filename
#: for the reason every other catalog puts it there: a second engine with base
#: assets of its own gets a second file, not a second key in this one.
ULTIMATE_RVC = "ultimate-rvc"

#: What `crucible rvc pull-base` fetches them for, and where a refusal sends a
#: reader. Spelled once so the job's refusal, the doctor line and the CLI agree.
PULL_COMMAND = "crucible rvc pull-base"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENGINE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HF_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

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
    # Required, and it is not decoration. These are four files with no obvious
    # relationship to each other, fetched into a tree nobody looks at until a
    # conversion fails; "why is fcpe.pt here" is a question somebody will ask,
    # and the answer belongs beside the pin rather than in a commit message.
    "why": str,
}


class RvcBaseError(CrucibleError):
    """The base-asset declaration is missing, unreadable, or wrong."""


@dataclass(frozen=True)
class BaseFile:
    """One file: where it comes from, where it goes, and what it must hash to.

    The field names are `crucible.weights.FileSource`'s, so `pull_files` takes
    these directly and there is one downloader rather than two.
    """

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
        """Every path this set puts under the base root, in declared order.

        **The one owner of "which files urvc needs"** — `crucible/jobs/rvc`
        reads this rather than keeping a list of its own, so the set that is
        pulled and the set that is checked for cannot drift (ARCHITECTURE.md
        R1). They used to be two lists and the job's was shorter: it checked for
        the embedder's weights and not for the `config.json` beside them,
        without which transformers will not load the directory at all.
        """
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


# ------------------------------------------------------------------ locating


def rvc_base_declarations_dir() -> Path:
    """Where `rvcbase/*.toml` live. Refuses by name if absent."""
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
    """`~/.crucible/rvc-base` — the `URVC_MODELS_DIR` half that is shared.

    The same directory `crucible/jobs/rvc/__init__.py` has always looked in, and
    it keeps its name: a host that already has the tree does not have to move it
    because a puller arrived.
    """
    return config.home / "rvc-base"


# ------------------------------------------------------------------ checking


def _check_table(where: str, table: dict[str, Any], required: dict[str, type]) -> None:
    unknown = sorted(set(table) - set(required))
    if unknown:
        raise RvcBaseError(
            f"{where}: unknown key(s) {unknown}; this table takes exactly "
            f"{sorted(required)}"
        )
    missing = sorted(set(required) - set(table))
    if missing:
        raise RvcBaseError(f"{where}: missing required key(s) {missing}")
    for key, kind in required.items():
        value = table[key]
        wrong = not isinstance(value, kind)
        if kind is int and isinstance(value, bool):
            wrong = True
        if wrong:
            raise RvcBaseError(
                f"{where}: {key} must be {kind.__name__}, got {type(value).__name__}"
            )


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
    _check_table(f"{path.name} [engine]", engine, _ENGINE_REQUIRED)

    engine_id = engine["id"]
    if not _ENGINE_ID.match(engine_id):
        raise RvcBaseError(
            f"{path.name}: engine.id {engine_id!r} must be lower-case and start "
            "with a letter or digit ([a-z0-9][a-z0-9._-]*)"
        )
    if engine_id != expected_id:
        raise RvcBaseError(
            f"{path.name}: engine.id is {engine_id!r} but the file is named "
            f"{expected_id!r}; the id and the filename are the same thing"
        )
    if not _HF_REPO.match(engine["hf_repo"]):
        raise RvcBaseError(
            f"{path.name}: hf_repo {engine['hf_repo']!r} is not an <owner>/<name> "
            "HuggingFace repo id"
        )
    if not _REVISION.match(engine["revision"]):
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
        _check_table(where, raw, _FILE_REQUIRED)
        if not _SHA256.match(raw["sha256"]):
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


# ------------------------------------------------------------------- loading


def parse_rvc_base(text: str, path: Path, expected_id: str) -> RvcBaseAssets:
    """Parse and validate one declaration. Raises RvcBaseError by name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RvcBaseError(f"{path.name}: not valid TOML: {exc}") from exc
    return _parse(document, path, expected_id)


def load_rvc_base(
    engine_id: str = ULTIMATE_RVC, directory: Path | None = None
) -> RvcBaseAssets:
    """Load `rvcbase/<engine_id>.toml`. Raises if it is not there."""
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


# ------------------------------------------------------------- on this host


def missing(config: Config, assets: RvcBaseAssets) -> list[str]:
    """Which declared targets are not on this host, in declared order.

    Presence, not digest. A file Crucible placed was verified when it was
    placed, and re-hashing 600 MB on every `check()` would make `crucible
    doctor` cost a minute — while a file somebody else put there is one Crucible
    has nothing to compare against, which is the state `rvc` has been in all
    along.
    """
    root = base_root(config)
    return [target for target in assets.targets if not (root / target).is_file()]


def installed(config: Config, assets: RvcBaseAssets) -> weights.InstalledWeights | None:
    """The pulled set at this pin, or None. A stamp at another pin is not this."""
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
    """Fetch and place every declared file, each verified before any is placed."""
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
