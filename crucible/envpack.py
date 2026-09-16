"""Environment packs — an env arrives as bytes from the release, not as a pip run.

PHASE14-ENVPACKS.md. A **pack** is a relocatable, self-contained Python tree: a
python-build-standalone CPython with one recipe's packages installed INTO it (not
a venv beside it — a venv records the base interpreter's absolute path in
`pyvenv.cfg` and in every console script's shebang, and does not survive a move).
Unpacked anywhere, `<dir>/bin/python` runs — `<dir>/python.exe` on the one
Windows pack, whose tree is python-build-standalone's Windows layout and has no
`bin/` at all. `pack_python(root, backend_kind)` is the one place that knows
which, so no caller ever spells either path (PHASE15 section 4.4).

Two verbs live here and they point in opposite directions:

* `build_pack()` PRODUCES a pack, on the backend it targets, from the recipe that
  is already the source of truth for `crucible install --build`. Developer and CI.
* `install_pack()` CONSUMES one: manifest, parts, sha, unpack, stamp. Every
  machine that is not building.

Why a standalone CPython and not the host's
-------------------------------------------
Nothing in Crucible may depend on a distro's Python, its version or its packages
(section 0). The interpreter is part of what a pack IS, which is also what lets
the server pack exist at all: a fresh machine has no `crucible` to build an env
with, so the first thing it downloads has to carry its own python.

Why zstd
--------
An 8 GB torch env unpacks in a fraction of gzip's time and compresses smaller.
`tar` and `zstd` are the two tools a POSIX target needs, and both are present
on Ubuntu >= 20.04 and on macOS 13+. Windows needs ONE: its `tar.exe` is
bsdtar with libzstd linked in, and no `zstd.exe` ships at all — so
`require_zstd_tar` asks that tar whether it carries zstd instead of demanding
a binary the OS does not have. A host that cannot unpack a pack is refused BY
NAME (`pack_no_zstd`) rather than silently falling back to a format nobody
built the asset in.

Why parts
---------
GitHub Releases caps one asset at 2 GiB. A pack over that is split at
`PART_MAX_BYTES` and the manifest's `sha256` describes the REASSEMBLED whole —
BookForge's rule, and the reason is that a per-part digest proves each part
arrived and proves nothing about the join. Parts are appended and deleted one at
a time, so peak extra disk is one part rather than the whole set.

What is NOT pruned, and why
---------------------------
`__pycache__` goes: it is regenerated on first import and costs hundreds of MB in
a torch env. A package's own `tests/` directory STAYS. Section 3.2 lists it as
prunable and it is not safe in general — several packages import their test
helpers from library code, and the failure then lands inside somebody's book
rather than in the smoke test. A pack that carries 40 MB of test files is worse
than nothing only in bandwidth; a pack that ImportErrors at chunk 900 is worse
than that.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import jobenv, workerenv
from .backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN
from .errors import CrucibleError
from .voices import NARRATOR_ENGINE_SAMPLING
from .weights import ProgressHook, directory_bytes, sha256_of

#: The pack that carries the server itself: CPython plus the crucible wheel and
#: its dependencies, so `<dir>/bin/crucible` runs on a machine with no Python.
#: It is not a job type and has no `envs/<type>/` recipe — `pyproject.toml` is
#: what declares what goes into it, and `recipe_sha256` hashes that file.
SERVER_PACK = "server"

# `LLAMA_WINDOWS` — the THIRD backend, and WINDOWS IS ONE (PHASE15 section 0's
# amendment, and 3.5). Structurally what `mlx-darwin` is: `llama-server`
# children serving GGUF weights, with the llm classes and `pages` answering
# from them. An earlier draft of 4.4 called this `host-windows` and called it
# "a pack backend, never a `backend_kind`" — that was written at a moment when
# Windows had no backend kind at all, and keeping it would have given one
# machine two names, which is the exact R1 shape the amendment strikes.
#
# `crucible/backend.py` IS THE OWNER of the name, beside `CUDA_LINUX` and
# `MLX_DARWIN`, and it is IMPORTED from there at the top of this module. This
# module spelled the string itself for one session, because the two constants
# landed on two branches at once; one name, one owner, so that spelling is
# gone and `envpack.LLAMA_WINDOWS` is `backend.LLAMA_WINDOWS`.

#: The one pack `llama-windows` publishes. Named for the directory it unpacks
#: into, like every other pack: `%LOCALAPPDATA%\\Crucible\\host\\`. Its
#: archive therefore falls straight out of the section 1 rule with no special
#: case — `crucible-env-host-llama-windows-<version>.tar.zst`.
HOST_PACK = "host"

#: The one console script `pyproject.toml` declares, and the only command any
#: pack of ours exists to carry. A pack without it is not a pack of Crucible,
#: whatever else pip put in it (`write_cmd_shims`).
OWN_CONSOLE_SCRIPT = "crucible"

#: The tray's two packages, installed into the `host` pack AFTER the wheel and
#: declared HERE rather than in `pyproject.toml`'s dependencies.
#:
#: They are desktop-only — `pystray` draws the icon and menu (4.1/4.2) and
#: `pillow` is what it renders the icon image with — and `pyproject.toml` is
#: what EVERY pack's server half is built from, so putting them there would
#: make every headless Linux server carry a GUI toolkit. The Windows host
#: and macOS server packs include them; Linux packs do not.
HOST_EXTRA_PACKAGES = ("pystray", "pillow")

#: Split here. GitHub Releases refuses an asset over 2 GiB; 1900 MiB leaves room
#: for the difference between a vendor's "2 GB" and 2 GiB without thinking about
#: which one they meant.
PART_MAX_BYTES = 1900 * 1024 * 1024

#: The manifest asset's name, on the release and in a build's `--out` directory.
MANIFEST_NAME = "envpacks.json"

#: The manifest's schema. A document that does not say `1` is refused rather
#: than read optimistically: a future field is fine, a future MEANING is not.
PACK_SCHEMA = 1

#: Where a download's parts and the reassembled archive live while `install` runs.
DOWNLOADS_DIRNAME = "downloads"

#: Override the manifest URL. An OPTION, for a test and for a mirror — never a
#: fallback: nothing reads it because the release URL failed.
MANIFEST_URL_ENV = "CRUCIBLE_ENVPACK_URL"

RELEASE_DOWNLOAD_BASE = (
    "https://github.com/telltaleatheist/crucible/releases/download"
)

#: `tar`'s zstd compression level when a pack is built. 10 rather than 19: on an
#: 8 GB torch env 19 costs tens of minutes for a few per cent, and the asset is
#: downloaded far more often than it is built but is built on a runner with a
#: clock on it.
ZSTD_BUILD_LEVEL = 10


class PackError(CrucibleError):
    """A pack refusal, by the name section 3.1 gives it.

    Every one of these is a name a person can search for: `pack_not_published`,
    `pack_manifest_unreadable`, `pack_download_failed`, `pack_sha_mismatch`,
    `pack_recipe_drift`, `pack_disk`, `pack_unpack_failed`, plus the build-side
    `pack_no_zstd`, `pack_unknown`, `pack_not_buildable_here`,
    `pack_smoke_failed` and `pack_build_failed`.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ------------------------------------------------- the pinned interpreter


@dataclass(frozen=True)
class StandalonePython:
    """One python-build-standalone asset, pinned by version AND by digest.

    ONE PLACE. The release tag, the asset name, the URL and the sha256 are all
    here and nowhere else, because a build that fetched a "latest" interpreter
    would produce two packs with one version number on them.

    `install_only` rather than the full build: it is the layout `uv` ships — a
    single `python/` directory with `bin/python3`, relocatable by construction
    (relative rpaths, no absolute paths baked in) — and the full archive carries
    debug symbols and a build manifest nothing here reads.

    `install_only` on WINDOWS is the same archive shape and a different tree:
    `python/python.exe`, `python/pythonw.exe`, `python/Scripts/`, `python/Lib/`,
    `python/DLLs/`, and no `bin/` whatsoever. That is why the interpreter is
    asked for by `pack_python(root, backend_kind)` and never spelled inline.
    """

    python_version: str
    release: str
    asset: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            "https://github.com/astral-sh/python-build-standalone/releases/"
            f"download/{self.release}/{self.asset}"
        )


#: The interpreter each backend's packs are built on. All THREE digests were
#: read from the release's own `SHA256SUMS` on 2026-09-14
#: (https://github.com/astral-sh/python-build-standalone/releases/download/
#: 20260901/SHA256SUMS), not from a download and not from memory.
#:
#: PHASE15 4.4 first wrote the Windows asset as
#: `x86_64-pc-windows-msvc-SHARED-install_only` and then corrected itself from
#: that file: the word "shared" appears ZERO times in the 20260901 SHA256SUMS
#: (against 90 "static"), because the `-shared` infix is retired and the
#: Windows `install_only` build IS the shared one. A pin nobody can download is
#: a build that fails on the runner and nowhere else, so the name below is the
#: one that is in the release and `tests/test_envpack.py` pins the absence of
#: "-shared" so the doc's original spelling cannot creep back.
#:
#: 3.11 and not 3.12+, because `requires-python = ">=3.11"` is the floor the
#: server declares and the original recipes were resolved by pip against 3.11
#: (`envs/asr/cuda-linux.txt` records one pin that moved for exactly that
#: reason). Building a pack on 3.12 would resolve a different set than the file
#: `crucible doctor` checks the env against.
STANDALONE_PYTHON: dict[str, StandalonePython] = {
    CUDA_LINUX: StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256="faa0758583a63f14c5eee516af82738403b59c13edda6fc0a21d953febd89eed",
    ),
    MLX_DARWIN: StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-aarch64-apple-darwin-install_only.tar.gz",
        sha256="50424fa409e8ae84b82a3052522f64695b47dff2158b70bb7358e0ebd6c085c9",
    ),
    # Same release and same CPython (3.11.16) as the two backends, which is the
    # property that mattered: the host pack runs the same server code.
    LLAMA_WINDOWS: StandalonePython(
        python_version="3.11.16",
        release="20260901",
        asset="cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz",
        sha256="6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6",
    ),
}

# Recipe-specific interpreters. The CUDA Higgs SGLang recipe declares 3.12
# through jobenv.EnvSpec; changing the server's interpreter would break other
# recipes. Digest read from the upstream 20260901 SHA256SUMS on 2026-09-16.
RECIPE_STANDALONE_PYTHON: dict[tuple[str, str], StandalonePython] = {
    (CUDA_LINUX, "3.12"): StandalonePython(
        python_version="3.12.14",
        release="20260901",
        asset="cpython-3.12.14+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256="936c246dfdbbfa7cb22dd01814a21f582a892689fae96b06071a5e433baffa22",
    ),
}


# ------------------------------------------------------------ the pack table


#: What a built pack must be able to IMPORT before it becomes an asset, keyed by
#: pack name and then by backend. The `server` pack is absent on purpose: it is
#: smoke-tested by running `bin/crucible --version`, which proves the console
#: script's shebang survived the move as well as the import.
#:
#: THIS TABLE IS BESIDE `cli.INSTALLER_FOR` IN MEANING, NOT IN FILE (R1): `cli`
#: owns "which command installs this job type" and cannot be imported from here
#: without a cycle, so the two are tied by a check instead of by an import —
#: `tests/test_envpack.py::test_smoke_table_covers_every_installable_name`
#: asserts that every name `cli.INSTALLABLE_JOB_TYPES` can install has a pack
#: and a smoke import on every backend whose recipe exists.
#:
#: The module name is not the distribution name and the difference is not
#: cosmetic: `mlx-lm` imports as `mlx_lm`, `faster-whisper` as `faster_whisper`,
#: `qwen-asr` as `qwen_asr`, `ultimate-rvc` as `ultimate_rvc`. A table written
#: from `HEADLINE_PACKAGE` would fail every smoke test on four of six packs.
SMOKE_IMPORT: dict[str, dict[str, str]] = {
    "llm": {CUDA_LINUX: "vllm", MLX_DARWIN: "mlx_lm"},
    # `asr` is TWO ENGINES, so the smoke import differs by backend: a Mac
    # pack that imported `faster_whisper` would fail every build, and one
    # that imported nothing would ship an env nobody had opened.
    "asr": {CUDA_LINUX: "faster_whisper", MLX_DARWIN: "mlx_whisper"},
    "align": {CUDA_LINUX: "qwen_asr", MLX_DARWIN: "qwen_asr"},
    "rvc": {CUDA_LINUX: "ultimate_rvc", MLX_DARWIN: "ultimate_rvc"},
    # The tts env's KEY is the pack's name, and it differs by backend for the
    # reason `jobenv.tts_env` gives: on cuda-linux two narrator engines cannot
    # share a venv, so the engine is in the name.
    "tts-higgs-v3": {CUDA_LINUX: "narrator"},
    "tts": {MLX_DARWIN: "narrator"},
}


@dataclass(frozen=True)
class PackTarget:
    """One (name, backend): what builds it, what it imports, where it lands.

    `name` is both the pack's name in the manifest and the directory under
    `~/.crucible/envs/` it unpacks into, which is what makes `install` a rename
    rather than a lookup. The server pack is the one exception and lands at
    `~/.crucible/server/` (section 4); the `host` pack is the second and lands
    at `<home>/host/` — `%LOCALAPPDATA%\\Crucible\\host\\` on the machine that
    has one (PHASE15 4.4). Neither is a job env, so neither belongs under
    `envs/`, where `crucible doctor` reads every directory as one.
    """

    name: str
    backend_kind: str
    job_type: str | None
    narrator_engine: str | None
    recipe: Path
    smoke_import: str | None

    def env_dir(self, home: Path) -> Path:
        if self.name == SERVER_PACK:
            return home / "server"
        if self.name == HOST_PACK:
            return home / "host"
        return home / "envs" / self.name

    def stamp_path(self, home: Path) -> Path:
        return self.env_dir(home) / jobenv.ENV_STAMP_NAME

    def archive_name(self, version: str) -> str:
        return pack_filename(self.name, self.backend_kind, version)


def pack_filename(name: str, backend_kind: str, version: str) -> str:
    """The naming rule, section 1, in the one place that states it."""
    return f"crucible-env-{name}-{backend_kind}-{version}.tar.zst"


def part_filename(archive: str, index: int) -> str:
    """`….tar.zst.part00`, `.part01`, … Two digits, zero padded, from 00."""
    return f"{archive}.part{index:02d}"


def repo_root() -> Path:
    """The checkout this build runs from. Where `pyproject.toml` is."""
    return Path(__file__).resolve().parent.parent


def server_recipe() -> Path:
    """What declares the server pack's contents: `pyproject.toml`.

    The server pack has no `envs/` recipe because its dependencies are the
    wheel's own, and a second list of them would be exactly the two-owners
    shape R1 is about. So the file that already owns them is the file whose
    digest goes into `recipe_sha256`, and `pack_recipe_drift` then means what it
    means everywhere else: the pack was built from a different declaration than
    the running server has.
    """
    path = repo_root() / "pyproject.toml"
    if not path.is_file():
        raise PackError(
            "pack_not_buildable_here",
            f"no {path}; the server pack is built from a checkout, and this "
            "crucible was installed as a wheel",
        )
    return path


#: `recipe_sha256` is `jobenv`'s, NOT `weights.sha256_of`. It hashes the
#: recipe's LINE-ENDING-NORMALISED bytes, so the column in `envpacks.json` is a
#: property of the RECIPE and not of the checkout a runner happened to build
#: from — measured 2026-09-15, and written up in that function. One name, one
#: implementation: the pack manifest, the env stamp and `crucible doctor`'s
#: drift line are the same fact and must not be able to disagree.
recipe_digest = jobenv.recipe_sha256


def build_backend_kind() -> str:
    """Which backend's packs THIS machine can build, from platform and arch only.

    Deliberately NOT `backend.detect_backend()`. That probe asks for a working
    `nvidia-smi` and a working `mlx`, because it answers "can this host SERVE" —
    and a pack build needs neither. pip resolves wheels from a platform tag, so
    a GPU-less hosted runner builds the cuda-linux pack perfectly well, and
    requiring a card to build one would mean every release waits on Owen's
    desk.

    Windows answers `llama-windows` — a backend in the full sense since
    PHASE15 section 0's amendment, and the one whose engine is `llama-server`
    rather than a pip package. ARM Windows answers nothing: there is no
    `aarch64-pc-windows` interpreter pin and no runner, and it says so by name
    rather than building x86 wheels on a machine that cannot run them.
    """
    system = sys.platform
    arch = platform.machine()
    if system == "linux" and arch in ("x86_64", "amd64"):
        return CUDA_LINUX
    if system == "darwin" and arch == "arm64":
        return MLX_DARWIN
    # `platform.machine()` says `AMD64` on Windows and `x86_64` on Linux for
    # the same silicon; both spellings are accepted because both are seen —
    # CPython reads the former from the registry and the latter from `uname`.
    if system == "win32" and arch in ("x86_64", "AMD64"):
        return LLAMA_WINDOWS
    raise PackError(
        "pack_not_buildable_here",
        f"{system}/{arch} builds no Crucible pack; packs are built on Linux "
        "x86_64 (cuda-linux), Apple Silicon macOS (mlx-darwin) and Windows "
        "x86_64 (llama-windows, the HOST pack — the tray, the installer and "
        "a server whose engine is llama-server rather than a pip env). There "
        "is no pack for this platform and arch",
    )


def _job_type_targets(backend_kind: str) -> list[PackTarget]:
    """Every job-type pack this backend has a recipe for.

    Derived from the same two modules `crucible install` uses, so a recipe added
    to `envs/` becomes a pack without anybody editing a list here. A backend
    with no `.txt` for a type gets no pack for it and that is a fact rather than
    a gap: `envs/asr/mlx-darwin.md` is prose explaining that CTranslate2 has no
    Metal backend.
    """
    targets: list[PackTarget] = []

    spec = jobenv.llm_env(backend_kind)
    targets.append(
        PackTarget(
            name=spec.key,
            backend_kind=backend_kind,
            job_type="llm",
            narrator_engine=None,
            recipe=jobenv.recipe_for(spec),
            smoke_import=SMOKE_IMPORT["llm"][backend_kind],
        )
    )

    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        tts_spec = jobenv.tts_env(engine, backend_kind)
        try:
            recipe = jobenv.recipe_for(tts_spec)
        except jobenv.EnvError:
            continue
        if any(target.name == tts_spec.key for target in targets):
            # mlx-darwin: every engine resolves to the same env and the same
            # recipe, so the second engine is the same pack, not another one.
            continue
        targets.append(
            PackTarget(
                name=tts_spec.key,
                backend_kind=backend_kind,
                job_type="tts",
                narrator_engine=engine,
                recipe=recipe,
                smoke_import=SMOKE_IMPORT[tts_spec.key][backend_kind],
            )
        )

    for job_type in workerenv.WORKER_JOB_TYPES:
        try:
            recipe = workerenv.recipe_for(job_type, backend_kind)
        except workerenv.WorkerEnvError:
            continue
        targets.append(
            PackTarget(
                name=job_type,
                backend_kind=backend_kind,
                job_type=job_type,
                narrator_engine=None,
                recipe=recipe,
                smoke_import=SMOKE_IMPORT[job_type][backend_kind],
            )
        )
    return targets


def pack_targets(backend_kind: str) -> dict[str, PackTarget]:
    """Every pack this backend publishes, by name. `server` first."""
    if backend_kind not in STANDALONE_PYTHON:
        raise PackError(
            "pack_not_buildable_here",
            f"{backend_kind!r} is not a Crucible backend; the backends are "
            f"{sorted(STANDALONE_PYTHON)}",
        )
    if backend_kind == LLAMA_WINDOWS:
        # ONE pack, and no job-type packs AT ALL — which is a fact about the
        # ENGINE, not a gap. `llama-windows` serves its llm classes and
        # `pages` from `llama-server` children over GGUF weights (PHASE15
        # 3.10), and a `llama-server` is a binary Crucible spawns, not a pip
        # env it installs; there is no `envs/llm/llama-windows.txt` and there
        # is nothing for one to contain. The Python job types (`tts asr align
        # rvc denoise`) need WSL2 on this machine and say so (3.5). Falling
        # through to `_job_type_targets` would ask `jobenv` for recipes that
        # do not exist and publish an empty-ish table by accident.
        #
        # Its recipe is `pyproject.toml` for the same reason the server
        # pack's is: what goes into it is the wheel's own dependencies, and a
        # second list of them would be the two-owners shape R1 is about. The
        # tray's extras (`HOST_EXTRA_PACKAGES`) ride along with the wheel and
        # are pinned by this module rather than by a recipe file.
        return {
            HOST_PACK: PackTarget(
                name=HOST_PACK,
                backend_kind=LLAMA_WINDOWS,
                job_type=None,
                narrator_engine=None,
                recipe=server_recipe(),
                # Smoke-tested by RUNNING `crucible.cmd --version`, like
                # `server`, because the thing most likely to be broken is the
                # shim that replaces pip's unrelocatable `.exe` launcher —
                # and an import would not touch it.
                smoke_import=None,
            )
        }
    targets = {
        SERVER_PACK: PackTarget(
            name=SERVER_PACK,
            backend_kind=backend_kind,
            job_type=None,
            narrator_engine=None,
            recipe=server_recipe(),
            smoke_import=None,
        )
    }
    for target in _job_type_targets(backend_kind):
        targets[target.name] = target
    return targets


def pack_target(name: str, backend_kind: str) -> PackTarget:
    """One pack, or `pack_unknown` naming what this backend does publish."""
    targets = pack_targets(backend_kind)
    found = targets.get(name)
    if found is None:
        raise PackError(
            "pack_unknown",
            f"there is no pack called {name!r} for {backend_kind}; this build "
            f"publishes {sorted(targets)}",
        )
    return found


def python_for_target(target: PackTarget) -> StandalonePython:
    """Select the recipe's interpreter before downloading or resolving packages."""
    default = STANDALONE_PYTHON[target.backend_kind]
    required = None
    if target.job_type == "tts":
        assert target.narrator_engine is not None
        required = jobenv.tts_env(target.narrator_engine, target.backend_kind).python_version
    elif target.job_type == "llm":
        required = jobenv.llm_env(target.backend_kind).python_version
    if required is None or default.python_version.startswith(required + "."):
        return default
    pin = RECIPE_STANDALONE_PYTHON.get((target.backend_kind, required))
    if pin is None:
        raise PackError(
            "pack_python_unavailable",
            f"{target.name}/{target.backend_kind} requires Python {required}, "
            "but no verified standalone interpreter is pinned for that recipe",
        )
    return pin


def target_for_env_key(key: str, backend_kind: str) -> PackTarget:
    """The pack that installs the env directory called `key` on this backend.

    `crucible install <job type>` resolves its env spec first and asks this,
    rather than asking for a pack named after the job type: on cuda-linux the
    `tts` job type's env is `tts-higgs-v3`, and the pack is named for the
    directory it becomes.
    """
    return pack_target(key, backend_kind)


def every_pack() -> list[tuple[str, str]]:
    """Every (name, backend) a tag carries. `scripts/release.sh` prints these."""
    rows: list[tuple[str, str]] = []
    for backend_kind in sorted(STANDALONE_PYTHON):
        for name in pack_targets(backend_kind):
            rows.append((name, backend_kind))
    return rows


# ---------------------------------------------------------------- the manifest


@dataclass(frozen=True)
class PackEntry:
    """One row of `envpacks.json`. Section 2."""

    name: str
    backend: str
    python: str
    bytes: int
    sha256: str
    parts: tuple[str, ...]
    #: `recipe_digest()`, NOT `sha256_of()`: line-ending-normalised, so this
    #: field is a property of the recipe and not of the checkout it was built
    #: from. See that function for the measurement.
    recipe_sha256: str
    unpacked_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "python": self.python,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "parts": list(self.parts),
            "recipe_sha256": self.recipe_sha256,
            "unpacked_bytes": self.unpacked_bytes,
        }


_ENTRY_FIELDS = (
    ("name", str),
    ("backend", str),
    ("python", str),
    ("bytes", int),
    ("sha256", str),
    ("parts", list),
    ("recipe_sha256", str),
    ("unpacked_bytes", int),
)


@dataclass(frozen=True)
class PackManifest:
    """`envpacks.json`: the single owner of what packs exist for one version."""

    version: str
    packs: tuple[PackEntry, ...]
    source: str = "<memory>"

    def find(self, name: str, backend_kind: str) -> PackEntry | None:
        for entry in self.packs:
            if entry.name == name and entry.backend == backend_kind:
                return entry
        return None

    def require(self, name: str, backend_kind: str) -> PackEntry:
        entry = self.find(name, backend_kind)
        if entry is not None:
            return entry
        published = sorted(f"{p.name}/{p.backend}" for p in self.packs)
        raise PackError(
            "pack_not_published",
            f"{self.source} names no pack {name!r} for {backend_kind} at "
            f"version {self.version}; it publishes {published}. A pack that is "
            "absent is refused rather than built quietly — `crucible install "
            f"{name} --build` is the explicit way to build it here",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PACK_SCHEMA,
            "version": self.version,
            "packs": [entry.to_dict() for entry in self.packs],
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def parse_manifest(text: str, *, source: str = "<memory>") -> PackManifest:
    """Read `envpacks.json`, or `pack_manifest_unreadable` saying which part.

    Every refusal names the field. A manifest is the thing that decides which
    bytes a machine downloads and unpacks into its env directory; "could not
    parse" would leave its reader with nothing to fix.
    """

    def refuse(detail: str) -> PackError:
        return PackError(
            "pack_manifest_unreadable", f"{source}: {detail}"
        )

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise refuse(f"is not JSON: {exc}") from None
    if not isinstance(document, dict):
        raise refuse(f"is a {type(document).__name__}, not an object")
    schema = document.get("schema")
    if schema != PACK_SCHEMA:
        raise refuse(
            f"says schema {schema!r}; this build reads schema {PACK_SCHEMA}"
        )
    version = document.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise refuse(f"has no version string (got {version!r})")
    rows = document.get("packs")
    if not isinstance(rows, list):
        raise refuse(f"has no `packs` array (got {type(rows).__name__})")

    entries: list[PackEntry] = []
    for index, row in enumerate(rows):
        where = f"packs[{index}]"
        if not isinstance(row, dict):
            raise refuse(f"{where} is a {type(row).__name__}, not an object")
        for field, kind in _ENTRY_FIELDS:
            if field not in row:
                raise refuse(f"{where} has no {field!r}")
            # bool is an int in Python and `bytes: true` must not read as 1.
            if isinstance(row[field], bool) or not isinstance(row[field], kind):
                raise refuse(
                    f"{where}.{field} is {row[field]!r}, not a {kind.__name__}"
                )
        parts = row["parts"]
        if not parts:
            raise refuse(f"{where}.parts is empty; a pack is at least one part")
        if not all(isinstance(part, str) and part for part in parts):
            raise refuse(f"{where}.parts holds something that is not a filename")
        if row["bytes"] <= 0 or row["unpacked_bytes"] <= 0:
            raise refuse(
                f"{where} declares bytes={row['bytes']} "
                f"unpacked_bytes={row['unpacked_bytes']}; both are sizes"
            )
        entries.append(
            PackEntry(
                name=row["name"],
                backend=row["backend"],
                python=row["python"],
                bytes=row["bytes"],
                sha256=row["sha256"].lower(),
                parts=tuple(parts),
                recipe_sha256=row["recipe_sha256"].lower(),
                unpacked_bytes=row["unpacked_bytes"],
            )
        )
    duplicates = _duplicates((entry.name, entry.backend) for entry in entries)
    if duplicates:
        raise refuse(
            "names the same pack twice: "
            + ", ".join(f"{name}/{backend}" for name, backend in duplicates)
        )
    return PackManifest(version=version, packs=tuple(entries), source=source)


def _duplicates(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    twice: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen and pair not in twice:
            twice.append(pair)
        seen.add(pair)
    return twice


def merge_manifests(
    version: str, documents: Sequence[tuple[str, str]]
) -> PackManifest:
    """Merge fragments into one manifest. CI's last job, section 3.3.

    `documents` is [(source name, text)]. A fragment for another version is
    refused rather than dropped: it would mean two builds' assets sharing one
    manifest, and `install` would hand a machine a pack built from a different
    recipe with a matching name.
    """
    entries: list[PackEntry] = []
    for source, text in documents:
        fragment = parse_manifest(text, source=source)
        if fragment.version != version:
            raise PackError(
                "pack_manifest_unreadable",
                f"{source} is version {fragment.version}, and this release is "
                f"{version}; fragments from two builds are never merged",
            )
        entries.extend(fragment.packs)
    duplicates = _duplicates((entry.name, entry.backend) for entry in entries)
    if duplicates:
        raise PackError(
            "pack_manifest_unreadable",
            "two fragments carry the same pack: "
            + ", ".join(f"{name}/{backend}" for name, backend in duplicates),
        )
    ordered = tuple(sorted(entries, key=lambda entry: (entry.backend, entry.name)))
    return PackManifest(version=version, packs=ordered, source=MANIFEST_NAME)


def manifest_url(version: str) -> str:
    """Where `install` reads the manifest for the RUNNING version.

    `$CRUCIBLE_ENVPACK_URL` replaces it. That is an OPTION — a mirror, a test,
    a release candidate staged somewhere else — and never a fallback: nothing in
    this module reads it because the release URL failed.
    """
    override = os.environ.get(MANIFEST_URL_ENV)
    if override is not None and override.strip() != "":
        return override.strip()
    return f"{RELEASE_DOWNLOAD_BASE}/v{version}/{MANIFEST_NAME}"


def asset_url(manifest_location: str, filename: str) -> str:
    """A part's URL: beside the manifest, always.

    The manifest and its parts are assets of one release, so the parts are
    resolved RELATIVE to wherever the manifest was read from. That is what makes
    `--manifest-url file:///tmp/packs/envpacks.json` work without a second flag,
    and what makes a mirror one URL rather than two.
    """
    base = manifest_location.rsplit("/", 1)[0]
    return f"{base}/{filename}"


def read_manifest(location: str, *, timeout: int = 60) -> PackManifest:
    """Fetch and parse the manifest at `location` (http(s) or file://)."""
    try:
        with urllib.request.urlopen(location, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise PackError(
            "pack_manifest_unreadable",
            f"{location} answered HTTP {exc.code} {exc.reason}. A release with "
            "no envpacks.json carries no packs for this version",
        ) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PackError(
            "pack_manifest_unreadable", f"could not read {location}: {exc}"
        ) from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackError(
            "pack_manifest_unreadable", f"{location} is not UTF-8: {exc}"
        ) from None
    return parse_manifest(text, source=location)


# ------------------------------------------------------------------ progress


#: A line `crucible install` prints so the install TASK can turn the download's
#: bytes into the `progress {bytes_done, bytes_total, file}` event the pull task
#: already emits (PHASE14 section 3.1, PHASE13 section 3.3).
#:
#: THIS IS NOT A LOG SCRAPE (R4). The install task runs the console script as a
#: CHILD PROCESS and reads its stdout — that pipe is the only channel between
#: the two, and the alternative is an extra inherited file descriptor, which is
#: a second transport to keep working on two platforms for one event shape. So
#: the line is a declared WIRE with one owner (this module writes it and parses
#: it), carrying JSON rather than prose, and the human-readable line an operator
#: reads is printed separately and is not parsed by anything.
PROGRESS_PREFIX = "crucible-progress "


def progress_line(bytes_done: int, bytes_total: int | None, file: str) -> str:
    return PROGRESS_PREFIX + json.dumps(
        {"bytes_done": bytes_done, "bytes_total": bytes_total, "file": file}
    )


def parse_progress_line(line: str) -> dict[str, Any] | None:
    """The three fields, or None when this is an ordinary line."""
    if not line.startswith(PROGRESS_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"bytes_done", "bytes_total", "file"}:
        return None
    return payload


# -------------------------------------------------------------------- tooling


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise PackError(
            "pack_no_zstd",
            f"no `{name}` on PATH. A pack is a zstd tarball, so `tar` and "
            "`zstd` are what a machine needs to unpack one — on Ubuntu "
            "`apt-get install zstd`, on macOS they ship with the OS, and on "
            "Windows `tar.exe` is in C:\\Windows\\System32 and carries zstd "
            "itself (see `require_zstd_tar`)",
        )
    return found


#: What `tar --version` has to name on Windows before a pack is written with
#: it. MEASURED on Owen's PC, 2026-09-14: `C:\\Windows\\System32\\tar.exe` is
#: `bsdtar 3.8.1 - libarchive 3.8.1 … libzstd/1.5.5 …`, so the marker is in
#: the first line. GNU tar prints `tar (GNU tar) 1.32` and names no library at
#: all, which is exactly the machine this check is for.
TAR_ZSTD_MARKER = "libzstd"


def _require_tar_with_zstd(tar: str) -> None:
    """On Windows the tar IS the zstd, so ask it rather than assuming it.

    `--version` and not a trial compression: the probe runs before an
    interpreter is fetched, and a machine that cannot write a pack should
    learn so in a hundred milliseconds rather than after a pip run.
    """
    try:
        probe = subprocess.run(
            [tar, "--version"], capture_output=True, text=True, timeout=60
        )
    except OSError as exc:
        raise PackError(
            "pack_no_zstd", f"could not run `{tar} --version`: {exc}"
        ) from None
    reported = (probe.stdout or probe.stderr).strip().splitlines()
    first = reported[0].strip() if reported else ""
    if probe.returncode != 0 or TAR_ZSTD_MARKER not in first.lower():
        raise PackError(
            "pack_no_zstd",
            f"`{tar}` reports {first or '(nothing)'!r}, which does not name "
            f"{TAR_ZSTD_MARKER}. Windows ships no `zstd.exe`, so the tar that "
            "writes a pack has to carry zstd itself — "
            "C:\\Windows\\System32\\tar.exe does (bsdtar 3.8.1, libzstd "
            "1.5.5, measured 2026-09-14). Put that one ahead of this one on "
            "PATH. A GNU tar here would shell out to a `zstd` that is not "
            "installed and fail at the compression step, which is after the "
            "interpreter download and the pip run — the expensive end of the "
            "build to find out at",
        )


def require_zstd_tar() -> None:
    """The tools, checked before anything is downloaded or built.

    TWO ON POSIX AND ONE ON WINDOWS, and that is a measured difference rather
    than a relaxation. Ubuntu's and macOS's `tar` reach zstd by launching the
    `zstd` binary, so both have to be there. Windows 10/11 ship `tar.exe` —
    bsdtar, with libzstd LINKED IN — and ship no `zstd.exe` at all, so
    demanding one would refuse a build that the stock machine can do. What
    replaces the demand is a question: `_require_tar_with_zstd` asks the tar
    it found whether it carries zstd, because an old libarchive or a GNU tar
    first on PATH is a real machine and must not reach the compression step.
    """
    tar = _tool("tar")
    if sys.platform == "win32":
        _require_tar_with_zstd(tar)
        return
    _tool("zstd")


def _run(
    command: Sequence[str],
    code: str,
    failure: str,
    on_line: Any = None,
    *,
    cwd: Path | None = None,
) -> None:
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=None if cwd is None else str(cwd),
    )
    tail: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        del tail[:-40]
        if on_line is not None:
            on_line(line)
    returncode = process.wait()
    if returncode != 0:
        raise PackError(
            code,
            f"{failure}: `{' '.join(command)}` exited {returncode}\n"
            + "\n".join(tail),
        )


def archive_argv(tar: str, root: Path, archive: Path) -> list[str]:
    """The `tar` command line that WRITES a pack. TWO SPELLINGS, both measured.

    They are not interchangeable and NEITHER IS A FALLBACK FOR THE OTHER —
    each is the only one its tar understands:

    * **POSIX** (`--use-compress-program "zstd -T0 -<level>"`). GNU tar reaches
      zstd by launching the binary, and `-T0` (all cores) is a `zstd` CLI flag
      that only exists on the command line. Untouched by the Windows work.
    * **Windows** (`--zstd --options zstd:compression-level=<level>`). `tar.exe`
      is bsdtar with libzstd linked in, so it compresses in-process and needs
      no `zstd.exe` — which is what makes the stock machine able to build at
      all, since Windows ships none. GNU tar has no `--options`.

    Measured on Owen's PC, 2026-09-14: the Windows form round-trips (create
    exit 0, extract exit 0, contents intact). No `zstd:threads`: libarchive's
    `threads=0` means SINGLE-threaded, the opposite of the CLI's `-T0`, and
    writing 0 in both spellings expecting the same thing is how a build gets
    eight times slower without anybody noticing.

    `ZSTD_BUILD_LEVEL` is the one owner of the level in both.
    """
    if sys.platform == "win32":
        return [
            tar,
            "--zstd",
            "--options",
            f"zstd:compression-level={ZSTD_BUILD_LEVEL}",
            "-C",
            str(root),
            "-cf",
            str(archive),
            ".",
        ]
    return [
        tar,
        "--use-compress-program",
        f"zstd -T0 -{ZSTD_BUILD_LEVEL}",
        "-C",
        str(root),
        "-cf",
        str(archive),
        ".",
    ]


def extract_argv(tar: str, archive: Path, into: Path) -> list[str]:
    """The `tar` command line that READS a pack. ONE spelling, and checked.

    `--zstd` on extract costs nothing on either tar: GNU tar launches the
    `zstd` binary that `require_zstd_tar` already proved is there, and bsdtar
    decompresses in-process — it accepts the flag in extract mode and would
    have auto-detected the format without it (both measured 2026-09-14, exit
    0). So the read path stays one argv, which is one fewer thing that can
    differ between the machine that builds a pack and the machine that
    installs it.
    """
    return [tar, "--zstd", "-xf", str(archive), "-C", str(into)]


def create_archive(root: Path, archive: Path, *, on_line: Any = None) -> None:
    """`tar --zstd` the CONTENTS of `root` into `archive`.

    The contents and not the directory: unpacking then fills whatever directory
    it is pointed at, so the pack's name on disk is the INSTALLER's decision
    (`<key>.partial` and then `<key>`) rather than a string baked into the tar.
    """
    require_zstd_tar()
    archive.parent.mkdir(parents=True, exist_ok=True)
    _run(
        archive_argv(_tool("tar"), root, archive),
        "pack_build_failed",
        f"could not tar {root} into {archive}",
        on_line,
    )


def extract_archive(archive: Path, into: Path, *, on_line: Any = None) -> None:
    """Unpack a pack archive into `into`, which must already exist."""
    require_zstd_tar()
    _run(
        extract_argv(_tool("tar"), archive, into),
        "pack_unpack_failed",
        f"could not unpack {archive.name} into {into}",
        on_line,
    )


# -------------------------------------------------------------- the download


def _fetch(
    url: str,
    destination: Path,
    *,
    already_done: int,
    grand_total: int | None,
    on_progress: ProgressHook | None,
    timeout: int = 120,
    chunk: int = 1 << 20,
) -> None:
    """One part, with RESUME, into `destination`.

    Resume is a `Range` request and it is believed only when the server ANSWERS
    206. A 200 means the whole body is coming whatever we asked for — which is
    what `file://` does, and what a proxy that ignores ranges does — and
    appending that to existing bytes would build a corrupt archive whose only
    symptom is `pack_sha_mismatch` after the whole download. So a 200 truncates
    first, and the digest at the end is what proves either path.
    """
    resume_from = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            partial = getattr(response, "status", None) == 206
            mode = "ab" if partial else "wb"
            done = resume_from if partial else 0
            with destination.open(mode) as handle:
                if not partial and resume_from > 0:
                    handle.truncate(0)
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if on_progress is not None:
                        on_progress(
                            already_done + done, grand_total, destination.name
                        )
    except urllib.error.HTTPError as exc:
        raise PackError(
            "pack_download_failed",
            f"{url} answered HTTP {exc.code} {exc.reason}",
        ) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PackError(
            "pack_download_failed", f"could not fetch {url}: {exc}"
        ) from None


def download_parts(
    urls: Sequence[str],
    archive: Path,
    *,
    total_bytes: int,
    downloads: Path,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> None:
    """Fetch each part, append it, delete it. Peak extra disk is ONE part.

    BookForge's pattern, and its reason: a per-part digest proves each part
    arrived and says nothing about the join, so the digest that matters is of
    the reassembled whole and the parts themselves are scratch. A part already
    on disk from an interrupted run is RESUMED (that is what `downloads/` is
    for); the growing archive is started from nothing, because appending onto a
    half-written archive from a previous attempt is the one way to build bytes
    that pass nothing and look like progress.
    """
    downloads.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    done = 0
    for url in urls:
        part = downloads / url.rsplit("/", 1)[-1]
        if on_line is not None:
            on_line(f"fetching {part.name}")
        _fetch(
            url,
            part,
            already_done=done,
            grand_total=total_bytes,
            on_progress=on_progress,
        )
        with archive.open("ab") as out, part.open("rb") as source:
            shutil.copyfileobj(source, out, 1 << 20)
        done += part.stat().st_size
        part.unlink()


def sha256_of_parts(parts: Sequence[Path], chunk: int = 1 << 20) -> str:
    """The digest of the reassembled whole, WITHOUT reassembling it.

    `envpack build --check` uses this: the parts are on disk beside each other
    and concatenating a 3 GB archive to hash it would need 3 GB nobody has a
    reason to spend.
    """
    digest = hashlib.sha256()
    for part in parts:
        with part.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
    return digest.hexdigest()


def split_archive(archive: Path, *, part_bytes: int = PART_MAX_BYTES) -> list[Path]:
    """Split into `.partNN` files and REMOVE the archive.

    Removed rather than kept: the parts are the assets, the archive is
    reconstructible from them, and on a hosted runner the difference is whether
    a 3 GB pack needs 3 GB or 6 GB at the end of the job.
    """
    parts: list[Path] = []
    with archive.open("rb") as source:
        index = 0
        while True:
            part = archive.parent / part_filename(archive.name, index)
            written = 0
            with part.open("wb") as handle:
                while written < part_bytes:
                    block = source.read(min(1 << 20, part_bytes - written))
                    if not block:
                        break
                    handle.write(block)
                    written += len(block)
            if written == 0:
                part.unlink()
                break
            parts.append(part)
            index += 1
            if written < part_bytes:
                break
    archive.unlink()
    return parts


# --------------------------------------------------------------------- install


def check_recipe(entry: PackEntry, target: PackTarget) -> None:
    """`pack_recipe_drift`, or nothing. R3: there is no "close enough".

    A pack built from an older recipe holds different package versions than the
    recipe `crucible doctor` checks the env against, so installing it produces an
    env that reads as broken the moment anybody asks — and, worse, an env whose
    numbers differ from the ones a manifest's memory estimates were measured on.
    """
    here = recipe_digest(target.recipe)
    if here != entry.recipe_sha256:
        raise PackError(
            "pack_recipe_drift",
            f"the {entry.name}/{entry.backend} pack was built from a "
            f"{target.recipe.name} whose sha256 is {entry.recipe_sha256[:12]}, "
            f"and this build's {target.recipe} hashes to {here[:12]}. The pack "
            "is not this recipe's. Install the release whose recipe it was "
            f"built from, or build here with `crucible install "
            f"{target.job_type or target.name} --build`",
        )


def check_disk(home: Path, entry: PackEntry) -> None:
    """`pack_disk`, with the numbers, BEFORE a byte is downloaded.

    The need is the unpacked tree plus the reassembled archive plus one part in
    flight — the three things that are on disk at the same moment, at the moment
    just before the archive is deleted.
    """
    largest_part = entry.bytes if len(entry.parts) == 1 else PART_MAX_BYTES
    need = entry.unpacked_bytes + entry.bytes + largest_part
    home.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(home).free
    if free < need:
        raise PackError(
            "pack_disk",
            f"{entry.name}/{entry.backend} needs "
            f"{need / 1e9:.1f} GB free under {home} "
            f"({entry.unpacked_bytes / 1e9:.1f} GB unpacked + "
            f"{entry.bytes / 1e9:.1f} GB archive + "
            f"{largest_part / 1e9:.1f} GB for the part in flight) and there is "
            f"{free / 1e9:.1f} GB. Nothing was downloaded",
        )


def _replace_directory(staged: Path, final: Path) -> None:
    """Make `staged` become `final`, atomically, whatever is at `final`.

    `os.replace` onto a NON-EMPTY directory is `ENOTEMPTY` on POSIX, so the old
    tree is moved aside first and removed after the rename. The window in which
    neither name holds the old env is one `rename`, and the new one is complete
    before it opens — which is the whole reason for `.partial`.
    """
    if final.exists():
        retired = final.with_name(f"{final.name}.old-{os.getpid()}")
        if retired.exists():
            shutil.rmtree(retired)
        os.replace(final, retired)
        try:
            os.replace(staged, final)
        except OSError:
            os.replace(retired, final)
            raise
        shutil.rmtree(retired, ignore_errors=True)
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)


def install_pack(
    home: Path,
    target: PackTarget,
    version: str,
    *,
    location: str | None = None,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
) -> PackEntry:
    """Download, verify and unpack one pack. Section 3.1, in order.

    Every failure leaves a name behind and nothing half-installed under the env
    directory: the unpack happens in `<key>.partial/`, which is removed on the
    next run, and only a complete tree is ever renamed into place.
    """
    require_zstd_tar()
    where = location if location is not None else manifest_url(version)
    manifest = read_manifest(where)
    entry = manifest.require(target.name, target.backend_kind)
    check_recipe(entry, target)
    check_disk(home, entry)

    downloads = home / DOWNLOADS_DIRNAME
    archive = downloads / pack_filename(
        target.name, target.backend_kind, manifest.version
    )
    urls = [asset_url(where, part) for part in entry.parts]
    started = time.monotonic()
    if on_line is not None:
        on_line(
            f"{entry.name}/{entry.backend}: {len(entry.parts)} part(s), "
            f"{entry.bytes / 1e9:.2f} GB -> {entry.unpacked_bytes / 1e9:.2f} GB "
            f"unpacked, from {where}"
        )
    download_parts(
        urls,
        archive,
        total_bytes=entry.bytes,
        downloads=downloads,
        on_line=on_line,
        on_progress=on_progress,
    )

    size = archive.stat().st_size
    if size != entry.bytes:
        archive.unlink()
        raise PackError(
            "pack_download_failed",
            f"{archive.name} reassembled to {size} bytes and the manifest says "
            f"{entry.bytes}. The archive was deleted",
        )
    digest = sha256_of(archive)
    if digest != entry.sha256:
        archive.unlink()
        raise PackError(
            "pack_sha_mismatch",
            f"{archive.name} hashes to {digest} and {where} pins "
            f"{entry.sha256}. The archive was deleted. Either the manifest is "
            "wrong or these are not the bytes it names, and both are worse than "
            "no env at all",
        )
    if on_line is not None:
        on_line(f"verified {digest[:12]} ({size / 1e9:.2f} GB)")

    final = target.env_dir(home)
    partial = final.with_name(f"{final.name}.partial")
    if partial.exists():
        # A half-done unpack from an interrupted run. Nothing has ever trusted
        # it — that is what the name is for — so it goes.
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        extract_archive(archive, partial, on_line=on_line)
    except PackError:
        shutil.rmtree(partial, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise
    python = pack_python(partial, target.backend_kind)
    if not python.is_file():
        shutil.rmtree(partial, ignore_errors=True)
        archive.unlink(missing_ok=True)
        raise PackError(
            "pack_unpack_failed",
            f"{archive.name} unpacked without a "
            f"{python.relative_to(partial).as_posix()}. A pack is an "
            "interpreter with an env installed into it; this archive is "
            "something else",
        )

    elapsed = time.monotonic() - started
    (partial / jobenv.ENV_STAMP_NAME).write_text(
        json.dumps(
            {
                "job_type": target.job_type,
                "backend": target.backend_kind,
                "recipe": target.recipe.name,
                "recipe_sha256": entry.recipe_sha256,
                "python_version": entry.python,
                "pack_sha256": entry.sha256,
                "pack_version": manifest.version,
                "seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _replace_directory(partial, final)
    archive.unlink(missing_ok=True)
    if on_line is not None:
        on_line(f"installed {final} in {elapsed:.0f}s")
    return entry


# ----------------------------------------------------------------- the build


def fetch_standalone_python(
    backend_kind: str,
    cache: Path,
    *,
    on_line: Callable[[str], None] | None = None,
    on_progress: ProgressHook | None = None,
    pin: StandalonePython | None = None,
) -> Path:
    """The pinned interpreter archive, verified. Cached so a rebuild is free."""
    if pin is None:
        pin = STANDALONE_PYTHON[backend_kind]
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / pin.asset
    if archive.is_file() and sha256_of(archive) == pin.sha256:
        return archive
    if archive.exists():
        archive.unlink()
    if on_line is not None:
        on_line(f"fetching {pin.asset}")
    _fetch(
        pin.url,
        archive,
        already_done=0,
        grand_total=None,
        on_progress=on_progress,
    )
    digest = sha256_of(archive)
    if digest != pin.sha256:
        archive.unlink()
        raise PackError(
            "pack_sha_mismatch",
            f"{pin.asset} hashes to {digest} and crucible/envpack.py pins "
            f"{pin.sha256}. The download was deleted",
        )
    return archive


def _prune(root: Path) -> None:
    """`__pycache__` and pip's own caches. See this module's header for what stays."""
    for directory in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(directory, ignore_errors=True)


#: The three lines that replace a baked-in shebang — distlib's own polyglot,
#: with the absolute interpreter path swapped for one derived from `$0`.
#:
#: `sh` reads line 2 as `exec <the python beside this script> <this script>
#: <the args>` — `'''exec'` is `''` then `'exec'`, which is the word `exec` —
#: and never reaches line 3, because `exec` has replaced it. Python reads lines
#: 2 and 3 as ONE triple-quoted string that it throws away.
#:
#: The single quotes are load-bearing. The obvious `"exec" "$(dirname -- "$0")
#: /python3" …` is a SyntaxError in Python, because the nested `"` inside the
#: command substitution ends the Python string early. (Written after watching
#: exactly that reach the smoke test, 2026-09-14.)
RELOCATABLE_SHEBANG = (
    "#!/bin/sh\n"
    "'''exec' \"$(dirname -- \"$0\")/python3\" \"$0\" \"$@\"\n"
    "' '''\n"
)


def relocate_console_scripts(root: Path) -> list[str]:
    """Make every `bin/` entry point survive the move. THE WHOLE PACK DEPENDS ON IT.

    **The interpreter is relocatable and the console scripts are not.**
    python-build-standalone bakes no absolute paths, but `pip install` writes
    each entry point with the *absolute* path of the interpreter that installed
    it, as a shebang. Unpack that anywhere else and the path is gone — and the
    failure is the worst-shaped one there is: `exec` on a script whose shebang
    does not resolve raises **ENOENT naming the SCRIPT**, so the message says
    `bin/crucible` does not exist while it is sitting right there.

    BookForge has been bitten by exactly this and wrote it down
    (`crucible/jobs/rvc/worker.py`'s header, from `electron/rvc-bridge.ts`):
    an env installed by extracting into a temp directory and moving it into
    place fails with exit 1 and *zero output*, because the launcher dies before
    python starts. Its answer was to never call a console script. That answer
    is not available here — PHASE14 section 4 has the bootstrap run
    `<server>/bin/crucible init` and point a systemd `ExecStart` at it — so the
    shebang is fixed instead of avoided.

    Found by building the `server` pack for real, 2026-09-14: the wheel
    installed, the archive tarred, and the smoke test died on
    `FileNotFoundError: .../pack/bin/crucible`.

    A script reached through a SYMLINK is not covered: `dirname "$0"` is the
    link's directory, not the pack's. That is deliberate rather than an
    oversight — `readlink -f` would cover it and is absent from macOS before
    12.3 — and nothing in Crucible symlinks into a pack: the service unit names
    the real path.

    THE TWO POSIX BACKENDS ONLY. `llama-windows` has no `bin/` and no shebang
    to rewrite; `build_pack` calls `write_cmd_shims` there instead, and this
    function is not reached. Nothing here was changed for Windows, because
    changing it would be changing what the two shipping backends do.

    Returns the names it rewrote, so a build can say so.
    """
    bin_dir = root / "bin"
    marker = str(root).encode("utf-8")
    rewritten: list[str] = []
    for entry in sorted(bin_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        body = entry.read_bytes()
        if not body.startswith(b"#!"):
            continue
        lines = body.split(b"\n")
        if marker not in b"\n".join(lines[:3]):
            # A shebang that does not name this build tree is somebody else's
            # business — `#!/usr/bin/env python3` is already relocatable, and
            # rewriting it would be this function inventing policy.
            continue
        if lines[0] == b"#!/bin/sh" and lines[1].startswith(b"'''exec'"):
            # distlib's long-path form: three lines of wrapper, then the code.
            rest = b"\n".join(lines[3:])
        else:
            rest = b"\n".join(lines[1:])
        mode = entry.stat().st_mode
        entry.write_bytes(RELOCATABLE_SHEBANG.encode("utf-8") + rest)
        entry.chmod(mode)
        rewritten.append(entry.name)
    return rewritten


def pack_python(root: Path, backend_kind: str) -> Path:
    """The interpreter inside a pack, whichever layout the pack has.

    PUBLIC because PHASE15 4.4 names it: python-build-standalone's Windows
    `install_only` tree is `python.exe` / `pythonw.exe` / `Scripts\\` / `Lib\\`
    / `DLLs\\` and has NO `bin/`, so a build, an install and a smoke test that
    each spell `root / "bin" / "python"` are three places that have to learn
    the same thing and two of them will not. One function, every caller.

    An unknown backend is refused rather than guessed at. Guessing `bin/python`
    would produce "unpacked without a bin/python" for a tree that is perfectly
    fine, which is a refusal that sends its reader to the wrong file.
    """
    if backend_kind == LLAMA_WINDOWS:
        return root / "python.exe"
    if backend_kind in (CUDA_LINUX, MLX_DARWIN):
        return root / "bin" / "python"
    raise PackError(
        "pack_not_buildable_here",
        f"{backend_kind!r} has no pack layout; the pack backends are "
        f"{sorted(STANDALONE_PYTHON)}",
    )


#: The `.cmd` shim's text, and the reason it exists at all.
#:
#: **7.2a's defect has no POSIX answer on Windows.** pip does not write a
#: shebang script into `Scripts\\`; it writes `Scripts\\<name>.exe`, a launcher
#: BINARY with the building interpreter's absolute path embedded inside the
#: executable. A move breaks it exactly as it breaks a shebang, and a shebang
#: rewrite cannot reach it — there is no text to rewrite. So the pack ships a
#: `.cmd` beside `python.exe` and the `.exe` launchers stay only as the dead
#: weight pip left: nothing in Crucible calls them (PHASE15 4.4).
#:
#: `%~dp0` is the directory of the running batch file, WITH a trailing
#: backslash — the Windows spelling of `$(dirname -- "$0")/`, which is why
#: `"%~dp0python.exe"` and not `"%~dp0\\python.exe"`. It is quoted because
#: `%LOCALAPPDATA%` contains the user's name and a user named "Owen Morgan"
#: would otherwise split the command in two.
#:
#: CRLF, not LF. `cmd.exe`'s batch parser is line-oriented on CRLF; an LF-only
#: `.cmd` misparses labels and can swallow its own last line, and the failure
#: shows up as a shim that silently does nothing rather than as a syntax error.
CMD_SHIM_HEADER = "@echo off\r\n"


def cmd_shim_text(module: str, function: str) -> str:
    """The two lines of one shim, for `module:function`.

    `-m <module>` when the function is `main`, because that is the form the
    console script and the module agree on: `crucible.cli` ends in
    `if __name__ == "__main__": raise SystemExit(main())`, so running it as a
    script and calling its `main` are the same act, and `-m` keeps the shim
    readable enough that an operator can see what it does.

    When the function is NOT named `main`, `-m` would run the module's own
    `__main__` behaviour — usually nothing at all — and the shim would exit 0
    having done none of the work the entry point names. That is the worst
    shape of failure there is, so the other form is spelled out in full and
    calls the function by name.
    """
    if function == "main":
        line = f'"%~dp0python.exe" -m {module} %*'
    else:
        line = (
            f'"%~dp0python.exe" -c "import sys; from {module} import '
            f'{function}; sys.exit({function}())" %*'
        )
    return CMD_SHIM_HEADER + line + "\r\n"


def _console_entry_points(root: Path) -> dict[str, tuple[str, str]]:
    """`{script name: (module, function)}`, READ from the pack's own metadata.

    Not hardcoded, and not guessed from the script's name. `crucible` is
    `crucible.cli:main` and `pip3.11` is `pip._internal.cli.main:main`; a table
    in this file would be a second owner of something `pyproject.toml` and
    every dependency already declare, and it would go stale the first time a
    dependency added a script.
    """
    site = root / "Lib" / "site-packages"
    found: dict[str, tuple[str, str]] = {}
    for metadata in sorted(site.glob("*.dist-info/entry_points.txt")):
        # `delimiters=("=",)` because configparser's default also splits on
        # `:`, and `crucible = crucible.cli:main` has one of each.
        parser = configparser.ConfigParser(delimiters=("=",))
        # Entry point names are case-sensitive; configparser lowercases keys
        # unless told not to.
        parser.optionxform = str  # type: ignore[method-assign, assignment]
        try:
            parser.read_string(metadata.read_text(encoding="utf-8"))
        except (configparser.Error, UnicodeDecodeError) as exc:
            raise PackError(
                "pack_build_failed",
                f"{metadata} is not a readable entry_points.txt: {exc}",
            ) from None
        if not parser.has_section("console_scripts"):
            continue
        for name, spec in parser.items("console_scripts"):
            # `name = module:function [extra]` — the extras marker is pip's
            # business and is not part of what we call.
            target = spec.split("[", 1)[0].strip()
            module, separator, function = target.partition(":")
            if not separator or not module.strip() or not function.strip():
                raise PackError(
                    "pack_build_failed",
                    f"{metadata} declares console script {name!r} as "
                    f"{spec!r}, which is not `module:function`. A shim cannot "
                    "be written for an entry point nobody can read, and "
                    "writing none would ship a pack whose command is missing",
                )
            found[name] = (module.strip(), function.strip())
    return found


def write_cmd_shims(root: Path) -> list[str]:
    """`<pack>\\<name>.cmd` for every console script, beside `python.exe`.

    The Windows half of `relocate_console_scripts`, and a different act for the
    reason `CMD_SHIM_HEADER` gives: there is no shebang to rewrite, only an
    `.exe` with an absolute path compiled into it.

    BESIDE `python.exe` — at the pack ROOT, not in `Scripts\\` — so the shim's
    `%~dp0python.exe` resolves without climbing, and so `install.ps1` and the
    Startup shortcut name `<pack>\\crucible.cmd` with no subdirectory in it.

    The NAMES come from `Scripts\\*.exe`, because that is the authority on what
    pip actually created in this tree; the MODULE comes from the installed
    distributions' `entry_points.txt`. A script pip wrote that no metadata
    explains is refused rather than skipped: skipping it would ship a pack
    missing a command, and the person who finds out is the operator.

    Returns the names it wrote, like `relocate_console_scripts`.
    """
    scripts = root / "Scripts"
    if not scripts.is_dir():
        raise PackError(
            "pack_build_failed",
            f"{root} has no Scripts\\ directory; pip installs console script "
            "launchers there on Windows, so this tree is not one pip has "
            "installed into",
        )
    declared = _console_entry_points(root)
    written: list[str] = []
    for launcher in sorted(scripts.glob("*.exe")):
        name = launcher.stem
        entry = declared.get(name)
        if entry is None:
            raise PackError(
                "pack_build_failed",
                f"{launcher} exists and no dist-info in {root} declares a "
                f"console script called {name!r}, so there is nothing to "
                f"point a {name}.cmd at. The pack's contents and its metadata "
                "disagree, and shipping the half that works is how a command "
                "goes missing quietly",
            )
        module, function = entry
        (root / f"{name}.cmd").write_text(
            cmd_shim_text(module, function), encoding="utf-8", newline=""
        )
        written.append(name)
    if OWN_CONSOLE_SCRIPT not in written:
        # MEASURED 2026-09-15, building the host pack on Owen's PC: with
        # `PYTHONPATH` pointing at the source checkout, the pack's pip found
        # the repo's `crucible.egg-info`, reported "Requirement already
        # satisfied: crucible", installed every DEPENDENCY and not the wheel,
        # and this function wrote twelve shims for other people's commands
        # and none for the one the pack exists to carry. The failure then
        # surfaced two steps later as `FileNotFoundError: [WinError 2]` out of
        # the smoke test — a traceback about a path, for a pack that is simply
        # empty of Crucible.
        raise PackError(
            "pack_build_failed",
            f"{root}\\Scripts has no {OWN_CONSOLE_SCRIPT}.exe, so the pack "
            f"carries {len(written)} other command(s) and not its own: "
            f"{', '.join(written) or 'none'}. pip installed this tree's "
            "dependencies and not the crucible wheel — the usual cause is a "
            "`PYTHONPATH` or a `*.egg-info` that made pip believe crucible "
            "was already satisfied. Build with crucible INSTALLED, not on a "
            "path.",
        )
    return written


def build_pack(
    target: PackTarget,
    version: str,
    out: Path,
    *,
    keep_tree: bool = False,
    on_line: Callable[[str], None] | None = None,
) -> PackEntry:
    """Produce one pack: interpreter, recipe, prune, tar, split, smoke, manifest.

    The ORDER is chosen for peak disk, because a hosted runner is the tightest
    machine this ever runs on (section 3.3). The build tree is deleted before
    the smoke test unpacks its own copy, and the archive is deleted by the split
    — so the high-water mark is `unpacked + archive` and never
    `unpacked + archive + unpacked`.
    """
    say = on_line if on_line is not None else (lambda line: None)
    require_zstd_tar()
    pin = python_for_target(target)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".cache"
    workspace = out / ".build" / target.name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    started = time.monotonic()
    interpreter = fetch_standalone_python(target.backend_kind, cache, on_line=say, pin=pin)
    say(f"unpacking {pin.asset}")
    _run(
        [_tool("tar"), "-xzf", str(interpreter), "-C", str(workspace)],
        "pack_build_failed",
        f"could not unpack {interpreter}",
        say,
    )
    root = workspace / "python"
    interpreter_path = pack_python(root, target.backend_kind)
    if not interpreter_path.is_file():
        raise PackError(
            "pack_build_failed",
            f"{pin.asset} did not unpack to a "
            f"python/{interpreter_path.relative_to(root).as_posix()} under "
            f"{workspace}; the pinned asset's layout is not install_only's",
        )

    python = str(interpreter_path)
    say("installing pip and wheel")
    _run(
        [python, "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        "pack_build_failed",
        f"could not upgrade pip in {root}",
        say,
    )
    if target.name in (SERVER_PACK, HOST_PACK):
        wheel = _build_wheel(out, say)
        say(f"installing {wheel.name}")
        _run(
            [python, "-m", "pip", "install", str(wheel)],
            "pack_build_failed",
            f"could not install {wheel} into {root}",
            say,
        )
        if target.name == HOST_PACK or (target.name == SERVER_PACK and target.backend_kind == "mlx-darwin"):
            # AFTER the wheel, so a resolver conflict between the tray and the
            # server's own pins fails while the server is already the thing
            # installed — and so the log reads in the order the pack was
            # assembled. See `HOST_EXTRA_PACKAGES` for why they are not
            # dependencies of the wheel.
            say(f"installing the tray: {', '.join(HOST_EXTRA_PACKAGES)}")
            _run(
                [python, "-m", "pip", "install", *HOST_EXTRA_PACKAGES],
                "pack_build_failed",
                f"could not install {HOST_EXTRA_PACKAGES} into {root}",
                say,
            )
    else:
        say(f"installing {target.recipe}")
        _run(
            [python, "-m", "pip", "install", "-r", str(target.recipe)],
            "pack_build_failed",
            f"could not install {target.recipe} into {root}",
            say,
        )

    python_version = subprocess.run(
        [python, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout.strip()
    if python_version != pin.python_version:
        raise PackError(
            "pack_build_failed",
            f"{root} reports python {python_version!r} and the pin says "
            f"{pin.python_version}",
        )

    if target.backend_kind == LLAMA_WINDOWS:
        shimmed = write_cmd_shims(root)
        say(f"wrote {len(shimmed)} .cmd shim(s): {', '.join(shimmed)}")
    else:
        rewritten = relocate_console_scripts(root)
        say(f"relocated {len(rewritten)} console script(s): {', '.join(rewritten)}")
    _prune(root)
    unpacked_bytes = directory_bytes(root)
    archive = out / target.archive_name(version)
    say(f"tarring {unpacked_bytes / 1e9:.2f} GB into {archive.name}")
    create_archive(root, archive, on_line=say)
    if not keep_tree:
        # Before the smoke test, so the high-water mark is one unpacked tree.
        shutil.rmtree(workspace, ignore_errors=True)

    smoke_test(target, archive, on_line=say)

    size = archive.stat().st_size
    digest = sha256_of(archive)
    parts = split_archive(archive)
    entry = PackEntry(
        name=target.name,
        backend=target.backend_kind,
        python=python_version,
        bytes=size,
        sha256=digest,
        parts=tuple(part.name for part in parts),
        recipe_sha256=recipe_digest(target.recipe),
        unpacked_bytes=unpacked_bytes,
    )
    write_manifest_entry(out, version, entry)
    say(
        f"built {target.name}/{target.backend_kind} in "
        f"{time.monotonic() - started:.0f}s: {size / 1e9:.2f} GB in "
        f"{len(parts)} part(s), sha {digest[:12]}"
    )
    return entry


def _build_wheel(out: Path, say: Callable[[str], None]) -> Path:
    """`python -m build --wheel` from the checkout, for the server pack."""
    root = repo_root()
    # ASKED BEFORE THE INTERPRETER IS FETCHED WOULD BE BETTER STILL, but this
    # is the first line of the only branch that needs it, and the refusal is
    # what matters: `python -m build` with no `build` installed prints "No
    # module named build", which is a true sentence that names neither the
    # package to install nor the fact that only the SERVER pack needs it.
    probe = subprocess.run(
        [sys.executable, "-c", "import build"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise PackError(
            "pack_build_failed",
            f"{sys.executable} has no `build` module, and the server pack is "
            "the one pack built from a wheel rather than from a recipe: "
            "`pip install build`. (scripts/release.sh refuses for the same "
            "reason and with the same fix.)",
        )
    destination = out / ".wheel"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    say("building the crucible wheel")
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(destination), str(root)],
        "pack_build_failed",
        f"could not build the crucible wheel from {root}",
        say,
        # NOT from the checkout, and this is not fussiness. `python -m` puts the
        # working directory on `sys.path`, and a checkout that has ever run
        # `python -m build` has a `build/` DIRECTORY in it — which shadows the
        # `build` module and fails with "No module named build.__main__", a
        # message that sends its reader to pip. The source directory is an
        # argument, so the build does not need to stand in it. (Hit for real
        # while building the server pack, 2026-09-14.)
        cwd=destination,
    )
    wheels = sorted(destination.glob("crucible-*.whl"))
    if len(wheels) != 1:
        raise PackError(
            "pack_build_failed",
            f"`python -m build` left {len(wheels)} wheels in {destination}; "
            "expected exactly one",
        )
    return wheels[0]


def smoke_test(
    target: PackTarget, archive: Path, *, on_line: Callable[[str], None] | None = None
) -> None:
    """Unpack to a temp dir and prove the pack RUNS THERE. Section 3.2.

    Somewhere else on purpose. A tree that works where it was built proves
    nothing about relocatability, which is the single property a pack has to
    have and the single property a venv does not.
    """
    say = on_line if on_line is not None else (lambda line: None)
    with tempfile.TemporaryDirectory(prefix="crucible-smoke-") as temporary:
        into = Path(temporary) / "pack"
        into.mkdir()
        extract_archive(archive, into, on_line=None)
        python = pack_python(into, target.backend_kind)
        if not python.is_file():
            raise PackError(
                "pack_smoke_failed",
                f"{archive.name} unpacked without a "
                f"{python.relative_to(into).as_posix()}",
            )
        if target.name == SERVER_PACK:
            command = [str(into / "bin" / "crucible"), "--version"]
            what = "bin/crucible --version"
        elif target.name == HOST_PACK:
            # THE WHOLE POINT OF THE SHIM, proved by running it from a tree
            # that is not the one it was written in. The `.exe` pip left in
            # `Scripts\` still names the build tree's interpreter and is dead
            # weight; `crucible.cmd` finds its neighbour through `%~dp0`, and
            # this is the only thing that can tell the two apart.
            #
            # `crucible envpack build host` is refused off win32, so this
            # branch is only ever reached on Windows. No `shell=True`: a
            # batch file handed to `CreateProcess` by its full name is run
            # through cmd.exe for us, measured 2026-09-14 — and `shell=True`
            # would hand the path to a command line that splits on the space
            # in `C:\Users\Owen Morgan\...`.
            command = [str(into / "crucible.cmd"), "--version"]
            what = "crucible.cmd --version"
        else:
            assert target.smoke_import is not None
            command = [str(python), "-c", f"import {target.smoke_import}"]
            what = f"import {target.smoke_import}"
        entry = Path(command[0])
        if entry.parent == into and not entry.is_file():
            # A pack missing the very command this test runs must say THAT,
            # not raise `FileNotFoundError: [WinError 2]` out of subprocess
            # and leave a person reading a traceback about a path (measured
            # 2026-09-15). `write_cmd_shims` now refuses this at build time;
            # this is the second door, for an archive built elsewhere.
            raise PackError(
                "pack_smoke_failed",
                f"{archive.name} unpacked into {into} and there is no "
                f"{entry.name} in it, so the pack carries no command. A pack "
                "that does not pass is not an asset",
            )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            # FROM SOMEWHERE THAT IS NOT THE BUILD TREE, explicitly. A shim
            # that resolved its interpreter relative to the CURRENT directory
            # rather than to its own would pass a test run from `<out>/.build`
            # and fail on the operator's machine; running it with the temp
            # unpack as the working directory is what closes that.
            cwd=temporary,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            raise PackError(
                "pack_smoke_failed",
                f"{archive.name} unpacked into {into} but `{what}` exited "
                f"{completed.returncode}: "
                + (detail[-1] if detail else "no output")
                + ". A pack that does not pass is not an asset",
            )
        say(f"smoke: {what} ok ({completed.stdout.strip() or 'no output'})")
        if target.name in (SERVER_PACK, HOST_PACK):
            # Installers consume these entrypoints. A version-only probe can
            # mistakenly bless an older core missing the lifecycle contract.
            # --help proves parser availability without installing or starting.
            for action in (None, "register", "install-cli", "install-desktop", "shutdown"):
                args = [command[0], "local"] + ([] if action is None else [action]) + ["--help"]
                checked = subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=temporary)
                if checked.returncode != 0:
                    detail = (checked.stderr or checked.stdout).strip()
                    raise PackError("pack_smoke_failed", f"{archive.name} lacks the lifecycle entrypoint "
                                    f"{' '.join(args[1:])}: {detail}")
            say("smoke: local lifecycle entrypoints ok")
            if target.backend_kind in (MLX_DARWIN, LLAMA_WINDOWS):
                # Version/help cannot reveal a missing runtime probe dependency.
                # Exercise the fresh-install path with real host detection, in a
                # new home, without starting a service or downloading any models.
                home = Path(temporary) / "fresh-home"
                environment = os.environ.copy()
                environment["CRUCIBLE_HOME"] = str(home)
                environment.pop("PYTHONPATH", None)
                environment.pop("PYTHONHOME", None)
                args = [command[0], "init", "--backend", target.backend_kind]
                checked = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                         cwd=temporary, env=environment)
                if checked.returncode != 0 or not (home / "config.toml").is_file():
                    detail = (checked.stderr or checked.stdout).strip()
                    raise PackError("pack_smoke_failed", f"{archive.name} failed fresh-home init: {detail}")
                say(f"smoke: fresh-home init with real {target.backend_kind} detection ok")
            else:
                # CUDA builds run on CPU-only hosted runners. Hardware acceptance
                # is separate; claiming to have probed NVIDIA here would be false.
                say("smoke: fresh-home CUDA init requires an NVIDIA acceptance host; not run here")


def write_manifest_entry(out: Path, version: str, entry: PackEntry) -> None:
    """Add or replace this pack's row in `<out>/envpacks.json`.

    Merged rather than overwritten, because a developer building three packs on
    one machine has one manifest and CI has one fragment per job — and both want
    the same file to come out right.
    """
    path = out / MANIFEST_NAME
    rows: list[PackEntry] = []
    if path.is_file():
        existing = parse_manifest(path.read_text(encoding="utf-8"), source=str(path))
        if existing.version != version:
            raise PackError(
                "pack_manifest_unreadable",
                f"{path} is version {existing.version} and this build is "
                f"{version}; move it aside rather than mixing two releases",
            )
        rows = [
            row
            for row in existing.packs
            if (row.name, row.backend) != (entry.name, entry.backend)
        ]
    rows.append(entry)
    manifest = PackManifest(
        version=version,
        packs=tuple(sorted(rows, key=lambda row: (row.backend, row.name))),
    )
    path.write_text(manifest.dumps(), encoding="utf-8")


def check_pack(out: Path, target: PackTarget, version: str) -> PackEntry:
    """`envpack build --check`: the parts on disk ARE what the manifest says.

    Three questions, and the third is the one that matters: are the parts all
    there, do they hash to the declared whole, and was the pack built from the
    recipe this checkout has now (`pack_recipe_drift`).
    """
    path = out / MANIFEST_NAME
    if not path.is_file():
        raise PackError(
            "pack_manifest_unreadable", f"no {path} to check against"
        )
    manifest = parse_manifest(path.read_text(encoding="utf-8"), source=str(path))
    if manifest.version != version:
        raise PackError(
            "pack_manifest_unreadable",
            f"{path} is version {manifest.version} and this build is {version}",
        )
    entry = manifest.require(target.name, target.backend_kind)
    parts = [out / name for name in entry.parts]
    missing = [part.name for part in parts if not part.is_file()]
    if missing:
        raise PackError(
            "pack_not_published",
            f"{path} names {len(entry.parts)} part(s) for "
            f"{entry.name}/{entry.backend} and {missing} are not in {out}",
        )
    size = sum(part.stat().st_size for part in parts)
    if size != entry.bytes:
        raise PackError(
            "pack_sha_mismatch",
            f"the parts of {entry.name}/{entry.backend} are {size} bytes and "
            f"{path} says {entry.bytes}",
        )
    digest = sha256_of_parts(parts)
    if digest != entry.sha256:
        raise PackError(
            "pack_sha_mismatch",
            f"the parts of {entry.name}/{entry.backend} hash to {digest} and "
            f"{path} pins {entry.sha256}",
        )
    check_recipe(entry, target)
    return entry
