"""The `engine` subject: llama.cpp's own release, pulled like weights.

PHASE15-HOST.md section 3.10, fact 1. `llama-windows` serves its models with
`llama-server.exe`, and that binary is not a Python package, not a pack and
not a model — it is a zip on a GitHub release. So it is a SUBJECT:
`{kind: "engine", id: "llama-cpp"}`, which means the page's Tasks panel draws
its download with bytes exactly as it draws a weights pull, `GET /v1/catalog`
says whether it is installed, and `DELETE /v1/catalog/engine/llama-cpp`
(3.5a) removes it. No new vocabulary anywhere; one more row.

THE BUILD IS PINNED, NEVER LISTED AT RUNTIME
---------------------------------------------
Foundry read the release listing and fell back to a pinned tag when the read
failed, which is two sources of truth for "which llama.cpp is this" and a
machine that quietly runs a different one on a bad network day. Here there is
ONE constant (`LLAMA_CPP_RELEASE`) and no listing call at all. Upgrading the
engine is an edit to this file with new digests beside it, which is what
pinning is for.

WHICH ASSETS, AND WHY THE CUDA ROW IS TWO OF THEM
---------------------------------------------------
Windows + NVIDIA takes `llama-<tag>-bin-win-cuda-12.4-x64.zip` **and**
`cudart-llama-bin-win-cuda-12.4-x64.zip`, unpacked into ONE directory: the
CUDA build links against the CUDA runtime DLLs and does not start without
them, and llama.cpp ships them as a separate asset rather than in the build.
A machine with no NVIDIA card takes the CPU build, one zip, and runs slowly —
which is allowed, and which the capability row says in words (Owen: *"a
crucible server will run on absolutely anything"*).

EVERY DIGEST IS CHECKED BEFORE ANYTHING IS PLACED, which is `pull_archive`'s
rule and `pull_files`' rule, for their reason: a half-placed engine directory
is one `llama-server` will start from and fail inside, and the failure then
arrives in the middle of somebody's book instead of here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import Config
from .errors import CrucibleError
from .weights import (
    PINNED,
    InstalledWeights,
    ProgressHook,
    PullCancelled,
    directory_bytes,
    sha256_of,
)

#: The subject's kind and id, the two words every door spells.
ENGINE_KIND = "engine"
LLAMA_CPP_ID = "llama-cpp"

#: **THE PIN.** `ggml-org/llama.cpp`, published 2026-09-14T20:53:17Z. One
#: constant, edited deliberately, never read from a listing at run time.
LLAMA_CPP_RELEASE = "b10970"

#: Where a release's assets are. The tag is interpolated; nothing else is.
RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"

#: The binary the residency spawns, inside the unpacked directory.
LLAMA_SERVER_EXE = "llama-server.exe"

#: The stamp, beside the unpacked files. Named for the subject, like every
#: other stamp in `crucible/weights.py`.
STAMP_NAME = ".crucible-engine.json"

#: How much of a download is read at a time. One megabyte, like `sha256_of`.
CHUNK_BYTES = 1 << 20

#: Seconds before a stalled connection is given up on. A 254 MB asset on a
#: slow line is minutes of transfer but never minutes between two packets.
CONNECT_TIMEOUT_SECONDS = 60.0


class EngineSubjectError(CrucibleError):
    """The engine could not be fetched, verified or unpacked."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Asset:
    """One zip on the release, with the size and digest the API published.

    The digests are the RELEASE'S OWN: the GitHub releases API publishes a
    `digest` per asset, read on 2026-09-14, so these are not a local
    measurement of a local download and a mismatch means the bytes are not
    the bytes ggml-org published.
    """

    name: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{RELEASE_URL}/{LLAMA_CPP_RELEASE}/{self.name}"


#: Windows + NVIDIA. TWO assets into one directory — the server does not start
#: without the cudart DLLs.
CUDA_ASSETS: tuple[Asset, ...] = (
    Asset(
        name=f"llama-{LLAMA_CPP_RELEASE}-bin-win-cuda-12.4-x64.zip",
        bytes=254_074_942,
        sha256="78c878ae30622a9e4be09e3831066454668ca70398114f23bf74ac814e52dad8",
    ),
    Asset(
        name="cudart-llama-bin-win-cuda-12.4-x64.zip",
        bytes=391_443_627,
        sha256="8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    ),
)

#: Windows without an NVIDIA card. Allowed, slow, and the capability row says
#: so rather than turning the class off.
CPU_ASSETS: tuple[Asset, ...] = (
    Asset(
        name=f"llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip",
        bytes=18_428_751,
        sha256="2c6d6516c04e95caa080d8eb917743e71858c73985acbb6739ad61b14e68b298",
    ),
)

#: The two words `doctor` and `/v1/info` print for a build.
CUDA_BUILD = "cuda-12.4"
CPU_BUILD = "cpu"


def build_for(gpu_vendor: str) -> str:
    """Which build this machine takes: `cuda-12.4` with an NVIDIA card, `cpu`.

    Takes the VENDOR rather than reading it, so the answer is testable off a
    Windows box — `crucible/backend.py` is the one thing that asks nvidia-smi.
    """
    return CUDA_BUILD if gpu_vendor == "nvidia" else CPU_BUILD


def assets_for(build: str) -> tuple[Asset, ...]:
    """The assets that build is made of. Refuses a build name it does not have."""
    if build == CUDA_BUILD:
        return CUDA_ASSETS
    if build == CPU_BUILD:
        return CPU_ASSETS
    raise EngineSubjectError(
        "engine_download_failed",
        f"{build!r} is not a llama.cpp build this server knows; they are "
        f"{CUDA_BUILD!r} and {CPU_BUILD!r}",
    )


def expected_bytes(build: str) -> int:
    """What the whole download weighs, for the catalog row and the page."""
    return sum(asset.bytes for asset in assets_for(build))


def engine_dir(config: Config) -> Path:
    """`<CRUCIBLE_HOME>/engines/llama-cpp/` — one directory, both zips in it.

    Under the home like every other subject, so 3.5's *"on Windows the
    server's home is `%LOCALAPPDATA%\\Crucible\\` and every subject lives
    under it"* is true of the engine too, and the host's migration deletes it
    through the same door.
    """
    return config.home / "engines" / LLAMA_CPP_ID


def stamp_path(config: Config) -> Path:
    return engine_dir(config) / STAMP_NAME


def server_path(config: Config) -> Path:
    """Where `llama-server.exe` is once the zips are unpacked.

    llama.cpp's Windows zips put every binary at the ROOT of the archive, so
    this is the directory itself. Found rather than assumed at install time
    (`_locate_server`), and recorded in the stamp — a release that changes its
    layout is then a refusal naming the directory, not a spawn of a path that
    is not there.
    """
    return engine_dir(config) / LLAMA_SERVER_EXE


def installed(config: Config, build: str) -> InstalledWeights | None:
    """The installed engine for THIS build, or None.

    A stamp naming a different tag or a different build is not installed, for
    `weights.installed`'s reason: the pin moved, or the machine did (a card
    was added), and running the old binaries under the new statement would be
    a silent substitution. `llama-server.exe` must be there too — a stamp
    beside a directory somebody emptied is a stamp that lies.
    """
    stamp = stamp_path(config)
    if not stamp.is_file():
        return None
    try:
        record = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("tag") != LLAMA_CPP_RELEASE or record.get("build") != build:
        return None
    if not server_path(config).is_file():
        return None
    return InstalledWeights(
        path=engine_dir(config),
        hf_repo=f"ggml-org/llama.cpp@{record['tag']}",
        revision=record["tag"],
        bytes=int(record["bytes"]),
        pulled=str(record["pulled"]),
        source=PINNED,
    )


def require_installed(config: Config, build: str) -> InstalledWeights:
    """The engine, or the refusal that names what to run."""
    found = installed(config, build)
    if found is not None:
        return found
    raise EngineSubjectError(
        "engine_not_installed",
        f"there is no llama.cpp {LLAMA_CPP_RELEASE} ({build}) at "
        f"{engine_dir(config)}. It is what serves every model on this "
        "backend — run `crucible install llm` (or `install pages`), which "
        "fetches the engine and nothing else",
    )


#: What `pull` uses to fetch one URL. Injected so a test drives a fake release
#: without a network, exactly as `crucible/weights.py` takes the hub's
#: `snapshot_download` from `huggingface_hub`.
Fetch = Callable[[str, Path, "ProgressHook | None"], None]


def download(url: str, destination: Path, on_progress: "ProgressHook | None") -> None:
    """One asset, streamed to disk, reporting bytes and honouring a cancel.

    The hook is the cancel point, as it is for a hub pull: it is called per
    chunk, and a `PullCancelled` out of it travels through here untouched
    (`crucible/weights.py`'s `PullCancelled` says why that matters).
    """
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=CONNECT_TIMEOUT_SECONDS) as response:
            declared = response.headers.get("Content-Length")
            total = int(declared) if declared is not None else None
            done = 0
            with destination.open("wb") as handle:
                while True:
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if on_progress is not None:
                        on_progress(done, total, destination.name)
    except PullCancelled:
        destination.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, OSError) as exc:
        raise EngineSubjectError(
            "engine_download_failed",
            f"{url} could not be fetched: {type(exc).__name__}: {exc}",
        ) from None


def _unzip(archive: Path, target: Path) -> None:
    """Extract a zip into `target`, refusing any member that escapes it.

    `weights._unpack`'s rule for the other archive format and for its reason:
    a member whose resolved destination is not under `target` is a refusal
    naming it, never a skip. Python's `ZipFile.extractall` sanitises absolute
    paths but not every `..`, so this is written out.
    """
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            destination = (root / member).resolve()
            if destination != root and root not in destination.parents:
                raise EngineSubjectError(
                    "engine_download_failed",
                    f"{archive.name} holds {member!r}, which would be written "
                    f"outside {target}. Nothing is unpacked from an archive "
                    "that reaches out of its own directory",
                )
        bundle.extractall(root)


def _locate_server(target: Path) -> Path:
    """`llama-server.exe` inside the unpacked tree, or a refusal.

    Searched rather than assumed. llama.cpp's Windows zips have put the
    binaries at the archive root for every release this pin has seen, but a
    layout is the release's to change and a spawn of a path that is not there
    is a worse sentence than this one.
    """
    direct = target / LLAMA_SERVER_EXE
    if direct.is_file():
        return direct
    found = sorted(target.rglob(LLAMA_SERVER_EXE))
    if not found:
        raise EngineSubjectError(
            "engine_download_failed",
            f"llama.cpp {LLAMA_CPP_RELEASE} unpacked into {target} and there is "
            f"no {LLAMA_SERVER_EXE} anywhere in it. The pinned assets are the "
            "Windows server builds; if this release moved its layout, the pin "
            "in crucible/llamacpp.py is what changes",
        )
    return found[0]


def pull(
    config: Config,
    build: str,
    *,
    force: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_progress: "ProgressHook | None" = None,
    fetch: Fetch | None = None,
) -> InstalledWeights:
    """Fetch, verify and unpack llama.cpp's pinned release for this build.

    EVERY DIGEST IS CHECKED BEFORE ANYTHING IS UNPACKED. Both CUDA assets are
    downloaded to a staging directory and hashed there; only when both pass
    does anything land in the engine directory. Otherwise a machine could end
    up with the server binaries and no cudart, which is a directory that looks
    installed and starts nothing.
    """
    existing = installed(config, build)
    if existing is not None and not force:
        return existing

    target = engine_dir(config)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    fetcher = download if fetch is None else fetch
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix="crucible-llamacpp-", dir=str(target.parent)))
    try:
        archives: list[tuple[Asset, Path]] = []
        for asset in assets_for(build):
            if on_line is not None:
                on_line(f"fetching {asset.name} ({asset.bytes / 1e6:.0f} MB)")
            path = staging / asset.name
            fetcher(asset.url, path, on_progress)
            measured = sha256_of(path)
            if measured != asset.sha256:
                raise EngineSubjectError(
                    "engine_sha_mismatch",
                    f"{asset.name} hashed {measured}, and the release published "
                    f"{asset.sha256}. Nothing is unpacked: these are not the "
                    "bytes ggml-org released, and a llama.cpp that is not the "
                    "pinned one is a server nobody can reason about",
                )
            archives.append((asset, path))
        for asset, path in archives:
            if on_line is not None:
                on_line(f"unpacking {asset.name} into {target}")
            _unzip(path, target)
    except (PullCancelled, EngineSubjectError):
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    server = _locate_server(target)
    if server != server_path(config):
        # Found somewhere else in the tree. Moved to the one place every other
        # door looks, rather than recorded as a second path a second reader
        # would have to know about.
        shutil.move(str(server), str(server_path(config)))

    elapsed = time.monotonic() - started
    size = directory_bytes(target)
    record = {
        "subject": f"{ENGINE_KIND}/{LLAMA_CPP_ID}",
        "tag": LLAMA_CPP_RELEASE,
        "build": build,
        "assets": [asset.name for asset in assets_for(build)],
        "bytes": size,
        "seconds": round(elapsed, 1),
        "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    stamp_path(config).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if on_line is not None:
        on_line(
            f"llama.cpp {LLAMA_CPP_RELEASE} ({build}) is at {target} "
            f"({size / 1e9:.2f} GB in {elapsed:.0f}s)"
        )
    result = installed(config, build)
    if result is None:  # pragma: no cover - the stamp was just written
        raise EngineSubjectError(
            "engine_download_failed",
            f"wrote {stamp_path(config)} but it does not read back as installed",
        )
    return result


def remove(config: Config) -> Path:
    """Delete the engine directory. `DELETE /v1/catalog/engine/llama-cpp`.

    Returns the directory that went, so the activity record and the CLI say
    where. Refuses nothing itself: whether the subject is installed and
    whether anything is using it are the catalog door's questions (3.5a), and
    asking them twice would be two answers to one.
    """
    target = engine_dir(config)
    shutil.rmtree(target)
    return target


def doctor_line(config: Config, gpu_vendor: str) -> str:
    """`crucible doctor`'s engine line on this backend (3.5).

    Reads `backend: llama-windows on windows/x86_64 — llama.cpp <tag>
    (cuda-12.4 | cpu)` when it is there, and says what to run when it is not.
    """
    build = build_for(gpu_vendor)
    found = installed(config, build)
    where = f"llama.cpp {LLAMA_CPP_RELEASE} ({build})"
    if found is None:
        return f"{where}: NOT INSTALLED — run `crucible install llm`"
    return f"{where}: {found.path} ({found.bytes / 1e9:.2f} GB, pulled {found.pulled})"


__all__ = [
    "CPU_ASSETS",
    "CPU_BUILD",
    "CUDA_ASSETS",
    "CUDA_BUILD",
    "ENGINE_KIND",
    "LLAMA_CPP_ID",
    "LLAMA_CPP_RELEASE",
    "LLAMA_SERVER_EXE",
    "Asset",
    "EngineSubjectError",
    "assets_for",
    "build_for",
    "doctor_line",
    "download",
    "engine_dir",
    "expected_bytes",
    "installed",
    "pull",
    "remove",
    "require_installed",
    "server_path",
    "stamp_path",
]
