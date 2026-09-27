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

ENGINE_KIND = "engine"
LLAMA_CPP_ID = "llama-cpp"

LLAMA_CPP_RELEASE = "b10970"

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"

LLAMA_SERVER_EXE = "llama-server.exe"

STAMP_NAME = ".crucible-engine.json"

CHUNK_BYTES = 1 << 20

CONNECT_TIMEOUT_SECONDS = 60.0


class EngineSubjectError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Asset:
    name: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{RELEASE_URL}/{LLAMA_CPP_RELEASE}/{self.name}"


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

CPU_ASSETS: tuple[Asset, ...] = (
    Asset(
        name=f"llama-{LLAMA_CPP_RELEASE}-bin-win-cpu-x64.zip",
        bytes=18_428_751,
        sha256="2c6d6516c04e95caa080d8eb917743e71858c73985acbb6739ad61b14e68b298",
    ),
)

CUDA_BUILD = "cuda-12.4"
CPU_BUILD = "cpu"


def build_for(gpu_vendor: str) -> str:
    return CUDA_BUILD if gpu_vendor == "nvidia" else CPU_BUILD


def assets_for(build: str) -> tuple[Asset, ...]:
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
    return sum(asset.bytes for asset in assets_for(build))


def engine_dir(config: Config) -> Path:
    return config.home / "engines" / LLAMA_CPP_ID


def stamp_path(config: Config) -> Path:
    return engine_dir(config) / STAMP_NAME


def server_path(config: Config) -> Path:
    return engine_dir(config) / LLAMA_SERVER_EXE


def installed(config: Config, build: str) -> InstalledWeights | None:
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


Fetch = Callable[[str, Path, "ProgressHook | None"], None]


def download(url: str, destination: Path, on_progress: "ProgressHook | None") -> None:
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
    if result is None:
        raise EngineSubjectError(
            "engine_download_failed",
            f"wrote {stamp_path(config)} but it does not read back as installed",
        )
    return result


def remove(config: Config) -> Path:
    target = engine_dir(config)
    shutil.rmtree(target)
    return target


def doctor_line(config: Config, gpu_vendor: str) -> str:
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
