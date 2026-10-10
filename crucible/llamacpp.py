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

from . import envpatches, hosttools, jobenv
from .backend import CUDA_LINUX
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

INSTALL_COMMAND = "crucible install llm"

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
        f"backend — run `{INSTALL_COMMAND}` (or `install pages`), which "
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


def _stage_release(
    build: str,
    staging: Path,
    fetcher: Fetch,
    on_line: Callable[[str], None] | None,
    on_progress: "ProgressHook | None",
) -> Path:
    downloads = staging / "downloads"
    unpacked = staging / "unpacked"
    downloads.mkdir()
    unpacked.mkdir()
    archives: list[tuple[Asset, Path]] = []
    for asset in assets_for(build):
        if on_line is not None:
            on_line(f"fetching {asset.name} ({asset.bytes / 1e6:.0f} MB)")
        path = downloads / asset.name
        fetcher(asset.url, path, on_progress)
        measured = sha256_of(path)
        if measured != asset.sha256:
            raise EngineSubjectError(
                "engine_sha_mismatch",
                f"{asset.name} hashed {measured}, and the release published "
                f"{asset.sha256}. Nothing is unpacked: these are not the "
                "bytes ggml-org released, and a llama.cpp that is not the "
                "pinned one is a server nobody can reason about. Run the pull "
                "again; a second mismatch means the download is being altered "
                "on its way here",
            )
        archives.append((asset, path))
    for asset, path in archives:
        if on_line is not None:
            on_line(f"unpacking {asset.name}")
        _unzip(path, unpacked)
    server = _locate_server(unpacked)
    if server != unpacked / LLAMA_SERVER_EXE:
        shutil.move(str(server), str(unpacked / LLAMA_SERVER_EXE))
    return unpacked


def _swap_in(unpacked: Path, target: Path, staging: Path) -> None:
    if not target.exists():
        unpacked.rename(target)
        return
    retired = staging / "retired"
    try:
        target.rename(retired)
    except OSError as exc:
        raise EngineSubjectError(
            "engine_replace_failed",
            f"the new llama.cpp {LLAMA_CPP_RELEASE} is downloaded and verified, "
            f"but {target} could not be moved aside ({type(exc).__name__}: {exc}). "
            "A llama-server started from it is most likely still running. The "
            "engine already there is untouched and still serves. Unload the "
            "resident model (the operator page's Unload, or POST /v1/jobs "
            '{"type": "unload-model"}), then pull the engine again',
        ) from None
    try:
        unpacked.rename(target)
    except OSError as exc:
        retired.rename(target)
        raise EngineSubjectError(
            "engine_replace_failed",
            f"the new llama.cpp could not be moved into {target} "
            f"({type(exc).__name__}: {exc}). The engine that was there is back "
            "in place and still serves; pull the engine again",
        ) from None


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
    target.parent.mkdir(parents=True, exist_ok=True)
    fetcher = download if fetch is None else fetch
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix="crucible-llamacpp-", dir=str(target.parent)))
    try:
        unpacked = _stage_release(build, staging, fetcher, on_line, on_progress)
        size = directory_bytes(unpacked)
        record = {
            "subject": f"{ENGINE_KIND}/{LLAMA_CPP_ID}",
            "tag": LLAMA_CPP_RELEASE,
            "build": build,
            "assets": [asset.name for asset in assets_for(build)],
            "bytes": size,
            "seconds": round(time.monotonic() - started, 1),
            "pulled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (unpacked / STAMP_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _swap_in(unpacked, target, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    elapsed = time.monotonic() - started
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
        return f"{where}: NOT INSTALLED — run `{INSTALL_COMMAND}`"
    return f"{where}: {found.path} ({found.bytes / 1e9:.2f} GB, pulled {found.pulled})"


# cuda-linux. ggml-org publishes no Linux CUDA build, so the binary is Crucible's own, pinned
# on its tools release and placed by `crucible install llm` (hosttools.LLAMA_SERVER_BUILDS).
# It is built against CUDA 13.0 and links cudart and cuBLAS dynamically: those come from
# the llm env, whose recipe pins PyPI's nvidia-cuda-runtime and nvidia-cublas for vLLM's own
# torch (crucible/envs/llm/cuda-linux.txt), so the heavy libraries reach a server from
# NVIDIA's mirror and only the binary from ours. The driver (libcuda.so.1) is the host's.
CUDA_LINUX_LIBRARY_REL = "nvidia/cu13/lib"

CUDA_LINUX_LIBRARIES: tuple[str, ...] = (
    "libcudart.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
)


@dataclass(frozen=True)
class LinuxEngine:
    installed: bool
    detail: str
    executable: Path | None
    library_dirs: tuple[Path, ...]


def cuda_linux_library_dir(home: Path) -> Path | None:
    packages = envpatches.site_packages(jobenv.env_dir(home, jobenv.llm_env(CUDA_LINUX)))
    return None if packages is None else packages / CUDA_LINUX_LIBRARY_REL


def cuda_linux_engine(home: Path) -> LinuxEngine:
    """llama-server on cuda-linux: the pinned binary and the llm env's CUDA libraries.
    Not installed, with the sentence that says which half is missing, until both are."""
    build = hosttools.llama_server_build()
    where = (
        "llama.cpp (no pinned build for this platform)"
        if build is None
        else f"llama.cpp {build.version}"
    )
    env = jobenv.env_status(home, jobenv.llm_env(CUDA_LINUX), CUDA_LINUX)
    if not env.installed:
        return LinuxEngine(
            installed=False,
            detail=(
                f"{where} loads its CUDA libraries from the llm env, which is not "
                f"ready: {env.detail}"
            ),
            executable=None,
            library_dirs=(),
        )
    libraries = cuda_linux_library_dir(home)
    if libraries is None:
        absent = list(CUDA_LINUX_LIBRARIES)
    else:
        absent = [name for name in CUDA_LINUX_LIBRARIES if not (libraries / name).is_file()]
    if libraries is None or absent:
        return LinuxEngine(
            installed=False,
            detail=(
                f"{where} links {list(CUDA_LINUX_LIBRARIES)}, and the llm env at "
                f"{env.path} has no {absent} under {CUDA_LINUX_LIBRARY_REL}. The "
                "env's recipe pins the nvidia wheels that carry them, so this env "
                f"was not built from it — run `{INSTALL_COMMAND} --force`"
            ),
            executable=None,
            library_dirs=(),
        )
    server = hosttools.llama_server_path(home)
    if build is None or not hosttools.llama_server_placed(home, build):
        return LinuxEngine(
            installed=False,
            detail=(
                f"{where} is not placed at {server} — run `{INSTALL_COMMAND}`, "
                "which keeps the env it has and fetches the engine"
            ),
            executable=None,
            library_dirs=(),
        )
    return LinuxEngine(
        installed=True,
        detail=f"{where} at {server}, CUDA libraries from {libraries}",
        executable=server,
        library_dirs=(libraries,),
    )


__all__ = [
    "CPU_ASSETS",
    "CPU_BUILD",
    "CUDA_LINUX_LIBRARIES",
    "CUDA_LINUX_LIBRARY_REL",
    "CUDA_ASSETS",
    "CUDA_BUILD",
    "ENGINE_KIND",
    "LLAMA_CPP_ID",
    "LLAMA_CPP_RELEASE",
    "LLAMA_SERVER_EXE",
    "Asset",
    "LinuxEngine",
    "EngineSubjectError",
    "assets_for",
    "build_for",
    "cuda_linux_engine",
    "cuda_linux_library_dir",
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
