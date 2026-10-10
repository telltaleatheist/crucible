from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ApiError, CrucibleError

PATH_ENV = "PATH"

TOOLS_DIR_NAME = "tools"

FFMPEG_STAMP_NAME = "ffmpeg.json"

FFMPEG_PROGRAMS: tuple[str, ...] = ("ffmpeg", "ffprobe")

ZIG_STAMP_NAME = "zig.json"

ZIG_DIR_NAME = "zig"

C_COMPILER_NAME = "cc"

C_COMPILER_ENV = "CC"


class HostToolError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolBuild:
    version: str
    url: str
    sha256: str
    bytes: int
    root: str
    provenance: str


FFMPEG_BUILDS: dict[str, ToolBuild] = {
    "linux-x86_64": ToolBuild(
        version="n8.1.3",
        url=(
            "https://github.com/telltaleatheist/crucible/releases/download/"
            "tools/ffmpeg-n8.1.3-linux64-lgpl-8.1.tar.xz"
        ),
        sha256="f179c8a7ea16ad8b79c188ff7d0d0ffd3a6e253614e54fcc4981e69442ca8d10",
        bytes=136_948_704,
        root="ffmpeg-n8.1.3-linux64-lgpl-8.1",
        provenance=(
            "BtbN/FFmpeg-Builds autobuild-2026-09-26-13-03, "
            "ffmpeg-n8.1.3-linux64-lgpl-8.1.tar.xz, re-hosted unchanged"
        ),
    ),
    "darwin-arm64": ToolBuild(
        version="n8.1.3",
        url=(
            "https://github.com/telltaleatheist/crucible/releases/download/"
            "tools/ffmpeg-n8.1.3-darwin-arm64-lgpl.tar.xz"
        ),
        sha256="f4216e6ff3c2db7de78ad9cb154a66aa2a98f5b4a3b060cfbc5af5a71bc0d7b4",
        bytes=8_090_236,
        root="ffmpeg-n8.1.3-darwin-arm64-lgpl",
        provenance=(
            "built on owens-mac-studio 2026-09-28 from ffmpeg.org ffmpeg-8.1.3.tar.xz "
            "(sha256 7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3, "
            "signature checked against key D67658D8); LGPL 2.1+, static, Apple clang, "
            "macOS 13+, system frameworks only, ad-hoc signed"
        ),
    ),
}


@dataclass(frozen=True)
class CompilerBuild(ToolBuild):
    """A pinned Zig whose ``zig cc`` is this host's C compiler.

    ``target`` is passed on every compile. Naming the glibc version makes zig link
    against its own glibc stubs and crt files, so the host needs no libc6-dev and no
    gcc; without it zig picks up whatever crt files the system has, and a fresh
    Crucible distro has none.
    """

    target: str


ZIG_BUILDS: dict[str, CompilerBuild] = {
    "linux-x86_64": CompilerBuild(
        version="0.17.0",
        url=(
            "https://github.com/telltaleatheist/crucible/releases/download/"
            "tools/zig-x86_64-linux-0.17.0.tar.xz"
        ),
        sha256="1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026",
        bytes=57_332_648,
        root="zig-x86_64-linux-0.17.0",
        provenance=(
            "ziglang.org/download/0.17.0 zig-x86_64-linux-0.17.0.tar.xz, the sha256 "
            "ziglang.org/download/index.json lists, re-hosted unchanged; MIT"
        ),
        target="x86_64-linux-gnu.2.28",
    ),
}


@dataclass(frozen=True)
class ToolFile:
    version: str
    url: str
    sha256: str
    bytes: int
    directory: str
    provenance: str
    licence: str


SILERO_VAD = ToolFile(
    version="6.2.1",
    url=(
        "https://github.com/telltaleatheist/crucible/releases/download/"
        "tools/silero_vad_op18_ifless-6.2.1.onnx"
    ),
    sha256="7671cd04b004e9076da0d4a7b1a5aec36adf161c39230c1cb94a4fd5db6bbd28",
    bytes=2_845_718,
    directory="silero-vad",
    provenance=(
        "silero-vad 6.2.1 from PyPI (silero_vad-6.2.1-py3-none-any.whl), "
        "silero_vad/data/silero_vad_op18_ifless.onnx, re-hosted unchanged"
    ),
    licence="MIT (Copyright (c) 2020-present Silero Team)",
)

_CHUNK_BYTES = 1 << 20

_WINDOWS_DRIVE = re.compile(r"^/mnt/[A-Za-z](/|$)")


def search_path() -> str:
    return os.environ.get(PATH_ENV, "")


def _home() -> Path | None:
    from .config import crucible_home
    from .errors import ConfigError

    try:
        return crucible_home()
    except ConfigError:
        return None


def tools_bin(home: Path | None = None) -> Path | None:
    root = _home() if home is None else home
    return None if root is None else root / TOOLS_DIR_NAME / "bin"


def _in_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.release().lower()


def _searched_entries(value: str) -> list[str]:
    entries = [entry for entry in value.split(os.pathsep) if entry]
    if _in_wsl():
        entries = [entry for entry in entries if not _WINDOWS_DRIVE.match(entry)]
    return entries


def which(tool: str) -> str | None:
    own = tools_bin()
    if own is not None:
        candidate = own / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(tool, path=os.pathsep.join(_searched_entries(search_path())))


def ffmpeg_path() -> str | None:
    return which("ffmpeg")


def ffprobe_path() -> str | None:
    return which("ffprobe")


def require_ffmpeg(job_type: str, why: str) -> str:
    found = ffmpeg_path()
    if found is None:
        raise ApiError(
            409,
            "ffmpeg_missing",
            f"there is no ffmpeg on this server's PATH, and {job_type} {why} "
            + searched_note(),
            {"path": search_path()},
        )
    return found


def worker_path(inherited: str) -> str:
    own = tools_bin()
    if own is None or not own.is_dir():
        return inherited
    return os.pathsep.join(entry for entry in (str(own), inherited) if entry)


def searched_note() -> str:
    own = tools_bin()
    first = (
        f"looked in {own}, which installing any job type fills, then "
        if own is not None
        else ""
    )
    value = search_path()
    if value == "":
        return f"({first}PATH searched: it is empty)"
    return f"({first}PATH searched: {value})"


def host_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def ffmpeg_build(platform_key: str | None = None) -> ToolBuild | None:
    return FFMPEG_BUILDS.get(host_platform() if platform_key is None else platform_key)


def ffmpeg_stamp(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / FFMPEG_STAMP_NAME


def ffmpeg_placed(home: Path, build: ToolBuild) -> bool:
    try:
        record = json.loads(ffmpeg_stamp(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("sha256") != build.sha256:
        return False
    bin_dir = home / TOOLS_DIR_NAME / "bin"
    return all((bin_dir / name).is_file() for name in FFMPEG_PROGRAMS)


def ffmpeg_report(home: Path) -> dict[str, object]:
    build = ffmpeg_build()
    placed = build is not None and ffmpeg_placed(home, build)
    found = which("ffmpeg")
    if placed:
        source = "crucible"
    elif found is None:
        source = "missing"
    else:
        source = "path"
    own = tools_bin(home)
    return {
        "path": found,
        "source": source,
        "platform": host_platform(),
        "pinned_version": None if build is None else build.version,
        "tools_bin": None if own is None else str(own),
    }


def _download(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with destination.open("wb") as handle:
                while True:
                    block = response.read(_CHUNK_BYTES)
                    if not block:
                        break
                    handle.write(block)
                    digest.update(block)
    except (urllib.error.URLError, OSError) as exc:
        raise HostToolError(
            "tool_download_failed",
            f"{url} could not be fetched: {type(exc).__name__}: {exc}",
        ) from None
    return digest.hexdigest()


def ensure_ffmpeg(
    home: Path,
    *,
    on_line: Callable[[str], None] | None = None,
    fetch: Callable[[str, Path], str] | None = None,
) -> str:
    build = ffmpeg_build()
    if build is None:
        found = which("ffmpeg")
        return (
            f"ffmpeg: no pinned build for {host_platform()}; "
            + (f"using this host's own ({found})" if found else "and there is none on PATH")
        )
    tools = home / TOOLS_DIR_NAME
    bin_dir = tools / "bin"
    if ffmpeg_placed(home, build):
        return f"ffmpeg: {build.version} already in {bin_dir}"

    bin_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ffmpeg-", dir=str(tools)))
    started = time.monotonic()
    try:
        archive = staging / build.url.rsplit("/", 1)[-1]
        if on_line is not None:
            on_line(f"fetching {archive.name} ({build.bytes / 1e6:.0f} MB)")
        measured = (_download if fetch is None else fetch)(build.url, archive)
        if measured != build.sha256:
            raise HostToolError(
                "tool_sha_mismatch",
                f"{archive.name} hashed {measured}, and Crucible pins "
                f"{build.sha256}. Nothing is placed: these are not the bytes on "
                "Crucible's tools release",
            )
        name = FFMPEG_PROGRAMS[0]
        try:
            with tarfile.open(archive, "r:xz") as bundle:
                for name in FFMPEG_PROGRAMS:
                    source = bundle.extractfile(f"{build.root}/bin/{name}")
                    if source is None:
                        raise KeyError(f"{build.root}/bin/{name} is not a file")
                    partial = bin_dir / f".{name}.partial"
                    with source, partial.open("wb") as handle:
                        shutil.copyfileobj(source, handle, _CHUNK_BYTES)
                    partial.chmod(0o755)
                    os.replace(partial, bin_dir / name)
        except (tarfile.TarError, KeyError, OSError) as exc:
            raise HostToolError(
                "tool_unpack_failed",
                f"{archive.name} matched its sha256, but {name} could not be "
                f"taken out of it: {type(exc).__name__}: {exc}",
            ) from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    ffmpeg_stamp(home).write_text(
        json.dumps(
            {
                "tool": "ffmpeg",
                "version": build.version,
                "platform": host_platform(),
                "url": build.url,
                "sha256": build.sha256,
                "provenance": build.provenance,
                "programs": list(FFMPEG_PROGRAMS),
                "seconds": round(time.monotonic() - started, 1),
                "placed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return f"ffmpeg: {build.version} placed in {bin_dir} (sha256 {build.sha256[:12]}...)"


def zig_build(platform_key: str | None = None) -> CompilerBuild | None:
    return ZIG_BUILDS.get(host_platform() if platform_key is None else platform_key)


def zig_stamp(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / ZIG_STAMP_NAME


def zig_dir(home: Path, build: CompilerBuild) -> Path:
    return home / TOOLS_DIR_NAME / ZIG_DIR_NAME / build.version


def zig_cache_dir(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / ZIG_DIR_NAME / "cache"


QUIETED_WARNING = "-Wno-macro-redefined"


def c_compiler_path(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / "bin" / C_COMPILER_NAME


def c_compiler_wrapper(home: Path, build: CompilerBuild) -> str:
    """The ``cc`` Crucible hands Triton: zig's clang, aimed at zig's own glibc.

    Zig's caches are kept beside it under the Crucible home, so a service whose
    $HOME is unusable still compiles, and nothing is written outside the home.

    ``-Wno-macro-redefined`` is the one warning it turns off. Zig's glibc headers
    define ``_POSIX_C_SOURCE`` as POSIX.1-2024 (``202405L``) and Python's pyconfig.h
    defines it again as ``200809L``, so every launcher Triton compiles (driver.c
    includes <dlfcn.h> before <Python.h>) printed that warning into the engine log.
    Clang cannot name one macro, so the class is turned off; every other warning
    still prints.
    """
    zig = shlex.quote(str(zig_dir(home, build) / "zig"))
    cache = shlex.quote(str(zig_cache_dir(home)))
    return (
        "#!/bin/sh\n"
        f"# Crucible's C compiler: zig {build.version} cc, placed by `crucible install`.\n"
        "# Triton JIT-compiles a small C launcher the first time a kernel runs.\n"
        f"ZIG_GLOBAL_CACHE_DIR={cache}\n"
        f"ZIG_LOCAL_CACHE_DIR={cache}\n"
        "export ZIG_GLOBAL_CACHE_DIR ZIG_LOCAL_CACHE_DIR\n"
        f'exec {zig} cc -target {build.target} {QUIETED_WARNING} "$@"\n'
    )


def _zig_unpacked(home: Path, build: CompilerBuild) -> bool:
    try:
        record = json.loads(zig_stamp(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("sha256") != build.sha256:
        return False
    return (zig_dir(home, build) / "zig").is_file()


def zig_placed(home: Path, build: CompilerBuild) -> bool:
    if not _zig_unpacked(home, build):
        return False
    try:
        written = c_compiler_path(home).read_text(encoding="utf-8")
    except OSError:
        return False
    return written == c_compiler_wrapper(home, build)


def _write_wrapper(home: Path, build: CompilerBuild) -> None:
    target = c_compiler_path(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    partial.write_text(c_compiler_wrapper(home, build), encoding="utf-8", newline="\n")
    partial.chmod(0o755)
    os.replace(partial, target)


def _unpack_zig(
    home: Path,
    build: CompilerBuild,
    on_line: Callable[[str], None] | None,
    fetch: Callable[[str, Path], str] | None,
) -> None:
    final = zig_dir(home, build)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".zig-", dir=str(home / TOOLS_DIR_NAME)))
    started = time.monotonic()
    try:
        archive = staging / build.url.rsplit("/", 1)[-1]
        if on_line is not None:
            on_line(f"fetching {archive.name} ({build.bytes / 1e6:.0f} MB)")
        measured = (_download if fetch is None else fetch)(build.url, archive)
        if measured != build.sha256:
            raise HostToolError(
                "tool_sha_mismatch",
                f"{archive.name} hashed {measured}, and Crucible pins "
                f"{build.sha256}. Nothing is placed: these are not the bytes on "
                "Crucible's tools release",
            )
        unpacked = staging / "unpacked"
        try:
            with tarfile.open(archive, "r:xz") as bundle:
                bundle.extractall(unpacked)
        except (tarfile.TarError, OSError) as exc:
            raise HostToolError(
                "tool_unpack_failed",
                f"{archive.name} matched its sha256, but could not be unpacked: "
                f"{type(exc).__name__}: {exc}",
            ) from None
        tree = unpacked / build.root
        if not (tree / "zig").is_file():
            raise HostToolError(
                "tool_unpack_failed",
                f"{archive.name} matched its sha256, but holds no {build.root}/zig",
            )
        if final.exists():
            # No stamp vouches for this tree (zig_placed said so before we got
            # here): an earlier place stopped between the rename and the stamp.
            # The verified tree replaces it whole.
            shutil.rmtree(final)
        os.replace(tree, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    zig_stamp(home).write_text(
        json.dumps(
            {
                "tool": "zig",
                "version": build.version,
                "platform": host_platform(),
                "url": build.url,
                "sha256": build.sha256,
                "provenance": build.provenance,
                "target": build.target,
                "seconds": round(time.monotonic() - started, 1),
                "placed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_zig(
    home: Path,
    *,
    on_line: Callable[[str], None] | None = None,
    fetch: Callable[[str, Path], str] | None = None,
) -> str:
    """Place Crucible's C compiler: the pinned Zig, and the ``cc`` that runs it.

    Idempotent. The tree is unpacked into a staging directory beside its home and
    renamed into place whole, so nothing ever reads half of it. The wrapper is
    rewritten whenever it is not exactly what this pin says, which also repairs a
    Crucible home that has moved. A failed fetch is not retried here, as with
    ffmpeg: installing again retries only what is missing.
    """
    build = zig_build()
    if build is None:
        return f"c compiler: none is placed on {host_platform()}"
    wrapper = c_compiler_path(home)
    if zig_placed(home, build):
        return f"c compiler: zig {build.version} cc already at {wrapper}"
    if _zig_unpacked(home, build):
        how = "was unpacked; its cc is rewritten"
    else:
        _unpack_zig(home, build, on_line, fetch)
        how = f"placed in {zig_dir(home, build)} (sha256 {build.sha256[:12]}...)"
    _write_wrapper(home, build)
    return f"c compiler: zig {build.version} {how}; {wrapper} runs it for {build.target}"


def env_runs_triton(env_dir: Path) -> bool:
    """Whether this env holds Triton, which JIT-compiles C the first time a kernel runs."""
    return any(env_dir.glob("lib/python*/site-packages/triton/__init__.py"))


def compiler_environment(env_dir: Path, home: Path | None = None) -> dict[str, str]:
    """``CC`` for a process started from ``env_dir``, or a refusal by name.

    The one owner of CC for Crucible's subprocesses: the engines
    (``engines/base.py`` ``start``) and the job workers (``workers._spawn``) both
    merge this over the environment they inherit, and every process they start
    inherits it in turn. Nothing is set on a host with no pinned compiler
    (mlx-darwin, Windows) or for an env without Triton.

    Crucible's compiler always wins over a ``CC`` the server inherited: this is an
    appliance, and the target that makes zig use its own glibc is what keeps the
    host free of a system toolchain. An env that runs Triton on a host whose
    compiler is not placed is misconfiguration, refused here before anything
    launches rather than minutes later at the first kernel.
    """
    build = zig_build()
    if build is None or not env_runs_triton(env_dir):
        return {}
    if home is None:
        from .config import crucible_home

        home = crucible_home()
    if not zig_placed(home, build):
        # A host updated by a deploy runs no `crucible install`, so the first process
        # that needs the compiler places it (57 MB, once), as install-on-submit places a
        # missing model. Only a placement that fails is refused, by name.
        try:
            ensure_zig(home)
        except HostToolError as exc:
            raise HostToolError(
                "c_compiler_missing",
                f"{env_dir} runs Triton, which compiles a C launcher the first time a "
                f"kernel runs, and Crucible's C compiler (zig {build.version} cc) could "
                f"not be placed at {c_compiler_path(home)}: {exc.message}. "
                f"{_install_sentence(home, env_dir)}",
            ) from exc
    return {C_COMPILER_ENV: str(c_compiler_path(home))}


def _envs_running_triton(home: Path) -> list[tuple[str, str]]:
    from . import jobenv
    from .backend import CUDA_LINUX

    return [
        (str(jobenv.env_dir(home, spec)), jobenv.install_command(spec))
        for spec in jobenv.every_env(CUDA_LINUX)
        if env_runs_triton(jobenv.env_dir(home, spec))
    ]


def _install_sentence(home: Path, env_dir: Path) -> str:
    for path, command in _envs_running_triton(home):
        if path == str(env_dir):
            return f"Run `{command}`, which places it"
    return (
        "That env is none of the ones `crucible install` builds under "
        f"{home / 'envs'}, so no install here places the compiler for it"
    )


def c_compiler_report(home: Path) -> dict[str, object]:
    build = zig_build()
    if build is None:
        return {
            "platform": host_platform(),
            "pinned_version": None,
            "placed": False,
            "path": None,
            "needed_by": [],
        }
    return {
        "platform": host_platform(),
        "pinned_version": build.version,
        "placed": zig_placed(home, build),
        "path": str(c_compiler_path(home)),
        "needed_by": [
            {"env": path, "fix": command} for path, command in _envs_running_triton(home)
        ],
    }


def needs_c_compiler(home: Path) -> bool:
    return zig_build() is not None and bool(_envs_running_triton(home))


def silero_vad_path(home: Path) -> Path:
    return (
        home / TOOLS_DIR_NAME / SILERO_VAD.directory / SILERO_VAD.url.rsplit("/", 1)[-1]
    )


def silero_vad_placed(home: Path) -> bool:
    path = silero_vad_path(home)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == SILERO_VAD.sha256
    except OSError:
        return False


def ensure_silero_vad(
    home: Path,
    *,
    fetch: Callable[[str, Path], str] | None = None,
) -> str:
    path = silero_vad_path(home)
    if silero_vad_placed(home):
        return f"speech detector: silero-vad {SILERO_VAD.version} already at {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        measured = (_download if fetch is None else fetch)(SILERO_VAD.url, partial)
        if measured != SILERO_VAD.sha256:
            raise HostToolError(
                "tool_sha_mismatch",
                f"{path.name} hashed {measured}, and Crucible pins "
                f"{SILERO_VAD.sha256}. Nothing is placed: these are not the bytes "
                "on Crucible's tools release",
            )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return (
        f"speech detector: silero-vad {SILERO_VAD.version} placed at {path} "
        f"(sha256 {SILERO_VAD.sha256[:12]}...)"
    )


# llama.cpp's llama-server for cuda-linux. ggml-org publishes CUDA builds for Windows only,
# so Crucible builds this one (scripts/build-llama-server-linux.sh) and re-hosts it on its
# own tools release. Only the binary is ours: it links cudart and cuBLAS dynamically and
# loads them from the llm env, which PyPI's nvidia-* wheels fill (crucible/llamacpp.py,
# CUDA_LINUX_LIBRARIES), never from this archive.
LLAMA_SERVER_PROGRAM = "llama-server"

LLAMA_SERVER_STAMP_NAME = "llama-server.json"

LLAMA_SERVER_BUILDS: dict[str, ToolBuild] = {
    "linux-x86_64": ToolBuild(
        version="b10970-cuda13.0",
        url=(
            "https://github.com/telltaleatheist/crucible/releases/download/"
            "tools/llama-server-b10970-cuda13.0-linux-x86_64.tar.xz"
        ),
        sha256="059b6d35b6e0b597e476c162d31a33b432d1c9386266c34169881290c1b3b73f",
        bytes=106_932_584,
        root="llama-server-b10970-cuda13.0-linux-x86_64",
        provenance=(
            "built on owens-pc (WSL2 Ubuntu 24.04, gcc 13.3) 2026-10-09 by "
            "scripts/build-llama-server-linux.sh from ggml-org/llama.cpp b10970 "
            "(bfdc32183d57f1e35bacf35c47d6311e2028bbbc); CUDA 13.0 from PyPI nvidia "
            "wheels (nvcc 13.0.88, cudart 13.0.96, cuBLAS 13.1.1.3), sm_75/80/86/89/90/120, "
            "static llama/ggml and libstdc++, links libcudart.so.13 and libcublas.so.13 "
            "(the llm env's) and libcuda.so.1 (the driver's), glibc 2.38+; MIT"
        ),
    ),
}


def llama_server_build(platform_key: str | None = None) -> ToolBuild | None:
    return LLAMA_SERVER_BUILDS.get(host_platform() if platform_key is None else platform_key)


def llama_server_stamp(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / LLAMA_SERVER_STAMP_NAME


def llama_server_path(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / "bin" / LLAMA_SERVER_PROGRAM


def llama_server_placed(home: Path, build: ToolBuild) -> bool:
    try:
        record = json.loads(llama_server_stamp(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("sha256") != build.sha256:
        return False
    return llama_server_path(home).is_file()


def ensure_llama_server(
    home: Path,
    *,
    on_line: Callable[[str], None] | None = None,
    fetch: Callable[[str, Path], str] | None = None,
) -> str:
    build = llama_server_build()
    if build is None:
        raise HostToolError(
            "tool_unpinned",
            f"there is no pinned llama-server build for {host_platform()}; Crucible "
            f"builds it for {sorted(LLAMA_SERVER_BUILDS)} only, and a GGUF model on "
            "cuda-linux has no engine without it",
        )
    tools = home / TOOLS_DIR_NAME
    bin_dir = tools / "bin"
    target = llama_server_path(home)
    if llama_server_placed(home, build):
        return f"llama-server: {build.version} already at {target}"

    bin_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".llama-server-", dir=str(tools)))
    started = time.monotonic()
    try:
        archive = staging / build.url.rsplit("/", 1)[-1]
        if on_line is not None:
            on_line(f"fetching {archive.name} ({build.bytes / 1e6:.0f} MB to download)")
        measured = (_download if fetch is None else fetch)(build.url, archive)
        if measured != build.sha256:
            raise HostToolError(
                "tool_sha_mismatch",
                f"{archive.name} hashed {measured}, and Crucible pins "
                f"{build.sha256}. Nothing is placed: these are not the bytes on "
                "Crucible's tools release",
            )
        member = f"{build.root}/bin/{LLAMA_SERVER_PROGRAM}"
        partial = bin_dir / f".{LLAMA_SERVER_PROGRAM}.partial"
        try:
            with tarfile.open(archive, "r:xz") as bundle:
                source = bundle.extractfile(member)
                if source is None:
                    raise KeyError(f"{member} is not a file")
                with source, partial.open("wb") as handle:
                    shutil.copyfileobj(source, handle, _CHUNK_BYTES)
            partial.chmod(0o755)
            os.replace(partial, target)
        except (tarfile.TarError, KeyError, OSError) as exc:
            raise HostToolError(
                "tool_unpack_failed",
                f"{archive.name} matched its sha256, but {member} could not be "
                f"taken out of it: {type(exc).__name__}: {exc}",
            ) from None
        finally:
            partial.unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    llama_server_stamp(home).write_text(
        json.dumps(
            {
                "tool": LLAMA_SERVER_PROGRAM,
                "version": build.version,
                "platform": host_platform(),
                "url": build.url,
                "sha256": build.sha256,
                "provenance": build.provenance,
                "seconds": round(time.monotonic() - started, 1),
                "placed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # The archive is xz: the binary it unpacks is a third larger than the download
    # (b10970: 106,932,584 B fetched, 139,514,616 B placed), so both are said.
    return (
        f"llama-server: {build.version} placed at {target} "
        f"({target.stat().st_size / 1e6:.1f} MB unpacked from a "
        f"{build.bytes / 1e6:.1f} MB download; sha256 {build.sha256[:12]}...)"
    )
