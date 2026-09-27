"""Where this server looks for the command-line tools it shells out to.

One fact, one owner (ARCHITECTURE.md R1): **`PATH`**. Three job types refuse when
ffmpeg is not on it (`asr` decodes through it, `align` decodes through it, `tts`
encodes through it), and `crucible service install` writes it into the unit or
the plist it generates. Those are the same fact read twice, and until this module
existed they were two answers nobody could compare.

Why this is not a detail
------------------------
Measured on Owen's Mac, 2026-09-13, against a real `crucible doctor` over a
**non-login** shell:

    job tts: NOT READY — there is no ffmpeg on PATH

ffmpeg was installed the whole time, at `/opt/homebrew/bin/ffmpeg`. The PATH that
shell searched was `/usr/bin:/bin:/usr/sbin:/sbin`, which is the same bare PATH a
**launchd agent and a systemd user unit are started with** — so a Crucible
installed as a service would have failed in exactly the same way, on a host where
every env was installed and every tool was present.

Two things follow, and this module is both of them:

- **A refusal names the PATH it searched.** "There is no ffmpeg on PATH" is true
  and useless; the reader's next question is always *which PATH*, and on a host
  with two shells and a service manager that is a real question with three
  answers. `searched_note()` is what answers it, at every site that says a tool
  is missing.
- **`crucible service install` records the installing shell's PATH**, because
  that shell is the one the operator proved the tools on — they just ran
  `crucible doctor` in it. Hardcoding `/opt/homebrew/bin` would fix one Mac and
  nothing else, and it would be a second owner of this fact besides the
  environment.

Crucible's own ffmpeg comes first (2026-09-26, fresh-install #25 and #43)
------------------------------------------------------------------------
kylies-pc's fresh WSL guest had no ffmpeg at all, and nobody there would have
known to `apt install` one. The rvc env got its ffmpeg from `static_ffmpeg`, a
pip package that downloads binaries from its author's GitHub the first time it
is asked. And the guest's PATH carried Windows' winget `/mnt/c/.../ffmpeg.exe`,
which could not even run while WSLInterop was broken.

Owen's ruling: *"that should be part of the rvc environment if we need it ...
downloading from gh releases is probably the most trustworthy/correct way."*
So there is ONE pinned build per platform (`FFMPEG_BUILDS`), hosted on
Crucible's own `tools` release and checked against a sha256 written in this
file. `crucible install` places it in `<home>/tools/bin/` (`ensure_ffmpeg`),
and `which()` looks THERE FIRST, for every job type. PATH is the fallback for a
platform with no pinned build (the Mac, which uses Homebrew's today). Inside a
WSL guest the Windows drives on PATH (`/mnt/<drive>/...`) are never searched.
No system package, and nothing is downloaded at run time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
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

from .errors import CrucibleError

PATH_ENV = "PATH"

#: Where Crucible's own host tools live, under its home.
TOOLS_DIR_NAME = "tools"

#: What records which ffmpeg build is placed there.
FFMPEG_STAMP_NAME = "ffmpeg.json"

#: The two programs the job types call. ffprobe is looked for BESIDE ffmpeg
#: (`jobs/asr/worker.py`, `jobs/alignlongform/stages.py`), so both are placed.
FFMPEG_PROGRAMS: tuple[str, ...] = ("ffmpeg", "ffprobe")


class HostToolError(CrucibleError):
    """A pinned host tool could not be fetched, verified or placed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolBuild:
    """One pinned archive of one tool, for one platform."""

    version: str
    url: str
    sha256: str
    bytes: int
    #: The archive's top directory; the programs are at `<root>/bin/<name>`.
    root: str
    #: Where these bytes came from.
    provenance: str


#: THE PINNED ffmpeg, keyed by `host_platform()`. 2026-09-26, #25 and #43.
#:
#: linux-x86_64: FFmpeg n8.1.3, BtbN/FFmpeg-Builds' static LGPL build. Its
#: binaries link only glibc (2.28 or newer) and libgcc_s, which every Ubuntu
#: has. LGPL rather than GPL because the job types need audio only: FLAC, PCM,
#: MP3 and the resampler are native or LGPL, and the GPL extras are video
#: codecs. The asset on our `tools` release is BtbN's, byte for byte: the same
#: sha256 is in their `checksums.sha256` for release
#: `autobuild-2026-09-26-13-03`, and GitHub's digest of their asset agrees.
#: Checked on Ubuntu 24.04 (WSL2) on 2026-09-26: `-version`, a FLAC encode and
#: decode, an MP3 encode, and ffprobe durations. BtbN prunes dated autobuilds,
#: which is one more reason the bytes are on our release and not theirs.
#:
#: darwin-arm64 has NO ROW YET. The Mac uses Homebrew's ffmpeg on PATH, and its
#: rvc env keeps `static_ffmpeg` as urvc's fallback. BtbN builds nothing for
#: macOS. A fresh Mac with no Homebrew needs a pinned row here too.
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
}

#: How much of a download is read at a time.
_CHUNK_BYTES = 1 << 20

#: A PATH entry that is a Windows drive seen from inside WSL. Never searched: an
#: `.exe` there runs only while WSLInterop works (#38, #43), and it is a Windows
#: program, not this guest's.
_WINDOWS_DRIVE = re.compile(r"^/mnt/[A-Za-z](/|$)")


def search_path() -> str:
    """The `PATH` this process searches. `""` when there is none.

    A module-level probe, for the reason `crucible/jobs/asr/__init__.py` gives
    about its own: a test replaces it and asserts on the refusal, instead of
    asserting on whatever the machine running the suite happens to have.
    """
    return os.environ.get(PATH_ENV, "")


def _home() -> Path | None:
    """Crucible's home, or None when this host cannot say where it is."""
    from .config import crucible_home
    from .errors import ConfigError

    try:
        return crucible_home()
    except ConfigError:
        return None


def tools_bin(home: Path | None = None) -> Path | None:
    """`<home>/tools/bin`, where `crucible install` places Crucible's own tools."""
    root = _home() if home is None else home
    return None if root is None else root / TOOLS_DIR_NAME / "bin"


def _in_wsl() -> bool:
    return sys.platform.startswith("linux") and "microsoft" in platform.release().lower()


def _searched_entries(value: str) -> list[str]:
    """PATH's entries, less the Windows drives when this is a WSL guest (#43)."""
    entries = [entry for entry in value.split(os.pathsep) if entry]
    if _in_wsl():
        entries = [entry for entry in entries if not _WINDOWS_DRIVE.match(entry)]
    return entries


def which(tool: str) -> str | None:
    """Where `tool` is: Crucible's own first, then PATH. None when not found.

    2026-09-26 (#25, #43): `<home>/tools/bin/<tool>` wins over anything on PATH,
    for every job type, so a host runs the pinned ffmpeg `crucible install`
    placed and not whichever one a shell happens to carry. Inside WSL, the
    Windows drives on PATH are never searched.
    """
    own = tools_bin()
    if own is not None:
        candidate = own / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(tool, path=os.pathsep.join(_searched_entries(search_path())))


def worker_path(inherited: str) -> str:
    """The PATH a worker process gets: Crucible's tools first, then `inherited`.

    2026-09-26 (#25). A worker's own libraries look for ffmpeg on PATH by
    themselves: urvc's `_add_ffmpeg_paths` asks `shutil.which("ffmpeg")`, and
    downloads one through `static_ffmpeg` when it finds none. The pinned build
    first on PATH is what keeps that download from ever happening.
    """
    own = tools_bin()
    if own is None or not own.is_dir():
        return inherited
    return os.pathsep.join(entry for entry in (str(own), inherited) if entry)


def searched_note() -> str:
    """`(looked in …, then PATH searched: …)`, carried by every "tool missing".

    An empty PATH is reported as empty rather than as nothing, because the two
    read identically in a sentence and only one of them is a configuration
    mistake somebody can fix. Crucible's own tools directory is named first
    because it is searched first, and because installing any job type fills it.
    """
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


# ------------------------------------------------------------ pinned ffmpeg


def host_platform() -> str:
    """`<system>-<machine>`, the key of `FFMPEG_BUILDS`: `linux-x86_64`, `darwin-arm64`."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def ffmpeg_build(platform_key: str | None = None) -> ToolBuild | None:
    """The pinned ffmpeg for this platform, or None when there is none."""
    return FFMPEG_BUILDS.get(host_platform() if platform_key is None else platform_key)


def ffmpeg_stamp(home: Path) -> Path:
    return home / TOOLS_DIR_NAME / FFMPEG_STAMP_NAME


def ffmpeg_placed(home: Path, build: ToolBuild) -> bool:
    """Is exactly this build in `<home>/tools/bin`? Read off the stamp and the files."""
    try:
        record = json.loads(ffmpeg_stamp(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("sha256") != build.sha256:
        return False
    bin_dir = home / TOOLS_DIR_NAME / "bin"
    return all((bin_dir / name).is_file() for name in FFMPEG_PROGRAMS)


def ffmpeg_report(home: Path) -> dict[str, object]:
    """What `crucible doctor` says about ffmpeg: which one, and from where.

    `source` is `crucible` (the pinned build is placed), `path` (found on PATH,
    which is right on a platform with no pinned build and a note on one that
    has one), or `missing`.
    """
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
    """Stream `url` to `destination`, returning the sha256 of what arrived."""
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
    """Place the pinned ffmpeg and ffprobe in `<home>/tools/bin`. Returns one line.

    2026-09-26, Owen's ruling on #25. `crucible install` calls this for every
    job type, so a server that can run an audio job has the ffmpeg it runs
    with. Nothing is fetched when the stamp already names this build.

    THE DIGEST IS CHECKED BEFORE ANYTHING IS PLACED. The archive is downloaded
    beside the tools directory and hashed as it arrives; a mismatch places
    nothing, and the download is removed either way. Only the two programs are
    taken out of it, each written under a temporary name and renamed into
    place, so a half-written ffmpeg is never at the path `which()` returns.

    A platform with no pinned build is not a refusal: the line says so, and
    `which()` falls back to PATH as it always did.
    """
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
