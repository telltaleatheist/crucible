"""Give a WSL guest's file cache back to Windows after an install or a download.

Victoria's laptop, 2026-10-24: after `crucible install asr` and a 1.9 GB model pull, the
WSL VM (vmmemWSL) held 7.6 GB, of which the guest's processes used 0.9 GB; the rest was
Linux's page cache of the files just written. Linux keeps that cache until something
needs the memory, but a WSL guest's cache is memory Windows cannot use, and her Godot
and Blender jobs were killed for low memory while it sat there. Dropping it by hand
(`echo 1 > /proc/sys/vm/drop_caches` as root) took the VM to 2.66 GB: WSL returns freed
pages to Windows by itself (free page reporting), it only never sees cached ones as free.

So after an install, a pull or a module, Crucible asks the kernel to drop the cached pages
of the files under its own home and the download caches (`posix_fadvise(DONTNEED)` on
each, after a `sync` so written pages are clean and can go). It needs no root, touches no
file's contents and stops nothing; a file read again is read from disk. Only under WSL: on
a Linux box or a Mac the cache is the machine's own and Linux reclaims it when it is
needed, so it is left to do its job.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .memorybudget import gib_text
from .service import in_wsl

DOWNLOAD_CACHES = (".cache/pip", ".cache/uv", ".cache/huggingface")


@dataclass(frozen=True)
class Released:
    files: int
    bytes: int
    skipped: int
    seconds: float

    def sentence(self) -> str:
        skipped = f"; {self.skipped} could not be opened and keep theirs" if self.skipped else ""
        return (
            f"gave the cached pages of {self.files:,} files ({gib_text(self.bytes)}) back to "
            f"Windows in {self.seconds:.1f}s{skipped}"
        )


def roots_for(home: Path) -> list[Path]:
    user = Path.home()
    return [home, *(user / cache for cache in DOWNLOAD_CACHES)]


def _files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for folder, _dirs, names in os.walk(root, followlinks=False):
            for name in names:
                path = Path(folder) / name
                if not path.is_symlink():  # its target is a file of its own, or not ours
                    yield path


def release(roots: Iterable[Path]) -> Released:
    """Drop the page cache of every regular file under `roots`. A file that cannot be
    opened (gone, or not ours) keeps its pages and is counted, never raised: the install
    it follows has already succeeded or failed on its own."""
    started = time.monotonic()
    os.sync()
    files = size = skipped = 0
    for path in _files(roots):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            skipped += 1
            continue
        try:
            stat = os.fstat(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            skipped += 1
            continue
        finally:
            os.close(fd)
        files += 1
        size += stat.st_size
    return Released(files, size, skipped, time.monotonic() - started)


def release_after_install(home: Path) -> str | None:
    """The sentence for what was given back, or None where there is nothing to give (not
    a WSL guest)."""
    if not in_wsl():
        return None
    return release(roots_for(home)).sentence()


__all__ = ["DOWNLOAD_CACHES", "Released", "release", "release_after_install", "roots_for"]
