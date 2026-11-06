"""Give a WSL guest's file cache back to Windows after an install or a download.

Victoria's laptop, 2026-10-24: after `crucible install asr` and a 1.9 GB model pull, the
WSL VM (vmmemWSL) held 7.6 GB, of which the guest's processes used 0.9 GB; the rest was
Linux's page cache of the files just written. Linux keeps that cache until something
needs the memory, but a WSL guest's cache is memory Windows cannot use, and her Godot
and Blender jobs were killed for low memory while it sat there. Dropping it manually
(`echo 1 > /proc/sys/vm/drop_caches` as root) took the VM to 2.66 GB: WSL returns freed
pages to Windows by itself (free page reporting), it only never sees cached ones as free.

A model load does the same thing, and nothing ended the cache it left. Victoria's laptop,
2026-11-06: one qwen3-asr-0.6b file raised Windows commit from 47.4 to 60.3 GB, within 0.8 GB
of the limit, and 4.9 GB of that was this cache; it was still there after the job, and Claude
Code killed the command waiting on it. `CacheWatcher` is the owner of that: on WSL the server
looks every few seconds and gives back what it finds over a limit, whatever left it. A cached
page mapped by a running engine is not counted and not dropped.

So after an install, a pull or a module, Crucible asks the kernel to drop the cached pages
of the files under its own home and the download caches (`posix_fadvise(DONTNEED)` on
each, after a `sync` so written pages are clean and can go). It needs no root, touches no
file's contents and stops nothing; a file read again is read from disk. Only under WSL: on
a Linux box or a Mac the cache is the machine's own and Linux reclaims it when it is
needed, so it is left to do its job.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .memorybudget import gib_text
from .service import in_wsl

DOWNLOAD_CACHES = (".cache/pip", ".cache/uv", ".cache/huggingface")

# Unmapped file cache the guest may hold before the watcher gives it back.
LIMIT_BYTES = 1 << 30
LOOK_EVERY_SECONDS = 10.0
# A release that left the guest over the limit means the cache is not under our roots, so
# looking again soon would walk the home for nothing.
AFTER_NO_GAIN_SECONDS = 300.0


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


def forget(fd: int, upto: int) -> None:
    """Drop the cache of the first `upto` bytes of a file this process has finished reading
    (a hash of every installed file reads hundreds of GiB nobody needs again). Off WSL the
    cache is the machine's own and is left alone."""
    if in_wsl():
        os.posix_fadvise(fd, 0, upto, os.POSIX_FADV_DONTNEED)


def droppable_bytes(meminfo: str) -> int:
    """The guest's file cache that no process has mapped, from /proc/meminfo's text."""
    fields = {}
    for line in meminfo.splitlines():
        name, _, rest = line.partition(":")
        if name in ("Cached", "Mapped", "Shmem"):
            fields[name] = int(rest.split()[0]) * 1024
    return max(0, fields["Cached"] - fields["Mapped"] - fields["Shmem"])


def _read_droppable() -> int:
    return droppable_bytes(Path("/proc/meminfo").read_text(encoding="ascii"))


class CacheWatcher:
    """Owns the WSL guest's file cache for as long as the server runs."""

    def __init__(
        self,
        home: Path,
        *,
        droppable: Callable[[], int] = _read_droppable,
        give_back: Callable[[list[Path]], Released] = lambda roots: release(roots),
    ) -> None:
        self._roots = roots_for(home)
        self._droppable = droppable
        self._give_back = give_back
        self._stop = threading.Event()

    def tick(self) -> float:
        """One look; the seconds until the next."""
        if self._droppable() < LIMIT_BYTES:
            return LOOK_EVERY_SECONDS
        print(f"crucible: file cache: {self._give_back(self._roots).sentence()}", file=sys.stderr)
        return LOOK_EVERY_SECONDS if self._droppable() < LIMIT_BYTES else AFTER_NO_GAIN_SECONDS

    def _run(self) -> None:
        wait = LOOK_EVERY_SECONDS
        while not self._stop.wait(wait):
            try:
                wait = self.tick()
            except (OSError, ValueError, KeyError, IndexError) as exc:
                print(
                    f"crucible: file cache: could not look: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                wait = AFTER_NO_GAIN_SECONDS

    def start(self) -> None:
        if in_wsl():
            threading.Thread(target=self._run, name="crucible-file-cache", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def release_after_install(home: Path) -> str | None:
    """The sentence for what was given back, or None where there is nothing to give (not
    a WSL guest)."""
    if not in_wsl():
        return None
    return release(roots_for(home)).sentence()


__all__ = [
    "CacheWatcher",
    "DOWNLOAD_CACHES",
    "Released",
    "droppable_bytes",
    "forget",
    "release",
    "release_after_install",
    "roots_for",
]
