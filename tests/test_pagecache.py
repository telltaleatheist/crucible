"""After an install or a pull, a WSL guest's file cache of what it wrote goes back to
Windows (Victoria's laptop, 2026-10-24: vmmemWSL held 7.6 GB, 0.9 GB of it processes, and
her Godot and Blender jobs were killed for low memory)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

from crucible import pagecache

LINUX_ONLY = pytest.mark.skipif(
    not hasattr(os, "posix_fadvise"), reason="posix_fadvise is Linux's"
)


@LINUX_ONLY
def test_every_regular_file_is_released_and_symlinks_are_left_to_their_targets(tmp_path: Path) -> None:
    (tmp_path / "env" / "bin").mkdir(parents=True)
    (tmp_path / "env" / "lib.so").write_bytes(b"x" * 4096)
    (tmp_path / "models.safetensors").write_bytes(b"y" * 8192)
    (tmp_path / "env" / "bin" / "python").symlink_to(sys.executable)
    released = pagecache.release([tmp_path, tmp_path / "absent"])
    assert (released.files, released.bytes, released.skipped) == (2, 4096 + 8192, 0)
    assert "2 files" in released.sentence()


@LINUX_ONLY
def test_a_file_that_cannot_be_opened_keeps_its_pages_and_is_counted(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.write_bytes(b"z")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running as root: every file opens")
        released = pagecache.release([tmp_path])
    finally:
        locked.chmod(0o600)
    assert released.skipped == 1 and "could not be opened" in released.sentence()


def test_off_wsl_nothing_is_released(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pagecache, "in_wsl", lambda: False)
    monkeypatch.setattr(pagecache, "release", lambda roots: pytest.fail("released off WSL"))
    assert pagecache.release_after_install(tmp_path) is None


def test_under_wsl_the_home_and_the_download_caches_are_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[list[Path]] = []
    monkeypatch.setattr(pagecache, "in_wsl", lambda: True)
    monkeypatch.setattr(pagecache, "release", lambda roots: seen.append(list(roots))
                        or pagecache.Released(3, 2 << 30, 0, 0.5))
    said = pagecache.release_after_install(tmp_path)
    assert said is not None and "back to Windows" in said
    assert seen[0][0] == tmp_path
    assert [root.name for root in seen[0][1:]] == ["pip", "uv", "huggingface"]


MEMINFO = "MemFree: 100 kB\nCached:    5242880 kB\nShmem:      1024 kB\nMapped:   1048576 kB\n"


def test_what_a_running_engine_has_mapped_is_not_droppable() -> None:
    assert pagecache.droppable_bytes(MEMINFO) == (5242880 - 1048576 - 1024) * 1024


def _watcher(tmp_path: Path, readings: list[int], gave: list[list[Path]]) -> pagecache.CacheWatcher:
    return pagecache.CacheWatcher(
        tmp_path,
        droppable=lambda: readings.pop(0),
        give_back=lambda roots: gave.append(roots) or pagecache.Released(1, 1, 0, 0.1),
    )


def test_under_the_limit_the_watcher_gives_nothing_back(tmp_path: Path) -> None:
    gave: list[list[Path]] = []
    watcher = _watcher(tmp_path, [pagecache.LIMIT_BYTES - 1], gave)
    assert watcher.tick() == pagecache.LOOK_EVERY_SECONDS
    assert gave == []


def test_over_the_limit_the_watcher_gives_back_the_home_and_keeps_looking(tmp_path: Path) -> None:
    gave: list[list[Path]] = []
    watcher = _watcher(tmp_path, [pagecache.LIMIT_BYTES, 0], gave)
    assert watcher.tick() == pagecache.LOOK_EVERY_SECONDS
    assert gave[0][0] == tmp_path


def test_cache_that_is_not_ours_does_not_walk_the_home_every_look(tmp_path: Path) -> None:
    gave: list[list[Path]] = []
    watcher = _watcher(tmp_path, [pagecache.LIMIT_BYTES * 2, pagecache.LIMIT_BYTES * 2], gave)
    assert watcher.tick() == pagecache.AFTER_NO_GAIN_SECONDS
    assert len(gave) == 1


def test_off_wsl_the_watcher_starts_no_thread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pagecache, "in_wsl", lambda: False)
    before = threading.active_count()
    pagecache.CacheWatcher(tmp_path).start()
    assert threading.active_count() == before


def test_a_failed_look_says_so_and_the_watcher_lives_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    looks: list[int] = []

    def broken() -> int:
        looks.append(1)
        watcher.stop()
        raise OSError("no /proc")

    watcher = pagecache.CacheWatcher(tmp_path, droppable=broken)
    monkeypatch.setattr(pagecache, "LOOK_EVERY_SECONDS", 0.0)
    watcher._run()
    assert looks == [1]
    assert "could not look: OSError: no /proc" in capsys.readouterr().err


@LINUX_ONLY
def test_forget_is_wsl_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[tuple[int, int, int]] = []
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda fd, start, end, advice: seen.append((start, end, advice)))
    target = tmp_path / "x"
    target.write_bytes(b"x")
    with target.open("rb") as handle:
        monkeypatch.setattr(pagecache, "in_wsl", lambda: False)
        pagecache.forget(handle.fileno(), 1)
        assert seen == []
        monkeypatch.setattr(pagecache, "in_wsl", lambda: True)
        pagecache.forget(handle.fileno(), 1)
    assert seen == [(0, 1, os.POSIX_FADV_DONTNEED)]
