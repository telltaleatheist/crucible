"""A model's footprint is what it took on this machine while serving, not its estimate.

Owen, 2026-11-27: "how much each model ACTUALLY takes in practice rather than in theory?"
and of host memory "that should probably be part of the calculation too"."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible import accelerator, footprints, procmemory
from crucible.accelerator import ProbeError
from crucible.errors import ApiError
from crucible.footprints import KEPT_RUNS, Run, Watch
from crucible.jobs import template

GIB = 1024**3


def _run(n: int, card: int | None, ram: int | None) -> Run:
    return Run(run_id=f"r{n}", card_bytes=card, ram_bytes=ram, measured_at=f"t{n}")


def test_nothing_is_known_before_a_served_run(tmp_path: Path) -> None:
    assert footprints.read(tmp_path, "dots-ocr", 12 * GIB) is None


def test_the_largest_of_the_last_runs_is_the_figure(tmp_path: Path) -> None:
    footprints.record(tmp_path, "dots-ocr", 12 * GIB, _run(1, 11 * GIB, 6 * GIB))
    footprints.record(tmp_path, "dots-ocr", 12 * GIB, _run(2, 9 * GIB, 7 * GIB))
    kept = footprints.read(tmp_path, "dots-ocr", 12 * GIB)
    assert kept is not None
    assert (kept.card_bytes, kept.ram_bytes) == (11 * GIB, 7 * GIB)


def test_only_the_last_runs_are_kept_so_a_bad_reading_is_outvoted(tmp_path: Path) -> None:
    footprints.record(tmp_path, "m", 1, _run(0, 20 * GIB, 1))
    for n in range(1, KEPT_RUNS + 1):
        footprints.record(tmp_path, "m", 1, _run(n, 10 * GIB, 1))
    kept = footprints.read(tmp_path, "m", 1)
    assert kept is not None and len(kept.runs) == KEPT_RUNS
    assert kept.card_bytes == 10 * GIB


def test_a_run_written_again_replaces_itself(tmp_path: Path) -> None:
    footprints.record(tmp_path, "m", 1, _run(1, 5 * GIB, 1 * GIB))
    footprints.record(tmp_path, "m", 1, _run(1, 6 * GIB, 2 * GIB))
    kept = footprints.read(tmp_path, "m", 1)
    assert kept is not None and len(kept.runs) == 1
    assert (kept.card_bytes, kept.ram_bytes) == (6 * GIB, 2 * GIB)


def test_one_model_at_two_shapes_is_two_footprints(tmp_path: Path) -> None:
    footprints.record(tmp_path, "asr", 6 * GIB, _run(1, 5 * GIB, 1))
    footprints.record(tmp_path, "asr", 7 * GIB, _run(1, 6 * GIB, 1))
    assert footprints.read(tmp_path, "asr", 6 * GIB).card_bytes == 5 * GIB  # type: ignore[union-attr]
    assert footprints.read(tmp_path, "asr", 7 * GIB).card_bytes == 6 * GIB  # type: ignore[union-attr]


def _proc(tmp_path: Path, pss_kb: dict[int, int]) -> Path:
    proc = tmp_path / "proc"
    for pid, kb in pss_kb.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "smaps_rollup").write_text(
            f"55a0-7fff ---p 00000000 00:00 0  [rollup]\nRss:  {kb * 2} kB\nPss:  {kb} kB\n",
            encoding="ascii",
        )
    return proc


def test_host_memory_is_the_summed_pss_and_a_gone_process_counts_nothing(
    tmp_path: Path,
) -> None:
    proc = _proc(tmp_path, {10: 4_540_804, 11: 2_242_764})
    assert procmemory.linux_pss_bytes([10, 11, 12], proc) == (4_540_804 + 2_242_764) * 1024


def test_meminfo_gives_available_and_total(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       13278392 kB\nMemFree:  100 kB\nMemAvailable:    6414940 kB\n",
        encoding="ascii",
    )
    assert procmemory.linux_host_memory(meminfo) == (6414940 * 1024, 13278392 * 1024)


class _Card:
    def __init__(self, used: list[int | Exception]) -> None:
        self.used = used

    def __call__(self) -> int:
        value = self.used.pop(0) if len(self.used) > 1 else self.used[0]
        if isinstance(value, Exception):
            raise value
        return value


def _watch(
    tmp_path: Path, card: _Card, ram_kb: int, working: list[bool], said: list[str]
) -> Watch:
    proc = _proc(tmp_path, {77: ram_kb})
    return Watch(
        home=tmp_path,
        model="dots-ocr",
        estimate_bytes=12 * GIB,
        card_before=1 * GIB,
        card_used=card,
        pids=lambda: [77],
        working=lambda: working[0],
        say=said.append,
        sample_s=3600.0,
        ram_of=lambda pids: procmemory.linux_pss_bytes(pids, proc),
    )


def test_an_idle_residency_keeps_nothing(tmp_path: Path) -> None:
    said: list[str] = []
    watch = _watch(tmp_path, _Card([13 * GIB]), 1_000_000, [False], said)
    watch.start()
    assert watch.finish() is None
    assert footprints.read(tmp_path, "dots-ocr", 12 * GIB) is None


def test_a_served_residency_keeps_its_peak_rise_and_its_host_memory(tmp_path: Path) -> None:
    said: list[str] = []
    working = [False]
    card = _Card([10 * GIB, 13 * GIB, 12 * GIB])
    watch = _watch(tmp_path, card, 6_000_000, working, said)
    watch.start()  # loaded: 9 GiB over the 1 GiB before
    working[0] = True
    watch.sample()  # serving: 12 GiB over
    kept = watch.finish()  # 11 GiB over: the peak stands
    assert kept is not None
    assert kept.card_bytes == 12 * GIB
    assert kept.ram_bytes == 6_000_000 * 1024
    assert watch.ram_now == 6_000_000 * 1024


def test_a_card_that_cannot_be_read_is_skipped_and_said_once_never_zero(
    tmp_path: Path,
) -> None:
    said: list[str] = []
    card = _Card([ProbeError("nvidia-smi gone"), ProbeError("nvidia-smi gone"), 11 * GIB])
    watch = _watch(tmp_path, card, 1_000, [True], said)
    watch.start()
    watch.sample()
    kept = watch.finish()
    assert kept is not None and kept.card_bytes == 10 * GIB
    assert len(said) == 1 and "the card" in said[0]


# --- The guard weighs what was measured -------------------------------------------------


def _config(home: Path, backend: str = "cuda-linux") -> Any:
    return SimpleNamespace(backend_kind=backend, home=home, desktop_allowance_bytes=GIB)


def _guarded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    asked: list[dict[str, Any]] = []
    monkeypatch.setattr(
        procmemory, "memory_words", lambda: "Crucible's Linux (WSL's share of this PC)"
    )

    def guard(backend: str, **kwargs: Any) -> str:
        asked.append(kwargs)
        return "state"

    monkeypatch.setattr(accelerator, "guard", guard)
    return asked


def test_before_any_measurement_the_card_is_weighed_by_the_estimate_and_memory_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (0, 1))  # would refuse if read
    template.card_guard(_config(tmp_path), model="dots-ocr", need_bytes=12 * GIB,
                        owned_pids=frozenset())
    assert asked[0]["need_bytes"] == 12 * GIB and asked[0]["need_basis"] == ""


def test_a_measured_model_is_weighed_by_what_it_took_even_below_its_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.record(tmp_path, "dots-ocr", 12 * GIB, _run(1, 11 * GIB, 6 * GIB))
    asked = _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (8 * GIB, 13 * GIB))
    template.card_guard(_config(tmp_path), model="dots-ocr", need_bytes=12 * GIB,
                        owned_pids=frozenset())
    assert asked[0]["need_bytes"] == 11 * GIB
    assert asked[0]["need_basis"] == template.MEASURED


def test_short_of_host_memory_the_load_waits_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.record(tmp_path, "dots-ocr", 12 * GIB, _run(1, 11 * GIB, 7 * GIB))
    _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (4 * GIB, 13 * GIB))
    with pytest.raises(ApiError) as caught:
        template.card_guard(_config(tmp_path), model="dots-ocr", need_bytes=12 * GIB,
                            owned_pids=frozenset(), reclaimable_ram_bytes=2 * GIB)
    error = caught.value
    assert error.code == accelerator.MEMORY_BUSY
    assert error.code in accelerator.WAITS_FOR_THE_CARD
    assert error.message.startswith(
        "dots-ocr needs 7.0 GiB of memory (the most its last loads here took) and "
        "Crucible's Linux (WSL's share of this PC) has 6.0 GiB available of 13.0 GiB "
        "(counting the 2.0 GiB the resident model gives back)."
    )


def test_what_the_resident_gives_back_makes_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.record(tmp_path, "dots-ocr", 12 * GIB, _run(1, 11 * GIB, 7 * GIB))
    _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (4 * GIB, 13 * GIB))
    template.card_guard(_config(tmp_path), model="dots-ocr", need_bytes=12 * GIB,
                        owned_pids=frozenset(), reclaimable_ram_bytes=3 * GIB)


def test_more_memory_than_the_machine_has_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.record(tmp_path, "big", 30 * GIB, _run(1, 20 * GIB, 16 * GIB))
    _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (10 * GIB, 13 * GIB))
    with pytest.raises(ApiError) as caught:
        template.card_guard(_config(tmp_path), model="big", need_bytes=30 * GIB,
                            owned_pids=frozenset())
    assert caught.value.code == "insufficient_host_memory"
    assert "16.0 GiB" in caught.value.message and "13.0 GiB in all" in caught.value.message


def test_on_a_mac_the_measured_footprint_is_the_one_pool_s_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owen, 2026-11-27: "some systems wont have wsl. some systems will be using a mac, or
    windows". On Apple silicon the card and host memory are one pool, weighed once."""
    footprints.record(tmp_path, "m", 14 * GIB, _run(1, None, 9 * GIB))
    asked = _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (0, 1))  # would refuse if read
    template.card_guard(_config(tmp_path, "mlx-darwin"), model="m", need_bytes=14 * GIB,
                        owned_pids=frozenset())
    assert asked[0]["need_bytes"] == 9 * GIB
    assert asked[0]["need_basis"] == template.MEASURED


def test_on_windows_the_card_and_host_memory_are_weighed_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.record(tmp_path, "dots-ocr", 5 * GIB, _run(1, 4 * GIB, 3 * GIB))
    asked = _guarded(monkeypatch)
    monkeypatch.setattr(procmemory, "host_memory", lambda: (2 * GIB, 32 * GIB))
    with pytest.raises(ApiError) as caught:
        template.card_guard(_config(tmp_path, "llama-windows"), model="dots-ocr",
                            need_bytes=5 * GIB, owned_pids=frozenset())
    assert asked[0]["need_bytes"] == 4 * GIB
    assert caught.value.code == accelerator.MEMORY_BUSY


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own readers")
def test_windows_reads_this_very_process_and_its_memory() -> None:
    held = procmemory.windows_working_set_bytes([os.getpid()])
    available, total = procmemory.windows_host_memory()
    assert 10 * 1024**2 < held < total
    assert 0 < available <= total
    assert procmemory.windows_working_set_bytes([4_000_000_000 // 4 * 4 + 3]) == 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's own readers")
def test_linux_reads_this_very_process_and_its_memory() -> None:
    held = procmemory.linux_pss_bytes([os.getpid()])
    available, total = procmemory.linux_host_memory()
    assert 10 * 1024**2 < held < total
    assert 0 < available <= total


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS's own readers")
def test_macos_reads_this_very_process_and_its_memory() -> None:
    held = procmemory.mac_footprint_bytes([os.getpid()])
    available, total = procmemory.host_memory()
    assert 10 * 1024**2 < held < total
    assert 0 < available <= total
    assert os.getpid() in procmemory.mac_group_members(frozenset({os.getpgid(0)}))


def test_an_unreadable_record_is_refused_by_name_with_the_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    footprints.footprints_path(tmp_path).write_text("not json", encoding="utf-8")
    _guarded(monkeypatch)
    with pytest.raises(ApiError) as caught:
        template.card_guard(_config(tmp_path), model="m", need_bytes=1,
                            owned_pids=frozenset())
    assert caught.value.code == "footprints_unreadable"
    assert "Delete that file" in caught.value.message
