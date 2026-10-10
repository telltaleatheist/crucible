from __future__ import annotations

import asyncio
import io
import threading
import time
from pathlib import Path

from crucible import loopwatch
from crucible.loopwatch import LoopWatch


def _block_for(seconds: float) -> None:
    time.sleep(seconds)


def test_a_blocked_loop_is_reported_with_the_blocking_stack() -> None:
    out = io.StringIO()

    async def main() -> None:
        watch = LoopWatch(asyncio.get_running_loop(), out=out)
        watch.start()
        await asyncio.sleep(0.3)
        _block_for(loopwatch.BLOCKED_S + 0.6)
        await asyncio.sleep(0.3)
        watch.stop()

    asyncio.run(main())
    said = out.getvalue()
    assert said.count("the event loop has not run for") == 1, said
    assert "_block_for" in said and "time.sleep" in said


def test_a_loop_that_keeps_beating_says_nothing() -> None:
    out = io.StringIO()

    async def main() -> None:
        watch = LoopWatch(asyncio.get_running_loop(), out=out)
        watch.start()
        for _ in range(12):
            await asyncio.sleep(0.1)
        watch.stop()

    asyncio.run(main())
    assert out.getvalue() == ""


def test_one_stall_is_said_at_most_twice() -> None:
    now = [0.0]
    out = io.StringIO()
    watch = LoopWatch(asyncio.new_event_loop(), out=out, clock=lambda: now[0])
    for t in (0.5, 1.2, 2.0, 4.0, 5.5, 9.0, 30.0):
        now[0] = t
        watch.check()
    assert out.getvalue().count("has not run for") == 2
    watch._beat()
    now[0] = 32.0
    watch.check()
    assert out.getvalue().count("has not run for") == 3, "a new stall is said again"


def test_a_report_says_where_the_other_threads_are_and_the_memory_state() -> None:
    out = io.StringIO()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        holding.set()
        release.wait(5)

    async def main() -> None:
        watch = LoopWatch(asyncio.get_running_loop(), out=out)
        watch.start()
        await asyncio.sleep(0.3)
        _block_for(loopwatch.BLOCKED_S + 0.6)
        await asyncio.sleep(0.3)
        watch.stop()

    threading.Thread(target=holder, name="the-holder", daemon=True).start()
    holding.wait(5)
    try:
        asyncio.run(main())
    finally:
        release.set()
    said = out.getvalue()
    assert "The other threads, innermost frame first:" in said, said
    assert "the-holder: " in said and " holder" in said, said
    assert "crucible-loopwatch" not in said, "the watcher does not report itself"
    assert "Memory: " in said


def test_the_memory_state_reads_pressure_ram_and_swap(tmp_path: Path) -> None:
    pressure = tmp_path / "pressure"
    pressure.write_text(
        "some avg10=71.20 avg60=40.00 avg300=9.00 total=1\n"
        "full avg10=64.05 avg60=30.00 avg300=7.00 total=1\n",
        encoding="utf-8",
    )
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       16384000 kB\nMemFree:          100000 kB\n"
        "MemAvailable:     209715 kB\nSwapTotal:       4194304 kB\nSwapFree:              0 kB\n",
        encoding="utf-8",
    )
    assert loopwatch.memory_state(pressure, meminfo) == (
        "pressure over 10 s some 71.20%, full 64.05%; available 0.2 GiB of 15.6 GiB; "
        "swap free 0.0 GiB of 4.0 GiB\n"
    )


def test_a_host_without_those_files_says_so(tmp_path: Path) -> None:
    assert loopwatch.memory_state(tmp_path / "no", tmp_path / "none") == (
        "not readable on this host\n"
    )
