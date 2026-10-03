from __future__ import annotations

import asyncio
import io
import time

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
