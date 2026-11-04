"""The Higgs stall guard: a row stuck on one silence code is nudged off it.

Owen, 2026-10-02 (training-pc, then crucible-pc): a runaway silence. Codebook 0
repeats a silence code frame after frame, and top-k / top-p cut every exit token
to zero, so nothing in the sampler can ever leave it - Mistborn MM02 rendered 90 s
with 98 % of frames identical to the one before. The render-door pause cap
(`serve/pause_cap.py`) makes such a chunk listenable after the fact; this keeps
the model from getting stuck in the first place. training-pc's prototype on the
sglang stack took the no-bed Mistborn A/B from 5 pauses over 5 s to 0, and the
longest pause from 108.6 s to 4.7 s, with coverage unchanged. The ring buffer
below is the refinement meant to bring 4.7 s toward 2 s.

THE CONTRACT, identical on every stack that implements it (crucible-pc owns the
Crucible side and the PC sglang-omni patch; this file is the MLX arm):

  HIGGS_STALL_GUARD="<frames>,<rate>,<max>,<window>"  read ONCE, at engine start.
  Unset or exactly `off` = no guard (the grammar is the PC patch's - see
  `stall_guard_from_env`). Crucible always sets it explicitly; the recommended
  value is "37,0.5,20,8". A malformed value is a startup error naming the
  variable.

  Per row: a ring R of the last <window> sampled codebook-0 codes, and `run`.

  Each frame, BEFORE temperature / top-k / top-p (so greedy is covered too), for
  a row that is active, past the delay window and not winding down after EOC:
  if run > frames, the cb0 logit of every code in R is lowered by
  min(max, rate * (run - frames)). Nothing is forced: the model's own next
  choice takes over.

  AFTER sampling, on a STEADY step (the row was counted, and this step did not
  finish it): if the cb0 code is already in R, run += 1 and away = 0. If it is
  not, a row whose guard is ENGAGED (run > frames) HOLDS its run and counts
  away += 1 - a return to a ring code is then penalised at once - until away
  reaches ESCAPE_FRAMES (10, 0.4 s), when the row has left: run = 0, away = 0,
  R emptied; a row not engaged just has run = 0. Then the code is pushed into R,
  evicting the oldest. (v2, training-pc 2026-10-02: v1 zeroed the run on ONE
  escaped frame, the model snapped straight back to the stuck code and sat out
  another `frames` before the guard engaged again - three cycles were the
  5.3-5.8 s pauses on the Mac arm. Holding only once ENGAGED leaves a row that
  was never stalling exactly as v1 had it.) On a step that is NOT steady - the
  delay window, the EOC wind-down, the step that finishes the row - run = 0 and
  R is emptied (crucible-pc, 2026-10-02: RESET), so a wind-down never carries a
  penalty and a new row starts clean. That is ONE line in `observe` to flip.

COST. Only cb0, and only for a row whose run is past `frames`: a row that is not
stalling touches no logits at all, and a stalling one scatters into at most
<window> entries - never a full-codebook bias per step, which is what cost the
prototype ~11 % of its speed.

PURE PYTHON. No mlx here, so the arithmetic is tested with fake logits on any
machine; `mlx_backend` does the one scatter.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass

ENV = 'HIGGS_STALL_GUARD'
#: The value Crucible sets (crucible-pc, 2026-10-02). Stated for the docs and the
#: tests; an UNSET variable is OFF, never this.
RECOMMENDED = '37,0.5,20,8'

#: Consecutive frames off the ring before an ENGAGED row counts as having left its
#: stall: 0.4 s at 25 fps. The PC patch's STALL_ESCAPE_FRAMES, the same number.
ESCAPE_FRAMES = 10


@dataclass(frozen=True)
class StallGuard:
    """The guard's four numbers."""
    frames: int
    rate: float
    max: float
    window: int

    def penalty(self, run: int) -> float:
        """How far each code in R is lowered for a row at `run`; 0 at or below `frames`."""
        if run <= self.frames:
            return 0.0
        return min(self.max, self.rate * (run - self.frames))


#: The PC sampler patch's grammar, exactly (crucible-pc, 2026-10-02), so a value reads the same on both stacks.
_GRAMMAR = re.compile(r'([0-9]+),([0-9]+(?:\.[0-9]+)?),([0-9]+(?:\.[0-9]+)?),([0-9]+)')
_RANGES = (('frames', 1, 10000), ('rate', 0.001, 100.0), ('max', 0.001, 1000.0), ('window', 1, 64))


def stall_guard_from_env(environ=None) -> StallGuard | None:
    """The guard `HIGGS_STALL_GUARD` asks for; None when it is UNSET or exactly `off`.

    THE SAME GRAMMAR AS THE PC (crucible-pc, 2026-10-02): after the surrounding whitespace is stripped the value is
    exactly `off` or `frames,rate,max,window` - frames and window `[0-9]+`, rate and max `[0-9]+(.[0-9]+)?` - within
    frames 1-10000, rate 0.001-100, max 0.001-1000, window 1-64. Anything else, the EMPTY string included, is refused
    naming the variable: one setting must not mean two things on two machines.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV)
    if raw is None:
        return None
    value = raw.strip()
    if value == 'off':
        return None
    m = _GRAMMAR.fullmatch(value)
    if m is None:
        raise ValueError(
            f'{ENV}={raw!r} is not "off" or "<frames>,<rate>,<max>,<window>" (e.g. {RECOMMENDED!r}): whole numbers '
            'for frames and window, plain decimals for rate and max, no spaces or signs.')
    numbers = (int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)))
    for (name, low, high), number in zip(_RANGES, numbers):
        if not low <= number <= high:
            raise ValueError(f'{ENV}={raw!r}: {name} is {number:g}, outside {low:g}-{high:g}.')
    frames, rate, top, window = numbers
    return StallGuard(frames=frames, rate=rate, max=top, window=window)

def counted(state, num_codebooks: int) -> bool:
    """Is this row counted on the step about to be taken - active, past the
    delay window, not winding down? Read from the sampler state BEFORE the step
    (mlx_audio `HiggsSamplerState`: `delay_count < n` is the delay window,
    `eoc_countdown is not None` the wind-down)."""
    return (not state.generation_done
            and state.delay_count >= num_codebooks
            and state.eoc_countdown is None)


class RowStall:
    """One row's ring R, its run, and how long an engaged row has been off R."""

    def __init__(self, guard: StallGuard):
        self.guard = guard
        self.ring: deque[int] = deque(maxlen=guard.window)
        self.run = 0
        self.away = 0

    def penalty(self) -> float:
        return self.guard.penalty(self.run)

    def codes(self) -> list[int]:
        """The distinct codes in R - each lowered once, never once per repeat."""
        return sorted(set(self.ring))

    def observe(self, code: int, steady: bool) -> None:
        """Record this step's sampled cb0 code; `steady` per the module docstring."""
        if not steady:
            # RESET (crucible-pc, 2026-10-02). To flip to "untouched", replace
            # the next line with `return`.
            self.run = 0; self.away = 0; self.ring.clear()  # noqa: E702 - the one line
            return
        if code in self.ring:
            self.run += 1
            self.away = 0
        elif self.run > self.guard.frames:
            self.away += 1
            if self.away >= ESCAPE_FRAMES:
                self.run = 0
                self.away = 0
                self.ring.clear()
        else:
            self.run = 0
        self.ring.append(int(code))

