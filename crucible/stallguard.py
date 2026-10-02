from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import CrucibleError
from .tomltable import check_table

ENV_VARIABLE = "HIGGS_STALL_GUARD"
OFF = "off"

KEY = "stall_guard"
NOTE_KEY = "stall_guard_note"

BASIS_DEFAULT = "default"
BASIS_MANIFEST = "manifest"

FRAMES_RANGE = (1, 10_000)
RATE_RANGE = (0.001, 100.0)
MAX_RANGE = (0.001, 1000.0)
WINDOW_RANGE = (1, 64)

_INTEGER = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"[0-9]+(\.[0-9]+)?")

_BLOCK_REQUIRED: dict[str, type] = {
    "frames": int,
    "rate": object,
    "max": object,
    "window": int,
}


class StallGuardError(CrucibleError):
    ...


@dataclass(frozen=True)
class StallGuard:
    frames: int
    rate: float
    max: float
    window: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "rate": self.rate,
            "max": self.max,
            "window": self.window,
        }


DEFAULT = StallGuard(frames=37, rate=1.0, max=20.0, window=16)

DEFAULT_NOTE = (
    "Crucible's default for every Higgs v3 voice that states no stall_guard "
    "(campaign 2026-10-02-stall-guard). A Higgs stall is codebook 0 emitting "
    "the same silence code frame after frame and never leaving: at top-k 50 / "
    "top-p 0.95 the exit tokens are cut to zero, so the state is absorbing "
    "(the no-bed Mistborn MM02 render: 98 % of frames repeated the previous "
    "cb0 code, 5 unique codes in 90 s). Past `frames` counted frames whose cb0 "
    "code is already among the last `window` cb0 codes, each of those codes "
    "loses min(max, rate * (run - frames)) of logit before temperature/top-k/"
    "top-p, and the model's own next choice takes over; nothing is forced. "
    "37 frames is ~1.5 s at 25 fps. Accepted 2026-10-02 (training-pc-1) from "
    "the A/B through Crucible on the PC, the stalling no-bed Mistborn ckpt-3510, "
    "Mutineers' Moon bank x 16 per arm: generated pauses over 5 s 5 -> 0, the "
    "longest 35.3 s -> 4.5 s, the dead air the render-side cap trims 108 s -> "
    "48 s, coverage 0.984 -> 0.985, speed -2.5 % (inside run noise). (37, 0.5, "
    "20, 8) reached 4.2 s with 65 s trimmed; the ~2 s target was dropped "
    "because the leftovers sit inside the voice's own pause range (corpus max "
    "4.44 s)."
)


def _number(where: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StallGuardError(f"{where}: {key} must be a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise StallGuardError(f"{where}: {key} must be finite, got {value}")
    return float(value)


def _in_range(where: str, key: str, value: float, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise StallGuardError(
            f"{where}: {key} must be in [{low:g}, {high:g}], got {value:g}"
        )


def checked(where: str, frames: Any, rate: Any, max_: Any, window: Any) -> StallGuard:
    for key, value in (("frames", frames), ("window", window)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise StallGuardError(
                f"{where}: {key} must be an integer, got {type(value).__name__}"
            )
    rate_f = _number(where, "rate", rate)
    max_f = _number(where, "max", max_)
    _in_range(where, "frames", frames, FRAMES_RANGE)
    _in_range(where, "rate", rate_f, RATE_RANGE)
    _in_range(where, "max", max_f, MAX_RANGE)
    _in_range(where, "window", window, WINDOW_RANGE)
    return StallGuard(frames=frames, rate=rate_f, max=max_f, window=window)


def check_block(where: str, value: Any) -> StallGuard | None:
    if value is False:
        return None
    if value is True:
        raise StallGuardError(
            f"{where}: {KEY} = true says nothing a missing key does not: an "
            "absent stall_guard IS the default guard. State the table "
            "{ frames, rate, max, window } to change it, or false to turn it off"
        )
    if not isinstance(value, dict):
        raise StallGuardError(
            f"{where}: {KEY} must be a table {{ frames, rate, max, window }} or "
            f"false, got {type(value).__name__}"
        )
    check_table(f"{where} {KEY}", value, _BLOCK_REQUIRED, {}, error=StallGuardError)
    return checked(
        f"{where} {KEY}", value["frames"], value["rate"], value["max"], value["window"]
    )


def _decimal(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    text = repr(float(value))
    if not _DECIMAL.fullmatch(text):
        raise StallGuardError(
            f"{value!r} does not write as a plain decimal ({text!r}); the "
            f"{ENV_VARIABLE} grammar has no exponent"
        )
    return text


def env_value(guard: StallGuard | None) -> str:
    if guard is None:
        return OFF
    return ",".join(
        (str(guard.frames), _decimal(guard.rate), _decimal(guard.max), str(guard.window))
    )


def parse_env(raw: str | None) -> StallGuard | None:
    if raw is None:
        return None
    value = raw.strip()
    if value == OFF:
        return None
    where = f"{ENV_VARIABLE}={raw!r}"
    parts = value.split(",")
    if len(parts) != 4:
        raise StallGuardError(
            f"{where}: must be 'off' or '<frames>,<rate>,<max>,<window>' "
            "(e.g. '37,0.5,20,8')"
        )
    frames_s, rate_s, max_s, window_s = parts
    for key, text, pattern in (
        ("frames", frames_s, _INTEGER),
        ("rate", rate_s, _DECIMAL),
        ("max", max_s, _DECIMAL),
        ("window", window_s, _INTEGER),
    ):
        if not pattern.fullmatch(text):
            raise StallGuardError(f"{where}: {key} {text!r} is not a plain decimal number")
    return checked(where, int(frames_s), float(rate_s), float(max_s), int(window_s))


__all__ = [
    "BASIS_DEFAULT",
    "BASIS_MANIFEST",
    "DEFAULT",
    "DEFAULT_NOTE",
    "ENV_VARIABLE",
    "FRAMES_RANGE",
    "KEY",
    "MAX_RANGE",
    "NOTE_KEY",
    "OFF",
    "RATE_RANGE",
    "StallGuard",
    "StallGuardError",
    "WINDOW_RANGE",
    "check_block",
    "checked",
    "env_value",
    "parse_env",
]
