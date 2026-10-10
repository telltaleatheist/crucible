"""`[video_desktop]`: what the Mac's video engine runs with so the desktop stays usable.

The keys, their defaults and their ranges live here, where both the engine's reader
(crucible/jobs/video/__init__.py, desktop_settings) and the Settings door read them. The
reader skips a value out of range and runs the default; Settings refuses one by name
(check_key), so nothing a person types through the page is silently not used.
"""

from __future__ import annotations

from typing import Any

from .backend import MLX_DARWIN
from .errors import ConfigError

TABLE = "video_desktop"

# Only the Mac's engine (ltx-2-mlx) reads the table.
BACKEND = MLX_DARWIN

# What the Mac's engine runs with when [video_desktop] does not say otherwise: the values
# for "responsive while somebody works" on an M1 Ultra (docs/internals/video.md,
# "Keeping the desktop responsive: [video_desktop]"). They are on by default; only
# `enabled = false` in the table turns them off.
DEFAULTS: dict[str, Any] = {
    "max_tile_tokens": 16000,
    "tile_spatial": 1,
    "tile_overlap": 2,
    "dit_eval_every": 1,
    "low_ram": True,
    "mlx_max_ops_per_buffer": 20,
    "mlx_max_mb_per_buffer": 40,
    "gpu_duty_pct": 85,
}

MAXIMUM: dict[str, int] = {"gpu_duty_pct": 100}

MINIMUM: dict[str, int] = {
    "gpu_duty_pct": 10,
    "max_tile_tokens": 0,
    "tile_spatial": 1,
    "tile_overlap": 0,
    "dit_eval_every": 0,
    "mlx_max_ops_per_buffer": 1,
    "mlx_max_mb_per_buffer": 1,
}

ENABLED = "enabled"
LOW_RAM = "low_ram"

# Judges a run rather than shaping it: the mean GPU busy a render should stay at or under.
GPU_BUSY_TARGET = "gpu_busy_target_pct"
GPU_BUSY_TARGET_DEFAULT = 90.0

# What each key does, in the words Settings shows beside it.
MEANING: dict[str, str] = {
    ENABLED: "keep the desktop responsive while a clip renders (off: render flat out)",
    "max_tile_tokens": "the most of the clip one pass works on at once (0: no tiling)",
    "tile_spatial": "how many pieces each frame is cut into",
    "tile_overlap": "how much neighbouring pieces overlap",
    "dit_eval_every": "pause for the desktop after this many transformer blocks (0: never)",
    LOW_RAM: "stream the model from disk instead of holding all of it in memory",
    "mlx_max_ops_per_buffer": "the most GPU operations sent in one go",
    "mlx_max_mb_per_buffer": "the most memory, in MB, one GPU send may touch",
    "gpu_duty_pct": "the share of time the render may keep the GPU busy, in percent",
    GPU_BUSY_TARGET: "the mean GPU busy, in percent, a render is judged against",
}

KEYS: tuple[str, ...] = (ENABLED, *DEFAULTS, GPU_BUSY_TARGET)


def int_in_range(key: str, value: Any) -> bool:
    """Whether `value` is a whole number `key` takes; the one range both doors read."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and MINIMUM[key] <= value <= MAXIMUM.get(key, value)
    )


def busy_target_in_range(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 100


def range_words(key: str) -> str:
    if key in (ENABLED, LOW_RAM):
        return "true or false"
    if key == GPU_BUSY_TARGET:
        return "a number above 0 and at most 100"
    if key in MAXIMUM:
        return f"a whole number from {MINIMUM[key]} to {MAXIMUM[key]}"
    return f"a whole number of at least {MINIMUM[key]}"


def check_key(key: str, value: Any) -> Any:
    """`value` for `key`, or a refusal that names the key and its range."""
    where = f"[{TABLE}] {key}"
    if key not in KEYS:
        raise ConfigError(
            f"video_desktop_unknown_key: {where} is not a key the video engine reads; "
            f"they are {list(KEYS)}"
        )
    if key in (ENABLED, LOW_RAM):
        ok = isinstance(value, bool)
    elif key == GPU_BUSY_TARGET:
        ok = busy_target_in_range(value)
    else:
        ok = int_in_range(key, value)
    if not ok:
        raise ConfigError(
            f"video_desktop_out_of_range: {where} is {range_words(key)}, got {value!r}"
        )
    return value


def rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per key for Settings: what the file says, what the engine runs, and a
    `problem` sentence for a value in the file the engine skips."""
    found: list[dict[str, Any]] = []
    for key in KEYS:
        if key == ENABLED:
            default: Any = True
        elif key == GPU_BUSY_TARGET:
            default = GPU_BUSY_TARGET_DEFAULT
        else:
            default = DEFAULTS[key]
        present = key in table
        problem = None
        if present:
            try:
                check_key(key, table[key])
            except ConfigError:
                problem = (
                    f"the file says {table[key]!r}, which is not {range_words(key)}, so "
                    f"the engine runs {default!r}"
                )
        found.append(
            {
                "key": key,
                "meaning": MEANING[key],
                "range": range_words(key),
                "default": default,
                "set": table[key] if present else None,
                "problem": problem,
            }
        )
    return found


__all__ = [
    "BACKEND",
    "DEFAULTS",
    "KEYS",
    "MAXIMUM",
    "MINIMUM",
    "TABLE",
    "busy_target_in_range",
    "check_key",
    "int_in_range",
    "range_words",
    "rows",
]
