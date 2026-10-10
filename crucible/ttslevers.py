"""`[tts.<engine>]`: what narrator's engine costs on this machine and how it is started.

`memory_bytes_estimate` (with `estimate_basis` and, for a declared number, its
`estimate_note`), `max_num_seqs` and its note, and the optional `mem_fraction` and
`context_length` with theirs. A voice reads them when it loads (voicerepo.merge), so a
voice already on the card keeps the numbers it was started with until it loads again;
Settings shows those and offers the reload.

Settings changes one engine's table through this module and the config reader's own
rules (config.tts_engine_record): a lever without its note, a fraction outside (0, 1)
or a declared estimate with no note is refused here exactly as the file would be.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, rewrite_config, tts_engine_record
from .errors import ConfigError

LEVER_KEYS: tuple[str, ...] = (
    "memory_bytes_estimate",
    "estimate_basis",
    "estimate_note",
    "max_num_seqs",
    "max_num_seqs_note",
    "mem_fraction",
    "mem_fraction_note",
    "context_length",
    "context_length_note",
)

UNSET = "engine_footprint_unset"
UNKNOWN = "tts_lever_unknown"
INVALID = "tts_lever_invalid"


def set_levers(config: Config, engine: str, patch: dict[str, Any]) -> Path:
    """Write `patch` into `[tts.<engine>]` (None removes an optional key) and keep the
    rest of the table and every other engine's as they were."""
    current = config.engine_footprint(engine)
    if current is None:
        raise ConfigError(
            f"{UNSET}: this config states no [tts.{engine}] table, so there is nothing "
            "to change. `crucible install tts` writes it with the numbers this build "
            "declares"
        )
    unknown = sorted(set(patch) - set(LEVER_KEYS))
    if unknown:
        raise ConfigError(
            f"{UNKNOWN}: {unknown} are not [tts.{engine}] keys; they are {list(LEVER_KEYS)}"
        )
    merged = current.to_dict()
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    try:
        written = tts_engine_record(engine, merged)
    except ConfigError as exc:
        raise ConfigError(f"{INVALID}: {exc}") from None
    footprints = tuple(
        written if entry.engine == engine else entry for entry in config.tts_engines
    )
    return rewrite_config(config, tts_engines=footprints)


__all__ = ["INVALID", "LEVER_KEYS", "UNKNOWN", "UNSET", "set_levers"]
