"""`[llm.concurrency]`: how many requests one chat model runs at once on this server.

A model's manifest states the width its memory was planned for (`--decode-concurrency`
on mlx-lm, `--max-num-seqs` on vLLM). A person may run it narrower here: on the Mac
Studio the 27B at 8 at once held the GPU at 100% and starved the displays (Owen,
2026-10-09: "running fewer requests at once would be fine … lets make it a
configuration thing"). Only lower, never higher, than the manifest; the width is read
when the model loads (crucible/engines/__init__.py, with_concurrency), so a model
already on the card keeps the width it was started with until it is loaded again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, rewrite_config
from .engines import concurrency_flag, stated_concurrency
from .engines.base import int_flag
from .errors import ConfigError
from .manifests import ManifestError, load_all_manifests


def _settable(backend_kind: str) -> dict[str, tuple[str, int]]:
    """id -> (display, manifest width) for every chat model on this backend whose engine
    reads its width from a flag."""
    try:
        manifests = load_all_manifests()
    except ManifestError as exc:
        raise ConfigError(str(exc)) from exc
    found: dict[str, tuple[str, int]] = {}
    for model_id, manifest in manifests.items():
        spec = manifest.backends.get(backend_kind)
        if spec is None or concurrency_flag(spec.engine) is None:
            continue
        stated = stated_concurrency(spec)
        if stated is not None:
            found[model_id] = (manifest.display or model_id, stated)
    return found


def rows(config: Config, resident: Any = None) -> list[dict[str, Any]]:
    """One row per settable model: the manifest's width, the person's (null = the
    manifest's), and, for the model on the card, the width its engine runs now."""
    running: tuple[str, int] | None = None
    if resident is not None:
        flag = concurrency_flag(resident.engine)
        if flag is not None:
            width = int_flag(resident.engine_args, flag)
            if width is not None:
                running = (resident.model_id, width)
    return [
        {
            "model": model_id,
            "display": display,
            "manifest": stated,
            "set": config.concurrency_for(model_id),
            "running": running[1] if running and running[0] == model_id else None,
        }
        for model_id, (display, stated) in sorted(_settable(config.backend_kind).items())
    ]


def set_concurrency(config: Config, model: str, width: int | None) -> Path:
    """Write `[llm.concurrency] model = width`, or remove it (None: the manifest's)."""
    settable = _settable(config.backend_kind)
    if model not in settable:
        raise ConfigError(
            f"concurrency_not_settable: {model!r} is not a chat model with a "
            f"concurrency on {config.backend_kind}; the settable ones are "
            f"{sorted(settable)}"
        )
    stated = settable[model][1]
    if width is not None and (isinstance(width, bool) or not 1 <= width <= stated):
        raise ConfigError(
            f"concurrency_out_of_range: {model} runs 1 to {stated} requests at once "
            f"here ({stated} is what its manifest was sized for); got {width!r}"
        )
    kept = tuple((m, w) for m, w in config.llm_concurrency if m != model)
    widths = kept if width is None or width == stated else (*kept, (model, width))
    return rewrite_config(config, llm_concurrency=tuple(sorted(widths)))


__all__ = ["rows", "set_concurrency"]
