"""Engines: one managed subprocess per resident model.

`build_engine()` is the only place an engine name becomes a class. A manifest
naming an engine this build does not have is refused by name, never substituted
for a different one.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    Engine,
    EngineError,
    SubprocessEngine,
    find_free_port,
    logs_dir,
)
from .mlx_lm import MlxLmEngine
from .vllm import VllmEngine

ENGINES: dict[str, type[SubprocessEngine]] = {
    VllmEngine.name: VllmEngine,
    MlxLmEngine.name: MlxLmEngine,
}


def engine_log_path(home: Path, model_id: str) -> Path:
    return logs_dir(home) / f"engine-{model_id}.log"


def build_engine(engine_name: str, python: Path, log_path: Path) -> SubprocessEngine:
    """The engine class for this name, instantiated. Refuses unknown names."""
    cls = ENGINES.get(engine_name)
    if cls is None:
        raise EngineError(
            f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
        )
    return cls(python=python, log_path=log_path)


def engine_model_name(engine_name: str, model_dir: Path, model_id: str) -> str:
    """The name *the engine* will answer to for this model.

    vLLM is told `--served-model-name <crucible id>`, so the two agree. mlx-lm has
    no such flag and reports the model directory it was given; the proxy rewrites
    the one `model` field after checking it against the resident Crucible id (see
    `crucible/engines/mlx_lm.py`).
    """
    if engine_name == VllmEngine.name:
        return model_id
    if engine_name == MlxLmEngine.name:
        return str(model_dir)
    raise EngineError(
        f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
    )


__all__ = [
    "ENGINES",
    "Engine",
    "EngineError",
    "MlxLmEngine",
    "SubprocessEngine",
    "VllmEngine",
    "build_engine",
    "engine_log_path",
    "engine_model_name",
    "find_free_port",
    "logs_dir",
]
