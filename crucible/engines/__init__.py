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


def build_voice_engine(
    narrator_engine: str, python: Path, log_path: Path
) -> SubprocessEngine:
    """The engine that serves a voice — **not written yet**.

    `crucible/engines/narrator.py` is the next builder's file and is where this
    becomes a real construction: a `SubprocessEngine` subclass that runs
    `python -m narrator.serve` from the tts env for `narrator_engine`, proves
    readiness from the `ready{device,backend}` line narrator prints on stdout
    (which is what `SubprocessEngine.announced_ready()` is the seam for), speaks
    newline-delimited JSON over its pipes, and tears down its SGLang-Omni or MLX
    engine on SIGTERM. `ENGINES` above gains an entry then, and this function
    becomes a lookup in it exactly as `build_engine` is.

    It refuses here rather than returning a stub that appears to work. A fake
    engine that answers `load-voice` successfully and produces no audio is
    exactly the kind of thing that ships: every test above it goes green and the
    failure surfaces as a silent book.
    """
    raise NotImplementedError(
        f"crucible cannot start narrator for the {narrator_engine!r} engine: "
        "crucible/engines/narrator.py is not written yet (PHASE3-TTS.md section "
        "4). Everything up to the spawn — the voice manifests, the refusals, the "
        "residency and the two lifecycle jobs — is in place and tested; this is "
        "the seam it stops at."
    )


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
        # mlx-lm's /v1/models reports `str(Path(--model).resolve())`, so this must
        # be resolved too or readiness would compare two spellings of one path.
        return str(Path(model_dir).resolve())
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
    "build_voice_engine",
    "engine_log_path",
    "engine_model_name",
    "find_free_port",
    "logs_dir",
]
