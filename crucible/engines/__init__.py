"""Engines: one managed subprocess per resident model.

`build_engine()` is the only place an engine name becomes a class. A manifest
naming an engine this build does not have is refused by name, never substituted
for a different one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    Engine,
    EngineError,
    SubprocessEngine,
    find_free_port,
    logs_dir,
)
from .mlx_lm import MlxLmEngine
from .narrator import NarratorEngine
from .vllm import VllmEngine

if TYPE_CHECKING:  # `crucible.narratorvoices` imports this package; no cycle at runtime
    from ..narratorvoices import VoicesDocument

ENGINES: dict[str, type[SubprocessEngine]] = {
    VllmEngine.name: VllmEngine,
    MlxLmEngine.name: MlxLmEngine,
}

#: The narrator engines this build can start. The keys are narrator's OWN engine
#: ids — what `NARRATOR_ENGINE` takes and what a voice manifest's
#: `narrator_engine` names — rather than Crucible engine names, because there is
#: one class here and the id is a parameter to it: narrator decides which stack
#: it runs underneath itself, and Crucible's business is only which env and which
#: value of that variable.
#:
#: It is a set rather than a `dict[str, type]` for that reason, and it is checked
#: against `crucible/voices.py`'s own table by a test, so a manifest can never
#: name an engine this file cannot start.
#:
#: A SET OF ONE since Owen's ruling of 2026-09-14 removed `orpheus`: Higgs is
#: the frontier and Orpheus will never be served here, so Crucible stops naming
#: it. `crucible/voices.py`'s `NARRATOR_ENGINE_SAMPLING` carries the ruling and
#: what a second engine has to add.
NARRATOR_ENGINES: frozenset[str] = frozenset({"higgs-v3"})


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
    narrator_engine: str,
    python: Path,
    log_path: Path,
    *,
    serving_stack: str | None,
    max_num_seqs: int | None,
    voices: "VoicesDocument | None",
) -> NarratorEngine:
    """The engine that serves a voice. Refuses an engine this build cannot run.

    One class for both narrator engines, because from Crucible's side they differ
    only in which env the interpreter comes from, what `NARRATOR_ENGINE` says and
    — since 2026-09-13 — what the server underneath is configured with, and
    — since 2026-09-14 — which document names its voices.

    THE THREE EXTRA FACTS HAVE DIFFERENT OWNERS, which is why they arrive as
    three arguments rather than one object: `serving_stack` belongs to the ENV
    RECIPE (`jobenv.tts_env` — vllm-omni on `cuda-linux` because that is what
    the recipe installs, None where narrator starts no server), `max_num_seqs`
    belongs to the VOICE MANIFEST (`[voice.serving]`), and `voices` is the
    document `crucible/narratorvoices.py` wrote from that manifest and the
    pulled weights for THIS load (None for an engine that resolves no voice by
    name — `narratorvoices.DOCUMENT_READERS`). All are mandatory keywords:
    `None` is a real answer and a default would hide a caller that forgot.
    """
    if narrator_engine not in NARRATOR_ENGINES:
        raise EngineError(
            f"unknown narrator engine {narrator_engine!r}; this build can start "
            f"{sorted(NARRATOR_ENGINES)}"
        )
    return NarratorEngine(
        narrator_engine=narrator_engine,
        python=python,
        log_path=log_path,
        serving_stack=serving_stack,
        max_num_seqs=max_num_seqs,
        voices=voices,
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
    "NARRATOR_ENGINES",
    "Engine",
    "EngineError",
    "MlxLmEngine",
    "NarratorEngine",
    "SubprocessEngine",
    "VllmEngine",
    "build_engine",
    "build_voice_engine",
    "engine_log_path",
    "engine_model_name",
    "find_free_port",
    "logs_dir",
]
