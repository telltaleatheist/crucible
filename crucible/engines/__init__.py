"""Engines: one managed subprocess per resident model.

`build_engine()` is the only place an engine name becomes a class. A manifest
naming an engine this build does not have is refused by name, never substituted
for a different one.

ONE ENGINE PER (BACKEND, CLASS FAMILY) since 2026-09-14, which is why there are
three classes for two backends: `cuda-linux` serves both text and pages with
vLLM, and `mlx-darwin` serves text with `mlx-lm` and pages with `mlx-vlm`.
`crucible/manifests.py` owns the pairing and refuses every other one; this
module owns only "which class does this name mean".
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    STOP_TIMEOUT_SECONDS,
    Engine,
    EngineError,
    SubprocessEngine,
    find_free_port,
    logs_dir,
)
from .llama_server import LlamaServerEngine
from .mlx_lm import MlxLmEngine
from .mlx_vlm import MlxVlmEngine
from .narrator import EngineWouldNotStop, NarratorEngine
from .vllm import VllmEngine

if TYPE_CHECKING:  # `crucible.narratorvoices` imports this package; no cycle at runtime
    from ..narratorvoices import VoicesDocument

#: Every server class this build can start, by the name a manifest uses.
#:
#: There are three and not two because `mlx-darwin` needs a SECOND class for
#: page reading: `mlx-lm` is a text server and cannot be handed an image, so
#: `crucible/manifests.py` maps (mlx-darwin, pages) to `mlx-vlm` and this is
#: where that name becomes a class. `cuda-linux` maps both families to vLLM,
#: which is why it needs only one.
#:
#: `residency.load()` picks by `spec.engine` and always has, so the family
#: never appears in the residency at all — the manifest names the engine, the
#: loader has already refused every pairing that is not allowed, and this dict
#: turns the surviving name into a process.
ENGINES: dict[str, type[SubprocessEngine]] = {
    VllmEngine.name: VllmEngine,
    MlxLmEngine.name: MlxLmEngine,
    MlxVlmEngine.name: MlxVlmEngine,
    LlamaServerEngine.name: LlamaServerEngine,
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
    mlx_total_bytes: int | None,
) -> NarratorEngine:
    """The engine that serves a voice. Refuses an engine this build cannot run.

    One class for both narrator engines, because from Crucible's side they differ
    only in which env the interpreter comes from, what `NARRATOR_ENGINE` says and
    — since 2026-09-13 — what the server underneath is configured with, and
    — since 2026-09-14 — which document names its voices.

    THE FOUR EXTRA FACTS HAVE DIFFERENT OWNERS, which is why they arrive as
    four arguments rather than one object: `serving_stack` belongs to the ENV
    RECIPE (`jobenv.tts_env` — sglang-omni on `cuda-linux` since Owen's ruling
    of 2026-09-15, because that is what the recipe installs; None where narrator
    starts no server), `max_num_seqs`
    belongs to the VOICE MANIFEST (`[voice.serving]`), `voices` is the
    document `crucible/narratorvoices.py` wrote from that manifest and the
    pulled weights for THIS load (None for an engine that resolves no voice by
    name — `narratorvoices.DOCUMENT_READERS`), and `mlx_total_bytes` belongs to
    the MACHINE (`accelerator.probe_unified_memory`, None off the in-process
    arm). All are mandatory keywords: `None` is a real answer and a default
    would hide a caller that forgot.

    THE LAST ONE IS THE MACHINE'S AND NOT THE VOICE'S, which is why it is not in
    the manifest beside `max_num_seqs`. The served arm's width is a property of
    the card the voice was certified on and travels with the voice; the
    in-process arm's is a property of how much unified memory THIS Mac has, and
    the same voice renders at 64 rows on a 64 GB machine and 24 on a small one
    (`narrator.MLX_TIERS`). Putting it in the manifest would make one number
    answer two questions.
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
        mlx_total_bytes=mlx_total_bytes,
    )


def engine_model_name(engine_name: str, model_dir: Path, model_id: str) -> str:
    """The name *the engine* will answer to for this model.

    vLLM is told `--served-model-name <crucible id>`, so the two agree. Neither
    MLX engine has such a flag and both report the model directory they were
    given; the proxy rewrites the one `model` field after checking it against
    the resident Crucible id (see `crucible/engines/mlx_lm.py`).

    THE TWO MLX ENGINES DIFFER BY ONE `resolve()`, and it is measured rather
    than assumed. mlx-lm reports `str(Path(--model).resolve())`, so this has to
    resolve too or readiness would compare two spellings of one path. mlx-vlm
    stores `model_path` exactly as handed over and reports that
    (`get_cached_model`, exercised on the Mac Studio 2026-09-14:
    `reports_the_path_verbatim` was true for the unresolved string), so
    resolving here would introduce the very mismatch the resolve prevents on
    the other one.
    """
    if engine_name == VllmEngine.name:
        return model_id
    if engine_name == MlxLmEngine.name:
        return str(Path(model_dir).resolve())
    if engine_name == LlamaServerEngine.name:
        # `--alias <crucible id>` (PHASE15-HOST.md 7.4, item 3): llama-server
        # would otherwise name the model after the GGUF file, so `/v1/models`
        # would answer `dots.ocr-Q8_0.gguf`. With the alias the name IS
        # the Crucible id, which makes readiness "the name equals the id this
        # server started" and the proxy verbatim — no rewrite, unlike mlx-lm.
        return model_id
    if engine_name == MlxVlmEngine.name:
        return str(model_dir)
    raise EngineError(
        f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
    )


__all__ = [
    "ENGINES",
    "NARRATOR_ENGINES",
    "STOP_TIMEOUT_SECONDS",
    "Engine",
    "EngineError",
    "EngineWouldNotStop",
    "LlamaServerEngine",
    "MlxLmEngine",
    "MlxVlmEngine",
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
