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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    STOP_TIMEOUT_SECONDS,
    Engine,
    EngineError,
    SubprocessEngine,
    find_free_port,
    int_flag,
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


#: What a chat door may admit at once, and why, for one engine name.
#:
#: `(limit, basis)` where `limit` is None when this engine states no
#: concurrency — in which case the door bounds nothing and says so, exactly as
#: it behaved before 2026-09-20.
ChatAdmission = tuple[int | None, str | None]


def chat_admission(
    engine_name: str, engine_args: "list[str] | tuple[str, ...]"
) -> ChatAdmission:
    """How many chat completions this engine's door admits at once, and why.

    `engine_args` is the argv the RESIDENT engine was started with
    (`ResidentModel.engine_args`). It is read only for an engine whose
    concurrency is a flag (`SubprocessEngine.chat_concurrency_flag`, mlx-lm's
    `--decode-concurrency` since 2026-09-24), so the number the door admits is
    the number the running engine was given and cannot be a second copy of it.

    **The engine's own concurrency, PLUS ONE.** The plus one is not a margin and
    not a guess: it is the request that is ready to begin the moment the running
    one finishes, so a serial engine's single generation thread never sits idle
    between two completions. Bounding at the concurrency itself would trade one
    defect for a slower version of the same door; bounding at concurrency + 1
    keeps the engine fed while capping the wait an ADMITTED request can inherit
    at a single completion ahead of it. For a batching engine the number is its
    batch width and the plus one changes nothing that matters.

    An engine that states no concurrency is not bounded. That is deliberate:
    vLLM batches, no starvation has ever been measured against it, and a limit
    invented here would cap work nobody showed needed capping. See
    `SubprocessEngine.chat_concurrency`.
    """
    cls = ENGINES.get(engine_name)
    if cls is None:
        raise EngineError(
            f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
        )
    concurrency = cls.chat_concurrency
    basis = cls.chat_concurrency_basis
    flag = cls.chat_concurrency_flag
    if flag is not None:
        if concurrency is not None:
            raise EngineError(
                f"{engine_name} states chat_concurrency {concurrency} AND reads "
                f"its concurrency from {flag}. One number, one owner: drop the "
                "constant, the argv is what the engine runs"
            )
        if basis is None:
            raise EngineError(
                f"{engine_name} reads its concurrency from {flag} and states no "
                "chat_concurrency_basis; say where that flag's meaning was read"
            )
        concurrency = int_flag(engine_args, flag)
        if concurrency is None:
            # MISCONFIGURATION, NOT WEATHER: the engine refuses to START without
            # the flag (`MlxLmEngine.start`), so a resident engine missing it is
            # a record that does not describe what is running.
            raise EngineError(
                f"{engine_name} was started without {flag} ({list(engine_args)}); "
                "its batch width is that flag and nothing else states it"
            )
        basis = f"{basis}; this engine was started with {flag} {concurrency}"
    if concurrency is None:
        if basis is not None:
            raise EngineError(
                f"{engine_name} states a chat_concurrency_basis and no "
                "chat_concurrency. The basis says where a number came from and "
                "there is no number; drop it, or state the number it describes"
            )
        return (None, None)
    if basis is None:
        raise EngineError(
            f"{engine_name} states chat_concurrency {concurrency} and no "
            "chat_concurrency_basis. A concurrency with no provenance is a "
            "number somebody typed; say where it was measured"
        )
    if concurrency < 1:
        raise EngineError(
            f"{engine_name} states chat_concurrency {concurrency}, which would "
            "admit nothing"
        )
    return (concurrency + 1, basis)


@dataclass(frozen=True)
class DecideReading:
    """Whether an engine serves a decision, how many top logprobs, and why.

    `max_logprobs` None with `served` True is an engine with no small cap;
    with `served` False it means nothing and `basis` is the refusal's reason.
    """

    served: bool
    max_logprobs: int | None
    basis: str


def decide_reading(engine_name: str) -> DecideReading:
    """What the decision door may ask this engine for (PHASE22 section 2.6).

    Checked for the same consistency `chat_admission` checks: a basis is
    required either way, a served engine's cap must hold at least a yes/no
    question's two letters, and a cap on an engine that serves nothing is a
    leftover.
    """
    cls = ENGINES.get(engine_name)
    if cls is None:
        raise EngineError(
            f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
        )
    basis = cls.decide_basis
    if basis is None:
        raise EngineError(
            f"{engine_name} states no decide_basis. Whether an engine returns top "
            "logprobs is read from its source, and the reading says where"
        )
    if not cls.decide_logprobs:
        if cls.max_logprobs is not None:
            raise EngineError(
                f"{engine_name} states max_logprobs {cls.max_logprobs} and serves no "
                "decision; drop the number, or state that it serves"
            )
        return DecideReading(served=False, max_logprobs=None, basis=basis)
    if cls.max_logprobs is not None and cls.max_logprobs < 2:
        raise EngineError(
            f"{engine_name} states max_logprobs {cls.max_logprobs}, which cannot "
            "read even a yes/no question"
        )
    return DecideReading(served=True, max_logprobs=cls.max_logprobs, basis=basis)


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
    mem_fraction: float | None,
    context_length: int | None,
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

    `mem_fraction` and `context_length` (2026-09-19) belong to the VOICE
    MANIFEST too, beside `max_num_seqs` in `[voice.serving]`, and `None` for
    each means "narrator's own launcher default" — 0.60 and the Higgs builder's
    4096 — which is a number stated in a file that has an owner rather than one
    chosen here. Unlike `max_num_seqs` they go to BOTH arms: Owen ruled on
    2026-09-19 that darwin is to be configured the same way, so an MLX backend
    with no knob for one of them is narrator's refusal to make, by name, and
    never this server's to hide.

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
        mem_fraction=mem_fraction,
        context_length=context_length,
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
    "ChatAdmission",
    "chat_admission",
    "DecideReading",
    "decide_reading",
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
