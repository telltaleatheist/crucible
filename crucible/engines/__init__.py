from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import EngineError
from ..narratorengines import NARRATOR_ENGINES, VoicesDocumentView
from .base import STOP_TIMEOUT_SECONDS, SubprocessEngine, find_free_port, int_flag, logs_dir
from .llama_server import LlamaServerEngine
from .mlx_lm import MlxLmEngine
from .mlx_vlm import MlxVlmEngine
from .narrator import EngineWouldNotStop, NarratorEngine
from .vllm import VllmEngine

ENGINES: dict[str, type[SubprocessEngine]] = {
    VllmEngine.name: VllmEngine,
    MlxLmEngine.name: MlxLmEngine,
    MlxVlmEngine.name: MlxVlmEngine,
    LlamaServerEngine.name: LlamaServerEngine,
}


def engine_class(engine_name: str) -> type[SubprocessEngine]:
    cls = ENGINES.get(engine_name)
    if cls is None:
        raise EngineError(
            f"unknown engine {engine_name!r}; this build has {sorted(ENGINES)}"
        )
    return cls


def engine_log_path(home: Path, model_id: str) -> Path:
    return logs_dir(home) / f"engine-{model_id}.log"


ChatAdmission = tuple[int | None, str | None]


def chat_admission(
    engine_name: str, engine_args: "list[str] | tuple[str, ...]"
) -> ChatAdmission:
    cls = engine_class(engine_name)
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
    served: bool
    max_logprobs: int | None
    basis: str


def decide_reading(engine_name: str) -> DecideReading:
    cls = engine_class(engine_name)
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


@dataclass(frozen=True)
class DecideItemsReading:
    batched: bool
    basis: str
    questions: bool = False
    """The question form is read through the items route too: every question
    a row of one batched request, no prime, no request per question."""


def decide_items_reading(engine_name: str) -> DecideItemsReading:
    cls = engine_class(engine_name)
    basis = cls.decide_items_basis
    if basis is None:
        raise EngineError(
            f"{engine_name} states no decide_items_basis. Whether an engine reads a "
            "list of items in one batched request is read from its source, and the "
            "reading says where"
        )
    questions = cls.decide_questions_batched
    if questions and not cls.decide_items_batched:
        raise EngineError(
            f"{engine_name} states decide_questions_batched with no batched items "
            "route; the question form can only ride a route the engine has"
        )
    if questions and cls.decide_questions_basis is None:
        raise EngineError(
            f"{engine_name} states decide_questions_batched and no "
            "decide_questions_basis; why the question form is better read as items "
            "is a measurement, and the reading says which"
        )
    return DecideItemsReading(
        batched=cls.decide_items_batched, basis=basis, questions=questions
    )


LIKELIHOOD_ROUTES = ("items", "prompt-logprobs")


@dataclass(frozen=True)
class LikelihoodReading:
    route: str | None
    """`items`, `prompt-logprobs`, or None when the engine cannot score candidates."""
    images: bool
    basis: str


def likelihood_reading(engine_name: str) -> LikelihoodReading:
    cls = engine_class(engine_name)
    basis = cls.decide_likelihood_basis
    route = cls.decide_likelihood_route
    if basis is None:
        raise EngineError(
            f"{engine_name} states no decide_likelihood_basis. Whether an engine returns "
            "the log-probability of every token of a candidate is read from its source, "
            "and the reading says where, or why it cannot"
        )
    if route is not None and route not in LIKELIHOOD_ROUTES:
        raise EngineError(
            f"{engine_name} states decide_likelihood_route {route!r}; the door reads "
            f"{list(LIKELIHOOD_ROUTES)}"
        )
    if route == "items" and not cls.decide_items_batched:
        raise EngineError(
            f"{engine_name} states the items route for likelihood and has no batched "
            "items route"
        )
    if route is None and cls.decide_likelihood_images:
        raise EngineError(
            f"{engine_name} states likelihood with images and scores no candidates"
        )
    return LikelihoodReading(route=route, images=cls.decide_likelihood_images, basis=basis)


def build_engine(
    engine_name: str,
    python: Path,
    log_path: Path,
    *,
    library_dirs: tuple[Path, ...] = (),
) -> SubprocessEngine:
    cls = engine_class(engine_name)
    return cls(python=python, log_path=log_path, library_dirs=library_dirs)


def build_voice_engine(
    narrator_engine: str,
    python: Path,
    log_path: Path,
    *,
    serving_stack: str | None,
    max_num_seqs: int | None,
    mem_fraction: float | None,
    context_length: int | None,
    stall_guard: str | None,
    voices: "VoicesDocumentView | None",
    mlx_total_bytes: int | None,
) -> NarratorEngine:
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
        stall_guard=stall_guard,
        voices=voices,
        mlx_total_bytes=mlx_total_bytes,
    )


def engine_model_name(engine_name: str, model_dir: Path, model_id: str) -> str:
    return engine_class(engine_name).served_name(model_dir, model_id)


def start_engine(
    engine: SubprocessEngine,
    weights_dir: Path,
    served: str,
    port: int,
    args: list[str],
    say: Callable[[str], None],
    timeout: float,
    *,
    cancelled: Callable[[], bool],
    confirm: Callable[[], Any] | None = None,
) -> None:
    try:
        engine.start(weights_dir, served, port, args)
        engine.ready(timeout, on_progress=say, cancelled=cancelled)
        if confirm is not None:
            confirm()
    except BaseException as start_failure:
        try:
            engine.stop()
        except EngineError as stop_failure:
            raise EngineError(
                f"{start_failure}\n...and stopping it also failed: {stop_failure}"
            ) from start_failure
        raise


def concurrency_flag(engine_name: str) -> str | None:
    """The flag an engine reads its batch width from; None for an engine whose width
    is a constant (llama-server runs one slot)."""
    return engine_class(engine_name).chat_concurrency_flag


def stated_concurrency(spec: Any) -> int | None:
    """How many requests at once the manifest starts this model with."""
    flag = concurrency_flag(spec.engine)
    return None if flag is None else int_flag(spec.engine_args, flag)


def with_concurrency(spec: Any, args: list[str], width: int, model_id: str) -> list[str]:
    """`args` with the engine's concurrency flag set to `width`: the person's
    [llm.concurrency] for this model on this server. It may only lower what the
    manifest states, because the manifest's number is what the model's memory was
    planned for (every in-flight sequence holds its own KV)."""
    flag = concurrency_flag(spec.engine)
    if flag is None:
        raise EngineError(
            f"concurrency_not_settable: config [llm.concurrency] sets {model_id} to "
            f"{width}, but {spec.engine} runs one request at a time and has no flag "
            f"for it. `crucible models concurrency {model_id} default` removes it"
        )
    stated = int_flag(args, flag)
    if stated is None:
        raise EngineError(
            f"{spec.engine} for {model_id} was given no {flag} ({args}); there is "
            "nothing for [llm.concurrency] to lower"
        )
    if width > stated:
        raise EngineError(
            f"concurrency_above_manifest: config [llm.concurrency] sets {model_id} to "
            f"{width} at once, above the {stated} its manifest was sized for. Set "
            f"{stated} or fewer, or `crucible models concurrency {model_id} default`"
        )
    kept: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        if arg.startswith(flag + "="):
            continue
        kept.append(arg)
    return [*kept, flag, str(width)]


def engine_load_args(
    manifest: Any,
    spec: Any,
    weights_dir: Path,
    plan: Any,
    *,
    context: int,
    card_args: tuple[str, ...] = (),
    concurrency: int | None = None,
) -> list[str]:
    args = engine_class(spec.engine).load_args(
        spec,
        weights_dir,
        context,
        plan,
        card_flags=card_args,
        source=manifest.path.name,
    )
    if concurrency is None:
        return args
    return with_concurrency(spec, args, concurrency, manifest.id)


__all__ = [
    "ChatAdmission",
    "chat_admission",
    "concurrency_flag",
    "stated_concurrency",
    "with_concurrency",
    "DecideItemsReading",
    "DecideReading",
    "LIKELIHOOD_ROUTES",
    "LikelihoodReading",
    "decide_items_reading",
    "decide_reading",
    "likelihood_reading",
    "ENGINES",
    "STOP_TIMEOUT_SECONDS",
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
    "engine_class",
    "engine_load_args",
    "engine_log_path",
    "engine_model_name",
    "find_free_port",
    "start_engine",
    "logs_dir",
]
