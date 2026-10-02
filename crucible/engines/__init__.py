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


def build_engine(engine_name: str, python: Path, log_path: Path) -> SubprocessEngine:
    cls = engine_class(engine_name)
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
    confirm: Callable[[], Any] | None = None,
) -> None:
    try:
        engine.start(weights_dir, served, port, args)
        engine.ready(timeout, on_progress=say)
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


def engine_load_args(
    manifest: Any,
    spec: Any,
    weights_dir: Path,
    plan: Any,
    *,
    context: int,
    card_args: tuple[str, ...] = (),
) -> list[str]:
    return engine_class(spec.engine).load_args(
        spec,
        weights_dir,
        context,
        plan,
        card_flags=card_args,
        source=manifest.path.name,
    )


__all__ = [
    "ChatAdmission",
    "chat_admission",
    "DecideItemsReading",
    "DecideReading",
    "decide_items_reading",
    "decide_reading",
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
