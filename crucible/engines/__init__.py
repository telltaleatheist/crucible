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

if TYPE_CHECKING:
    from ..narratorvoices import VoicesDocument

ENGINES: dict[str, type[SubprocessEngine]] = {
    VllmEngine.name: VllmEngine,
    MlxLmEngine.name: MlxLmEngine,
    MlxVlmEngine.name: MlxVlmEngine,
    LlamaServerEngine.name: LlamaServerEngine,
}

NARRATOR_ENGINES: frozenset[str] = frozenset({"higgs-v3"})


def engine_log_path(home: Path, model_id: str) -> Path:
    return logs_dir(home) / f"engine-{model_id}.log"


ChatAdmission = tuple[int | None, str | None]


def chat_admission(
    engine_name: str, engine_args: "list[str] | tuple[str, ...]"
) -> ChatAdmission:
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
    if engine_name == VllmEngine.name:
        return model_id
    if engine_name == MlxLmEngine.name:
        return str(Path(model_dir).resolve())
    if engine_name == LlamaServerEngine.name:
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
