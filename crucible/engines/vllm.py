from __future__ import annotations

from pathlib import Path

from ..backend import BF16, CUDA_GRAPHS, VLLM_STARTS, CardFacts
from .base import EngineError, SubprocessEngine, int_flag

MODULE = "vllm.entrypoints.openai.api_server"

MAX_LOGPROBS = 32

LOGPROBS_MODE = "raw_logprobs"

DECIDE_ARGS: tuple[str, ...] = (
    "--max-logprobs",
    str(MAX_LOGPROBS),
    "--logprobs-mode",
    LOGPROBS_MODE,
    "--enable-prompt-tokens-details",
)


ENVIRONMENT: dict[str, str] = {
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


ENGINE_NAME = "vllm"

AUTO_DTYPE = "auto"


def dtype_of(engine_args: "tuple[str, ...] | list[str]") -> str:
    args = list(engine_args)
    for index, arg in enumerate(args):
        if arg == "--dtype" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--dtype="):
            return arg.partition("=")[2]
    return AUTO_DTYPE


def stated_dtype(spec: object) -> str:
    stated = getattr(spec, "dtype", None)
    if stated is not None:
        return stated
    return dtype_of(getattr(spec, "engine_args", ()))


BF16_FALLBACK_DTYPE = "float16"


def bf16_fallback(spec: object) -> str | None:
    if getattr(spec, "engine", None) != ENGINE_NAME:
        return None
    return BF16_FALLBACK_DTYPE if stated_dtype(spec) == "bfloat16" else None


def run_dtype(spec: object, card: "CardFacts | None") -> str:
    fallback = bf16_fallback(spec)
    if fallback is not None and card is not None and card.has(BF16) is False:
        return fallback
    return stated_dtype(spec)


def card_args(spec: object, card: "CardFacts | None") -> tuple[str, ...]:
    if getattr(spec, "engine", None) != ENGINE_NAME:
        return ()
    args: list[str] = []
    dtype = run_dtype(spec, card)
    if dtype != stated_dtype(spec):
        args += ["--dtype", dtype]
    if card is not None and card.has(CUDA_GRAPHS) is False:
        args.append("--enforce-eager")
    return tuple(args)


def card_needs(spec: object) -> tuple[str, ...]:
    if getattr(spec, "engine", None) != ENGINE_NAME:
        return ()
    return (VLLM_STARTS,)


class VllmEngine(SubprocessEngine):
    name = "vllm"

    decide_logprobs = True
    max_logprobs = MAX_LOGPROBS
    decide_basis = (
        "vLLM 0.29.0's /v1/chat/completions returns "
        "choices[0].logprobs.content[].top_logprobs as {token, logprob, bytes} "
        "(vllm/entrypoints/openai/chat_completion/protocol.py L81-95), capped by "
        f"--max-logprobs, which Crucible starts it with at {MAX_LOGPROBS} and "
        f"--logprobs-mode {LOGPROBS_MODE}"
    )

    chat_concurrency_flag = "--max-num-seqs"
    chat_concurrency_basis = (
        "vLLM 0.29.0 schedules at most --max-num-seqs sequences per step "
        "(vllm/engine/arg_utils.py L586, L2409-2429, read 2026-09-24; every "
        "cuda-linux block states it, and the "
        "number of CUDA graphs captured follows it) and queues the rest"
    )

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        if int_flag(args, "--max-num-seqs") is None:
            raise EngineError(
                f"vllm_flags_unstated: this cuda-linux block's engine_args state "
                f"no --max-num-seqs ({args}). It is the batch the chat door "
                "admits against and the CUDA graphs vLLM captures; state it in "
                "the manifest (engines/vllm.py)"
            )
        super().start(model_dir, served_name, port, args)

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        return [
            str(self._python),
            "-m",
            MODULE,
            "--model",
            str(model_dir),
            "--served-model-name",
            served_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *args,
        ]

    def environment(self) -> dict[str, str]:
        return dict(ENVIRONMENT)
