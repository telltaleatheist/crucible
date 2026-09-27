from __future__ import annotations

from .backend import BF16, CUDA_GRAPHS, VLLM_STARTS, CardFacts

VLLM_ENGINE = "vllm"

AUTO_DTYPE = "auto"

BF16_FALLBACK_DTYPE = "float16"

UNSTATED_ENGINE_CONCURRENCY = 16


def flag_value(args: "list[str] | tuple[str, ...]", flag: str) -> str | None:
    found: str | None = None
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            found = args[index + 1]
        elif arg.startswith(flag + "="):
            found = arg.split("=", 1)[1]
    return found


def dtype_of(engine_args: "tuple[str, ...] | list[str]") -> str:
    stated = flag_value(engine_args, "--dtype")
    return AUTO_DTYPE if stated is None else stated


def declared_dtype(spec: object) -> str | None:
    stated = getattr(spec, "dtype", None)
    if stated is None:
        stated = dtype_of(getattr(spec, "engine_args", None) or ())
    return None if stated == AUTO_DTYPE else stated


def stated_dtype(spec: object) -> str:
    declared = declared_dtype(spec)
    return AUTO_DTYPE if declared is None else declared


def _runs_on_vllm(spec: object) -> bool:
    return getattr(spec, "engine", None) == VLLM_ENGINE


def bf16_fallback(spec: object) -> str | None:
    if not _runs_on_vllm(spec):
        return None
    return BF16_FALLBACK_DTYPE if declared_dtype(spec) == "bfloat16" else None


def dtype_on(stated: str, fallback: str | None, card: "CardFacts | None") -> str:
    if fallback is not None and card is not None and card.has(BF16) is False:
        return fallback
    return stated


def run_dtype(spec: object, card: "CardFacts | None") -> str:
    return dtype_on(stated_dtype(spec), bf16_fallback(spec), card)


def card_args(spec: object, card: "CardFacts | None") -> tuple[str, ...]:
    if not _runs_on_vllm(spec):
        return ()
    args: list[str] = []
    dtype = run_dtype(spec, card)
    if dtype != stated_dtype(spec):
        args += ["--dtype", dtype]
    if card is not None and card.has(CUDA_GRAPHS) is False:
        args.append("--enforce-eager")
    return tuple(args)


def card_needs(spec: object) -> tuple[str, ...]:
    if not _runs_on_vllm(spec):
        return ()
    return (VLLM_STARTS,)
