from __future__ import annotations

from pathlib import Path
from typing import Any

from ..enginespec import VLLM_ENGINE
from ..errors import EngineError
from .base import SubprocessEngine, int_flag, plan_flags

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


class VllmEngine(SubprocessEngine):
    name = VLLM_ENGINE

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

    @classmethod
    def load_args(
        cls,
        spec: Any,
        weights_dir: Path,
        context: int,
        plan: Any,
        *,
        card_flags: tuple[str, ...] = (),
        source: str = "",
    ) -> list[str]:
        return [
            *spec.engine_args,
            "--max-model-len",
            str(context),
            *DECIDE_ARGS,
            *card_flags,
            *plan_flags(plan),
        ]

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
