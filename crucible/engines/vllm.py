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

# The grammar behind response_format json_schema. vLLM 0.29's "auto" picks xgrammar, whose
# rule for a string with `maxLength` admits no escape sequence at all: a capped string can
# never hold a newline, a quote or a backslash. B-Side's describe caps its lyrics at 2,400
# chars, so on the PC every song came back as one paragraph per section while the same
# request on the Mac (mlx-lm) and uncapped on the PC wrote the lines (2026-10-05).
# llguidance 1.7.6, in the same env, accepts the escapes capped and uncapped (measured with
# both matchers on Qwen3.5's tokenizer); tests/test_chat_admission.py pins the choice.
STRUCTURED_OUTPUTS_ARGS: tuple[str, ...] = (
    "--structured-outputs-config",
    '{"backend": "guidance"}',
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

    decide_items_basis = (
        "vLLM 0.29.0 batches concurrent requests itself (up to --max-num-seqs) and "
        "reuses the shared state through its prefix cache, so the items go as one "
        "request each after the state is sent alone"
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
            *STRUCTURED_OUTPUTS_ARGS,
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
