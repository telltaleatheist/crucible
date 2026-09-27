"""vLLM, the `cuda-linux` engine.

Started as `python -m vllm.entrypoints.openai.api_server` from the llm venv, so
the server process itself never imports torch. The args are the manifest's
`engine_args` plus the two Crucible always sets: `--served-model-name <id>`, so
the engine answers to Crucible's model id rather than a filesystem path, and
`--max-model-len <context>`, so the context the manifest promises is the context
the engine actually allows. That context is `ModelManifest.context_for(backend)`
— this backend's own number when it declares one, the model's otherwise.
"""

from __future__ import annotations

from pathlib import Path

from ..backend import BF16
from .base import EngineError, SubprocessEngine, int_flag

MODULE = "vllm.entrypoints.openai.api_server"

#: `--max-logprobs`, composed by `Residency._engine_args` (PHASE22 section 2.6).
#: 26 letters plus `decide.LABEL_MARGIN`'s 4 is 30; 32 is that rounded to a
#: power of two, and it is the engine's own cap on a request, not a cost — vLLM
#: 0.29.0 defaults it to 20 (`vllm/config/model.py` L250), which would make a
#: 17-option choice unreadable with its margin and a 21-option one a 400.
MAX_LOGPROBS = 32

#: `--logprobs-mode`, stated rather than defaulted. 0.29.0's default IS
#: `raw_logprobs` (`vllm/config/model.py` L255), and the flag is here anyway: a
#: build that defaulted to a `processed_*` mode would report the distribution
#: AFTER temperature, and a decision is sent at temperature 0, so every answer
#: would read as certain. Raw means "before any logits processor"
#: (the same file, L256-262).
LOGPROBS_MODE = "raw_logprobs"

#: What a decision needs from the engine, in the order they are appended.
#: `--enable-prompt-tokens-details` (a `FrontendArgs` bool, so argparse's
#: `BooleanOptionalAction`: `vllm/entrypoints/launchers/cli_args.py` L132,
#: `vllm/engine/arg_utils.py` L387-389) is what makes
#: `usage.prompt_tokens_details.cached_tokens` a number; without it
#: `_make_prompt_tokens_details` returns None
#: (`vllm/entrypoints/openai/chat_completion/serving.py` L90-108) and a
#: decision's `cached_tokens` is null.
DECIDE_ARGS: tuple[str, ...] = (
    "--max-logprobs",
    str(MAX_LOGPROBS),
    "--logprobs-mode",
    LOGPROBS_MODE,
    "--enable-prompt-tokens-details",
)


#: The environment every vLLM process Crucible starts runs under — the resident
#: engine below, and the `asr` job type's Qwen3-ASR worker
#: (`crucible/jobs/asr/qwen.py`), which runs the same vLLM out of the same llm
#: env in-process. ONE OWNER, because each line below is a measured failure on
#: Owen's PC and a second copy is a copy that misses the next one.
ENVIRONMENT: dict[str, str] = {
    # vLLM's usage-stats ping is an outbound call Crucible never makes
    # on the operator's behalf.
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    # Without this, no model loads under WSL2 at all.
    #
    # vLLM 0.29.0's V2 model runner (`Using V2 Model Runner`, the
    # default on this build) keeps its request state in a UVA buffer —
    # page-locked host memory the GPU addresses directly — and raises
    # `RuntimeError: UVA is not available` in `UvaBuffer.__init__` if it
    # cannot have one. Under WSL, `CudaPlatformBase.is_pin_memory_available`
    # returns `envs.VLLM_WSL2_ENABLE_PIN_MEMORY`, which defaults to 0. So
    # on Owen's PC the 9B died 45 s into its first load, with the reason
    # 60 lines above the 40 the failure quotes (measured 2026-09-12).
    #
    # This is not a workaround for a broken host: pinned memory works
    # here. Measured in the pinned llm env on WSL2 kernel 6.6.87.1,
    # driver 591.86 — `torch.zeros(..., pin_memory=True).is_pinned()` is
    # True, `get_accelerator_view_from_cpu_tensor` returns a cuda:0 view
    # that reads back what the host wrote, and a non_blocking copy
    # arrives. vLLM's default is conservative for WSL2 in general, not
    # true of this one.
    #
    # It is also not a blanket claim about every host: vLLM reads this
    # variable only under WSL, and only after its own kernel gate
    # (>= 4.19.121). On bare-metal Linux nothing reads it; on a WSL2
    # kernel too old to pin, the gate refuses before it is consulted; and
    # on a host where pinning is somehow unavailable anyway, the engine
    # fails to start and says so rather than running degraded.
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
    # The llm env has no CUDA compiler, and FlashInfer's sampler wants one.
    #
    # vLLM defaults `VLLM_USE_FLASHINFER_SAMPLER` to True and reaches
    # FlashInfer's top-k/top-p kernel during warm-up. That kernel is not
    # in the wheel: FlashInfer JIT-builds it on first use, and the build
    # ends `RuntimeError: Could not find nvcc and default
    # cuda_home='/usr/local/cuda' doesn't exist` — measured on Owen's PC
    # 2026-09-12, after the KV cache had already been allocated, so the
    # engine died two minutes into an otherwise healthy load.
    #
    # That is a property of the env, not of the host: `crucible install
    # llm` builds `~/.crucible/envs/llm` from pinned pip wheels
    # (PHASE2-LLM.md section 2), and a CUDA toolkit is not one of them.
    # A WSL2 CUDA install ships the driver shim and no nvcc at all.
    #
    # So Crucible asks for the sampler that needs no compiler. If the
    # recipe ever gains a CUDA compiler, delete this line and measure
    # what FlashInfer's sampler is worth; do not delete it before, because
    # the failure it prevents is a load that dies at the last step.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


#: What the model catalog names this engine (`manifests.BACKEND_ENGINES`), and
#: what the asr catalog names it (`asrmodels.VLLM_ENGINE`). The two catalogs
#: spell it the same; `card_needs` is the one question that reads it from both.
ENGINE_NAME = "vllm"

#: The dtype vLLM runs when a block states none: `--dtype` defaults to `auto`.
AUTO_DTYPE = "auto"


def dtype_of(engine_args: "tuple[str, ...] | list[str]") -> str:
    """The `--dtype` a block's engine args state, or `auto` when they state none.

    Both spellings argparse takes: `--dtype bfloat16` and `--dtype=bfloat16`.
    """
    args = list(engine_args)
    for index, arg in enumerate(args):
        if arg == "--dtype" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--dtype="):
            return arg.partition("=")[2]
    return AUTO_DTYPE


def card_needs(spec: object) -> tuple[str, ...]:
    """What a backend block needs from the CARD for vLLM to start it at all.

    Fresh-install #48, 2026-09-26. Read in vLLM 0.29.0, the llm env's pin, not
    recalled:

    * **A STATED `bfloat16` IS A HARD REFUSAL BELOW 8.0.** `v1/worker/
      gpu_worker.py` L414 calls `check_if_supports_dtype` in `init_device`,
      and `platforms/cuda.py` L622-640 raises *"Bfloat16 is only supported on
      GPUs with compute capability of at least 8.0"* — after the process has
      started, before a weight is read. The qwen3.5 blocks state
      `--dtype bfloat16` in their `engine_args`, and the Qwen3-ASR blocks'
      `dtype = "bfloat16"` reaches `LLM(dtype=...)` the same way
      (`jobs/asr/qwen_worker.py`, `load_vllm`), so on a Turing card every one
      of them is a job that fails at its first line.
    * **`auto` is not.** `config/model.py` L2268-2309 (`_resolve_auto_dtype`)
      falls back from a bfloat16 checkpoint to float16 with a warning — the
      same two bytes a parameter, so the memory arithmetic is unchanged — and
      `dots-ocr` and the 27B-4bit state no `--dtype`. Whether a bf16-trained
      checkpoint reads pages or translates as well in float16 has not been
      measured; that is a quality question for the ladder
      (docs/PROPOSAL-GPU-LADDER.md), not a refusal this function may invent.
    * **FlashAttention 2 is not needed.** Below 8.0 vLLM selects its Triton
      attention (`v1/attention/backends/triton_attn.py` L373-374 allows every
      capability; FlashInfer is floored at 8.0 in `flashinfer.py` L506-510).

    The feature names are `backend.FEATURE_FLOORS`', so the need and the card's
    answer are compared by one vocabulary. Every other engine answers `()`:
    nothing this build runs elsewhere refuses a card by its generation (the
    torch workers emulate bf16 below 8.0 — `torch.cuda.is_bf16_supported`,
    `including_emulation` — which is slower and not a refusal).

    `spec` is any catalog's backend block: a model block carries its dtype in
    `engine_args`, an asr block in `dtype`.
    """
    if getattr(spec, "engine", None) != ENGINE_NAME:
        return ()
    stated = getattr(spec, "dtype", None)
    dtype = stated if stated is not None else dtype_of(getattr(spec, "engine_args", ()))
    return (BF16,) if dtype == "bfloat16" else ()


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

    #: HOW MANY SEQUENCES vLLM SCHEDULES AT ONCE: `--max-num-seqs`, read off
    #: the resident engine's argv (2026-09-24). Until today vLLM stated no
    #: concurrency, so `chat.max_in_flight` was null on the PC and Foundry's
    #: placement fell back to `CRUCIBLE_CHAT_CONCURRENCY = 4` against an engine
    #: every one of whose blocks runs `--max-num-seqs 16` — a quarter of its
    #: batch. The door now admits `--max-num-seqs + 1`: the batch, and one
    #: request ready for the next free slot (past the batch vLLM queues
    #: internally, which is safe, but the wait belongs on the wire). `start()`
    #: refuses an argv without it, as mlx-lm refuses its own flags.
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
        """Refuse an argv that does not state its batch.

        vLLM picks its own default by device and usage context
        (`vllm/engine/arg_utils.py` L2676-2760, 0.29.0, read in the WSL llm env
        2026-09-24), and the chat door reads its admission off this flag; a block that leaves it to vLLM would be a door
        with no stated width. Misconfiguration, refused by name.
        """
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
