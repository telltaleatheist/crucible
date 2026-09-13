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

from .base import SubprocessEngine

MODULE = "vllm.entrypoints.openai.api_server"


class VllmEngine(SubprocessEngine):
    name = "vllm"

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
        return {
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
