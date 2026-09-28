from __future__ import annotations

from pathlib import Path

from ..errors import EngineError
from .base import SubprocessEngine, weights_subject_id
from .mlx_vlm_serve import MAX_TOP_LOGPROBS

SERVE_SCRIPT = Path(__file__).resolve().with_name("mlx_vlm_serve.py")

WIDTH_FLAG = "--width"


class MlxVlmEngine(SubprocessEngine):
    name = "mlx-vlm"

    decide_logprobs = True
    max_logprobs = MAX_TOP_LOGPROBS
    decide_basis = (
        "Crucible's Mac server (engines/mlx_vlm_serve.py) answers a question body "
        "with choices[0].logprobs.content[].top_logprobs from mlx-vlm 0.7.1's "
        "BatchGenerator(compute_logprobs=True, top_logprobs_k=k), which argsorts the "
        "whole vocabulary and caps nothing (mlx_vlm/generate/ar.py "
        "GenerationBatch._step L1229-1254, PromptProcessingBatch.generate "
        "L2182-2267, read on the Mac Studio 2026-09-28); the server caps "
        "top_logprobs at MAX_TOP_LOGPROBS to match mlx-lm and normalises them in "
        "float32 (logits_in_float32)"
    )

    decide_items_batched = True
    decide_items_basis = (
        "Crucible's Mac server (engines/mlx_vlm_serve.py) answers POST "
        "/v1/crucible/items with engines/items_forward.py: the images embedded "
        "once with the shared state, then every item as a batched row over a copy "
        "of its cache, at the rope positions the lone question would have"
    )

    @classmethod
    def served_name(cls, weights_dir: Path, model_id: str) -> str:
        return str(weights_dir)

    def subject_id(self, model_dir: Path, served_name: str) -> str:
        return weights_subject_id(model_dir)

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        if WIDTH_FLAG not in args:
            raise EngineError(
                f"{self.name} was started without {WIDTH_FLAG}: the manifest's "
                f"[backends.mlx-darwin] engine_args must state how many pages "
                "one batch carries (models/dots-ocr.toml cites the measurement). "
                "This engine does not default it"
            )
        return [
            str(self._python),
            str(SERVE_SCRIPT),
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *args,
        ]
