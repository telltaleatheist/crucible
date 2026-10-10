from __future__ import annotations

from pathlib import Path

from ..errors import EngineError
from .base import SubprocessEngine, weights_subject_id
from .mlx_vlm_serve import MAX_TOP_LOGPROBS

SERVE_SCRIPT = Path(__file__).resolve().with_name("mlx_vlm_serve.py")

WIDTH_FLAG = "--width"


class MlxVlmEngine(SubprocessEngine):
    name = "mlx-vlm"

    chat_prefill = False
    chat_prefill_basis = (
        "Crucible's Mac server (engines/mlx_vlm_serve.py) answers a page or a "
        "decision, and refuses a body field it does not know (KNOWN_FIELDS): it "
        "has no way to be told to continue a message"
    )

    structured_output_basis = (
        "Crucible's Mac server (engines/mlx_vlm_serve.py) reads a page or a decision "
        "and refuses any body field it does not know (KNOWN_FIELDS, QUESTION_FIELDS), "
        "response_format among them: it enforces no structured output. mlx-vlm 0.7.1 "
        "ships an llguidance processor (mlx_vlm/structured.py), which this server does "
        "not use"
    )

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

    decide_likelihood_route = "items"
    decide_likelihood_images = True
    decide_likelihood_basis = (
        "Crucible's Mac server (engines/mlx_vlm_serve.py) answers a candidates body on "
        "POST /v1/crucible/items with engines/items_forward.py: mlx-vlm 0.7.1's "
        "apply_chat_template passes continue_final_message through to the tokenizer "
        "(mlx_vlm/prompt_utils.py get_chat_template, read on the Mac Studio "
        "2026-10-10), the images are embedded once with the shared state, and each "
        "candidate is scored at every token from a copy of its cache"
    )

    decide_likelihood_prompt_basis = (
        "Crucible's Mac reader (engines/mlx_vlm_serve.py parse_items_job) refuses the "
        "prompt form: it renders every context through mlx-vlm's chat template to place "
        "the images, and no reranker manifest names an mlx-vlm block"
    )

    embed_basis = (
        "Crucible's Mac reader (engines/mlx_vlm_serve.py parse_items_job) refuses an "
        "embed request: it serves the vision forms, and no embedding manifest names an "
        "mlx-vlm block"
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
