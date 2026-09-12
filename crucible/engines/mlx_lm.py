"""mlx-lm, the `mlx-darwin` engine.

Started as `python -m mlx_lm.server` from the llm venv.

What mlx_lm.server does and does not have (measured, mlx-lm 0.31.3)
------------------------------------------------------------------
It speaks the OpenAI surface Crucible needs: `GET /v1/models`, `POST
/v1/chat/completions`, and `stream: true` framed as `data: {...}` SSE with a
`data: [DONE]` terminator.

It has **no `--served-model-name`**. `/v1/models` lists what is in the HF cache
and in `--model-path`-adjacent directories, and a chat request's `model` field is
treated as a path or repo id to load. So Crucible cannot ask mlx-lm to answer to
the id `qwen3.5-9b`; the engine only knows the directory the weights sit in.

Crucible closes that gap at the proxy, not here, and does it by checking rather
than by substituting: `/v1/openai/chat/completions` requires the request's
`model` to equal the resident Crucible id (409 `model_not_resident` otherwise),
and only then rewrites that one field to the name this engine answers to before
forwarding. The rewrite is recorded in `engine_model_name`, reported by
`GET /v1/openai/models`, so nothing is silent. On vLLM the two names are equal
and the rewrite is the identity.
"""

from __future__ import annotations

from pathlib import Path

from .base import SubprocessEngine

MODULE = "mlx_lm.server"


class MlxLmEngine(SubprocessEngine):
    name = "mlx-lm"

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        # `served_name` here is what mlx-lm will actually report, which is the
        # model directory; the caller passes that, not the Crucible id.
        return [
            str(self._python),
            "-m",
            MODULE,
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *args,
        ]
