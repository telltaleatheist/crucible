from __future__ import annotations

from pathlib import Path

from .base import EngineError, SubprocessEngine

SERVE_SCRIPT = Path(__file__).resolve().with_name("mlx_vlm_serve.py")

WIDTH_FLAG = "--width"


class MlxVlmEngine(SubprocessEngine):
    name = "mlx-vlm"

    decide_logprobs = False
    decide_basis = (
        "the Mac page server computes no logprobs "
        "(engines/mlx_vlm_serve.py: batch_generate(compute_logprobs=False)) and "
        "refuses a body carrying logprobs (KNOWN_FIELDS)"
    )

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
