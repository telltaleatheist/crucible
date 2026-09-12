"""vLLM, the `cuda-linux` engine.

Started as `python -m vllm.entrypoints.openai.api_server` from the llm venv, so
the server process itself never imports torch. The args are the manifest's
`engine_args` plus the two Crucible always sets: `--served-model-name <id>`, so
the engine answers to Crucible's model id rather than a filesystem path, and
`--max-model-len <context_default>`, so the context the manifest promises is the
context the engine actually allows.
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
        # vLLM's usage-stats ping is an outbound call Crucible never makes on the
        # operator's behalf.
        return {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
