from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .. import envpatches
from ..envpatches import PatchError
from ..errors import EngineError
from .base import SubprocessEngine, int_flag, weights_subject_id

MODULE = "mlx_lm"
SUBCOMMAND = "server"

CONFIRM_POLL_SECONDS = 5.0

REQUIRED_FLAGS: tuple[str, ...] = (
    "--decode-concurrency",
    "--prompt-concurrency",
    "--prompt-cache-size",
)


class MlxLmEngine(SubprocessEngine):
    name = "mlx-lm"

    chat_concurrency_flag = "--decode-concurrency"
    chat_concurrency_basis = (
        "mlx-lm 0.31.3 batches on its one generation thread: BatchGenerator with "
        "completion_batch_size = --decode-concurrency (mlx_lm/server.py "
        "L813-830, taken whenever is_batchable and no seed, L685-686; read on "
        "the Mac Studio 2026-09-24)"
    )

    chat_prefill = False
    chat_prefill_basis = (
        "mlx-lm 0.31.3's server renders every chat with add_generation_prompt=True "
        "and reads no continue_final_message (mlx_lm/server.py _tokenize L573-586, "
        "read on the Mac Studio 2026-10-10): a final assistant message is closed "
        "and a new answer opened after it, so a prefill would not be continued"
    )

    decide_logprobs = True
    max_logprobs = 40
    decide_basis = (
        "mlx-lm 0.31.3 returns choices[0].logprobs.content[].top_logprobs on its "
        "chat route; its validator caps top_logprobs at 11 (mlx_lm/server.py "
        "L1245, read on the Mac Studio 2026-09-23) and Crucible's "
        "mlx-lm-top-logprobs-40 patch raises it to 40, checked at engine start"
    )

    decide_items_batched = True
    decide_items_basis = (
        "mlx-lm 0.31.3 answers one prompt per request; Crucible's "
        "mlx-lm-decide-items patch (applied by this engine at start) adds POST "
        "/v1/crucible/items, which runs engines/items_forward.py on the generation "
        "thread: the shared state once, then every item as a batched row over a "
        "copy of its cache"
    )

    decide_questions_batched = True
    decide_questions_basis = (
        "measured on the Mac Studio M1 Ultra, qwen3.5-9b bf16, 2026-10-01: MLX's "
        "bf16 matmul leaves its matrix-vector kernel past one row (1.6 ms vs 6.0 ms "
        "for 0.8 GB of weights at 1 vs 2-64 rows), so any forward of 2-64 tokens "
        "costs ~115 ms and a chat request is three of them (mlx-lm prefills the "
        "system, user and thinking-tail segments apart) plus a token and a "
        "pipelined token nobody reads; the items route reads every question as "
        "a row of one forward over a state it keeps between decisions"
    )

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        missing = [flag for flag in REQUIRED_FLAGS if int_flag(args, flag) is None]
        if missing:
            raise EngineError(
                "mlx_lm_flags_unstated: this mlx-darwin block's engine_args "
                f"state no {', '.join(missing)} ({args}). Each is a memory "
                "decision for THIS model -- every in-flight sequence holds its "
                "own KV and recurrent state -- so Crucible states it in the "
                "manifest rather than inherit mlx-lm's default "
                "(engines/mlx_lm.py, REQUIRED_FLAGS)"
            )
        if self._python.is_file():
            self._require_patched(self._python.parent.parent)
        super().start(model_dir, served_name, port, args)

    def _require_patched(self, env_dir: Path) -> None:
        self_applied = envpatches.SELF_APPLIED_LLM_PATCHES
        gated = [p for p in envpatches.LLM_PATCHES if p not in self_applied]
        try:
            for patch in gated:
                envpatches.require_applied(patch, env_dir)
            envpatches.ensure_applied(
                self_applied, env_dir, self._python,
                scripts_dir=envpatches.LLM_SCRIPTS_DIR,
            )
            for patch in self_applied:
                envpatches.require_applied(patch, env_dir)
        except PatchError as exc:
            raise EngineError(
                f"llm_env_unpatched: {exc}. This engine states what it "
                f"serves (max_logprobs {self.max_logprobs}, logprobs "
                "computed in float32, the decide door's items route) because "
                "of the llm env's patches and will not start without every "
                "one; run `crucible env patch llm` (or `crucible install llm "
                "--force`) and load again"
            ) from exc

    @classmethod
    def served_name(cls, weights_dir: Path, model_id: str) -> str:
        return str(Path(weights_dir).resolve())

    def subject_id(self, model_dir: Path, served_name: str) -> str:
        return weights_subject_id(model_dir)

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        return [
            str(self._python),
            "-m",
            MODULE,
            SUBCOMMAND,
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *args,
        ]

    def confirm(
        self,
        deadline: float,
        on_progress: Callable[[str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        url = f"{self.base_url}/v1/chat/completions"
        payload = json.dumps(
            {
                "model": self._served_name,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode("utf-8")
        attempt = 0
        last: str = "no attempt made"
        while True:
            self.raise_if_cancelled(cancelled)
            if self._process is not None and self._process.poll() is not None:
                raise EngineError(
                    f"{self.name} exited {self._process.returncode} while loading "
                    "its weights. " + self.log_report()
                )
            request = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                remaining = max(1.0, deadline - time.monotonic())
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if body.get("choices"):
                    if on_progress is not None:
                        on_progress(
                            f"{self.name} generated its first token; the weights "
                            "are in memory"
                        )
                    return
                last = f"the engine answered without choices: {body}"
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"

            if time.monotonic() >= deadline:
                raise EngineError(
                    f"{self.name} answered /v1/models but could not generate a "
                    f"token before the timeout: {last}. " + self.log_report()
                )
            attempt += 1
            if on_progress is not None:
                on_progress(
                    f"{self.name} reading weights into memory "
                    f"({attempt * CONFIRM_POLL_SECONDS:.0f}s so far)"
                )
            time.sleep(CONFIRM_POLL_SECONDS)
