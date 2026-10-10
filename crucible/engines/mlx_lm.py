from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

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

PREFILL_STEP_FLAG = "--prefill-step-size"

MLX_LM_PREFILL_STEP = 2048
"""mlx-lm 0.31.3's own --prefill-step-size default: the most tokens one prefill
step reads, and the most a step here is ever given."""

EVAL_BUDGET_FLOPS = 2 * 27e9 * 256
"""The GPU work one evaluation may hold: a prompt is read in steps of at most
this many FLOPs (2 x params x tokens). macOS cannot preempt a running Metal
workload, and while one runs the window server's compositing can wait for ALL of
it: measured on the Mac Studio M1 Ultra (2026-10-10), a 2048-token step of
qwen3.8-27b-8bit is 12-13 s of GPU and the desktop froze for 12.6 s at a time
(the window server's own IPC unanswered, a 60 Hz Metal client blocked 13.2 s),
while the 9B's 3.7 s steps never froze it. The budget is the 27B at 256 tokens:
1.59 s per step, at the same throughput as 2048 (164 vs 162-170 tok/s; 128
tokens cost 8%, 64 cost 18%). Every model's step is derived from it, so each
step is about the same stretch of GPU time (docs/internals/engines-and-capability.md,
"Prefill steps hold the GPU")."""


def attention_flops_per_position(weights_dir: Path) -> float:
    """FLOPs one new token spends per token already in the cache: QK^T and the
    weighted sum over V, 4 x heads x head_dim per full-attention layer (a linear-
    attention layer's cost does not grow with the cache). Read from the weights'
    config.json, the only place the architecture is stated."""
    path = Path(weights_dir) / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(
            f"model_config_unreadable: {path} cannot be read ({exc}); the prefill step "
            "is sized from the model's attention layers"
        ) from None
    text = config.get("text_config", config)
    try:
        layers = int(text["num_hidden_layers"])
        heads = int(text["num_attention_heads"])
        head_dim = int(text.get("head_dim") or int(text["hidden_size"]) // heads)
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineError(
            f"model_config_unreadable: {path} states no {exc} for its attention layers"
        ) from None
    types = text.get("layer_types")
    if isinstance(types, list):
        full = sum(1 for kind in types if kind == "full_attention")
    else:
        full = layers // int(text.get("full_attention_interval") or 1)
    return 4.0 * full * heads * head_dim


def prefill_step(params_b: float, attention_per_position: float, context: int) -> int:
    """Tokens per prefill step: EVAL_BUDGET_FLOPS of work for a step at the deepest
    position the context allows (2 x params per token, plus attention over every
    token already cached), never more than mlx-lm's own step. Measured on the 27B
    at a 24k-token prompt: 2048-token steps grew 12.2 -> 13.7 s along it."""
    if params_b <= 0 or context <= 0:
        raise EngineError(
            f"prefill_step: params_b and context must be positive, got {params_b}, {context}"
        )
    per_token = 2 * params_b * 1e9 + attention_per_position * context
    return max(1, min(MLX_LM_PREFILL_STEP, int(EVAL_BUDGET_FLOPS / per_token)))


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

    structured_output_formats = frozenset({"json_object", "json_schema"})
    structured_output_fields = frozenset({"structured_outputs"})
    structured_output_basis = (
        "stock mlx-lm 0.31.3 reads no response_format (mlx_lm/server.py builds its "
        "logits processors from logit_bias and the penalties only, "
        "_make_logits_processors L414-423); Crucible's mlx-lm-structured-output patch "
        "(applied by this engine at start) reads response_format json_object and "
        "json_schema and structured_outputs json, json_object, regex, choice and "
        "grammar, compiles them with llguidance as vLLM 0.29.0's guidance backend "
        "does, gives each constrained sequence its own matcher as a logits processor, "
        "and refuses by name anything else, guided_* and grammar included "
        "(engines/structured_mlx.py)"
    )
    json_whitespace_compact = True
    json_whitespace_basis = (
        "Crucible's mlx-lm-structured-output patch compiles a json schema with "
        "llguidance's grammar_from_json_schema(schema, defaults={whitespace_flexible: "
        "true}) as vLLM does (engines/structured_mlx.py compile_grammar), and "
        "llguidance 1.8.0 (the Mac llm env's pin) takes the schema's own x-guidance "
        "options over those defaults (measured on the Mac Studio, 2026-10-10), so the "
        "door writes x-guidance.whitespace_flexible false into the schema; a "
        "json_object goes as the schema {type: object}, which is what the patch "
        "compiles a json_object to (structured_mlx.ANY_OBJECT)"
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

    decide_likelihood_route = "items"
    decide_likelihood_basis = (
        "Crucible's items route (engines/items_forward.py, ITEMS_VERSION 5) reads a "
        "candidates body: every candidate's prompt is the chat template's open "
        "assistant reply (continue_final_message), the shared state runs once, each "
        "question's context once over it, every candidate's first token is read "
        "off that question's last position, and the rest of a candidate is a "
        "batched row whose every token is scored off the head in float32"
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

    @classmethod
    def model_args(
        cls, manifest: Any, args: list[str], weights_dir: Path, context: int
    ) -> list[str]:
        """The prefill step, derived from the model's size and attention at the
        context it is loaded with (`prefill_step`). One owner: a manifest that
        states the flag itself is refused."""
        if int_flag(args, PREFILL_STEP_FLAG) is not None:
            raise EngineError(
                f"prefill_step_stated: {manifest.path.name} states {PREFILL_STEP_FLAG} "
                f"({args}); Crucible derives it from params_b so every model's step "
                "holds the GPU about as long (engines/mlx_lm.py, EVAL_BUDGET_FLOPS). "
                "Remove it from the manifest"
            )
        step = prefill_step(
            manifest.params_b, attention_flops_per_position(weights_dir), context
        )
        return [*args, PREFILL_STEP_FLAG, str(step)]

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        if int_flag(args, PREFILL_STEP_FLAG) is None:
            raise EngineError(
                f"mlx_lm_prefill_step_unset: this argv states no {PREFILL_STEP_FLAG} "
                f"({args}); mlx-lm's own {MLX_LM_PREFILL_STEP} would hold the GPU for "
                "up to 13 s per step on a 27B and freeze the desktop. Start the "
                "engine through engine_load_args, which derives it"
            )
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
                "computed in float32, the decide door's items route, "
                "response_format enforced with llguidance) because "
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
