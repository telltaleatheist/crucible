"""mlx-lm, the `mlx-darwin` engine.

Started as `python -m mlx_lm server` from the llm env.

What mlx_lm.server has and has not (READ from its source and exercised on the
Mac Studio, mlx-lm 0.31.3)
--------------------------------------------------------------------------
It speaks the OpenAI surface Crucible needs: `GET /v1/models`, `POST
/v1/chat/completions`, and `stream: true` framed as `data: {...}` SSE with a
`data: [DONE]` terminator. Three things about it are not what the phase-2
contract assumes, and Crucible handles each explicitly rather than papering over
it:

1. **No `--served-model-name`.** `/v1/models` lists every mlx-looking repo in the
   HuggingFace cache plus, when `--model` is a path, `str(Path(--model).resolve())`.
   So the name this engine answers to is the *resolved weights directory*, never
   the Crucible id. `engines.engine_model_name()` returns that path, and the
   proxy rewrites the request's one `model` field to it — **after** checking it
   against the resident Crucible id, so a wrong name is still a 409. The
   substitution is reported in `GET /v1/openai/models` as `engine_model_name`.

2. **A chat request's `model` is a thing it will load.** `ResponseGenerator`
   calls `model_provider.load(args.model.model, ...)` per request: hand mlx-lm a
   different path or repo id and it fetches and loads it. Crucible's 409 gate is
   what stops that — the proxy only ever forwards the one directory it started
   the engine on, so this engine can never be talked into loading a second model
   behind the server's back.

3. **`/v1/models` answers before the weights are in memory.** `load_default()`
   runs on the generation thread while the HTTP server is already serving. So a
   200 from the list route does not mean the model is resident, and `ready()`
   would otherwise return with 19 GB still to read. `confirm()` closes that: a
   one-token completion, which forces the load and proves generation works. That
   is a deliberate addition to the readiness check in PHASE2-LLM.md section 3,
   which specifies `/v1/models` alone.

It also has no `--max-model-len`: the context comes from the model's own config,
so `Residency._engine_args` sends that flag to vLLM only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .. import envpatches
from ..narratorpatches import PatchError
from .base import SubprocessEngine, EngineError

#: `python -m mlx_lm.server` still runs in 0.31.3 but prints a deprecation
#: notice; `python -m mlx_lm server` is the form it asks for, and needs nothing
#: on PATH.
MODULE = "mlx_lm"
SUBCOMMAND = "server"

CONFIRM_POLL_SECONDS = 5.0


class MlxLmEngine(SubprocessEngine):
    name = "mlx-lm"

    #: ONE. mlx-lm serves HTTP on a `ThreadingHTTPServer`, so it ACCEPTS any
    #: number of chat requests at once and looks concurrent from outside — but
    #: `ResponseGenerator` has a single `self.requests = Queue()` drained by a
    #: single `self._generation_thread = Thread(target=self._generate)`
    #: (`mlx_lm/server.py:444,451`, mlx-lm 0.31.3, read in
    #: `~/.crucible/envs/llm` on the Mac Studio on 2026-09-20). Generation is
    #: strictly FIFO through that one thread, so the Nth request waits for all
    #: N-1 before it and nothing about the socket says so.
    #:
    #: That is what starved Foundry's clean pass: 12 in flight, a 300 s client
    #: deadline, and a request that had not started when the deadline passed.
    #: The accepting is what makes it dangerous — a serial engine that refused
    #: the connection would have told the client the truth immediately.
    #:
    #: AND A CLOSED SOCKET DOES NOT STOP A NON-STREAMED REQUEST HERE. Read in
    #: mlx-lm 0.31.3's `mlx_lm/server.py` (the PyPI wheel, 2026-09-24): the
    #: only thing that stops generation is `GenerationContext._should_stop`,
    #: set by `ctx.stop()` in `handle_completion`'s `finally` (L1552-1553) —
    #: i.e. when the HANDLER thread leaves, which for a non-streamed request is
    #: after its one `wfile.write` of the whole answer (L1549). Until then that
    #: thread sits in `response_queue.get()` (L1037, L1048) and never touches
    #: the socket; the prefill keepalive writes only `if self.stream` (L1413).
    #: So when Crucible cancels its request (`api._unless_the_caller_leaves`)
    #: and closes the socket, mlx-lm still prefills and answers every request it
    #: has already ACCEPTED, running or queued — and even a stop that did land
    #: is read only between tokens (`_serve_single`, L1008) or between prefill
    #: chunks (the batched path, L861). What Crucible's cancel buys on this
    #: engine is the rest: nothing more is SENT (a decision's waiting questions
    #: are taken back at its gate), the `InFlight` row closes, and the
    #: settlement's SIGTERM — which does stop a prefill — is free to come as
    #: soon as nothing else holds the card. At most `chat_admission`'s limit of
    #: requests per door is ever at the engine, so that is the whole overrun.
    chat_concurrency = 1
    chat_concurrency_basis = (
        "mlx-lm 0.31.3 generates on one thread draining one queue "
        "(mlx_lm/server.py ResponseGenerator, read on the Mac Studio "
        "2026-09-20); its ThreadingHTTPServer accepts concurrently but "
        "generation is strictly serial"
    )

    #: A DECISION IS SERVED, AT MOST FORTY LOGPROBS WIDE — BECAUSE OF A PATCH.
    #:
    #: Read in mlx-lm 0.31.3's installed `mlx_lm/server.py` in the Mac's
    #: `~/.crucible/envs/llm` on 2026-09-23 (no model run): the chat body's
    #: `logprobs`/`top_logprobs` are read at L1189-1190 and validated at L1245;
    #: `_format_top_logprobs` (L426-435) emits `{id, token, logprob}`;
    #: L1317-1321 writes `choices[0].logprobs.content = [dict(top[0],
    #: top_logprobs=top), ...]` — the same OpenAI path vLLM and llama-server
    #: answer on, so one reader.
    #:
    #: UNPATCHED, THE VALIDATOR SAYS 11: `top_logprobs int, min 0, max 11,
    #: whitelist [-1]`, and 12 is a 400 — a question with more than 7 options
    #: (K = labels + 4, PHASE22-DECIDE.md section 2.4) could not be read on the
    #: Mac at all. `envs/llm/patches/patch_mlx_lm_top_logprobs.py` raises it to
    #: 40 (26 letters + the margin of 4, with room); that validator is the only
    #: ceiling (`_format_top_logprobs` takes any `top_n`). `crucible install llm`
    #: applies it before the env's stamp, and every install/upgrade re-applies it
    #: through the installer's `env-patch-llm` step (`crucible env patch llm`)
    #: before the service starts.
    #:
    #: THE NUMBER IS TIED TO THE CHECK AT ENGINE START, not merely asserted:
    #: `start()` below runs the patch's own check against the env this engine is
    #: spawned from and refuses by name (`llm_env_unpatched`) unless it says
    #: `applied`. So a resident mlx-lm is always a patched one, and the door
    #: (`engines.decide_reading`, a class-level reading) can never be told 40 by
    #: an engine that would answer 400.
    #:
    #: Two things differ from vLLM and neither moves a letter. The token strings
    #: are RAW pieces (`tokenizer.convert_ids_to_tokens`), not decoded text,
    #: which for a bare capital letter is the same string. And the logprobs are
    #: taken AFTER the logits processors (`mlx_lm/generate.py` L409-420), which
    #: is why a decision states its own sampling and takes no manifest default
    #: (`crucible/decide.py`, `request_body`).
    decide_logprobs = True
    max_logprobs = 40
    decide_basis = (
        "mlx-lm 0.31.3 returns choices[0].logprobs.content[].top_logprobs on its "
        "chat route; its validator caps top_logprobs at 11 (mlx_lm/server.py "
        "L1245, read on the Mac Studio 2026-09-23) and Crucible's "
        "mlx-lm-top-logprobs-40 patch raises it to 40, checked at engine start"
    )

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        """Refuse to start on an env without the `top_logprobs` patch.

        The env is the one this engine's interpreter lives in
        (`<env>/bin/python`, `jobenv.env_python`). A missing interpreter is left
        to the base class, which already names it.
        """
        if self._python.is_file():
            env_dir = self._python.parent.parent
            try:
                envpatches.require_applied(envpatches.MLX_LM_TOP_LOGPROBS, env_dir)
            except PatchError as exc:
                raise EngineError(
                    f"llm_env_unpatched: {exc}. This engine states "
                    f"max_logprobs {self.max_logprobs} because of that patch and "
                    "will not start without it; run `crucible env patch llm` "
                    "(or `crucible install llm --force`) and load again"
                ) from exc
        super().start(model_dir, served_name, port, args)

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        # `served_name` is what mlx-lm will report for this model, which is the
        # resolved weights directory; `engines.engine_model_name()` computes it.
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
        self, deadline: float, on_progress: Callable[[str], None] | None
    ) -> None:
        """One token, to force the load and prove generation works.

        Without this, `ready()` returns the moment the HTTP thread binds — see
        point 3 in this module's docstring.
        """
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
            if self._process is not None and self._process.poll() is not None:
                raise EngineError(
                    f"{self.name} exited {self._process.returncode} while loading "
                    f"its weights. Last lines of {self._log_path}:\n"
                    + self.log_tail()
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
                    f"token before the timeout: {last}. Last lines of "
                    f"{self._log_path}:\n" + self.log_tail()
                )
            attempt += 1
            if on_progress is not None:
                on_progress(
                    f"{self.name} reading weights into memory "
                    f"({attempt * CONFIRM_POLL_SECONDS:.0f}s so far)"
                )
            time.sleep(CONFIRM_POLL_SECONDS)
