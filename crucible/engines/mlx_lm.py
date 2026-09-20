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
    chat_concurrency = 1
    chat_concurrency_basis = (
        "mlx-lm 0.31.3 generates on one thread draining one queue "
        "(mlx_lm/server.py ResponseGenerator, read on the Mac Studio "
        "2026-09-20); its ThreadingHTTPServer accepts concurrently but "
        "generation is strictly serial"
    )

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
