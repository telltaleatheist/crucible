"""mlx-vlm, the `mlx-darwin` engine for the `pages` class family.

Started as `python -m mlx_vlm server` from the llm env — the same env `mlx-lm`
runs out of, because mlx-vlm's pins are compatible with it (see
`envs/llm/mlx-darwin.txt`). It exists because `mlx-lm` is a TEXT server: it has
no vision tower and no image content part, so a Mac that reads pages needs a
second server class. `crucible/manifests.py`'s `BACKEND_ENGINES` is what makes
room for one — one engine per (backend, class family) since 2026-09-14.

> **THIS ENGINE IS NOT YET REACHABLE, AND THAT IS A MEASUREMENT, NOT AN
> OVERSIGHT.** No manifest names `mlx-vlm`, so `build_engine` will never be
> asked for it today. `models/dots-ocr.toml` carries the run that stopped the
> block from being written: mlx-vlm's `/v1/chat/completions` does not put the
> image into the prompt for `dots_ocr`, and the page comes back as one
> `Picture`. The summary is in "What is broken" below. The class is here,
> complete and tested against a double, so that the day the upstream bug is
> fixed the change is a `[backends.mlx-darwin]` block and nothing else.

What mlx_vlm.server has and has not (READ from mlx-vlm 0.6.10's own source and
EXERCISED on the Mac Studio on 2026-09-14, both 0.6.10 and 0.7.1)
------------------------------------------------------------------------------
It speaks the OpenAI surface Crucible needs — `GET /v1/models`, `POST
/v1/chat/completions` with `image_url` content parts, SSE streaming — and its
CLI (`mlx_vlm/server/cli.py`) takes `--model`, `--host`, `--port`,
`--max-tokens`, `--trust-remote-code` and a long tail of sampling and KV flags.
Four of its properties matter to this class, and each is measured rather than
assumed:

1. **`/v1/models` reports the model DIRECTORY, verbatim as given.** The route
   lists every mlx-looking repo in the HuggingFace cache plus the `model_path`
   of whatever is loaded, and `get_cached_model` stores that string exactly as
   it was handed over — it does not resolve it, which is where this differs
   from `mlx-lm` (`engines/mlx_lm.py` resolves, because mlx-lm calls
   `Path(...).resolve()` itself). So `engines.engine_model_name()` returns
   `str(model_dir)` unresolved for this engine, and the two spellings can never
   disagree because only one is ever produced.

2. **A 200 from `/v1/models` DOES mean the weights are in memory**, unlike
   mlx-lm. The preload runs in FastAPI's `lifespan`, which uvicorn completes
   before it accepts a connection; the server log shows "Model and processor
   loaded successfully" then "Application startup complete" then the first
   request. Measured cold: 4.15 s from spawn to a 200, against 1.93 s for the
   in-process load alone. So this class needs NO `confirm()` override — the
   base class's `/v1/models` poll is the honest readiness check here, and
   adding a one-token completion would be asking the engine to prove something
   it has already proven.

3. **`--trust-remote-code` is NOT needed for dots.ocr**, though vLLM requires
   it for the same weights. mlx-vlm ships its own `dots_ocr` model class
   (`mlx_vlm/models/dots_ocr/`), so the repo's `auto_map` is never followed.
   The flag exists on the CLI and is simply not passed; a manifest that wants
   it can put it in `engine_args`.

4. **SIGTERM is honoured.** Measured: the server exited on the first SIGTERM
   inside the base class's wait, so `stop()` never has to consider escalating —
   which it would not do anyway (`engines/base.py` never SIGKILLs).

What is broken, and why no manifest names this engine yet
----------------------------------------------------------
**The image never reaches the model through `/v1/chat/completions`.** Measured
on the Mac Studio on 2026-09-14 against `mlx-community/dots.ocr-4bit` @
`4ab989e4`, on mlx-vlm **0.6.10 and 0.7.1 alike**, with a 1300x2112 synthetic
book page:

    in process   `mlx_vlm.generate(...)`  -> five blocks, the three body
                 paragraphs verbatim, 16.32 s, peak 4,927,004,359 B
    over HTTP    the same weights, the same image, the same prompt
                 -> `[{"bbox": [1, 0, 1008, 1008], "category": "Picture"}]`
                 in 0.72 s

The discriminator is the token count, and it is unambiguous. The server logs
the request as `images=1` and then `prompt_tokens=216` — the TEXT alone. Asked
a 21-token question instead, it logs `prompt_tokens=21`. On the same machine,
in the same env, `mlx_vlm.utils.prepare_inputs(processor, images=[the page],
prompts=the same formatted prompt)` returns `input_ids` of length **3,464**
with `pixel_values` of shape (13800, 588) — the 13,800 patches a 1300x2112 page
makes at patch 14, merged 2x2 into 3,450 image tokens. So the image placeholder
is never expanded on the server's path, and the model answers about a page it
was not shown.

Four request shapes were tried and all four produced the byte-identical wrong
answer: the image part first, the text part first, a local file path instead of
a data URI, and an explicit `resize_shape`. `apply_chat_template` was ruled out
directly — it produces the same `<|img|><|imgpad|><|endofimg|>` prompt whether
it is given a string or the message list the server passes.

Shipping the manifest block anyway would turn `pages` on for every Mac and
answer every page with a single `Picture` block, which is precisely the kind of
silent wrongness this repo refuses. So the block is not written, the reason is
in `models/dots-ocr.toml` beside the numbers, and `docs/PHASE15-HOST.md` 7c
says what has to be true to write it.
"""

from __future__ import annotations

from pathlib import Path

from .base import SubprocessEngine

#: `python -m mlx_vlm server`. `mlx_vlm/__main__.py` dispatches the subcommand
#: itself; there is no `mlx_vlm.server` module entry point to run instead.
MODULE = "mlx_vlm"
SUBCOMMAND = "server"


class MlxVlmEngine(SubprocessEngine):
    name = "mlx-vlm"

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        """`--model <dir>` preloads in the lifespan, before uvicorn accepts.

        That is the whole reason `--model` is passed rather than letting the
        first request load it: a server that loads on demand would answer
        `/v1/models` while empty, and `ready()` would return with gigabytes
        still to read — mlx-lm's problem, which this engine does not have only
        because of this flag.

        `served_name` is not passed to anything. mlx-vlm has no
        `--served-model-name` and reports the directory it was given, which is
        what `engines.engine_model_name()` computes and the proxy rewrites the
        request's `model` field to.
        """
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
