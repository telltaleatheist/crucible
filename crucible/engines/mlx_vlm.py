"""mlx-vlm, the `mlx-darwin` engine for the `pages` class family.

Crucible's OWN page server (`crucible/engines/mlx_vlm_serve.py`), started from
the llm env's python — the same env `mlx-lm` runs out of, because mlx-vlm's
pins resolve under it (see `envs/llm/mlx-darwin.txt`). It exists because
`mlx-lm` is a TEXT server: it has no vision tower and no image content part,
so a Mac that reads pages needs a second server class.
`crucible/manifests.py`'s `BACKEND_ENGINES` is what makes room for one — one
engine per (backend, class family) since 2026-09-14.

WHY THE SERVER IS CRUCIBLE'S AND NOT `python -m mlx_vlm server`
-----------------------------------------------------------------
This class used to spawn mlx-vlm's own HTTP server, and that server never put
the image into the prompt for `dots_ocr`. Measured on the Mac Studio on
2026-09-14 against `mlx-community/dots.ocr-4bit`, on mlx-vlm 0.6.10 and 0.7.1
alike, with a 1300x2112 synthetic book page:

    in process   `mlx_vlm.generate(...)`  -> five blocks, the three body
                 paragraphs verbatim, 16.32 s, peak 4,927,004,359 B
    over HTTP    the same weights, the same image, the same prompt
                 -> `[{"bbox": [1, 0, 1008, 1008], "category": "Picture"}]`
                 in 0.72 s

The discriminator was the token count: the server logged `prompt_tokens=216`
— the text alone — where `prepare_inputs` on the same page makes 3,464. Four
request shapes produced the byte-identical wrong answer. So no manifest named
this engine, and Macs read no pages, until the in-process path could be put
behind the proxy's wire. That is what `mlx_vlm_serve.py` is: the in-process
path — `batch_generate`'s own call order, validated byte for byte on real
pages on 2026-09-21 — behind exactly the two routes the proxy speaks. Its
module docstring carries the measurements and what it refuses.

What the class relies on, each measured rather than assumed
-------------------------------------------------------------
1. **`/v1/models` reports the model DIRECTORY, verbatim as given.** The server
   answers with the `--model` string it was handed and resolves nothing, which
   is where this differs from `mlx-lm` (`engines/mlx_lm.py` resolves, because
   mlx-lm calls `Path(...).resolve()` itself). So `engines.engine_model_name()`
   returns `str(model_dir)` unresolved for this engine, and the two spellings
   can never disagree because only one is ever produced.

2. **A 200 from `/v1/models` DOES mean the weights are in memory**, unlike
   mlx-lm. `mlx_vlm_serve.main()` loads BEFORE it binds the socket — the same
   order mlx-vlm's own server had (a preload in FastAPI's lifespan) and the
   reason this class needs NO `confirm()` override. The base class's
   `/v1/models` poll is the honest readiness check here.

3. **`--width` is the manifest's, never this class's.** How many pages one
   batch carries is a fact about the weights on a card, measured and cited in
   `models/dots-ocr.toml`'s `[backends.mlx-darwin]` block; the server refuses
   to start without it and so does `command()`, by name, so a block that
   forgot it fails at load rather than serially forever.

4. **SIGTERM is honoured.** The server installs a handler that shuts the HTTP
   loop down, so `stop()` never has to consider escalating — which it would
   not do anyway (`engines/base.py` never SIGKILLs).
"""

from __future__ import annotations

from pathlib import Path

from .base import EngineError, SubprocessEngine

#: The server, shipped inside this package and run from the llm env's python.
#: A FILE PATH and not `-m`, for `jobs/asr/mlx_worker.py`'s reason: the env it
#: runs in has no `crucible` installed, so there is no module to name.
SERVE_SCRIPT = Path(__file__).resolve().with_name("mlx_vlm_serve.py")

#: The flag the manifest must carry. See point 3.
WIDTH_FLAG = "--width"


class MlxVlmEngine(SubprocessEngine):
    name = "mlx-vlm"

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        """`--model <dir>` is loaded before the socket binds (point 2).

        `served_name` is not passed to anything: the server reports the
        directory it was given, which is what `engines.engine_model_name()`
        computes and the proxy rewrites the request's `model` field to.
        """
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
