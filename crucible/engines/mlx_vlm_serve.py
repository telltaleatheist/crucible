"""Crucible's own page server for `mlx-darwin`: dots.ocr through mlx-vlm, IN PROCESS.

Run out of the llm env as `python <this file> --model <dir> --host --port --width N`
by `crucible/engines/mlx_vlm.py`. It is **standalone** — no `crucible` import —
for `jobs/asr/mlx_worker.py`'s reason: the env it runs in has no `crucible`
installed and never will.

WHY THIS EXISTS INSTEAD OF `python -m mlx_vlm server`
------------------------------------------------------
mlx-vlm's own HTTP server never puts the image into the prompt for `dots_ocr`:
measured on the Mac Studio on 2026-09-14 (0.6.10 and 0.7.1 alike), the server
logged `prompt_tokens=216` — the text alone — where the in-process path makes
3,464 for the same page, and answered every page as one `Picture` block. The
in-process path is correct, so this process IS the in-process path behind the
OpenAI surface the proxy already speaks: `GET /v1/models` and
`POST /v1/chat/completions` with one `image_url` data-URI part. Nothing about
the wire changes; a client cannot tell which engine read its page.

BATCHING, AND WHAT IS AND IS NOT VALIDATED (Mac Studio, 2026-09-21)
--------------------------------------------------------------------
Two shapes were tried on four real 739x1259 book pages against
`mlx-community/dots.ocr-bf16`:

* **Continuous insertion** — one `BatchGenerator`, each page prefilled on its
  own and `insert()`ed while others were mid-generation — read page one
  correctly and produced garbage for the other three (one an 8192-token
  runaway, one a 7-token page). NOT USED, and do not reintroduce it without
  the byte comparison below passing.
* **Static micro-batches** — every row of a batch inserted BEFORE the first
  `next()`, which is exactly `mlx_vlm.generate.ar._generate_batch`'s own
  order — matched `batch_generate` byte for byte on all four pages. That is
  the path here. `read_batch()` is that function with one difference: it
  keeps each row's `finish_reason` and token count, which `batch_generate`
  throws away and which the page contract needs (`finish_reason: "length"`
  is what tells a client to re-read a cut-off page at the full ceiling).

So the queue is drained `--width` rows at a time. Rows in one batch must share
an image shape (upstream prefills with `pad_to_uniform_size=False`), so a batch
is the oldest waiting row plus up to width-1 more of the same shape; a page of
another shape simply waits for the next batch. Width 1 is serial, and the
number is the manifest's (`models/dots-ocr.toml`), not this file's.

THE VISION TOWER RUNS ONE IMAGE AT A TIME, AND THE METAL WATCHDOG IS WHY.
Upstream embeds a whole batch in one `get_input_embeddings` call. Twelve
739x1259 pages (1,385 prompt tokens each) survive that; twelve 1300x2232 pages
(3,895 tokens each — the size a 468x760 pt paperback is at 200 dpi) do not:
macOS kills the command buffer with `[METAL] Command buffer execution failed:
Impacting Interactivity (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)`,
measured 2026-09-21, whatever the language model's prefill batch was. So
`read_batch()` runs `prepare_inputs` + `get_input_embeddings` per row and hands
each row its own embeddings; the language model still prefills and decodes the
rows as one batch. Cost: ~15% on small pages (6.5 vs 5.7 s/page at width 8);
the same text on every page tried. What it buys: twelve 200-dpi paperback pages
at width 12 read in 14.5 s/page against 20.2 serial, peak 13.96 GB, no watchdog.

A PREFILL BATCH SMALLER THAN THE ROW COUNT IS NOT A KNOB. `BatchGenerator`
takes `prefill_batch_size` separately from `completion_batch_size`, and setting
it lower was tried as the watchdog fix: with twelve rows, prefill 1 read ONE
page correctly and prefill 4 read four, the rest garbage or 8192-token
runaways — prompts joining a live decode batch, the continuous-insertion
failure wearing a different flag. Prefill 12 read all twelve identically to
serial. `prefill_batch_size == completion_batch_size == len(batch)` is
upstream's invariant and this file keeps it.

WHAT IS REFUSED, BY NAME
------------------------
This is a page reader, and the request it reads is the one `crucible/pages.py`
publishes as `pages_engine.request`. Anything else is a misconfiguration, not
weather, and is refused in a sentence rather than quietly answered
differently:

* `temperature` other than 0 (dots is decoded greedily here; there is no
  sampler on this path), `top_p` other than 1, `n` other than 1,
  `stream: true`, and any sampling field this file does not understand;
* a `messages` list that is not exactly one user turn carrying exactly one
  `image_url` (a `data:image/...;base64,` URI) and exactly one `text` part;
* a `model` that is not the one loaded.

THE PRIVATE NAMES, PINNED. `read_batch()` calls four names from
`mlx_vlm.generate.ar` that upstream does not export (`BatchGenerator`,
`_split_prompt_kwargs_per_row`, `_chunked_prefill_enabled`,
`DEFAULT_PREFILL_STEP_SIZE`) plus `_default_prefill_step_size_for_offload`
where present. That is deliberate and it is safe only because
`envs/llm/mlx-darwin.txt` pins mlx-vlm to one exact version; `_check_upstream()`
refuses to start if any of them is missing, so a moved pin fails at load by
name rather than at the first page.
"""

from __future__ import annotations

import os
import sys

# THIS FILE LIVES BESIDE `mlx_vlm.py`, THE ENGINE CLASS, and python puts a
# script's own directory first on `sys.path` — so run as a file, `import
# mlx_vlm` below would find Crucible's engine module and not the library, and
# fail on its first relative import. Measured, not feared: the first run of
# `tests/test_mlx_vlm_serve.py` did exactly that. Only the script directory is
# removed, and only when this file is the program; imported as
# `crucible.engines.mlx_vlm_serve` (the tests do), nothing is touched.
if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != _HERE
    ]

import argparse  # noqa: E402
import base64  # noqa: E402
import binascii  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from typing import Any, Callable  # noqa: E402

#: The one media-type family a page arrives as. `crucible/pages.py`'s
#: `data_uri()` writes `data:image/png;base64,`; JPEG is accepted too because
#: it is a picture, not a different request.
DATA_URI_PREFIX = "data:image/"

#: The request fields this reader understands. Anything else is refused, so a
#: client that thinks it set `top_k` learns that it did not.
KNOWN_FIELDS = frozenset(
    {"model", "messages", "temperature", "top_p", "n", "max_tokens", "stream", "user"}
)


class Refusal(Exception):
    """A request this reader will not answer, with the HTTP status it earns."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


# ------------------------------------------------------------------- the rows


@dataclass
class Row:
    """One page's answer, in the batch's order."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Job:
    """One request waiting for the reader thread."""

    image: Any  # PIL.Image.Image, RGB
    prompt: str
    max_tokens: int
    shape: tuple[int, int]
    done: threading.Event = field(default_factory=threading.Event)
    row: Row | None = None
    error: Exception | None = None


class Reader:
    """The mlx-vlm calls, behind one method, so the server above is testable
    with a fake `mlx_vlm` package on PYTHONPATH and the real one is exercised
    on a Mac."""

    def __init__(self, model_dir: str) -> None:
        import mlx_vlm

        self._mlx_vlm = mlx_vlm
        self._check_upstream()
        self.model, self.processor = mlx_vlm.load(model_dir)

    def _check_upstream(self) -> None:
        """Every private name `read_batch` depends on, or a refusal naming it."""
        from mlx_vlm.generate import ar

        missing = [
            name
            for name in (
                "BatchGenerator",
                "_split_prompt_kwargs_per_row",
                "_chunked_prefill_enabled",
                "DEFAULT_PREFILL_STEP_SIZE",
            )
            if not hasattr(ar, name)
        ]
        if missing:
            raise RuntimeError(
                f"mlx-vlm {getattr(self._mlx_vlm, '__version__', '?')} does not "
                f"have {missing} in mlx_vlm.generate.ar; this reader was written "
                "against the version envs/llm/mlx-darwin.txt pins and will not "
                "guess at another one"
            )

    def _embed(self, job: Job) -> tuple[Any, Any, dict[str, Any]]:
        """One row through the processor and the vision tower.

        `(input_ids, embedding, gen_kwargs)` for ONE image — the per-image
        half of the module docstring's watchdog finding. `mx.eval` so the
        tower's work is done here, one image at a time, and not deferred
        into the language model's first step as one twelve-image buffer.
        """
        import mlx.core as mx
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        model, processor = self.model, self.processor
        formatted = apply_chat_template(processor, model.config, job.prompt, num_images=1)
        inputs = prepare_inputs(
            processor,
            images=[job.image],
            audio=None,
            prompts=[formatted],
            image_token_index=getattr(model.config, "image_token_index", None),
            resize_shape=None,
            add_special_tokens=should_add_special_tokens(
                model.config.model_type, processor
            ),
            pad_to_uniform_size=False,
        )
        input_ids = inputs["input_ids"]
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        embedding = model.get_input_embeddings(
            input_ids,
            inputs.get("pixel_values"),
            mask=inputs.get("attention_mask"),
            **data_kwargs,
        )
        mx.eval(embedding.inputs_embeds)
        gen_kwargs = {
            **data_kwargs,
            **{k: v for k, v in embedding.to_dict().items() if v is not None},
        }
        return input_ids, embedding, gen_kwargs

    def read_batch(self, jobs: list[Job]) -> list[Row]:
        """`mlx_vlm.generate.ar._generate_batch`, with the finish reasons kept
        and the vision tower run per image.

        Every row is inserted before the first `next()`, and the language
        model prefills and decodes all of them as one batch — the order the
        2026-09-21 byte comparisons validated at widths 1 to 16.
        """
        import mlx.core as mx
        from mlx_vlm.generate import ar

        model, processor = self.model, self.processor
        n = len(jobs)
        rows = [self._embed(job) for job in jobs]
        first_ids, first_embedding, first_kwargs = rows[0]
        step = ar.DEFAULT_PREFILL_STEP_SIZE
        if hasattr(ar, "_default_prefill_step_size_for_offload"):
            step = ar._default_prefill_step_size_for_offload(
                model, step, None, ar.DEFAULT_PREFILL_STEP_SIZE
            )
        if step is not None and not ar._chunked_prefill_enabled(
            model,
            input_ids=first_ids,
            inputs_embeds=first_embedding.inputs_embeds,
            draft_model=None,
            draft_kind=None,
            prefill_kwargs=dict(first_kwargs),
        ):
            step = None
        generator = ar.BatchGenerator(
            model.language_model,
            processor,
            # BOTH n. See the module docstring: a smaller prefill batch reads
            # that many pages correctly and the rest as garbage.
            prefill_batch_size=n,
            completion_batch_size=n,
            compute_logprobs=False,
            max_tokens=max(job.max_tokens for job in jobs),
            greedy_sampling=True,
            prefill_step_size=step,
        )
        try:
            uids: list[int] = []
            for job, (input_ids, _embedding, gen_kwargs) in zip(jobs, rows):
                uids += generator.insert(
                    input_ids.tolist(),
                    [job.max_tokens],
                    prompt_kwargs=ar._split_prompt_kwargs_per_row(gen_kwargs, 1),
                )
            tokens: dict[int, list[int]] = {uid: [] for uid in uids}
            finish: dict[int, str] = {}
            while generator.has_work:
                _prompt_stats, responses = generator.next()
                for response in responses:
                    if response.finish_reason != "stop":
                        tokens[response.uid].append(response.token)
                    if response.finish_reason is not None:
                        finish[response.uid] = response.finish_reason
        finally:
            generator.close()
        detokenizer = processor.detokenizer
        answers: list[Row] = []
        for uid, (input_ids, _embedding, _kwargs) in zip(uids, rows):
            detokenizer.reset()
            for token in tokens[uid]:
                detokenizer.add_token(token)
            detokenizer.finalize()
            answers.append(
                Row(
                    text=detokenizer.text,
                    # A row the generator never finished is a bug in the loop
                    # above, not a page; say so rather than call it "stop".
                    finish_reason=finish[uid],
                    prompt_tokens=int(input_ids.shape[1]),
                    completion_tokens=len(tokens[uid]),
                )
            )
        mx.clear_cache()
        return answers


# ---------------------------------------------------------------- the batcher


class Batcher:
    """Drains the queue `width` same-shape rows at a time on one thread.

    ONE thread, because there is one accelerator and the reader holds it for a
    whole batch; a second thread would not overlap anything, it would
    interleave two batches' allocations.
    """

    def __init__(self, reader: Reader, width: int, log: Callable[[str], None]) -> None:
        if width < 1:
            raise ValueError(f"--width must be at least 1, not {width}")
        self._reader = reader
        self._width = width
        self._log = log
        self._waiting: list[Job] = []
        self._lock = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="reader", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        with self._lock:
            self._waiting.append(job)
            self._lock.notify()

    def _take(self) -> list[Job]:
        """The oldest waiting row and up to width-1 more of its shape."""
        with self._lock:
            while not self._waiting:
                self._lock.wait()
            first = self._waiting[0]
            batch = [job for job in self._waiting if job.shape == first.shape][: self._width]
            for job in batch:
                self._waiting.remove(job)
            return batch

    def _run(self) -> None:
        while True:
            batch = self._take()
            started = time.monotonic()
            try:
                rows = self._reader.read_batch(batch)
            except Exception as exc:  # the model raised; the rows are told, the thread lives
                self._log(f"batch of {len(batch)} failed: {exc!r}")
                for job in batch:
                    job.error = exc
                    job.done.set()
                continue
            elapsed = time.monotonic() - started
            self._log(
                f"batch of {len(batch)} at {batch[0].shape[0]}x{batch[0].shape[1]}: "
                f"{elapsed:.1f}s, {elapsed / len(batch):.1f}s/page, "
                f"finish={[row.finish_reason for row in rows]}, "
                f"tokens={[row.completion_tokens for row in rows]}"
            )
            for job, row in zip(batch, rows):
                job.row = row
                job.done.set()


# ---------------------------------------------------------------- the request


def decode_image(uri: str) -> Any:
    """The PIL image a data URI carries, as RGB.

    `.convert("RGB")` always: a PNG that decodes to RGBA or a palette trips
    the image processor (the Mac's finding, 2026-09-21), and it is the same
    picture either way.
    """
    from PIL import Image

    if not uri.startswith(DATA_URI_PREFIX):
        raise Refusal(
            400,
            "not_a_data_uri",
            "image_url.url must be a data:image/...;base64, URI; this reader "
            "fetches nothing",
        )
    comma = uri.find(",")
    if comma < 0 or not uri[:comma].endswith(";base64"):
        raise Refusal(400, "not_a_data_uri", "image_url.url is not base64 data")
    try:
        raw = base64.b64decode(uri[comma + 1 :], validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (binascii.Error, ValueError, OSError) as exc:
        raise Refusal(400, "bad_image", f"the image would not decode: {exc}") from exc
    return image.convert("RGB")


def parse_request(body: dict[str, Any], served: str) -> tuple[Any, str, int]:
    """(image, prompt, max_tokens) from a chat body, or a refusal by name."""
    unknown = sorted(set(body) - KNOWN_FIELDS)
    if unknown:
        raise Refusal(
            400,
            "unknown_field",
            f"this page reader does not understand {unknown}; it reads "
            f"{sorted(KNOWN_FIELDS)}",
        )
    if body.get("model") != served:
        raise Refusal(
            404, "model_not_found", f"{body.get('model')!r} is not loaded; {served!r} is"
        )
    if body.get("stream") is True:
        raise Refusal(400, "no_streaming", "a page is answered whole, never streamed")
    temperature = body.get("temperature")
    if temperature != 0:
        raise Refusal(
            400,
            "not_greedy",
            f"temperature must be 0 and was {temperature!r}: this reader decodes "
            "greedily, which is what crucible/pages.py publishes",
        )
    if body.get("top_p", 1) != 1:
        raise Refusal(400, "not_greedy", f"top_p must be 1 and was {body['top_p']!r}")
    if body.get("n", 1) != 1:
        raise Refusal(400, "one_choice", f"n must be 1 and was {body['n']!r}")
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise Refusal(
            400, "max_tokens_required", f"max_tokens must be a positive int, not {max_tokens!r}"
        )
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise Refusal(
            400, "one_user_turn", "a page request is exactly one user message"
        )
    turn = messages[0]
    if not isinstance(turn, dict) or turn.get("role") != "user":
        raise Refusal(400, "one_user_turn", "the one message must have role 'user'")
    parts = turn.get("content")
    if not isinstance(parts, list):
        raise Refusal(
            400, "content_parts", "content must be a list of parts: one image_url and one text"
        )
    images = [p for p in parts if isinstance(p, dict) and p.get("type") == "image_url"]
    texts = [p for p in parts if isinstance(p, dict) and p.get("type") == "text"]
    if len(images) != 1 or len(texts) != 1 or len(parts) != 2:
        raise Refusal(
            400,
            "content_parts",
            f"a page request carries exactly one image_url part and one text "
            f"part; this one has {len(images)} and {len(texts)} of {len(parts)}",
        )
    url = (images[0].get("image_url") or {}).get("url")
    if not isinstance(url, str):
        raise Refusal(400, "not_a_data_uri", "image_url.url is missing")
    prompt = texts[0].get("text")
    if not isinstance(prompt, str) or not prompt:
        raise Refusal(400, "empty_prompt", "the text part is empty")
    return decode_image(url), prompt, max_tokens


def completion_document(served: str, row: Row) -> dict[str, Any]:
    """The OpenAI chat-completion shape, non-streaming, one choice."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": row.text},
                "finish_reason": row.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.prompt_tokens + row.completion_tokens,
        },
    }


# ----------------------------------------------------------------- the server


class _Handler(BaseHTTPRequestHandler):
    served: str = ""
    batcher: Batcher | None = None
    log: Callable[[str], None] = lambda line: None

    def log_message(self, *_args: object) -> None:  # noqa: D102 - stderr is the log
        return

    def _send(self, status: int, document: dict[str, Any]) -> None:
        payload = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _refuse(self, refusal: Refusal) -> None:
        self._send(
            refusal.status,
            {"error": {"code": refusal.code, "message": str(refusal), "type": "invalid_request_error"}},
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/v1/models", "/models"):
            self._send(
                200, {"object": "list", "data": [{"id": self.served, "object": "model"}]}
            )
            return
        self._refuse(Refusal(404, "not_found", f"no route {self.path}"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._refuse(Refusal(404, "not_found", f"no route {self.path}"))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise Refusal(400, "bad_json", "the body is not a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._refuse(Refusal(400, "bad_json", f"the body is not JSON: {exc}"))
            return
        except Refusal as refusal:
            self._refuse(refusal)
            return
        try:
            image, prompt, max_tokens = parse_request(body, self.served)
        except Refusal as refusal:
            self._refuse(refusal)
            return
        assert self.batcher is not None
        job = Job(image=image, prompt=prompt, max_tokens=max_tokens, shape=(image.height, image.width))
        self.batcher.submit(job)
        job.done.wait()
        if job.error is not None:
            self._send(
                500,
                {"error": {"code": "engine_error", "message": repr(job.error), "type": "server_error"}},
            )
            return
        assert job.row is not None
        self._send(200, completion_document(self.served, job.row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="the weights directory, reported verbatim")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--width",
        required=True,
        type=int,
        help="rows per batch; the manifest states it, this file never defaults it",
    )
    args = parser.parse_args(argv)

    def log(line: str) -> None:
        sys.stderr.write(f"[mlx_vlm_serve] {line}\n")
        sys.stderr.flush()

    # LOAD BEFORE BIND. A 200 from /v1/models must mean the weights are in
    # memory — that is the readiness contract `engines/mlx_vlm.py` relies on
    # and the one property that lets it skip a confirm() completion.
    started = time.monotonic()
    reader = Reader(args.model)
    log(f"loaded {args.model} in {time.monotonic() - started:.1f}s; width {args.width}")
    batcher = Batcher(reader, args.width, log)

    _Handler.served = args.model
    _Handler.batcher = batcher
    _Handler.log = staticmethod(log)  # type: ignore[assignment]
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True

    def on_term(_signum: int, _frame: Any) -> None:
        log("SIGTERM, stopping")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, on_term)
    log(f"serving on {args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
