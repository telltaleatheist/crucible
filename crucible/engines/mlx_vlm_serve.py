from __future__ import annotations

import os
import sys

def _stop_engine_module_shadowing_mlx_vlm_library() -> None:
    engines_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != engines_dir
    ]


if __name__ == "__main__":
    _stop_engine_module_shadowing_mlx_vlm_library()

import argparse
import base64
import binascii
import io
import json
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

DATA_URI_PREFIX = "data:image/"

KNOWN_FIELDS = frozenset(
    {"model", "messages", "temperature", "top_p", "n", "max_tokens", "stream", "user"}
)


class Refusal(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Row:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Job:
    image: Any
    prompt: str
    max_tokens: int
    grid: tuple[int, ...]
    done: threading.Event = field(default_factory=threading.Event)
    row: Row | None = None
    error: Exception | None = None


class Reader:
    def __init__(self, model_dir: str) -> None:
        import mlx_vlm

        self._mlx_vlm = mlx_vlm
        self._check_upstream()
        self.model, self.processor = mlx_vlm.load(model_dir)

    def grid_of(self, image: Any) -> tuple[int, ...]:
        out = self.processor.image_processor(images=[image], return_tensors="np")
        grid = out["image_grid_thw"]
        first = grid[0] if hasattr(grid, "__getitem__") else grid
        return tuple(int(v) for v in (first.tolist() if hasattr(first, "tolist") else first))

    def _check_upstream(self) -> None:
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
                    finish_reason=finish[uid],
                    prompt_tokens=int(input_ids.shape[1]),
                    completion_tokens=len(tokens[uid]),
                )
            )
        mx.clear_cache()
        return answers


class Batcher:
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

    def key_of(self, image: Any) -> tuple[int, ...]:
        return self._reader.grid_of(image)

    def _take(self) -> list[Job]:
        with self._lock:
            while not self._waiting:
                self._lock.wait()
            first = self._waiting[0]
            batch = [job for job in self._waiting if job.grid == first.grid][: self._width]
            for job in batch:
                self._waiting.remove(job)
            return batch

    def _run(self) -> None:
        while True:
            batch = self._take()
            started = time.monotonic()
            try:
                rows = self._reader.read_batch(batch)
            except Exception as exc:
                self._log(f"batch of {len(batch)} failed: {exc!r}")
                for job in batch:
                    job.error = exc
                    job.done.set()
                continue
            elapsed = time.monotonic() - started
            for job, row in zip(batch, rows):
                job.row = row
                job.done.set()
            try:
                self._log(self._describe(batch, rows, elapsed))
            except Exception as exc:
                self._log(f"batch of {len(batch)} done in {elapsed:.1f}s (describing it failed: {exc!r})")

    @staticmethod
    def _describe(batch: list[Job], rows: list[Row], elapsed: float) -> str:
        grid = "x".join(str(v) for v in batch[0].grid)
        sizes = sorted({f"{job.image.width}x{job.image.height}" for job in batch})
        return (
            f"batch of {len(batch)} at grid {grid} ({len(sizes)} raw size"
            f"{'s' if len(sizes) != 1 else ''}): "
            f"{elapsed:.1f}s, {elapsed / len(batch):.1f}s/page, "
            f"finish={[row.finish_reason for row in rows]}, "
            f"tokens={[row.completion_tokens for row in rows]}"
        )


def decode_image(uri: str) -> Any:
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


class _Handler(BaseHTTPRequestHandler):
    served: str = ""
    batcher: Batcher | None = None
    log: Callable[[str], None] = lambda line: None

    def log_message(self, *_args: object) -> None:
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

    def do_GET(self) -> None:
        if self.path in ("/v1/models", "/models"):
            self._send(
                200, {"object": "list", "data": [{"id": self.served, "object": "model"}]}
            )
            return
        self._refuse(Refusal(404, "not_found", f"no route {self.path}"))

    def do_POST(self) -> None:
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
        try:
            grid = self.batcher.key_of(image)
        except Exception as exc:
            self._send(
                500,
                {"error": {"code": "engine_error", "message": f"the processor could not grid this image: {exc!r}", "type": "server_error"}},
            )
            return
        job = Job(image=image, prompt=prompt, max_tokens=max_tokens, grid=grid)
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
    parser = argparse.ArgumentParser(
        description="Crucible's own page server for mlx-darwin: dots.ocr through mlx-vlm, in process."
    )
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

    started = time.monotonic()
    reader = Reader(args.model)
    log(f"loaded {args.model} in {time.monotonic() - started:.1f}s; width {args.width}")
    batcher = Batcher(reader, args.width, log)

    _Handler.served = args.model
    _Handler.batcher = batcher
    _Handler.log = staticmethod(log)
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
