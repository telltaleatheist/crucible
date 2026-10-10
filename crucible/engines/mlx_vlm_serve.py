from __future__ import annotations

import os
import sys


def _stop_engine_module_shadowing_mlx_vlm_library() -> None:
    engines_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [
        entry for entry in sys.path if os.path.realpath(entry or os.getcwd()) != engines_dir
    ]


if __name__ == "__main__":
    _stop_engine_module_shadowing_mlx_vlm_library()

import argparse
import base64
import binascii
import hashlib
import importlib.util
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

QUESTION_ONLY_FIELDS = frozenset({"logprobs", "top_logprobs", "chat_template_kwargs"})

QUESTION_FIELDS = KNOWN_FIELDS | QUESTION_ONLY_FIELDS

TEMPLATE_KWARGS = frozenset({"enable_thinking"})

ROLES = frozenset({"system", "user", "assistant"})

MAX_TOP_LOGPROBS = 40

MAX_IMAGES = 8

VISION_CACHED_MODEL_TYPES = frozenset({"qwen3_5"})

VISION_CACHE_ENTRIES = 16


def _load_items_forward() -> Any:
    try:
        from . import items_forward

        return items_forward
    except ImportError:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "items_forward.py")
        spec = importlib.util.spec_from_file_location("crucible_items_forward", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


ITEMS = _load_items_forward()


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
    logprobs: list[dict[str, Any]] | None = None


@dataclass
class Job:
    image: Any
    prompt: str
    max_tokens: int
    grid: tuple[int, ...]
    done: threading.Event = field(default_factory=threading.Event)
    row: Row | None = None
    error: Exception | None = None


@dataclass
class Asked:
    messages: list[dict[str, Any]]
    images: list[Any]
    image_key: str | None
    max_tokens: int
    top_logprobs: int | None
    template_kwargs: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    row: Row | None = None
    error: Exception | None = None


@dataclass
class ItemsJob:
    ask: Any
    images: list[Any]
    image_key: str | None
    done: threading.Event = field(default_factory=threading.Event)
    row: dict[str, Any] | None = None
    error: Exception | None = None


def logits_in_float32(_tokens: Any, logits: Any) -> Any:
    import mlx.core as mx

    return logits.astype(mx.float32)


def token_entry(text: str, logprob: float) -> dict[str, Any]:
    return {"token": text, "logprob": float(logprob), "bytes": list(text.encode("utf-8"))}


class Reader:
    def __init__(self, model_dir: str) -> None:
        import mlx_vlm

        self._mlx_vlm = mlx_vlm
        self._check_upstream()
        self.model, self.processor = mlx_vlm.load(model_dir)
        self._vision_features: Any = None

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
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            self.processor, self.model.config, job.prompt, num_images=1
        )
        return self._embedded([job.image], formatted, {})

    def _embedded(
        self, images: list[Any], formatted: Any, vision: dict[str, Any]
    ) -> tuple[Any, Any, dict[str, Any]]:
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        model, processor = self.model, self.processor
        inputs = prepare_inputs(
            processor,
            images=images or None,
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
            **vision,
        )
        mx.eval(embedding.inputs_embeds)
        gen_kwargs = {
            **data_kwargs,
            **{k: v for k, v in embedding.to_dict().items() if v is not None},
        }
        return input_ids, embedding, gen_kwargs

    def _prefill_step(
        self, first_ids: Any, first_embedding: Any, first_kwargs: dict[str, Any]
    ) -> int | None:
        from mlx_vlm.generate import ar

        model = self.model
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
        return step

    def read_batch(self, jobs: list[Job]) -> list[Row]:
        import mlx.core as mx
        from mlx_vlm.generate import ar

        model, processor = self.model, self.processor
        n = len(jobs)
        rows = [self._embed(job) for job in jobs]
        step = self._prefill_step(*rows[0])
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

    def items(self, job: ItemsJob) -> dict[str, Any]:
        import mlx.core as mx

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prepared: list[dict[str, Any]] = []

        def tokenize(messages: list[dict[str, Any]], reply: bool = False) -> list[int]:
            inputs = self._prepared(messages, job, reply)
            prepared.append(inputs)
            return inputs["input_ids"][0].tolist()

        if isinstance(job.ask, ITEMS.LikelihoodAsk):
            split = ITEMS.likelihood_prompts(
                job.ask, tokenize, lambda messages: tokenize(messages, True)
            )
            shared_pass, item_pass, head = self._row_passes(split.shared, split.suffixes, prepared[0], job)
            document = ITEMS.likelihood_document(
                split,
                ITEMS.read_likelihood(
                    split, shared_pass, item_pass,
                    lambda hidden, targets: ITEMS.token_logprobs(head, hidden, targets),
                ),
                ITEMS.row_read_tokens(split),
            )
            mx.clear_cache()
            return document
        document = ITEMS.answer_items(
            job.ask,
            tokenize,
            lambda split: self._item_rows(split, prepared[0], job),
            lambda token: tokenizer.decode([token]),
        )
        mx.clear_cache()
        return document

    def _prepared(
        self, messages: list[dict[str, Any]], job: ItemsJob, reply: bool = False
    ) -> dict[str, Any]:
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        model, processor = self.model, self.processor
        # A reply is the candidate rendered as the open assistant message
        # (`continue_final_message`, which mlx-vlm passes through to the
        # tokenizer's template), so its prompt ends on the candidate.
        opened = {"continue_final_message": True} if reply else {}
        formatted = apply_chat_template(
            processor, model.config, messages, add_generation_prompt=not reply,
            num_images=len(job.images), **opened, **job.ask.template_kwargs,
        )
        return prepare_inputs(
            processor,
            images=job.images or None,
            audio=None,
            prompts=[formatted],
            image_token_index=getattr(model.config, "image_token_index", None),
            resize_shape=None,
            add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
            pad_to_uniform_size=False,
        )

    def _item_rows(self, split: Any, inputs: dict[str, Any], job: ItemsJob) -> list[list[tuple[int, float]]]:
        shared_pass, item_pass, head = self._row_passes(split.shared, split.suffixes, inputs, job)
        return ITEMS.read_items(split, shared_pass, item_pass, head, job.ask.top_logprobs)

    def _row_passes(
        self, shared_tokens: list[int], suffixes: list[list[int]], inputs: dict[str, Any], job: ItemsJob
    ) -> tuple[Callable[[], list[Any]], Callable[[list[Any], list[int]], Any], Callable[[Any], Any]]:
        import mlx.core as mx
        from mlx_vlm.models.cache import make_prompt_cache

        model = self.model
        language = model.language_model
        inner, head = ITEMS.text_parts(model)
        extra = {
            k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        whole = mx.array([shared_tokens + max(suffixes, key=len)])
        embedding = model.get_input_embeddings(
            whole, inputs.get("pixel_values"), **extra, **self._vision(job)
        )
        embeds = embedding.inputs_embeds
        positions = getattr(embedding, "position_ids", None)
        shared = len(shared_tokens)

        def forward(ids: Any, embedded: Any, placed: Any, cache: list[Any]) -> Any:
            kwargs: dict[str, Any] = {"inputs_embeds": embedded, "cache": cache}
            if placed is not None:
                kwargs["position_ids"] = placed
            return inner(ids, **kwargs)

        def shared_pass() -> list[Any]:
            cache = make_prompt_cache(language)
            for start in range(0, shared, ITEMS.CHUNK_TOKENS):
                end = min(shared, start + ITEMS.CHUNK_TOKENS)
                placed = None if positions is None else positions[..., start:end]
                forward(whole[:, start:end], embeds[:, start:end], placed, cache)
                mx.eval([entry.state for entry in cache])
            return cache

        def item_pass(cache: list[Any], suffix: list[int]) -> Any:
            ids = mx.array([suffix])
            own = ITEMS.copied(make_prompt_cache(language), cache)
            placed = None if positions is None else positions[..., shared:shared + len(suffix)]
            return forward(ids, inner.embed_tokens(ids), placed, own)

        return shared_pass, item_pass, head

    def _vision(self, job: Asked | ItemsJob) -> dict[str, Any]:
        if job.image_key is None or self.model.config.model_type not in VISION_CACHED_MODEL_TYPES:
            return {}
        if self._vision_features is None:
            from mlx_vlm.vision_cache import VisionFeatureCache

            self._vision_features = VisionFeatureCache(max_size=VISION_CACHE_ENTRIES)
        return {"vision_cache": self._vision_features, "_image_key": job.image_key}

    def answer(self, job: Asked) -> Row:
        import mlx.core as mx
        from mlx_vlm.generate import ar
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = self.model, self.processor
        formatted = apply_chat_template(
            processor,
            model.config,
            job.messages,
            num_images=len(job.images),
            **job.template_kwargs,
        )
        input_ids, embedding, gen_kwargs = self._embedded(
            job.images, formatted, self._vision(job)
        )
        reading = job.top_logprobs is not None
        generator = ar.BatchGenerator(
            model.language_model,
            processor,
            prefill_batch_size=1,
            completion_batch_size=1,
            compute_logprobs=reading,
            top_logprobs_k=job.top_logprobs or 0,
            max_tokens=job.max_tokens,
            greedy_sampling=True,
            prefill_step_size=self._prefill_step(input_ids, embedding, gen_kwargs),
        )
        try:
            (uid,) = generator.insert(
                input_ids.tolist(),
                [job.max_tokens],
                prompt_kwargs=ar._split_prompt_kwargs_per_row(gen_kwargs, 1),
                logits_processors=[[logits_in_float32]] if reading else None,
            )
            steps: list[Any] = []
            while generator.has_work:
                _prompt_stats, responses = generator.next()
                steps.extend(response for response in responses if response.uid == uid)
        finally:
            generator.close()
        row = self._row_of(steps, int(input_ids.shape[1]), reading)
        mx.clear_cache()
        return row

    def _row_of(self, steps: list[Any], prompt_tokens: int, reading: bool) -> Row:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        detokenizer = self.processor.detokenizer
        detokenizer.reset()
        spoken = [step.token for step in steps if step.finish_reason != "stop"]
        for token in spoken:
            detokenizer.add_token(token)
        detokenizer.finalize()
        return Row(
            text=detokenizer.text,
            finish_reason=steps[-1].finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(spoken),
            logprobs=[self._step_entry(tokenizer, step) for step in steps] if reading else None,
        )

    @staticmethod
    def _step_entry(tokenizer: Any, step: Any) -> dict[str, Any]:
        entry = token_entry(tokenizer.decode([step.token]), step.token_logprob)
        entry["top_logprobs"] = [
            token_entry(tokenizer.decode([token]), logprob)
            for token, logprob in (step.top_logprobs or [])
        ]
        return entry


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
            if isinstance(first, (Asked, ItemsJob)):
                batch = [first]
            else:
                batch = [
                    job
                    for job in self._waiting
                    if isinstance(job, Job) and job.grid == first.grid
                ][: self._width]
            for job in batch:
                self._waiting.remove(job)
            return batch

    def _run(self) -> None:
        while True:
            batch = self._take()
            started = time.monotonic()
            try:
                rows = self._read(batch)
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

    def _read(self, batch: list[Any]) -> list[Any]:
        if isinstance(batch[0], ItemsJob):
            return [self._reader.items(batch[0])]
        if isinstance(batch[0], Asked):
            return [self._reader.answer(batch[0])]
        return self._reader.read_batch(batch)

    @staticmethod
    def _describe(batch: list[Any], rows: list[Any], elapsed: float) -> str:
        if isinstance(batch[0], ItemsJob):
            document = rows[0]
            if "groups" in document:
                candidates = sum(len(group["candidates"]) for group in document["groups"])
                return (
                    f"likelihood pass with {len(batch[0].images)} image(s): {elapsed:.2f}s, "
                    f"{document['shared_tokens']} shared tokens, {candidates} candidates"
                )
            return (
                f"items pass with {len(batch[0].images)} image(s): {elapsed:.2f}s, "
                f"{document['shared_tokens']} shared tokens, {len(document['slots'])} items"
            )
        if isinstance(batch[0], Asked):
            asked, row = batch[0], rows[0]
            return (
                f"question with {len(asked.images)} image(s): {elapsed:.2f}s, "
                f"prompt {row.prompt_tokens} tokens, top_logprobs "
                f"{asked.top_logprobs}, finish={row.finish_reason}"
            )
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


def is_page_request(body: dict[str, Any]) -> bool:
    if QUESTION_ONLY_FIELDS & set(body):
        return False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return True
    if len(messages) != 1 or not isinstance(messages[0], dict):
        return False
    parts = messages[0].get("content")
    return messages[0].get("role") == "user" and isinstance(parts, list) and any(
        isinstance(part, dict) and part.get("type") == "image_url" for part in parts
    )


def _check_sampling(body: dict[str, Any], served: str) -> int:
    if body.get("model") != served:
        raise Refusal(
            404, "model_not_found", f"{body.get('model')!r} is not loaded; {served!r} is"
        )
    if body.get("stream") is True:
        raise Refusal(400, "no_streaming", "a question is answered whole, never streamed")
    if body.get("temperature") != 0:
        raise Refusal(
            400,
            "not_greedy",
            f"temperature must be 0 and was {body.get('temperature')!r}: this server "
            "decodes greedily, which is what crucible/decide.py sends",
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
    return max_tokens


def _top_logprobs(body: dict[str, Any]) -> int | None:
    wanted = body.get("logprobs", False)
    if not isinstance(wanted, bool):
        raise Refusal(400, "bad_logprobs", f"logprobs must be true or false, not {wanted!r}")
    top = body.get("top_logprobs")
    if not wanted:
        if top is not None:
            raise Refusal(
                400,
                "top_logprobs_without_logprobs",
                "top_logprobs was sent without logprobs: true; send both",
            )
        return None
    if top is None:
        return 0
    if not isinstance(top, int) or isinstance(top, bool) or top < 0:
        raise Refusal(400, "bad_top_logprobs", f"top_logprobs must be an int >= 0, not {top!r}")
    if top > MAX_TOP_LOGPROBS:
        raise Refusal(
            400,
            "too_many_top_logprobs",
            f"top_logprobs is {top} and this server returns at most "
            f"{MAX_TOP_LOGPROBS} (MAX_TOP_LOGPROBS in crucible/engines/mlx_vlm_serve.py, "
            f"the max_logprobs MlxVlmEngine states); ask for {MAX_TOP_LOGPROBS} or fewer",
        )
    return top


def _template_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise Refusal(400, "bad_template_kwargs", "chat_template_kwargs must be an object")
    unknown = sorted(set(kwargs) - TEMPLATE_KWARGS)
    if unknown:
        raise Refusal(
            400,
            "unknown_template_kwarg",
            f"chat_template_kwargs carries {unknown}; this server passes only "
            f"{sorted(TEMPLATE_KWARGS)} to the chat template",
        )
    return dict(kwargs)


def _turn_images(turn: dict[str, Any], index: int) -> list[str]:
    parts = turn.get("content")
    if isinstance(parts, str):
        return []
    if not isinstance(parts, list):
        raise Refusal(
            400, "content_parts", f"messages[{index}].content must be a string or a list of parts"
        )
    urls: list[str] = []
    for part in parts:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "image_url":
            urls.append(_part_url(part, index))
        elif kind != "text" or not isinstance(part.get("text"), str):
            raise Refusal(
                400,
                "content_parts",
                f"messages[{index}] carries a part that is neither text nor image_url: {kind!r}",
            )
    if urls and turn["role"] != "user":
        raise Refusal(400, "content_parts", f"messages[{index}] is {turn['role']!r}; only a user turn carries images")
    return urls


def _part_url(part: dict[str, Any], index: int) -> str:
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str):
        raise Refusal(400, "not_a_data_uri", f"messages[{index}]: image_url.url is missing")
    return url


def _image_urls(messages: Any) -> list[str]:
    if not isinstance(messages, list) or not messages:
        raise Refusal(400, "no_messages", "messages must be a non-empty list of turns")
    urls: list[str] = []
    for index, turn in enumerate(messages):
        if not isinstance(turn, dict) or turn.get("role") not in ROLES:
            raise Refusal(
                400, "bad_turn", f"messages[{index}] must be an object whose role is one of {sorted(ROLES)}"
            )
        urls += _turn_images(turn, index)
    if len(urls) > MAX_IMAGES:
        raise Refusal(
            400,
            "too_many_images",
            f"the question carries {len(urls)} images; this server reads at most {MAX_IMAGES}",
        )
    return urls


def parse_question(body: dict[str, Any], served: str) -> Asked:
    unknown = sorted(set(body) - QUESTION_FIELDS)
    if unknown:
        raise Refusal(
            400,
            "unknown_field",
            f"this server does not understand {unknown}; a question reads "
            f"{sorted(QUESTION_FIELDS)}",
        )
    max_tokens = _check_sampling(body, served)
    top_logprobs = _top_logprobs(body)
    template_kwargs = _template_kwargs(body)
    urls = _image_urls(body.get("messages"))
    images = [decode_image(url) for url in urls]
    return Asked(
        messages=body["messages"],
        images=images,
        image_key=hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest() if urls else None,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
        template_kwargs=template_kwargs,
    )


def parse_items_job(body: dict[str, Any], served: str) -> ItemsJob:
    try:
        ask = ITEMS.parse_request(body, (served,))
    except ITEMS.ItemsRefusal as refusal:
        raise Refusal(refusal.status, refusal.code, str(refusal)) from None
    if isinstance(ask, ITEMS.EmbedAsk) or getattr(ask, "messages", None) is None:
        raise Refusal(
            400,
            "not_served_by_mlx_vlm",
            "this reader scores chat-form questions and candidates only; vectors and "
            "the prompt form are mlx-lm's (engines/mlx_vlm.py states what it serves)",
        )
    urls = _image_urls(ask.messages)
    return ItemsJob(
        ask=ask,
        images=[decode_image(url) for url in urls],
        image_key=hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest() if urls else None,
    )


def completion_document(served: str, row: Row) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": row.text},
    }
    if row.logprobs is not None:
        choice["logprobs"] = {"content": row.logprobs}
    choice["finish_reason"] = row.finish_reason
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served,
        "choices": [choice],
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
        if self.path not in ("/v1/chat/completions", "/chat/completions", ITEMS.ITEMS_PATH):
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
        if self.path == ITEMS.ITEMS_PATH:
            self._read_items(body)
        elif is_page_request(body):
            self._read_page(body)
        else:
            self._answer_question(body)

    def _answer_question(self, body: dict[str, Any]) -> None:
        try:
            asked = parse_question(body, self.served)
        except Refusal as refusal:
            self._refuse(refusal)
            return
        assert self.batcher is not None
        self.batcher.submit(asked)
        self._reply(asked)

    def _read_items(self, body: dict[str, Any]) -> None:
        try:
            job = parse_items_job(body, self.served)
        except Refusal as refusal:
            self._refuse(refusal)
            return
        assert self.batcher is not None
        self.batcher.submit(job)
        self._reply(job)

    def _read_page(self, body: dict[str, Any]) -> None:
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
        self._reply(job)

    def _reply(self, job: Job | Asked | ItemsJob) -> None:
        job.done.wait()
        if isinstance(job.error, ITEMS.ItemsRefusal):
            self._send(job.error.status, ITEMS.refusal_document(job.error))
            return
        if job.error is not None:
            self._send(
                500,
                {"error": {"code": "engine_error", "message": repr(job.error), "type": "server_error"}},
            )
            return
        assert job.row is not None
        if isinstance(job, ItemsJob):
            self._send(200, job.row)
            return
        self._send(200, completion_document(self.served, job.row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Crucible's own mlx-vlm server for mlx-darwin, in process: dots.ocr "
            "pages, and the decision door's questions with their top logprobs."
        )
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
