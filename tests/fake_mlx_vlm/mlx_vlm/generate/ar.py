from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_PREFILL_STEP_SIZE = 2048
STOP_TOKEN = 151643

QUESTION_ANSWER = "A"

QUESTION_PROBS: tuple[tuple[str, float], ...] = (
    ("A", 0.6), ("B", 0.3), ("C", 0.05), ("D", 0.02), ("E", 0.01),
    ("F", 0.005), ("G", 0.002), ("H", 0.001),
)


@dataclass
class _Response:
    uid: int
    token: int
    token_logprob: float
    finish_reason: Optional[str]
    top_logprobs: Optional[list[tuple[int, float]]] = None


class _Logits:
    dtype = "bfloat16"

    def astype(self, dtype: str) -> "_Logits":
        widened = _Logits()
        widened.dtype = dtype
        return widened


def _record(entry: dict) -> None:
    record = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_BATCHES")
    if record:
        with open(record, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> list[dict]:
    return [dict(prompt_kwargs) for _ in range(batch_size)]


def _chunked_prefill_enabled(model, *, input_ids, inputs_embeds, draft_model, draft_kind, prefill_kwargs) -> bool:
    return True


def _default_prefill_step_size_for_offload(model, prefill_step_size, draft_model, default):
    return prefill_step_size


def _logits_dtype(processors) -> str:
    logits = _Logits()
    for processor in processors or []:
        logits = processor([], logits)
    return logits.dtype


class BatchGenerator:
    def __init__(self, model, processor, *, prefill_batch_size, completion_batch_size,
                 compute_logprobs, max_tokens, greedy_sampling, prefill_step_size,
                 top_logprobs_k=0, **kwargs) -> None:
        assert greedy_sampling is True, "the reader decodes greedily"
        assert prefill_batch_size == completion_batch_size, "prefill and completion batch differ"
        self.batch_size = completion_batch_size
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self._live: list[dict[str, Any]] = []
        self._uid = 0
        self.closed = False
        self._stepped = False

    def _question(self, prompt: dict, cap: int, processors) -> dict[str, Any]:
        _record({
            "question": True,
            "images": prompt["images"],
            "prompt": prompt["prompt"],
            "max_tokens": cap,
            "compute_logprobs": self.compute_logprobs,
            "top_logprobs_k": self.top_logprobs_k,
            "logits_dtype": _logits_dtype(processors),
        })
        return {"uid": self._uid, "codes": [ord(c) for c in QUESTION_ANSWER], "at": 0, "cap": cap,
                "question": True}

    def insert(self, prompts, max_tokens, prompt_kwargs=None, logits_processors=None, **kwargs) -> list[int]:
        assert not self._stepped, "insert() after next(): continuous insertion"
        if isinstance(max_tokens, int) or max_tokens is None:
            max_tokens = [max_tokens] * len(prompts)
        processors = logits_processors or [None] * len(prompts)
        uids = []
        for row, (prompt, cap, chosen) in enumerate(zip(prompts, max_tokens, processors)):
            if isinstance(prompt, dict):
                self._live.append(self._question(prompt, cap, chosen))
                uids.append(self._uid)
                self._uid += 1
                continue
            height, width, _text = prompt
            raise_on = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_RAISE_ON")
            if raise_on and int(raise_on) == height:
                raise RuntimeError(f"the fake model refuses a page {height} tall")
            spelled = f"page {height}x{width} row {row}"
            self._live.append({"uid": self._uid, "codes": [ord(c) for c in spelled], "at": 0, "cap": cap,
                               "shape": (height, width)})
            uids.append(self._uid)
            self._uid += 1
        return uids

    @property
    def has_work(self) -> bool:
        return bool(self._live)

    def _top(self) -> Optional[list[tuple[int, float]]]:
        if self.top_logprobs_k <= 0:
            return None
        return [(ord(text), math.log(p)) for text, p in QUESTION_PROBS[: self.top_logprobs_k]]

    def _step(self, row: dict[str, Any]) -> _Response:
        finish = None
        if row["at"] >= len(row["codes"]):
            token, finish = STOP_TOKEN, "stop"
        else:
            token = row["codes"][row["at"]]
            row["at"] += 1
            if row["at"] >= row["cap"]:
                finish = "length"
        if not row.get("question"):
            return _Response(uid=row["uid"], token=token, token_logprob=0.0, finish_reason=finish)
        chosen = dict((ord(text), math.log(p)) for text, p in QUESTION_PROBS)
        return _Response(
            uid=row["uid"],
            token=token,
            token_logprob=chosen.get(token, -30.0) if self.compute_logprobs else 0.0,
            finish_reason=finish,
            top_logprobs=self._top(),
        )

    def next(self, **kwargs):
        if not self._stepped:
            self._stepped = True
            pages = [row for row in self._live if not row.get("question")]
            if pages:
                _record({
                    "rows": len(pages),
                    "shapes": [row["shape"] for row in pages],
                    "batch_size": self.batch_size,
                })
        responses = [self._step(row) for row in self._live]
        self._live = [row for row, response in zip(self._live, responses) if response.finish_reason is None]
        return [], responses

    def close(self) -> None:
        self.closed = True
