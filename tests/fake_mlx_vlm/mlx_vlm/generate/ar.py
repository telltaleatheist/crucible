from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_PREFILL_STEP_SIZE = 2048
STOP_TOKEN = 151643


@dataclass
class _Response:
    uid: int
    token: int
    token_logprob: float
    finish_reason: Optional[str]


def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> list[dict]:
    return [dict(prompt_kwargs) for _ in range(batch_size)]


def _chunked_prefill_enabled(model, *, input_ids, inputs_embeds, draft_model, draft_kind, prefill_kwargs) -> bool:
    return True


def _default_prefill_step_size_for_offload(model, prefill_step_size, draft_model, default):
    return prefill_step_size


class BatchGenerator:
    def __init__(self, model, processor, *, prefill_batch_size, completion_batch_size,
                 compute_logprobs, max_tokens, greedy_sampling, prefill_step_size, **kwargs) -> None:
        assert greedy_sampling is True, "the reader decodes greedily"
        assert prefill_batch_size == completion_batch_size, "prefill and completion batch differ"
        self.batch_size = completion_batch_size
        self._live: list[dict[str, Any]] = []
        self._uid = 0
        self.closed = False
        self._stepped = False

    def insert(self, prompts, max_tokens, prompt_kwargs=None, **kwargs) -> list[int]:
        assert not self._stepped, "insert() after next(): continuous insertion"
        if isinstance(max_tokens, int) or max_tokens is None:
            max_tokens = [max_tokens] * len(prompts)
        uids = []
        rows = []
        for row, (prompt, cap) in enumerate(zip(prompts, max_tokens)):
            height, width, _text = prompt
            rows.append((height, width))
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

    def next(self, **kwargs):
        if not self._stepped:
            self._stepped = True
            record = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_BATCHES")
            if record:
                with open(record, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({
                            "rows": len(self._live),
                            "shapes": [row["shape"] for row in self._live],
                            "batch_size": self.batch_size,
                        }) + "\n"
                    )
        responses = []
        still = []
        for row in self._live:
            finish = None
            if row["at"] >= len(row["codes"]):
                token, finish = STOP_TOKEN, "stop"
            else:
                token = row["codes"][row["at"]]
                row["at"] += 1
                if row["at"] >= row["cap"]:
                    finish = "length"
            responses.append(_Response(uid=row["uid"], token=token, token_logprob=0.0, finish_reason=finish))
            if finish is None:
                still.append(row)
        self._live = still
        return [], responses

    def close(self) -> None:
        self.closed = True
