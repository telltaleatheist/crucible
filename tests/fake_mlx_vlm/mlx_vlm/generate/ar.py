"""`mlx_vlm.generate.ar`'s batch surface, faithful to the calls the reader makes.

`BatchGenerator.next()` returns `(prompt_responses, generation_responses)` and
each generation response carries `uid`, `token`, `token_logprob` and
`finish_reason` — `"stop"` on the row's last step (a stop token the reader must
NOT append, exactly as upstream's `_generate_batch` skips it), `"length"` when
the row reaches its own `max_tokens`, else None. One token per `next()` per
live row, rows dropping out as they finish, which is the shape upstream has.

What a row says: `page <height>x<width> row <index>` — the fake's whole
"reading" — followed by a stop token.
"""

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
        # Upstream's invariant, and the one a smaller prefill batch broke on
        # the Mac (1 of 12 pages right at prefill 1): both sizes are the batch.
        assert prefill_batch_size == completion_batch_size, "prefill and completion batch differ"
        self.batch_size = completion_batch_size
        self._live: list[dict[str, Any]] = []
        self._uid = 0
        self.closed = False
        self._stepped = False

    def insert(self, prompts, max_tokens, prompt_kwargs=None, **kwargs) -> list[int]:
        # THE ORDER THE READER MUST KEEP: every row before the first next().
        # Inserting into a stepped generator is the continuous-insertion path
        # that read one page in four correctly; the fake refuses it outright.
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
            # The batch is complete the moment the first step runs; that is
            # what a test reads to prove the width and the same-shape rule.
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
