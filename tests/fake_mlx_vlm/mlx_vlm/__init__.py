"""A fake `mlx_vlm` package, importable only through `tests/fake_mlx_vlm` on
PYTHONPATH, for `crucible/engines/mlx_vlm_serve.py`.

It stands in for exactly the names `mlx_vlm_serve.Reader` calls — `load`,
`prompt_utils.apply_chat_template`, `utils.prepare_inputs`,
`utils.should_add_special_tokens`, and `generate.ar`'s `BatchGenerator`,
`_split_prompt_kwargs_per_row`, `_chunked_prefill_enabled`,
`DEFAULT_PREFILL_STEP_SIZE` — in the shapes mlx-vlm 0.7.1 gives them, so the
call sequence under test is the one the real reader makes on a Mac. What it
does NOT do is read a page: the "model" answers every row with a text derived
from the image's size, one token per character, so a test can predict the
answer, count the tokens and set `max_tokens` below it to see `length`.

What it records, and where: every batch's row count and image shape goes to
the file `CRUCIBLE_FAKE_MLX_VLM_BATCHES` names, one JSON line per batch —
that is how a test proves the width and the same-shape rule without reading
the server's stderr.

Two behaviours are selectable by environment, because the tests about them
are tests of the SERVER's handling and not of the model:

    CRUCIBLE_FAKE_MLX_VLM_LOAD_S     seconds `load()` sleeps, so a readiness
                                     test can see /v1/models refuse before the
                                     weights are "in".
    CRUCIBLE_FAKE_MLX_VLM_RAISE_ON   an image height; a batch containing a row
                                     of that height raises from the generator,
                                     which is how a test sees the reader
                                     thread survive a failed batch.
"""

from __future__ import annotations

import os
import time

__version__ = "0.7.1+fake"


class _Config:
    model_type = "dots_ocr"
    image_token_index = 151665


class _Embedding:
    def __init__(self, inputs_embeds: object) -> None:
        self.inputs_embeds = inputs_embeds

    def to_dict(self) -> dict:
        return {"inputs_embeds": self.inputs_embeds}


class FakeModel:
    config = _Config()
    language_model = "the language model"

    def get_input_embeddings(self, input_ids: object, pixel_values: object, mask=None, **kwargs):
        return _Embedding(("embeds", input_ids))


class _Detokenizer:
    """Tokens are code points; `text` is what they spell."""

    def __init__(self) -> None:
        self._codes: list[int] = []
        self.text = ""

    def reset(self) -> None:
        self._codes = []
        self.text = ""

    def add_token(self, token: int) -> None:
        self._codes.append(token)

    def finalize(self) -> None:
        self.text = "".join(chr(code) for code in self._codes)


class FakeProcessor:
    def __init__(self) -> None:
        self.detokenizer = _Detokenizer()


def load(path_or_hf_repo: str, **kwargs):
    delay = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_LOAD_S")
    if delay:
        time.sleep(float(delay))
    return FakeModel(), FakeProcessor()
