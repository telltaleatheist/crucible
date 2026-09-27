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


class _ImageProcessor:

    def __call__(self, images, return_tensors=None):
        grids = []
        for image in images:
            h = max(28, round(image.height / 28) * 28)
            w = max(28, round(image.width / 28) * 28)
            grids.append([1, h // 14, w // 14])
        return {"image_grid_thw": grids}


class FakeProcessor:
    def __init__(self) -> None:
        self.detokenizer = _Detokenizer()
        self.image_processor = _ImageProcessor()


def load(path_or_hf_repo: str, **kwargs):
    delay = os.environ.get("CRUCIBLE_FAKE_MLX_VLM_LOAD_S")
    if delay:
        time.sleep(float(delay))
    return FakeModel(), FakeProcessor()
