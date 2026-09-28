from __future__ import annotations

from .prompt_utils import QUESTION_MARK


class FakeArray:
    def __init__(self, rows: list[list[object]], prompt_tokens: int) -> None:
        self._rows = rows
        self.shape = (len(rows), prompt_tokens)

    def tolist(self) -> list[list[object]]:
        return self._rows


def _question_inputs(images, prompt: str) -> dict:
    seen = [[image.width, image.height, list(image.getpixel((0, 0)))] for image in images or []]
    prompt_tokens = 20 + len(prompt) + sum((w * h) // 1024 for w, h, _ in seen)
    return {
        "input_ids": FakeArray([{"images": seen, "prompt": prompt}], prompt_tokens),
        "pixel_values": "pixels" if seen else None,
        "attention_mask": "mask",
    }


def prepare_inputs(processor, images, audio, prompts, image_token_index, resize_shape,
                   add_special_tokens, pad_to_uniform_size):
    if len(prompts) == 1 and prompts[0].startswith(QUESTION_MARK):
        return _question_inputs(images, prompts[0])
    if len(images) != 1:
        raise AssertionError(f"prepare_inputs was handed {len(images)} images; the reader embeds one at a time")
    height, width = images[0].height, images[0].width
    height = max(28, round(height / 28) * 28)
    width = max(28, round(width / 28) * 28)
    prompt_tokens = 34 + (height * width) // 196
    rows = [[height, width, prompt] for prompt in prompts]
    return {
        "input_ids": FakeArray(rows, prompt_tokens),
        "pixel_values": "pixels",
        "attention_mask": "mask",
        "image_grid_thw": "grid",
    }


def should_add_special_tokens(model_type: str, processor) -> bool:
    return False
