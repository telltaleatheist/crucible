"""`prepare_inputs` and `should_add_special_tokens`, in the shape 0.7.1 gives them.

The fake `input_ids` carries the images' sizes and the prompts, because that is
all the fake generator needs to spell its answer; `shape[1]` is what the reader
reports as `prompt_tokens`, so it is a number a test can predict:
34 + (height * width) // 196, the way a real 14-pixel patch grid merged 2x2
comes out (the constant is not upstream's; it only needs to be deterministic).
"""

from __future__ import annotations


class FakeArray:
    def __init__(self, rows: list[list[object]], prompt_tokens: int) -> None:
        self._rows = rows
        self.shape = (len(rows), prompt_tokens)

    def tolist(self) -> list[list[object]]:
        return self._rows


def prepare_inputs(processor, images, audio, prompts, image_token_index, resize_shape,
                   add_special_tokens, pad_to_uniform_size):
    # ONE image per call is the reader's contract since the watchdog finding
    # (the vision tower runs per image); a batch here is the old path.
    if len(images) != 1:
        raise AssertionError(f"prepare_inputs was handed {len(images)} images; the reader embeds one at a time")
    height, width = images[0].height, images[0].width
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
