"""`apply_chat_template`, in the shape 0.7.1 gives it."""

from __future__ import annotations


def apply_chat_template(processor, config, prompt, add_generation_prompt=True, num_images=0, **kwargs):
    return ("<|img|><|imgpad|><|endofimg|>" * num_images) + prompt
