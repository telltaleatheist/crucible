from __future__ import annotations

QUESTION_MARK = "Q|"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(part["text"] for part in content if part.get("type") == "text")


def apply_chat_template(processor, config, prompt, add_generation_prompt=True, num_images=0, **kwargs):
    if isinstance(prompt, list):
        turns = "|".join(f"{turn['role']}:{_text_of(turn['content'])}" for turn in prompt)
        thinking = kwargs.get("enable_thinking")
        return f"{QUESTION_MARK}images={num_images}|thinking={thinking}|{turns}"
    return ("<|img|><|imgpad|><|endofimg|>" * num_images) + prompt
