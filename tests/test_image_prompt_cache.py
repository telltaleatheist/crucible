from __future__ import annotations

from .fake_image_worker import worker

BASE = {
    "request_id": "r",
    "prompt": "a red apple",
    "negative_prompt": None,
    "width": 512,
    "height": 512,
    "seed": 1,
    "steps": 4,
    "guidance": 1.0,
    "image_path": None,
    "image_strength": None,
    "mask_path": None,
    "mask_blur": None,
    "output_path": "out.png",
    "revision": "rev-a",
    "backend": "cuda-linux",
}


def key(**changed: object) -> tuple:
    return worker.Job({**BASE, **changed}).prompt_key


def test_the_key_names_prompt_negative_guidance_revision_and_backend() -> None:
    same = key()
    assert key(seed=9, steps=40, width=1024) == same
    assert key(prompt="a green apple") != same
    assert key(negative_prompt="blurry", guidance=4.0) != same
    assert key(revision="rev-b") != same
    assert key(backend="mlx-darwin") != same


def test_the_least_recently_used_entry_goes_first() -> None:
    cache = worker.PromptCache(entries=2, byte_cap=1000)
    cache.put("a", "A", 1)
    cache.put("b", "B", 1)
    assert cache.get("a") == "A"
    cache.put("c", "C", 1)
    assert "b" not in cache and "a" in cache and "c" in cache
    assert len(cache) == 2


def test_the_byte_cap_evicts_and_refuses_what_could_never_fit() -> None:
    cache = worker.PromptCache(entries=10, byte_cap=100)
    cache.put("a", "A", 60)
    cache.put("b", "B", 60)
    assert "a" not in cache and cache.bytes == 60
    cache.put("huge", "H", 101)
    assert "huge" not in cache and cache.bytes == 60


def test_clear_empties_it() -> None:
    cache = worker.PromptCache()
    cache.put("a", "A", 5)
    cache.clear()
    assert len(cache) == 0 and cache.bytes == 0 and cache.get("a") is None
