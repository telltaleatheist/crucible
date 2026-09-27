from __future__ import annotations

import re
from typing import Any

from .enginespec import declared_dtype

MIN_WEIGHT_BITS = 4

_GGUF_FLOAT = re.compile(r"(?:^|[-_.])(BF16|F16|F32)(?:[-_.]|$)", re.IGNORECASE)
_GGUF_QUANT = re.compile(r"(?:^|[-_.])(?:UD-)?I?Q(\d)(?:_[A-Z0-9]+)*(?:[-_.]|$)", re.IGNORECASE)

_REPO_BITS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"(?:^|[-_/])(?:bf16|fp16|f16)(?:$|[-_])", re.IGNORECASE), 16),
    (re.compile(r"(?:^|[-_/])(?:8bit|int8|fp8|w8a\d+)(?:$|[-_])", re.IGNORECASE), 8),
    (re.compile(r"(?:^|[-_/])(?:4bit|int4|awq|gptq|w4a\d+|nvfp4|mxfp4)(?:$|[-_])", re.IGNORECASE), 4),
    (re.compile(r"(?:^|[-_/])(?:3bit|int3)(?:$|[-_])", re.IGNORECASE), 3),
    (re.compile(r"(?:^|[-_/])(?:2bit|int2)(?:$|[-_])", re.IGNORECASE), 2),
)

_DTYPE_BITS: dict[str, int] = {
    "bfloat16": 16,
    "float16": 16,
    "half": 16,
    "float32": 32,
    "float": 32,
}


def gguf_bits(file: str) -> int | None:
    stem = file[: -len(".gguf")] if file.lower().endswith(".gguf") else file
    floating = _GGUF_FLOAT.search(stem)
    if floating is not None:
        return 32 if floating.group(1).upper() == "F32" else 16
    quant = _GGUF_QUANT.search(stem)
    if quant is not None:
        return int(quant.group(1))
    return None


def repo_bits(hf_repo: str) -> int | None:
    name = hf_repo.rsplit("/", 1)[-1]
    for pattern, bits in _REPO_BITS:
        if pattern.search(name):
            return bits
    return None


def weight_bits(spec: Any) -> int | None:
    file = getattr(spec, "file", None)
    if file:
        found = gguf_bits(file)
        if found is not None:
            return found
    hf_repo = getattr(spec, "hf_repo", None)
    if hf_repo:
        found = repo_bits(hf_repo)
        if found is not None:
            return found
    dtype = declared_dtype(spec)
    if dtype is not None:
        return _DTYPE_BITS.get(dtype)
    return None


def below_floor(bits: int | None) -> bool:
    return bits is not None and bits < MIN_WEIGHT_BITS


def label(bits: int | None, run_dtype: str | None) -> str:
    if bits is not None and bits < 16:
        return f"{bits}-bit"
    if run_dtype in ("bfloat16",):
        return "bf16"
    if run_dtype in ("float16", "half"):
        return "fp16"
    if run_dtype in ("float32", "float"):
        return "fp32"
    if bits == 16:
        return "16-bit"
    if bits == 32:
        return "fp32"
    return "its native precision"


__all__ = [
    "MIN_WEIGHT_BITS",
    "below_floor",
    "gguf_bits",
    "label",
    "repo_bits",
    "weight_bits",
]
