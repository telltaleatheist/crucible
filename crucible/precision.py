"""How many bits a backend block's weights are, and the floor under it.

Owen, 2026-09-26 (fresh-install #48, after package G refused bf16 blocks on a
Turing card): *"we can quantize if we need to. no less than 4."* So a card that
cannot run a model at its best precision is offered the best LOWER one it can
run — float16 where the card has no bf16, then 8-bit, then 4-bit — and nothing
under 4 bits is ever a candidate.

Why this is DERIVED and not a manifest key
------------------------------------------
PHASE9's first decision is that "how heavily is this quantized" already has an
owner and must not grow a second (ARCHITECTURE.md R1): within a family the
declared size answers it, and the best-first walk orders by size. What the floor
needs is narrower — a yes or no to "is this under 4 bits" — and every block
already SAYS its precision in the one place its engine reads it:

    llama-server   the GGUF `file`'s quant tag      Qwen3.5-9B-Q8_0.gguf -> 8
    mlx-lm / -vlm  the repo's precision suffix      mlx-community/...-4bit -> 4
    vllm           `--dtype` in engine_args, or a   `--dtype bfloat16` -> 16,
                   quantization named in the repo   ...-AWQ-INT4 -> 4
    qwen asr/align the block's `dtype`              bfloat16 -> 16

This module reads those and nothing else, so the bits cannot disagree with what
the engine loads. A block whose precision none of them states (faster-whisper's
CTranslate2 float16, the RVC and separator checkpoints) answers None — unknown,
which the floor does not refuse, because nothing in this build is under 4 bits
by any reading and a refusal on a guess is the thing #48 is about.
"""

from __future__ import annotations

import re
from typing import Any

#: THE FLOOR. Owen, 2026-09-26: *"we can quantize if we need to. no less than
#: 4."* A block whose weights are fewer bits than this is never a candidate
#: (`capability.CatalogCandidates`) and a GGUF naming one is refused at load
#: (`manifests._gguf_name`).
MIN_WEIGHT_BITS = 4

#: GGUF quant tags, as llama.cpp names them in file names: `Q4_K_M`, `IQ3_XXS`,
#: `Q8_0`, `BF16`, `F16`, `F32`, with Unsloth's `UD-` prefix allowed in front.
_GGUF_FLOAT = re.compile(r"(?:^|[-_.])(BF16|F16|F32)(?:[-_.]|$)", re.IGNORECASE)
_GGUF_QUANT = re.compile(r"(?:^|[-_.])(?:UD-)?I?Q(\d)(?:_[A-Z0-9]+)*(?:[-_.]|$)", re.IGNORECASE)

#: Precision named in a repo id, as the hubs this catalog pins name it.
_REPO_BITS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"(?:^|[-_/])(?:bf16|fp16|f16)(?:$|[-_])", re.IGNORECASE), 16),
    (re.compile(r"(?:^|[-_/])(?:8bit|int8|fp8|w8a\d+)(?:$|[-_])", re.IGNORECASE), 8),
    (re.compile(r"(?:^|[-_/])(?:4bit|int4|awq|gptq|w4a\d+|nvfp4|mxfp4)(?:$|[-_])", re.IGNORECASE), 4),
    (re.compile(r"(?:^|[-_/])(?:3bit|int3)(?:$|[-_])", re.IGNORECASE), 3),
    (re.compile(r"(?:^|[-_/])(?:2bit|int2)(?:$|[-_])", re.IGNORECASE), 2),
)

#: A dtype a block states, as the bits it stores a weight in.
_DTYPE_BITS: dict[str, int] = {
    "bfloat16": 16,
    "float16": 16,
    "half": 16,
    "float32": 32,
    "float": 32,
}


def gguf_bits(file: str) -> int | None:
    """The weight bits a GGUF file name states, or None if it states none."""
    stem = file[: -len(".gguf")] if file.lower().endswith(".gguf") else file
    floating = _GGUF_FLOAT.search(stem)
    if floating is not None:
        return 32 if floating.group(1).upper() == "F32" else 16
    quant = _GGUF_QUANT.search(stem)
    if quant is not None:
        return int(quant.group(1))
    return None


def repo_bits(hf_repo: str) -> int | None:
    """The weight bits a repo id's name states, or None."""
    name = hf_repo.rsplit("/", 1)[-1]
    for pattern, bits in _REPO_BITS:
        if pattern.search(name):
            return bits
    return None


def weight_bits(spec: Any) -> int | None:
    """What one backend block's weights are stored in, or None if it does not say.

    Asked in the order the engine itself decides: a named file first (that is
    the weights llama-server loads), then a quantization the repo is named for
    (a compressed-tensors or MLX quant is the weights whatever `--dtype` says
    about activations), then a stated dtype.
    """
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
    dtype = getattr(spec, "dtype", None)
    if dtype is None:
        # The engine owns the reading of its own args. Imported here and not at
        # the top because the manifest loader imports this module for its
        # GGUF floor, and the loader must not pull every engine in with it.
        from .engines.vllm import dtype_of

        engine_args = getattr(spec, "engine_args", None)
        if engine_args:
            stated = dtype_of(engine_args)
            dtype = None if stated == "auto" else stated
    if dtype is not None:
        return _DTYPE_BITS.get(dtype)
    return None


def below_floor(bits: int | None) -> bool:
    """Is this under Owen's floor? Unknown is not."""
    return bits is not None and bits < MIN_WEIGHT_BITS


def label(bits: int | None, run_dtype: str | None) -> str:
    """The precision a person reads: `bf16`, `fp16`, `fp32`, `8-bit`, `4-bit`.

    `run_dtype` is what the engine is started in on THIS card, which is what
    tells bf16 from its float16 fallback; `bits` is what the weights are.
    """
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
