from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)

import json
import time
import wave

from workerio import SAMPLE_RATE, decode, send

MIN_PIECE_SECONDS = 0.5

SEARCH_SECONDS = 10.0
ENERGY_WINDOW_MS = 100.0

ENGINES = ("vllm", "mlx-audio", "qwen-asr")

_STATE: dict = {
    "engine": None,
    "model": None,
    "prompt": None,
    "language": None,
    "context": None,
    "max_batch": None,
    "max_new_tokens": None,
    "decoded_source": None,
    "decoded": None,
}


def require(request: dict, key: str, kind):
    return workerio.require(
        request,
        key,
        kind,
        "qwen asr",
        "every parameter is required because every one of them changes the "
        "transcript",
    )


def split_points(wav, max_piece_s: float, joins=()) -> list:
    import numpy

    total = int(wav.shape[0])
    max_len = int(max_piece_s * SAMPLE_RATE)
    if total <= max_len:
        return [(0, total)]
    search = int(min(SEARCH_SECONDS, max_piece_s / 2) * SAMPLE_RATE)
    window = max(4, int(ENERGY_WINDOW_MS / 1000.0 * SAMPLE_RATE))
    spans = []
    start = 0
    while total - start > max_len:
        cut = start + max_len
        left = max(start, cut - search)
        right = cut
        inside = [point for point in joins if left < point <= right]
        if inside:
            boundary = max(inside)
        elif right - left <= window:
            boundary = cut
        else:
            magnitude = numpy.abs(wav[left:right])
            sums = numpy.convolve(
                magnitude, numpy.ones(window, dtype=numpy.float32), mode="valid"
            )
            quietest = int(numpy.argmin(sums))
            boundary = left + quietest + window // 2
        boundary = min(max(boundary, start + 1), total)
        spans.append((start, boundary))
        start = boundary
    spans.append((start, total))
    return spans


def write_wav(path: str, samples) -> None:
    import numpy

    pcm = (numpy.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def read_wav(path: str):
    import numpy

    with wave.open(path, "rb") as handle:
        shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        if shape != (1, 2, SAMPLE_RATE):
            raise ValueError(
                f"{path} is {shape} (channels, sample width, rate); a piece is "
                f"(1, 2, {SAMPLE_RATE}) because this worker wrote it"
            )
        frames = handle.readframes(handle.getnframes())
    return numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32) / 32768.0


def official_prompt(context: str | None, language: str) -> str:
    return (
        f"<|im_start|>system\n{context or ''}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {language}<asr_text>"
    )


def load_vllm(request: dict) -> tuple:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            f"the llm env in {sys.executable} cannot import vllm ({exc}); build "
            "it with `crucible install llm`"
        ) from None
    llm = LLM(
        model=require(request, "model_dir", str),
        dtype=require(request, "dtype", str),
        max_model_len=require(request, "max_model_len", int),
        max_num_seqs=require(request, "max_batch", int),
        kv_cache_memory_bytes=require(request, "kv_cache_memory_bytes", int),
        gpu_memory_utilization=float(
            require(request, "gpu_memory_utilization", (int, float))
        ),
        limit_mm_per_prompt={"audio": 1},
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer, "cuda"


def load_mlx(request: dict) -> tuple:
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_audio.stt.utils import load
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"the llm env in {sys.executable} cannot import mlx-audio ({exc}); "
            "build it with `crucible install llm`"
        ) from None
    model_dir = require(request, "model_dir", str)
    dtype = require(request, "dtype", str)
    if require(request, "max_batch", int) != 1:
        raise RuntimeError(
            "mlx-audio is given one piece per call here; a max_batch other than "
            "1 is a server that thinks this is a different engine"
        )
    model = load(model_dir, lazy=False, strict=True)
    wanted = getattr(mx, dtype, None)
    if wanted is None:
        raise RuntimeError(f"mlx has no dtype {dtype!r}")
    found = {
        str(value.dtype)
        for _, value in tree_flatten(model.parameters())
        if value.dtype in (mx.float16, mx.bfloat16, mx.float32)
    }
    if found != {str(wanted)}:
        raise RuntimeError(
            f"the model loaded in {sorted(found)}, and this job runs it in "
            f"{dtype} only"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer, str(mx.default_device())


def load_qwen_asr(request: dict) -> tuple:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            f"the env in {sys.executable} cannot import qwen_asr/torch ({exc}); "
            "it is the align env's (`crucible install align`)"
        ) from None
    dtype = require(request, "dtype", str)
    device = require(request, "device", str)
    if require(request, "max_batch", int) != 1:
        raise RuntimeError(
            "qwen-asr is given one piece per call here; a max_batch other than "
            "1 is a server that thinks this is a different engine"
        )
    wanted = getattr(torch, dtype, None)
    if not isinstance(wanted, torch.dtype):
        raise RuntimeError(f"torch has no dtype {dtype!r}")
    model = Qwen3ASRModel.from_pretrained(
        require(request, "model_dir", str),
        dtype=wanted,
        device_map=device,
        max_new_tokens=require(request, "max_new_tokens", int),
        max_inference_batch_size=1,
    )
    found = {
        str(p.dtype) for p in model.model.parameters() if p.dtype.is_floating_point
    }
    if found != {str(wanted)}:
        raise RuntimeError(
            f"the model loaded in {sorted(found)}, and this job runs it in {dtype} only"
        )
    return model, model.processor.tokenizer, str(model.model.device)


def load(request: dict) -> None:
    engine = require(request, "engine", str)
    if engine not in ENGINES:
        raise KeyError(f"engine {engine!r} is not one this worker runs; {ENGINES}")
    language = require(request, "language", str)
    if "context" not in request:
        raise KeyError(
            "the qwen asr request has no 'context'; null means none, and the "
            "server sends the key either way"
        )
    context = request["context"]
    if context is not None and not isinstance(context, str):
        raise KeyError(f"'context' must be a string or null, got {type(context).__name__}")
    ceiling = require(request, "context_max_tokens", int)

    started = time.time()
    if engine == "vllm":
        model, tokenizer, device = load_vllm(request)
    elif engine == "qwen-asr":
        model, tokenizer, device = load_qwen_asr(request)
    else:
        model, tokenizer, device = load_mlx(request)
    seconds = time.time() - started

    counted = len(tokenizer.encode(context)) if context else 0
    if counted > ceiling:
        raise RuntimeError(
            f"context is {counted} tokens and this engine was sized for at most "
            f"{ceiling}; send the instruction and the names, not the document"
        )

    _STATE.update(
        engine=engine,
        model=model,
        prompt=official_prompt(context, language),
        language=language,
        context=context,
        max_batch=require(request, "max_batch", int),
        max_new_tokens=require(request, "max_new_tokens", int),
    )
    send(
        "ready",
        seconds=seconds,
        engine=engine,
        device=device,
        dtype=require(request, "dtype", str),
        context_tokens=counted,
    )
    send("done")


def _decoded(ffmpeg: str, source: str, speech, speechonly, progress):
    key = (source, json.dumps(speech, sort_keys=True))
    if _STATE["decoded_source"] == key:
        return _STATE["decoded"]
    wav = decode(ffmpeg, source, progress)
    samples, kept = int(wav.shape[0]), None
    if samples <= 0:
        raise RuntimeError(f"{source} decoded to zero length")
    if speech is not None:
        wav, kept = speechonly.cut_for_worker(wav, speech, source, progress)
    _STATE["decoded_source"], _STATE["decoded"] = key, (wav, samples, kept)
    return wav, samples, kept


def _region_bounds(region, total_samples: int, source: str) -> tuple[int, int]:
    if region is None:
        return 0, total_samples
    region_first = max(0, int(round(float(region[0]) * SAMPLE_RATE)))
    region_last = min(total_samples, int(round(float(region[1]) * SAMPLE_RATE)))
    if region_last <= region_first:
        total = total_samples / float(SAMPLE_RATE)
        raise ValueError(f"region_s {region} holds no audio in {total:.1f}s of {source}")
    return region_first, region_last


def _region_joins(speechonly, kept, region_first: int, region_last: int) -> list:
    if kept is None:
        return []
    return [
        point - region_first
        for point in speechonly.joins(kept)
        if region_first < point < region_last
    ]


def _write_piece(wav, path: str, core: tuple[int, int], pad: int) -> None:
    import numpy

    core_first, core_last = core
    total_samples = int(wav.shape[0])
    audio_first = max(0, core_first - pad)
    audio_last = min(total_samples, core_last + pad)
    samples = wav[audio_first:audio_last]
    minimum = int(MIN_PIECE_SECONDS * SAMPLE_RATE)
    if samples.shape[0] < minimum:
        samples = numpy.pad(samples, (0, minimum - samples.shape[0]))
    write_wav(path, samples)
    send(
        "result",
        offset_s=core_first / float(SAMPLE_RATE),
        duration_s=(core_last - core_first) / float(SAMPLE_RATE),
        audio_offset_s=audio_first / float(SAMPLE_RATE),
        audio_duration_s=(audio_last - audio_first) / float(SAMPLE_RATE),
        wav=path,
    )


def split(request: dict) -> None:
    ffmpeg = require(request, "ffmpeg", str)
    source = require(request, "source", str)
    max_piece_s = float(require(request, "max_piece_s", (int, float)))
    out_dir = require(request, "out_dir", str)
    if "region_s" not in request:
        raise KeyError("the qwen asr request has no 'region_s'; null means the whole source")
    region = request["region_s"]
    overlap_s = float(require(request, "overlap_s", (int, float)))
    if overlap_s < 0:
        raise ValueError(f"overlap_s is {overlap_s}; it is seconds of real audio, at least 0")
    speechonly = workerio.load_sibling("speechonly", __file__)
    speech = speechonly.from_request(request)
    os.makedirs(out_dir, exist_ok=True)
    wav, samples, kept = _decoded(ffmpeg, source, speech, speechonly, workerio.decode_reporter())
    total = int(wav.shape[0]) / float(SAMPLE_RATE)
    region_first, region_last = _region_bounds(region, int(wav.shape[0]), source)
    joins = _region_joins(speechonly, kept, region_first, region_last)
    spans = split_points(wav[region_first:region_last], max_piece_s, joins)
    send(
        "ready",
        duration_s=total,
        pieces=len(spans),
        samples=samples,
        speech_s=None if kept is None else total,
        kept=kept,
    )
    pad = int(round(overlap_s * SAMPLE_RATE))
    stem = os.path.splitext(os.path.basename(source))[0]
    tag = "" if region is None else f".r{region_first}"
    for position, (first, last_sample) in enumerate(spans):
        path = os.path.join(out_dir, f"{stem}{tag}.{position:05d}.wav")
        _write_piece(wav, path, (region_first + first, region_first + last_sample), pad)
    send("done")


def transcribe_vllm(batch: list) -> list:
    from vllm import SamplingParams

    llm = _STATE["model"]
    prompts = [
        {
            "prompt": _STATE["prompt"],
            "multi_modal_data": {"audio": [(read_wav(piece["wav"]), SAMPLE_RATE)]},
        }
        for piece in batch
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=int(piece["max_tokens"]))
        for piece in batch
    ]
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    rows = []
    for piece, output in zip(batch, outputs):
        completion = output.outputs[0]
        rows.append(
            {
                "text": completion.text.strip(),
                "tokens": len(completion.token_ids),
                "hit_token_limit": completion.finish_reason == "length",
            }
        )
    return rows


def transcribe_mlx(batch: list) -> list:
    import mlx.core as mx

    model = _STATE["model"]
    rows = []
    for piece in batch:
        budget = int(piece["max_tokens"])
        result = model.generate(
            read_wav(piece["wav"]),
            max_tokens=budget,
            batch_size=1,
            temperature=0.0,
            language=_STATE["language"],
            system_prompt=_STATE["context"],
            chunk_duration=1200.0,
            verbose=False,
        )
        tokens = int(result.generation_tokens)
        rows.append(
            {
                "text": str(result.text).strip(),
                "tokens": tokens,
                "hit_token_limit": tokens >= budget,
            }
        )
        mx.clear_cache()
    return rows


def transcribe_qwen_asr(batch: list) -> list:
    import torch

    asr = _STATE["model"]
    prompt = asr._build_text_prompt(
        context=_STATE["context"] or "", force_language=_STATE["language"]
    )
    rows = []
    for piece in batch:
        budget = int(piece["max_tokens"])
        inputs = asr.processor(
            text=[prompt], audio=[read_wav(piece["wav"])], return_tensors="pt", padding=True
        )
        inputs = inputs.to(asr.model.device).to(asr.model.dtype)
        with torch.inference_mode():
            out = asr.model.generate(**inputs, max_new_tokens=budget)
        new = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = asr.processor.batch_decode(
            new.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        tokens = int(new.shape[0])
        rows.append(
            {"text": text.strip(), "tokens": tokens, "hit_token_limit": tokens >= budget}
        )
        if asr.model.device.type == "mps":
            torch.mps.empty_cache()
    return rows


def transcribe(request: dict) -> None:
    if _STATE["model"] is None:
        raise RuntimeError("a transcribe request arrived before a load request")
    pieces = require(request, "pieces", list)
    if not pieces:
        raise RuntimeError("a transcribe request with no pieces is not a request")
    for position, piece in enumerate(pieces):
        if not isinstance(piece, dict):
            raise KeyError(f"piece {position} is not an object")
        require(piece, "wav", str)
        require(piece, "max_tokens", int)
    send("ready", pieces=len(pieces))

    size = int(_STATE["max_batch"])
    run = {
        "vllm": transcribe_vllm,
        "mlx-audio": transcribe_mlx,
        "qwen-asr": transcribe_qwen_asr,
    }[_STATE["engine"]]
    done = 0
    for first in range(0, len(pieces), size):
        batch = pieces[first : first + size]
        for row in run(batch):
            send("result", **row)
        done += len(batch)
        send("progress", stage="transcribing", processed=done, total=len(pieces))
    send("done")


OPS = {"load": load, "split": split, "transcribe": transcribe}


def main() -> int:
    workerio.claim_stdout()
    return workerio.serve("qwen asr", OPS)


if __name__ == "__main__":
    sys.exit(main())
