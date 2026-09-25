"""The Qwen3-ASR worker: vLLM on cuda-linux; on mlx-darwin Qwen's own `qwen_asr`
on torch MPS (`qwen3-asr-1.7b`) or the mlx-audio port (`qwen3-asr-1.7b-mlx`,
the fast one). One wire for all three.

THE THIRD ENGINE, 2026-09-24: on identical pieces, Qwen's own package heard a
few more fillers than the MLX port (11-13 against 9-10 on a 10-minute window)
at 2.5x the time, so the Mac's official id runs the package (in the align env,
which has it) and the port kept an id of its own for speed. `load` takes a `device` key, the torch
device for `qwen-asr` and null for the other two.

(The paragraphs below were written for the two-engine worker and still hold
for vLLM and mlx-audio.)

docs/PHASE25-QWEN-ASR.md section 4. Run as `<llm env python> qwen_worker.py`,
held open for ONE JOB by `crucible/jobs/asr/qwen.py` (a `WorkerSession` the job
owns and stops in its `finally`), and never resident beyond it.

This module is **standalone**: it imports the standard library, `numpy`, and
then EITHER `vllm` OR `mlx_audio` — never `crucible`, which the llm env does not
have. It is one file for both engines, unlike whisper's two workers, because
here the engines differ in about sixty lines (load and decode) and share
everything else: the ffmpeg decode, the quiet-point split, the wav files both
models read, and the wire. Each engine's import happens inside its own branch,
so the Mac never imports vLLM and the PC never imports MLX.

The wire, in full
-----------------
    stdin   one object per line; the op is required. Nothing has a default.

            {"op": "load", "engine", "model_dir", "dtype", "max_batch",
             "max_new_tokens", "max_model_len", "kv_cache_memory_bytes",
             "gpu_memory_utilization", "language", "context",
             "context_max_tokens"}
                -> ready {seconds, engine, device, dtype, context_tokens}, done
            The last three vLLM keys are null on mlx-audio. `language` is the
            model's English NAME ("English"), mapped by the server. The context
            is fixed for the session because the session is one job.

            {"op": "split", "ffmpeg", "source", "max_piece_s", "out_dir"}
                -> ready {duration_s, pieces}
                   result {offset_s, duration_s, wav}     one per piece
                   done
            Decode `source` with ffmpeg to 16 kHz mono, cut it into pieces of
            at most `max_piece_s` at the quietest point near each boundary, and
            write each piece to `out_dir` as a 16-bit PCM wav. `offset_s` is
            relative to `source`. The server calls this on the job's input,
            and again on one piece's own wav when that piece is re-decoded at a
            smaller window (`loopguard.WINDOW_LADDER_SECONDS`).

            {"op": "transcribe", "pieces": [{"wav", "max_tokens"}]}
                -> ready {pieces}
                   result {text, tokens, hit_token_limit}  one per piece
                   done

    fd 1    `ready` / `progress` / `result` / `failed` / `done`, and nothing
            else. No index anywhere: results are matched by position.

The prompt, and the one place the two engines differ
----------------------------------------------------
On vLLM this worker writes the prompt itself, in the OFFICIAL format — the
model repo's own `chat_template.json` at the pinned revision, which always
emits a system turn and puts the context verbatim before `<|im_end|>`, followed
by `language {Name}<asr_text>` to force the language exactly as `qwen_asr`
0.0.6's `_build_text_prompt` does. That is the prompt ContentStudio measured.
(vLLM's own `/v1/audio/transcriptions` omits the system turn when there is no
context; this does not.)

On mlx-audio the prompt is the library's (`Qwen3ASRModel._build_prompt`), and
it differs in ONE character: it writes `{context}\\n` before `<|im_end|>`. Not
patched here — a patch to a library's private prompt builder is a patch that
breaks silently on its next release — and written down in PHASE25 section 8 as
a comparison owed on the first live run.

Neither engine applies `qwen_asr`'s `detect_and_fix_repetitions`, which
silently collapses a run of repeated characters or short patterns in the
official SDK's `parse_asr_output`. What the model wrote is what comes back; a
loop is the server's to detect and name (`loopguard.py`), not this worker's to
tidy away.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import wave

#: fd 1, once `claim_stdout` has moved it out of every library's reach.
_RESULTS = None


def claim_stdout() -> None:
    """fd 1 is results, stderr is everything else — before any engine import.

    The first thing `main` does. vLLM logs to stdout through its own handler at
    import time and tqdm writes progress bars; either would land in the middle
    of a JSON line otherwise. It is a function rather than module-level code
    (the other workers' form) only so the pure parts of this file can be
    imported by a test without taking the test runner's stdout: nothing above
    it imports anything that prints, and `vllm` and `mlx_audio` are imported
    only inside the `load` op, long after `main` has called this.
    """
    global _RESULTS
    results_fd = os.dup(1)
    os.dup2(2, 1)
    _RESULTS = os.fdopen(results_fd, "w", encoding="utf-8", buffering=1)

#: Both models work at 16 kHz mono. Not a parameter: it is the rate the feature
#: extractor was trained at.
SAMPLE_RATE = 16_000

#: `qwen_asr` 0.0.6's `MIN_ASR_INPUT_SECONDS`: a piece shorter than this is
#: zero-padded at its tail to this length, because the audio tower has nothing
#: to encode below it. The padding is silence and the REPORTED duration stays
#: the real one, so no timestamp moves.
MIN_PIECE_SECONDS = 0.5

#: The quiet-point search: the last 10 s before each nominal boundary is
#: searched for the 100 ms window with the least energy, and the cut is made at
#: its quietest sample. `qwen_asr` 0.0.6's `split_audio_into_chunks` searches
#: 5 s EITHER side, so its pieces run up to 5 s past the limit it names; this
#: searches the same 10 s entirely on the near side, so a piece is never longer
#: than `max_piece_s` — the figure the aligner's trust, the token budget and
#: `max_model_len` were all computed from.
SEARCH_SECONDS = 10.0
ENERGY_WINDOW_MS = 100.0

DECODE_REPORT_SECONDS = 1.0

ENGINES = ("vllm", "mlx-audio", "qwen-asr")

_STATE: dict = {
    "engine": None,
    "model": None,
    "prompt": None,
    "language": None,
    "context": None,
    "max_batch": None,
    "max_new_tokens": None,
}


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> None:
    send("failed", message=message)


def require(request: dict, key: str, kind):
    """One required key of the stated type, or a refusal naming it."""
    if key not in request:
        raise KeyError(
            f"the qwen asr request has no {key!r}; every parameter is required "
            "because every one of them changes the transcript"
        )
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        names = "/".join(getattr(k, "__name__", str(k)) for k in kinds)
        raise KeyError(
            f"the qwen asr request's {key!r} must be {names}, got "
            f"{type(value).__name__}"
        )
    return value


# ------------------------------------------------------------------- decoding


def decode(ffmpeg: str, source: str, on_progress):
    """Any container -> mono float32 at 16 kHz, through ffmpeg. Raises on failure.

    The same decode every `asr` and `align` worker uses, for their reason: PyAV
    silently truncates some assembled m4b files. stderr is drained on a thread
    so an error-spewing decode cannot fill the pipe and deadlock.
    """
    import numpy

    process = subprocess.Popen(
        [
            ffmpeg, "-nostdin", "-v", "error", "-i", source, "-map", "0:a:0",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errors: list = []

    def drain() -> None:
        for block in iter(lambda: process.stderr.read(65536), b""):
            errors.append(block)

    pump = threading.Thread(target=drain, daemon=True)
    pump.start()
    buffer = bytearray()
    while True:
        block = process.stdout.read(1 << 20)
        if not block:
            break
        buffer += block
        on_progress(len(buffer) / (SAMPLE_RATE * 4))
    process.stdout.close()
    code = process.wait()
    pump.join(timeout=5)
    if code != 0:
        tail = b"".join(errors).decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg exited {code} on {source}: {tail}")
    return numpy.frombuffer(buffer, dtype=numpy.float32)


def split_points(wav, max_piece_s: float) -> list:
    """`[(start_sample, end_sample)]` covering `wav` exactly, no gaps, no overlap.

    After `qwen_asr` 0.0.6's `split_audio_into_chunks` (Apache-2.0, Alibaba
    Qwen team), ported rather than imported because the llm env this runs in
    does not have `qwen_asr`, with the search moved to the near side of each
    cut (`SEARCH_SECONDS`): every boundary is the quietest sample of the
    quietest 100 ms window in the 10 s before the nominal cut.
    """
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
        if right - left <= window:
            boundary = cut
        else:
            magnitude = numpy.abs(wav[left:right])
            sums = numpy.convolve(
                magnitude, numpy.ones(window, dtype=numpy.float32), mode="valid"
            )
            quietest = int(numpy.argmin(sums))
            inner = int(numpy.argmin(magnitude[quietest : quietest + window]))
            boundary = left + quietest + inner
        boundary = min(max(boundary, start + 1), total)
        spans.append((start, boundary))
        start = boundary
    spans.append((start, total))
    return spans


def write_wav(path: str, samples) -> None:
    """16-bit PCM mono at 16 kHz, with the standard library's own writer.

    16-bit because it is what the aligner worker has always fed its model
    (`crucible/jobs/align/worker.py` writes PCM_16 before `model.align`), so
    the ASR model and the aligner hear the same samples.
    """
    import numpy

    pcm = (numpy.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def read_wav(path: str):
    """What `write_wav` wrote, back as float32 in [-1, 1]. Refuses anything else."""
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


# -------------------------------------------------------------------- engines


def official_prompt(context: str | None, language: str) -> str:
    """The model repo's own chat template, rendered, plus the forced language.

    `chat_template.json` at the pinned revision: a system turn ALWAYS (empty
    when there is no context), the user turn holding the audio placeholder
    vLLM expands (`Qwen3ASRForConditionalGeneration.get_placeholder_str`), and
    the generation prompt. `language {Name}<asr_text>` is `qwen_asr` 0.0.6's
    `_build_text_prompt` forcing the language, which makes the output the
    transcript alone.
    """
    return (
        f"<|im_start|>system\n{context or ''}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {language}<asr_text>"
    )


def load_vllm(request: dict) -> tuple:
    """vLLM, in-process, with every number from the manifest."""
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
        # With `kv_cache_memory_bytes` set this is only vLLM's startup gate —
        # free memory must be at least this share of the card
        # (`v1/worker/utils.py` `request_memory`) — and the server sets it to
        # the manifest's estimate over the card, the same figure its guard used.
        gpu_memory_utilization=float(
            require(request, "gpu_memory_utilization", (int, float))
        ),
        limit_mm_per_prompt={"audio": 1},
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer, "cuda"


def load_mlx(request: dict) -> tuple:
    """mlx-audio's Qwen3-ASR, reading the official checkpoint and converting it."""
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
    # STRICT: every tensor the model has must come from the checkpoint and
    # every tensor in the checkpoint must land somewhere. The pinned revision
    # holds `thinker.audio_tower.*` (397), `thinker.model.*` (310) and the tied
    # `thinker.lm_head.weight` that `sanitize` drops, and nothing else; a
    # release that renamed a layer must be a refusal, not a model running
    # with a freshly initialised one.
    model = load(model_dir, lazy=False, strict=True)
    # FULL PRECISION IS CHECKED, NOT ASSUMED (Owen, 2026-09-24). The weights
    # are bf16 on disk and mlx-audio keeps a checkpoint's dtype; a library that
    # quietly cast would be running a different model under the same id.
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
    """Qwen's own `qwen_asr` package on torch, as ContentStudio ran it.

    The Mac's official engine since 2026-09-24: on identical pieces it heard a
    few more fillers than the MLX port (asr/qwen3-asr-1.7b.toml has the table).
    It runs in the ALIGN env, which pins exactly ContentStudio's versions.

    NO FORCED ALIGNER HERE. The package can load one beside the model; Crucible
    runs the aligner as its own session (`qwen.py`), so this loads the ASR
    model alone. `max_inference_batch_size=1` is stated rather than left at the
    package's 32, which aborted an MPS process outright.
    """
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
    # FULL PRECISION IS CHECKED, NOT ASSUMED (Owen, 2026-09-24).
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

    # THE CONTEXT'S LENGTH, in this model's own tokens, before a second of
    # audio is decoded. `max_model_len` was sized for at most `ceiling`, and a
    # longer one would end a 180 s piece's decode early with nothing to say so.
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


# ---------------------------------------------------------------------- split


def split(request: dict) -> None:
    ffmpeg = require(request, "ffmpeg", str)
    source = require(request, "source", str)
    max_piece_s = float(require(request, "max_piece_s", (int, float)))
    out_dir = require(request, "out_dir", str)
    os.makedirs(out_dir, exist_ok=True)

    last = [0.0]

    def progress(decoded_s: float) -> None:
        now = time.time()
        if now - last[0] < DECODE_REPORT_SECONDS:
            return
        last[0] = now
        send("progress", stage="decoding", processed_s=round(decoded_s, 1))

    wav = decode(ffmpeg, source, progress)
    total = wav.shape[0] / float(SAMPLE_RATE)
    if total <= 0:
        raise RuntimeError(f"{source} decoded to zero length")
    spans = split_points(wav, max_piece_s)
    send("ready", duration_s=total, pieces=len(spans))

    import numpy

    minimum = int(MIN_PIECE_SECONDS * SAMPLE_RATE)
    stem = os.path.splitext(os.path.basename(source))[0]
    for position, (first, last_sample) in enumerate(spans):
        samples = wav[first:last_sample]
        if samples.shape[0] < minimum:
            samples = numpy.pad(samples, (0, minimum - samples.shape[0]))
        path = os.path.join(out_dir, f"{stem}.{position:05d}.wav")
        write_wav(path, samples)
        send(
            "result",
            offset_s=first / float(SAMPLE_RATE),
            duration_s=(last_sample - first) / float(SAMPLE_RATE),
            wav=path,
        )
    send("done")


# ----------------------------------------------------------------- transcribe


def transcribe_vllm(batch: list) -> list:
    """One `generate` over up to `max_batch` pieces; vLLM batches them."""
    from vllm import SamplingParams

    llm = _STATE["model"]
    prompts = [
        {
            "prompt": _STATE["prompt"],
            # With the rate stated, vLLM's parser resamples nothing
            # (`AudioResampler.resample` returns the input when the rates
            # match), which is why the llm env needs none of vllm's [audio]
            # extras for this.
            "multi_modal_data": {"audio": [(read_wav(piece["wav"]), SAMPLE_RATE)]},
        }
        for piece in batch
    ]
    params = [
        # Greedy. Temperature 0 is the SDK's own (`qwen_asr`'s vLLM backend
        # builds `SamplingParams(temperature=0.0, max_tokens=...)`), and it is
        # what makes a re-decode of the same audio a reproduction.
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
    """One piece per `generate` call, for the loop guard's sake (asrmodels.py)."""
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
            # Longer than any piece, so mlx-audio never cuts one again: the
            # pieces are already cut, at the aligner's limit.
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
        # MLX keeps freed buffers for reuse until told otherwise; between
        # pieces they are memory nothing will ask for again.
        mx.clear_cache()
    return rows


def transcribe_qwen_asr(batch: list) -> list:
    """One piece per `generate`, the way `qwen_asr` 0.0.6 runs its transformers
    backend (`Qwen3ASRModel._infer_asr_transformers`), minus two things.

    The prompt is the package's own `_build_text_prompt` (the repo's chat
    template plus `language {Name}<asr_text>`), and the inputs go through its
    own processor, so the model hears exactly what it heard in ContentStudio's
    run. What is NOT the package's: the token count, which its public
    `transcribe` does not return and the loop guard needs, and its
    `parse_asr_output`, whose `detect_and_fix_repetitions` silently collapses
    repeats (the module docstring).
    """
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
        # torch's MPS caching allocator keeps every freed block; between pieces
        # they are memory nothing will ask for again.
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


# ----------------------------------------------------------------------- main


OPS = {"load": load, "split": split, "transcribe": transcribe}


def main() -> int:
    claim_stdout()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"the qwen asr request is not JSON: {exc}")
            return 1
        if not isinstance(request, dict):
            fail(f"the qwen asr request must be an object, got {type(request).__name__}")
            return 1
        handler = OPS.get(request.get("op"))
        if handler is None:
            fail(f"the qwen asr request's op is {request.get('op')!r}; this worker takes {sorted(OPS)}")
            return 1
        try:
            handler(request)
        except KeyError as exc:
            fail(str(exc.args[0]))
            return 1
        except Exception as exc:
            # A whole request failed. The session is one job, and the job is
            # over, so this exits rather than waiting for a line it cannot use.
            fail(f"{type(exc).__name__}: {exc}")
            return 1
    # EOF: the job stopped the session politely. Exit 0 so `stop()` sees a
    # worker that went when asked, releasing the card the way its library
    # expects to.
    return 0


if __name__ == "__main__":
    sys.exit(main())
