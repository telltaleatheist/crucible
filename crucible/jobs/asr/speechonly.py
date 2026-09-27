"""Speech only: Silero VAD on the CPU, the audio it keeps, and the way back.

Owen, 2026-09-27: *"Sending silences through an asr model produces
hallucination and nonsense"* — in Qwen3-ASR and in whisper both — and *"Go
ahead and write it. We can use the speech detector, that's fine."* A caller
who sends `speech_only: true` (`AsrParams`) has every long stretch without
speech taken out of the audio before any model hears it, and gets its
timestamps back on the ORIGINAL timeline with a list of what was taken out.

A SPEECH DETECTOR, NOT A LOUDNESS THRESHOLD. auto-editor-style cutting keeps
whatever is louder than a level, so it cannot tell music, room tone or a breath
from a sentence; a model trained to hear speech can. Silero VAD is that model
(MIT, snakers4/silero-vad), the same detector faster-whisper ships.

This module is STANDALONE, like the workers that import it: the standard
library and numpy, nothing from `crucible`. The workers run as
`<env python> <worker>.py` with this directory first on `sys.path`, so they
import it as `speechonly`; the server imports it as `crucible.jobs.asr.
speechonly`. numpy is imported inside the functions that need it, so the
server can read the constants and `Timeline` without it.

THE RUNTIME IS NUMPY, AND WHY (2026-09-27)
-------------------------------------------
The detector has to run inside whichever env the job's ASR worker runs in, on
the CPU, never on the card (it is busy):

    faster-whisper   asr env, cuda-linux    onnxruntime 1.30.0, no torch
    mlx-whisper      asr env, mlx-darwin    torch 2.14.0, no onnxruntime
    vllm             llm env, cuda-linux    torch 2.13.0, no onnxruntime
    qwen-asr         align env, mlx-darwin  torch 2.14.0, no onnxruntime
    mlx-audio        llm env, mlx-darwin    neither

No runtime is in all five, and adding one to a recipe makes every installed
env of it read as not installed until it is rebuilt (`jobenv.env_status`
compares every pin) — a vLLM reinstall on the PC to add a 1.2 MB detector that
is off by default. numpy is in all five. Silero publishes the network written
out in plain tensor operations (`silero_vad/tinygrad_model.py` in the wheel);
`probabilities` below is that forward pass in numpy, over the weights of the
release's own ONNX export, read straight out of the file's protobuf.

CHECKED, 2026-09-27: on real speech, these probabilities against onnxruntime
1.30.0 running the release's DEFAULT model (`silero_vad.onnx`, what Silero's
`load_silero_vad(onnx=True)` loads) differ by at most 8.6e-6. One trap found
doing it: the `silero_vad_16k.safetensors` in the same wheel holds OLDER
weights (up to 0.42 apart from the default model on the same audio), so it is
not what is pinned.

THE PIN. silero-vad 6.2.1 (PyPI wheel `silero_vad-6.2.1-py3-none-any.whl`),
file `silero_vad/data/silero_vad_op18_ifless.onnx` — the default model's
weights as plain initializers (onnxruntime puts it within 2.7e-7 of
`silero_vad.onnx`) — re-hosted byte for byte on Crucible's `tools` release
(`hosttools.SILERO_VAD`), its sha256 checked when it is placed and again by
the worker that reads it.

WHAT IS KEPT: GENEROUS, BECAUSE DROPPED OPENINGS ARE A BUG WE JUST FIXED
-----------------------------------------------------------------------
docs/PHASE25-QWEN-ASR.md: a piece whose first word sits at sample zero loses
that word. So every rule here leans towards keeping audio:

- Every 32 ms frame the detector scores at or above `threshold` is speech. The
  default is low (`DEFAULT_THRESHOLD` 0.3; Silero's own is 0.5), and there is
  no minimum speech length: a single frame of speech is kept.
- Every speech run is widened by `pad_s` (0.3 s) of real audio on both sides.
- A stretch between two kept regions is REMOVED only if it is at least
  `min_gap_s` (2 s) long AFTER that padding; any shorter pause stays in, as
  context. So no removed region is shorter than `min_gap_s`.

THE TIMELINE
------------
The kept regions are concatenated into one shorter signal, and the worker cuts
and transcribes THAT, exactly as it would the original. Every time that comes
back is on the shortened timeline, and `Timeline` maps it back — sample-exact,
because the table is integer samples — the way a piece's offset already maps
its own times. A caller never sees the shortened version.

A join is where two kept regions meet in the shortened signal: `2 * pad_s` of
non-speech, the best place there is to cut. The Qwen piece cutter cuts at a
join whenever one falls in its search span (`qwen_worker.split_points`).
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

#: The detector's rate and frame: 512 new samples a call at 16 kHz (32 ms),
#: read with the 64 samples before them. Silero's, not a choice.
SAMPLE_RATE = 16_000
FRAME = 512
CONTEXT = 64

#: The caller's three knobs: default, lowest, highest. The bounds are refusals
#: in `AsrParams`, each with a sentence saying what to send instead.
#:
#: threshold: 0.3 and not Silero's 0.5, because the cost of calling speech
#: silence is a little silence kept, and the cost of calling silence speech's
#: opposite is a lost line. Below 0.1 nearly every frame of a real recording
#: scores as speech and nothing is removed; above 0.7 quiet speech is lost.
DEFAULT_THRESHOLD = 0.3
MIN_THRESHOLD = 0.1
MAX_THRESHOLD = 0.7
#: pad_s: ~300 ms of real audio each side of every speech run (Owen's figure).
#: Never under 0.1 s: a region cut tight against its first word is the
#: dropped-opening bug by another route.
DEFAULT_PAD_S = 0.3
MIN_PAD_S = 0.1
MAX_PAD_S = 2.0
#: min_gap_s: the shortest stretch that is ever removed. 2 s (Owen's figure):
#: a narrator's pause between sentences, and a breath, stay in.
DEFAULT_MIN_GAP_S = 2.0
MIN_MIN_GAP_S = 1.0
MAX_MIN_GAP_S = 60.0

#: What the detector is, for the transcript's `speech` block.
DETECTOR = "silero-vad 6.2.1 (silero_vad_op18_ifless.onnx weights), numpy on the CPU"

#: The tensors the network is, and their shapes. A file with anything else is
#: refused rather than half-read.
_SHAPES: dict[str, tuple[int, ...]] = {
    "stft_conv.weight": (258, 1, 256),
    "conv1.weight": (128, 129, 3),
    "conv1.bias": (128,),
    "conv2.weight": (64, 128, 3),
    "conv2.bias": (64,),
    "conv3.weight": (64, 64, 3),
    "conv3.bias": (64,),
    "conv4.weight": (128, 64, 3),
    "conv4.bias": (128,),
    "lstm_cell.weight_ih": (512, 128),
    "lstm_cell.weight_hh": (512, 128),
    "lstm_cell.bias_ih": (512,),
    "lstm_cell.bias_hh": (512,),
    "final_conv.weight": (1, 128, 1),
    "final_conv.bias": (1,),
}

#: Silero's names for them in the ONNX export (under `model.`, the 16 kHz
#: network; `model_8k.` is the 8 kHz one, which is never read).
_ONNX_NAMES: dict[str, str] = {
    "stft.forward_basis_buffer": "stft_conv.weight",
    "encoder.0.reparam_conv.weight": "conv1.weight",
    "encoder.0.reparam_conv.bias": "conv1.bias",
    "encoder.1.reparam_conv.weight": "conv2.weight",
    "encoder.1.reparam_conv.bias": "conv2.bias",
    "encoder.2.reparam_conv.weight": "conv3.weight",
    "encoder.2.reparam_conv.bias": "conv3.bias",
    "encoder.3.reparam_conv.weight": "conv4.weight",
    "encoder.3.reparam_conv.bias": "conv4.bias",
    "decoder.rnn.weight_ih": "lstm_cell.weight_ih",
    "decoder.rnn.weight_hh": "lstm_cell.weight_hh",
    "decoder.rnn.bias_ih": "lstm_cell.bias_ih",
    "decoder.rnn.bias_hh": "lstm_cell.bias_hh",
    "decoder.decoder.2.weight": "final_conv.weight",
    "decoder.decoder.2.bias": "final_conv.bias",
}

#: Frames encoded at once. The encoder is per frame and runs as matrix
#: products over a block; only the LSTM walks frame by frame. 8,192 frames is
#: 262 s of audio and about 60 MB of intermediate arrays.
_BLOCK = 8192


# ---------------------------------------------------------------- the request


def from_request(request: dict):
    """A worker request's `speech`: None (off), or `settings()`'s. Required key."""
    if "speech" not in request:
        raise KeyError(
            "the asr request has no 'speech'; null means transcribe everything, "
            "and the server sends the key either way"
        )
    value = request["speech"]
    if value is None:
        return None
    if not isinstance(value, dict):
        raise KeyError(f"the asr request's 'speech' must be an object or null, got {type(value).__name__}")
    return settings(value)


def cut_for_worker(wav, speech: dict, source: str, report) -> tuple:
    """`detect`, as every worker runs it: progress reported, no speech refused.

    `report(seconds)` is called as the detector moves through the audio. A
    source in which the detector hears no speech at all is a failure with a
    sentence, not an empty transcript: the caller asked to have silence taken
    out, not to be told a book has no words in it.
    """
    total = wav.shape[0] / float(SAMPLE_RATE)
    cut, kept = detect(wav, speech, report)
    if not kept:
        raise RuntimeError(
            f"the speech detector heard no speech anywhere in {total:.0f}s of "
            f"{source} at speech_threshold {speech['threshold']:g}, so there is "
            "nothing to transcribe. Send a lower speech_threshold, or "
            "speech_only: false to transcribe all of it"
        )
    return cut, kept


def settings(request: dict) -> dict:
    """The `speech` object a worker is sent, checked. Refuses by name.

    `{"weights", "sha256", "threshold", "pad_s", "min_gap_s"}`, every key
    required; the server has already refused a value out of bounds, so one
    arriving here is a server bug.
    """
    out: dict = {}
    for key, kind in (
        ("weights", str),
        ("sha256", str),
        ("threshold", (int, float)),
        ("pad_s", (int, float)),
        ("min_gap_s", (int, float)),
    ):
        if key not in request:
            raise KeyError(f"the speech request has no {key!r}; every key is required")
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, kind):
            raise KeyError(f"the speech request's {key!r} is {type(value).__name__}")
        out[key] = value
    for key, low, high in (
        ("threshold", MIN_THRESHOLD, MAX_THRESHOLD),
        ("pad_s", MIN_PAD_S, MAX_PAD_S),
        ("min_gap_s", MIN_MIN_GAP_S, MAX_MIN_GAP_S),
    ):
        if not low <= float(out[key]) <= high:
            raise KeyError(f"the speech request's {key!r} is {out[key]}; it is {low:g} to {high:g}")
    return out


# ------------------------------------------------------------------- weights


def _varint(blob: bytes, at: int) -> tuple:
    value = shift = 0
    while True:
        byte = blob[at]
        at += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, at
        shift += 7


def _fields(blob: bytes, first: int, last: int):
    """One protobuf message's `(field, wire type, value)`, in order.

    A varint is an int; a length-delimited field is its `(first, last)` byte
    range, so a nested message is read in place without a copy.
    """
    at = first
    while at < last:
        key, at = _varint(blob, at)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, at = _varint(blob, at)
        elif wire == 1:
            value, at = blob[at : at + 8], at + 8
        elif wire == 2:
            length, at = _varint(blob, at)
            value, at = (at, at + length), at + length
        elif wire == 5:
            value, at = blob[at : at + 4], at + 4
        else:
            raise RuntimeError(f"protobuf wire type {wire} at byte {at}")
        yield field, wire, value


def read_weights(path: str, sha256: str) -> dict:
    """The 16 kHz network's tensors, from the pinned file, or a refusal.

    The digest is checked before a byte is parsed: different bytes under the
    pinned name are a different detector. The file is Silero's own ONNX
    export (`silero_vad_op18_ifless.onnx`), whose weights are plain graph
    initializers; only those are read, with the ONNX protobuf's four field
    numbers that matter (ModelProto.graph 7, GraphProto.initializer 5,
    TensorProto dims 1, data_type 2, name 8, raw_data 9), so no env needs the
    `onnx` package. The graph itself is not run: `probabilities` is.
    """
    import numpy

    with open(path, "rb") as handle:
        blob = handle.read()
    measured = hashlib.sha256(blob).hexdigest()
    if measured != sha256:
        raise RuntimeError(
            f"{path} hashed {measured}, and the speech detector is pinned at {sha256}"
        )
    wanted = {f"model.{theirs}": ours for theirs, ours in _ONNX_NAMES.items()}
    tensors: dict = {}
    for field, _, graph in _fields(blob, 0, len(blob)):
        if field != 7:
            continue
        for inner, _, tensor in _fields(blob, *graph):
            if inner != 5:
                continue
            dims: list = []
            name, dtype, raw = "", 0, None
            for key, wire, value in _fields(blob, *tensor):
                if key == 1 and wire == 0:
                    dims.append(value)
                elif key == 1 and wire == 2:
                    at = value[0]
                    while at < value[1]:
                        dim, at = _varint(blob, at)
                        dims.append(dim)
                elif key == 2:
                    dtype = value
                elif key == 8:
                    name = blob[value[0] : value[1]].decode("utf-8")
                elif key == 9:
                    raw = value
            ours = wanted.get(name)
            if ours is None:
                continue
            shape = _SHAPES[ours]
            if dtype != 1 or raw is None or tuple(dims) != shape:
                raise RuntimeError(
                    f"{path}: {name} is not a float32 tensor of shape {shape} "
                    f"(data_type {dtype}, dims {dims})"
                )
            tensors[ours] = numpy.frombuffer(
                blob[raw[0] : raw[1]], dtype="<f4"
            ).reshape(shape)
    missing = sorted(set(_SHAPES) - set(tensors))
    if missing:
        raise RuntimeError(f"{path} holds no {missing}; it is not the pinned detector")
    return tensors


# ------------------------------------------------------------------- network


def _conv1d(x, weight, bias, stride: int):
    """`torch.nn.Conv1d(kernel_size=3, padding=1)` over `(frames, channels, length)`."""
    import numpy

    padded = numpy.pad(x, ((0, 0), (0, 0), (1, 1)))
    length = (x.shape[2] + 2 - 3) // stride + 1
    taps = numpy.stack(
        [padded[:, :, k : k + stride * (length - 1) + 1 : stride] for k in range(3)],
        axis=-1,
    )  # (frames, channels, length, 3)
    out = numpy.einsum("nclk,ock->nol", taps, weight, optimize=True)
    return out + bias[None, :, None]


def probabilities(weights: dict, wav, on_progress=None):
    """Silero's speech probability for every 512-sample frame of `wav`.

    `silero_vad/tinygrad_model.py` at 6.2.1, step for step: each frame is read
    with the 64 samples before it (zeros before the first), reflect-padded by
    64 on the right, put through the STFT convolution, reduced to magnitudes,
    four convolutions with ReLU, one LSTM step carrying its state from the
    frame before, ReLU, a 1x1 convolution and a sigmoid. The last frame is
    zero-padded to 512, as Silero's own `get_speech_timestamps` does.
    """
    import numpy

    samples = numpy.asarray(wav, dtype=numpy.float32)
    frames = -(-samples.shape[0] // FRAME)
    padded = numpy.zeros(CONTEXT + frames * FRAME, dtype=numpy.float32)
    padded[CONTEXT : CONTEXT + samples.shape[0]] = samples

    stft = weights["stft_conv.weight"][:, 0, :].T  # (256, 258)
    w_ih = weights["lstm_cell.weight_ih"].T  # (128, 512)
    w_hh = weights["lstm_cell.weight_hh"].T  # (128, 512)
    b = weights["lstm_cell.bias_ih"] + weights["lstm_cell.bias_hh"]
    w_out = weights["final_conv.weight"][0, :, 0]
    b_out = float(weights["final_conv.bias"][0])

    h = numpy.zeros(128, dtype=numpy.float32)
    c = numpy.zeros(128, dtype=numpy.float32)
    out = numpy.empty(frames, dtype=numpy.float32)
    window = FRAME + CONTEXT
    for first in range(0, frames, _BLOCK):
        count = min(_BLOCK, frames - first)
        starts = (first + numpy.arange(count)) * FRAME
        x = padded[starts[:, None] + numpy.arange(window)[None, :]]  # (n, 576)
        x = numpy.concatenate([x, x[:, -2 : -CONTEXT - 2 : -1]], axis=1)  # reflect, (n, 640)
        spec = numpy.stack([x[:, i * 128 : i * 128 + 256] @ stft for i in range(4)], axis=2)
        magnitude = numpy.sqrt(spec[:, :129, :] ** 2 + spec[:, 129:, :] ** 2)  # (n, 129, 4)
        y = numpy.maximum(_conv1d(magnitude, weights["conv1.weight"], weights["conv1.bias"], 1), 0)
        y = numpy.maximum(_conv1d(y, weights["conv2.weight"], weights["conv2.bias"], 2), 0)
        y = numpy.maximum(_conv1d(y, weights["conv3.weight"], weights["conv3.bias"], 2), 0)
        y = numpy.maximum(_conv1d(y, weights["conv4.weight"], weights["conv4.bias"], 1), 0)
        gates_in = (y[:, :, 0] @ w_ih + b).astype(numpy.float32)  # (n, 512)
        hidden = numpy.empty((count, 128), dtype=numpy.float32)
        for t in range(count):
            gates = gates_in[t] + h @ w_hh
            i, f, g, o = gates[:128], gates[128:256], gates[256:384], gates[384:]
            c = c / (1.0 + numpy.exp(-f)) + numpy.tanh(g) / (1.0 + numpy.exp(-i))
            h = numpy.tanh(c) / (1.0 + numpy.exp(-o))
            hidden[t] = h
        logits = numpy.maximum(hidden, 0) @ w_out + b_out
        out[first : first + count] = 1.0 / (1.0 + numpy.exp(-logits))
        if on_progress is not None:
            on_progress((first + count) * FRAME / float(SAMPLE_RATE))
    return out


# ------------------------------------------------------------------- regions


def kept_regions(probs, total_samples: int, threshold: float, pad_s: float, min_gap_s: float) -> list:
    """`[[first, last)]` sample spans to keep, in order, disjoint. Empty: no speech.

    Every frame at or above `threshold` is speech; each run of them is widened
    by `pad_s` both sides (clamped to the file) and runs that then touch or
    overlap merge. A stretch between two regions is removed only when it is at
    least `min_gap_s` long; a shorter one is kept, so the two regions merge.
    The same rule at the file's two ends: leading or trailing non-speech is
    removed only when it is at least `min_gap_s` long.
    """
    pad = int(round(pad_s * SAMPLE_RATE))
    gap = int(round(min_gap_s * SAMPLE_RATE))
    runs: list = []
    run_start = None
    for index, value in enumerate(probs):
        if value >= threshold:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start, index))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(probs)))
    kept: list = []
    for first_frame, end_frame in runs:
        first = max(0, first_frame * FRAME - pad)
        last = min(total_samples, end_frame * FRAME + pad)
        if kept and first - kept[-1][1] < gap:
            kept[-1][1] = max(kept[-1][1], last)
        else:
            kept.append([first, last])
    if kept and kept[0][0] < gap:
        kept[0][0] = 0
    if kept and total_samples - kept[-1][1] < gap:
        kept[-1][1] = total_samples
    return kept


def detect(wav, speech: dict, on_progress=None) -> tuple:
    """`(shortened wav, kept spans)` for one decoded source, under `speech`.

    What a worker calls between its decode and its cut. `speech` is
    `settings()`'s. When nothing is removed the signal is `wav` itself.

    IN PLACE WHEN IT CAN BE: an 18-hour book is 4 GB of float32, and a copy of
    it beside the original is 4 GB more. A writable `wav` (the workers' decode
    is a view of a bytearray) is compacted in place, each kept region moved
    down to where the one before it ended, and a view of its front is
    returned, so the caller must not read `wav` again. A read-only one is
    copied.
    """
    import numpy

    weights = read_weights(speech["weights"], speech["sha256"])
    probs = probabilities(weights, wav, on_progress)
    total = int(wav.shape[0])
    kept = kept_regions(
        probs, total, float(speech["threshold"]), float(speech["pad_s"]), float(speech["min_gap_s"])
    )
    if kept == [[0, total]]:
        return wav, kept
    if not kept:
        return wav[:0], kept
    if not wav.flags.writeable:
        return numpy.concatenate([wav[first:last] for first, last in kept]), kept
    at = 0
    for first, last in kept:
        size = last - first
        if first != at:
            # Overlapping ranges moving down: numpy buffers the overlap.
            wav[at : at + size] = wav[first:last]
        at += size
    return wav[:at], kept


def joins(kept: list) -> list:
    """Where the kept regions meet in the shortened signal, in its samples."""
    points = []
    at = 0
    for first, last in kept[:-1]:
        at += last - first
        points.append(at)
    return points


# ------------------------------------------------------------------ timeline


@dataclass(frozen=True)
class Timeline:
    """The shortened signal's seconds, back to the source's. Sample-exact.

    Built from the worker's `kept` table (integer samples of the source). A
    time on the shortened timeline lies in one kept region, and moves by that
    region's offset — the whole of the arithmetic, as for a piece's offset.
    """

    kept: tuple  # ((first, last), ...) source samples
    total_samples: int

    @classmethod
    def from_ready(cls, ready: dict) -> "Timeline":
        """A worker's `ready` (`kept`, `samples`) as a timeline. ValueError if absent.

        A worker that was sent `speech` and reports no table did not do it,
        and its times would be read on the wrong timeline.
        """
        kept, samples = ready.get("kept"), ready.get("samples")
        if not isinstance(kept, list) or not isinstance(samples, int) or isinstance(samples, bool):
            raise ValueError(
                "the worker was asked for speech only and reported no kept regions, "
                "so its times cannot be put back on the audio's own timeline"
            )
        return cls.from_kept(kept, samples)

    @classmethod
    def from_kept(cls, kept: list, total_samples: int) -> "Timeline":
        spans = tuple((int(first), int(last)) for first, last in kept)
        previous = 0
        for first, last in spans:
            if not previous <= first < last <= total_samples:
                raise ValueError(
                    f"the kept regions {kept} are not ordered, disjoint spans of "
                    f"{total_samples} samples"
                )
            previous = last
        return cls(kept=spans, total_samples=int(total_samples))

    def __post_init__(self) -> None:
        # Where each region starts on the shortened timeline, once: a book's
        # transcript maps a hundred thousand words through this.
        at, starts = 0, []
        for first, last in self.kept:
            starts.append(at)
            at += last - first
        object.__setattr__(self, "_starts", tuple(starts))

    def region(self, t: float, *, end: bool = False) -> int:
        """Which kept region a shortened-timeline time is in.

        At a join the time belongs to the region AFTER it when it is a start
        and to the region BEFORE it when it is an end (`end=True`), so a span
        that ends exactly at a join does not reach across a removed stretch.
        """
        sample = t * SAMPLE_RATE
        find = bisect.bisect_left if end else bisect.bisect_right
        return max(0, find(self._starts, sample) - 1)

    def original(self, t: float, *, end: bool = False, region: int | None = None) -> float:
        """A shortened-timeline time in source seconds, clamped to its region."""
        index = self.region(t, end=end) if region is None else region
        first, last = self.kept[index]
        start = self._starts[index]
        shift = (first - start) / float(SAMPLE_RATE)
        low, high = first / float(SAMPLE_RATE), last / float(SAMPLE_RATE)
        return min(max(t + shift, low), high)

    def span(self, start: float, end: float) -> tuple:
        """A segment's `(start, end)`: each end mapped on its own side.

        A segment may hold a removed stretch (it can hold several sentences),
        and then it spans it in source time, which is true.
        """
        return self.original(start), self.original(end, end=True)

    def word(self, start: float, end: float) -> tuple:
        """A word's `(start, end)`: both ends in the region holding its middle.

        A word cannot straddle a removed stretch (there was no speech in it),
        so an end that a timing model put past a join is clamped to the edge of
        the region the word is mostly in, rather than stretched across seconds
        of removed audio.
        """
        index = self.region((start + end) / 2.0)
        return self.original(start, region=index), self.original(end, end=True, region=index)

    def removed(self) -> list:
        """What was taken out, as `[{"start", "end"}]` in source seconds."""
        out = []
        previous = 0
        for first, last in self.kept:
            if first > previous:
                out.append({"start": previous / SAMPLE_RATE, "end": first / SAMPLE_RATE})
            previous = last
        if previous < self.total_samples:
            out.append({"start": previous / SAMPLE_RATE, "end": self.total_samples / SAMPLE_RATE})
        return out

    def speech_seconds(self) -> float:
        return sum(last - first for first, last in self.kept) / float(SAMPLE_RATE)


def report(speech, timeline) -> dict:
    """`speech_only`, `speech` and `removed`: three keys every transcript has.

    `speech` and `removed` are null when speech_only was off, which is a
    different statement from `removed: []` (the detector ran and found no
    stretch long enough to take out).
    """
    if speech is None or timeline is None:
        return {"speech_only": False, "speech": None, "removed": None}
    removed = timeline.removed()
    return {
        "speech_only": True,
        "speech": {
            "detector": DETECTOR,
            "sha256": speech["sha256"],
            "threshold": speech["threshold"],
            "pad_s": speech["pad_s"],
            "min_gap_s": speech["min_gap_s"],
            "speech_s": timeline.speech_seconds(),
            "removed_s": sum(row["end"] - row["start"] for row in removed),
        },
        "removed": removed,
    }
