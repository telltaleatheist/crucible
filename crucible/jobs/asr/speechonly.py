from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

SAMPLE_RATE = 16_000
FRAME = 512
CONTEXT = 64

DEFAULT_THRESHOLD = 0.3
MIN_THRESHOLD = 0.1
MAX_THRESHOLD = 0.7
DEFAULT_PAD_S = 0.3
MIN_PAD_S = 0.1
MAX_PAD_S = 2.0
DEFAULT_MIN_GAP_S = 2.0
MIN_MIN_GAP_S = 1.0
MAX_MIN_GAP_S = 60.0

DETECTOR = "silero-vad 6.2.1 (silero_vad_op18_ifless.onnx weights), numpy on the CPU"

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

_BLOCK = 8192


def from_request(request: dict):
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


def _conv1d(x, weight, bias, stride: int):
    import numpy

    padded = numpy.pad(x, ((0, 0), (0, 0), (1, 1)))
    length = (x.shape[2] + 2 - 3) // stride + 1
    taps = numpy.stack(
        [padded[:, :, k : k + stride * (length - 1) + 1 : stride] for k in range(3)],
        axis=-1,
    )
    out = numpy.einsum("nclk,ock->nol", taps, weight, optimize=True)
    return out + bias[None, :, None]


def probabilities(weights: dict, wav, on_progress=None):
    import numpy

    samples = numpy.asarray(wav, dtype=numpy.float32)
    frames = -(-samples.shape[0] // FRAME)
    padded = numpy.zeros(CONTEXT + frames * FRAME, dtype=numpy.float32)
    padded[CONTEXT : CONTEXT + samples.shape[0]] = samples

    stft = weights["stft_conv.weight"][:, 0, :].T
    w_ih = weights["lstm_cell.weight_ih"].T
    w_hh = weights["lstm_cell.weight_hh"].T
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
        x = padded[starts[:, None] + numpy.arange(window)[None, :]]
        x = numpy.concatenate([x, x[:, -2 : -CONTEXT - 2 : -1]], axis=1)
        spec = numpy.stack([x[:, i * 128 : i * 128 + 256] @ stft for i in range(4)], axis=2)
        magnitude = numpy.sqrt(spec[:, :129, :] ** 2 + spec[:, 129:, :] ** 2)
        y = numpy.maximum(_conv1d(magnitude, weights["conv1.weight"], weights["conv1.bias"], 1), 0)
        y = numpy.maximum(_conv1d(y, weights["conv2.weight"], weights["conv2.bias"], 2), 0)
        y = numpy.maximum(_conv1d(y, weights["conv3.weight"], weights["conv3.bias"], 2), 0)
        y = numpy.maximum(_conv1d(y, weights["conv4.weight"], weights["conv4.bias"], 1), 0)
        gates_in = (y[:, :, 0] @ w_ih + b).astype(numpy.float32)
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


def kept_regions(probs, total_samples: int, threshold: float, pad_s: float, min_gap_s: float) -> list:
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
            wav[at : at + size] = wav[first:last]
        at += size
    return wav[:at], kept


def joins(kept: list) -> list:
    points = []
    at = 0
    for first, last in kept[:-1]:
        at += last - first
        points.append(at)
    return points


@dataclass(frozen=True)
class Timeline:

    kept: tuple
    total_samples: int

    @classmethod
    def from_ready(cls, ready: dict) -> "Timeline":
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
        at, starts = 0, []
        for first, last in self.kept:
            starts.append(at)
            at += last - first
        object.__setattr__(self, "_starts", tuple(starts))

    def region(self, t: float, *, end: bool = False) -> int:
        sample = t * SAMPLE_RATE
        find = bisect.bisect_left if end else bisect.bisect_right
        return max(0, find(self._starts, sample) - 1)

    def original(self, t: float, *, end: bool = False, region: int | None = None) -> float:
        index = self.region(t, end=end) if region is None else region
        first, last = self.kept[index]
        start = self._starts[index]
        shift = (first - start) / float(SAMPLE_RATE)
        low, high = first / float(SAMPLE_RATE), last / float(SAMPLE_RATE)
        return min(max(t + shift, low), high)

    def span(self, start: float, end: float) -> tuple:
        return self.original(start), self.original(end, end=True)

    def word(self, start: float, end: float) -> tuple:
        index = self.region((start + end) / 2.0)
        return self.original(start, region=index), self.original(end, end=True, region=index)

    def removed(self) -> list:
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


def timeline_for(ready: dict, speech) -> "Timeline | None":
    if speech is None:
        return None
    return Timeline.from_ready(ready)


def span(timeline, start: float, end: float) -> tuple:
    if timeline is None:
        return start, end
    return timeline.span(start, end)


def to_source(segment: dict, timeline) -> dict:
    if timeline is None:
        return segment
    moved = dict(segment)
    moved["start"], moved["end"] = timeline.span(segment["start"], segment["end"])
    if "words" in segment:
        words = []
        for word in segment["words"]:
            start, end = timeline.word(word["start"], word["end"])
            words.append({**word, "start": start, "end": end})
        moved["words"] = words
    return moved


def report(speech, timeline) -> dict:
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
