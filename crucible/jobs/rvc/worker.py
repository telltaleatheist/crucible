"""The `rvc` worker: ultimate-rvc, run in its own interpreter, in batches of pieces.

PHASE4-AUDIO.md sections 0 and 4. This module is **standalone**. It imports
nothing from `crucible`, because the env it runs in has no `crucible` installed
and never will, and it does not import ultimate-rvc, which it *runs*. Since
2026-09-26 it does import numpy, soundfile and scipy: they are urvc's own
dependencies, pinned in both rvc recipes, and the cutting and stitching below
cannot be done without reading audio.

Why it spawns urvc instead of importing it
------------------------------------------
**Batching is a memory bound, not a throughput choice.** BookForge recycles a
convert process every 96 files, proven necessary on a 64 GB Mac on 2026-07-17: a
ten-minute chunk grew the process by about 1.5 GB and never released it, unified
memory ballooned to ~50 GB, the machine hit swap and RVC slowed about 5x. The env
carries a per-file `torch.mps.empty_cache` patch and it is *not sufficient*. What
is sufficient is the process dying, because then the OS reclaims everything it
leaked regardless of what leaked. So each batch is its own child that exits, and
that is only possible if the engine is a command rather than an import.

This is engine knowledge. The client never sees `batch_size`, never sees a batch
boundary, and gets one output per input either way (`crucible/jobs/rvc/__init__.py`).

**Never `urvc.exe`, and never the `urvc` console script.** pip's console-script
launcher bakes the interpreter path in at install time as a shebang, and an env
that was installed by extracting into a temp directory and moving it into place
then points at a python that no longer exists — which fails with exit 1 and
*zero output*, because the launcher dies before python starts
(`electron/rvc-bridge.ts:53`). `sys.executable -m ultimate_rvc.cli.main` is the
same program reached a way that cannot go stale, and this paragraph is here so
that nobody simplifies it back.

Long inputs are cut here, converted in pieces, and stitched back (#5)
---------------------------------------------------------------------
Fresh-install snag #5, 2026-09-26: the job used to hand each input to urvc
WHOLE, so a 7-12 h master had to be cut into ten-minute files and rejoined by
the client, because one ten-minute file already grows the process ~1.5 GB. Owen:
Crucible must be idiot proof; a caller who sends the 12-hour master whole gets
the right answer. So every input, of any length, now goes through the same
four steps, and a sentence is simply an input that is one piece:

1. **Plan (`plan_input`).** One streaming read finds the cuts: every boundary
   is the CENTRE of the quietest 100 ms in the 10 s before the nominal cut —
   `asr`'s rule (`jobs/asr/qwen_worker.py`, `split_points`), ported rather than
   imported because the two workers run in different envs and neither may
   import the other's. Two differences, both forced by the input's size: the
   energy is a running sum instead of a convolution (at 48 kHz the convolution
   is 9x asr's cost, per cut), and nothing is held but the search window, so a
   12 h master is never in memory. A final piece is never shorter than
   `MIN_TAIL_SECONDS`: the last cut moves earlier instead.
2. **Cut (`read_pieces`).** A second streaming read writes each piece with
   `overlap_s` of the REAL neighbouring audio on both sides, as a float WAV at
   the input's own rate and channels, into the batch's staging directory.
   Pieces are written per batch, so the disk holds one batch of them. The
   staging directory is inside the JOB's directory (`staging_dir`), never the
   system temp: a cancel stops this worker with a signal that skips its
   `finally`, and what it leaves there is then the retention collector's
   (`JobStore.reap`), like everything else the job wrote (2026-09-26).
3. **Convert.** urvc's `convert-dir`, one process per batch. A batch closes at
   `batch_size` pieces or at a seconds bound derived from this host's memory,
   and a process whose own memory passes the budget is stopped after the piece
   it is on (`memory_plan`, the 2026-09-26 incident). Pieces are counted,
   never input files: 96 ten-minute files in one process is the leak this
   module exists to prevent.
4. **Stitch (`Stitcher`).** Each converted piece is resampled from urvc's rate
   to the OUTPUT rate (below), then trimmed or zero-padded to exactly the
   frames its span maps to, its overlap is dropped, and neighbours are joined
   with a `crossfade_s` raised-cosine fade centred on the seam. The output
   lasts exactly as long as the input (`out_frame`); the writer checks the
   count and fails if it is wrong.
   Each input's result is sent the moment its last piece is stitched, and the
   server publishes it then, so a cancel or a later failure leaves every input
   already finished as a readable artifact (2026-09-26: a cancel at 26 of 35
   ten-minute files on the Mac lost all 26).

Why an overlap AND a crossfade, and why the crossfade is short
--------------------------------------------------------------
- **The overlap is context.** rmvpe's pitch model ends in a bidirectional GRU
  and contentvec is a transformer: the frames near a piece's edge are decided
  by audio on both sides of them. With no overlap they see RVC's own reflect
  padding (time-reversed audio) instead of what was really there. The overlap
  also absorbs urvc's length error — about 20 ms short per 60 s, measured on
  kylies-pc 2026-09-26, place in the piece unmeasured — into audio that is
  thrown away, so the zero padding that makes the length exact never reaches
  the output while the error is smaller than the overlap. The one seam with no
  overlap past it is the input's own end, so an input's last few milliseconds
  (the last piece's shortfall, ~20 ms per minute of it, if urvc's loss is at
  the end) are zero-padded silence; an input almost always ends in silence
  anyway, and padding the file's end would add `overlap_s` of conversion to
  every one-piece sentence to save it. **Owen accepted this end padding as it
  is, 2026-09-26:** *"i guess we can accept 20 ms of padding"* (up to ~20 ms per minute of the last piece, at the very end
  of an input only): not a defect to fix, and nobody should add an end pad
  to "fix" it. Cost: `2 * overlap_s`
  more audio through the model per piece, 1.7% at the defaults.
- **The crossfade is click removal, not blending.** Two pieces converted
  separately do not meet at the same waveform value, and a hard join at a
  sample is a step, which is a click. But RVC's vocoder (NSF) builds its voiced
  source from a sine whose phase is integrated from the START of each piece, so
  the two conversions of one overlap agree in pitch and not in phase, and a
  long crossfade through voiced sound is two out-of-phase copies partly
  cancelling. Cuts land in the quietest point near each boundary, so the fade
  is usually through a pause; it is kept at 20 ms so that when it is not, it
  is too short to hear as anything but the absence of a click.

Both are the caller's (Owen's asr ruling, 2026-09-26: "make it so the caller
can determine how big the chunks are and whether they overlap. and by how
much"); `crossfade_s` cannot exceed `2 * overlap_s`, because a fade needs
converted audio from both pieces on both sides of the seam.

The input's NAME carries nothing (#45)
--------------------------------------
Fresh-install snag #45, 2026-09-26: `--input c000=chunk.flac` was refused
because the NAME `c000` had no extension, although the bytes were plainly
FLAC. The worker now reads what an input is from its bytes (libsndfile), stages
pieces under names it makes itself, and writes each output in the input's own
format, so the name is only a name: the artifact comes back under it,
extension or not.

The output format, stated
-------------------------
**Container and sample format**: the input's (a 24-bit FLAC comes back a
24-bit FLAC; RF64 instead of WAV past WAV's 4 GiB). The converted signal's
own resolution is urvc's 16-bit (`convert-dir` writes 16-bit WAV at the
model's rate); a 24-bit container holds that signal and adds no detail the
engine did not produce.

**Sample rate** (Owen's ruling, 2026-09-26: "i would like to keep 48 khz but
if we cant then we cant"): by default (`output_rate: "native"`) the output
rate is `max(input rate, the rate urvc wrote)`, so it is never below the
model's own. The rate urvc wrote is READ from its first converted piece of
each input, not assumed; for the published models it is 48 kHz. Until this
ruling the output came back at the input's rate, and a 24 kHz input threw away the
model's whole band above 12 kHz. `output_rate: "input"` keeps the input's
rate, the old behaviour, for a caller that must have it. The length is exact
in TIME, not in frames: `out_frames = round(in_frames * out_rate / in_rate)`,
rounded half up in integers (`out_frame`), so it is the same number on every
host. Each piece's span maps through the same function, so the pieces still
tile the output with no gap and no overlap.

**Channels** (Owen, 2026-09-26: "i only ever use mono but others might need
stereo"): by default (`output_channels: "input"`) the output has the input's
channel count. **It is NOT a per-channel conversion.** RVC converts a mono
mix and produces one voice, and for a stereo input that one converted
signal is written to every channel, identical in each. `output_channels:
"mono"` writes one channel. A true per-channel conversion is not offered:
it would double the conversion for speech that is the same voice in both
channels, and two independently converted channels differ in phase (see the
crossfade note above), which smears the voice across the stereo image.

The wire, in full
-----------------
    stdin   one object: {models_dir, model_name, output_dir, input_dir, inputs,
                         index_rate, protect_rate, n_semitones, batch_size,
                         memory_fraction, leak_bytes_per_audio_s,
                         max_batch_audio_s, fallback_budget_bytes,
                         piece_s, overlap_s, crossfade_s,
                         output_rate ("native"|"input"),
                         output_channels ("input"|"mono"), staging_dir,
                         ffmpeg, ffprobe, f0_method?, hop_length?}
            `f0_method` and `hop_length` are THE ONLY optional keys in any phase
            4 worker's request, and their absence is meaningful rather than a
            refusal — see `_convert_args`.

    fd 1    {"type": "progress", "stage": "cutting", "processed", "total"}
                                                  one per input planned
            {"type": "ready", "files", "pieces", "batches", "batch_size",
             "batch_audio_s", "memory_budget_bytes", "memory_basis", "model"}
            {"type": "progress", "stage": "converting", "processed", "total",
             "batch", "batches"}                  processed/total are PIECES
            {"type": "result", "bytes", "frames", "sample_rate", "channels",
             "format", "subtype"}                 one per input, BY POSITION,
                                                  sent as each input finishes
            {"type": "failed", "message"}         the whole run
            {"type": "done"}

fd 1 is results and nothing else
--------------------------------
The first thing this file does is dup fd 1 somewhere safe and point the original
at stderr. Whose lesson it is: narrator's aligner, Owen's 401-chunk witches book,
2026-09-05 — a library logged a warning to stdout between two result lines and
the parent's `json.loads` died with "Extra data". Here the risk is even plainer,
because urvc's own progress goes to *its* stdout and this worker reads it; that
stream is parsed, never forwarded.
"""

from __future__ import annotations

import os
import sys

# ---- fd 1 is results, stderr is everything else. Before any other import. ----
_RESULTS_FD = os.dup(1)
os.dup2(2, 1)
_RESULTS = os.fdopen(_RESULTS_FD, "w", encoding="utf-8", buffering=1)

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
import soundfile  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

#: urvc's own progress line, and the app's regex verbatim
#: (`electron/rvc-bridge.ts:257`). It counts within the batch, so the worker adds
#: the offset of the batches already done.
PROGRESS_LINE = re.compile(r"^\[RVC\]\s+(\d+)/(\d+)")

#: The quiet-point search, `asr`'s constants (`jobs/asr/qwen_worker.py`): the
#: last 10 s before each nominal cut, or half the piece if that is less, is
#: searched for the 100 ms window with the least energy, and the cut is made at
#: that window's CENTRE, so it lands in the middle of a pause and not against
#: the next word.
SEARCH_SECONDS = 10.0
ENERGY_WINDOW_MS = 100.0

#: No final piece shorter than this. Without it a master one second longer than
#: a whole number of pieces ends in a 1-sample piece, and a pitch model handed
#: one sample has nothing to find a pitch in. The last cut moves earlier instead.
MIN_TAIL_SECONDS = 1.0

#: Frames per streaming read: about 5 s at 48 kHz. What the planner and the
#: cutter hold at once is this plus one search window or one piece.
READ_BLOCK_FRAMES = 1 << 18

#: How far a converted piece may be from the length it was given before it is
#: a failure rather than a rounding: urvc runs about 20 ms short per 60 s
#: (0.03%), and anything past 100 ms plus 0.5% is not this piece's conversion.
LENGTH_TOLERANCE_SECONDS = 0.1
LENGTH_TOLERANCE_FRACTION = 0.005

#: WAV and AIFF sizes are 32-bit. An output WAV whose data would not fit is
#: written as RF64, which every reader of long WAVs reads.
WAV_DATA_LIMIT_BYTES = 0xFFFFFFFF - (1 << 20)

#: Bytes per sample of the PCM subtypes, for the RF64 decision. Unknown
#: subtypes count as 8, which errs toward RF64.
SUBTYPE_BYTES = {
    "PCM_S8": 1, "PCM_U8": 1, "ULAW": 1, "ALAW": 1,
    "PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8,
}

#: Seconds between `cutting` progress messages while one long input is planned.
CUT_REPORT_SECONDS = 5.0


def send(message_type: str, **fields: object) -> None:
    """One JSON object, one line, flushed, on the real fd 1."""
    _RESULTS.write(json.dumps({"type": message_type, **fields}) + "\n")
    _RESULTS.flush()


def fail(message: str) -> int:
    send("failed", message=message)
    return 1


def require(request: dict, key: str, kind):
    """One required key, or a refusal naming it."""
    if key not in request:
        raise KeyError(
            f"the rvc request has no {key!r}; every parameter except 'f0_method' "
            "and 'hop_length' is required, and those two are absent on purpose"
        )
    value = request[key]
    kinds = kind if isinstance(kind, tuple) else (kind,)
    wrong = not isinstance(value, kinds) or (
        isinstance(value, bool) and bool not in kinds
    )
    if wrong:
        raise KeyError(
            f"the rvc request's {key!r} must be "
            f"{'/'.join(k.__name__ for k in kinds)}, got {type(value).__name__}"
        )
    return value


def _convert_args(request: dict, input_dir: str, output_dir: str) -> list[str]:
    """The `python -m ultimate_rvc.cli.main generate convert-dir ...` argv.

    **An absent `f0_method` or `hop_length` means the flag is OMITTED**, so urvc
    keeps its own default. It does not mean a value Crucible chose. This is the
    one place in the whole server where "absent" is a meaningful wire value
    rather than a refusal, and the reason is that urvc's defaults are the tuned
    ones: a server that filled them in would be substituting its guess for the
    engine's measurement and the output would not say which it got.

    `hop_length` is tested with `is not None` and not for truthiness, because its
    range is 1-512 and nothing meaningful in it is falsy — but read the way the
    others are, a future 0 would silently mean "unset" instead of being refused
    by urvc's own CLI, which is the louder failure.

    The glob and the output extension are always `wav`: they select the pieces
    this worker staged, never the client's files (#45, 2026-09-26).
    """
    argv = [
        sys.executable,
        "-m",
        "ultimate_rvc.cli.main",
        "generate",
        "convert-dir",
        input_dir,
        output_dir,
        require(request, "model_name", str),
        "--index-rate",
        str(require(request, "index_rate", (int, float))),
        # protect_rate's scale is INVERTED — see `jobs/rvc/__init__.py`, where it
        # is validated. Nothing is adjusted here; the number goes across as the
        # client sent it.
        "--protect-rate",
        str(require(request, "protect_rate", (int, float))),
        "--input-glob",
        "*.wav",
        "--output-ext",
        "wav",
    ]
    f0_method = request.get("f0_method")
    if f0_method is not None:
        argv += ["--f0-method", str(f0_method)]
    hop_length = request.get("hop_length")
    if hop_length is not None:
        argv += ["--hop-length", str(hop_length)]
    n_semitones = require(request, "n_semitones", int)
    if n_semitones:
        # Zero is urvc's own default and the flag is omitted for it, matching the
        # app (`rvc-bridge.ts`): a pitch shift of zero and no pitch shift are the
        # same thing, and sending `--n-semitones 0` would only add a way for the
        # two to stop being the same thing.
        argv += ["--n-semitones", str(n_semitones)]
    return argv


# ------------------------------------------------------------ plan and cut


def _quietest(magnitude, window: int) -> int:
    """Offset of the CENTRE of the quietest `window` samples in `magnitude`.

    `asr`'s rule (`split_points`), with the sliding sum taken from a cumulative
    sum rather than `numpy.convolve`: the same numbers up to rounding, at a cost
    that does not grow with the window. A span no longer than the window is cut
    at its end, as asr's is.
    """
    if magnitude.shape[0] <= window:
        return int(magnitude.shape[0])
    running = numpy.concatenate(([0.0], numpy.cumsum(magnitude, dtype=numpy.float64)))
    sums = running[window:] - running[:-window]
    return int(numpy.argmin(sums)) + window // 2


def _cut_lengths(rate: int, piece_s: float) -> tuple[int, int, int, int]:
    """`(max_len, search, window, min_tail)` in frames at `rate`."""
    max_len = int(piece_s * rate)
    search = int(min(SEARCH_SECONDS, piece_s / 2) * rate)
    window = max(4, int(ENERGY_WINDOW_MS / 1000.0 * rate))
    min_tail = int(MIN_TAIL_SECONDS * rate)
    return max_len, search, window, min_tail


def plan_input(path: str, piece_s: float, on_progress=None) -> tuple[list, int]:
    """`([(first, last)], total_frames)`: the cuts, found in one streaming read.

    The spans cover the input exactly, with no gaps and no overlap, and none is
    longer than `piece_s`. Only the search window's magnitudes are kept, never
    the audio, and the frame count is COUNTED rather than taken from the header,
    because a VBR MP3's header can be wrong about it and the output has to match
    what the samples are.
    """
    info = soundfile.info(path)
    rate = int(info.samplerate)
    max_len, search, window, min_tail = _cut_lengths(rate, piece_s)

    spans: list[tuple[int, int]] = []
    start = 0
    # Mono magnitudes for frames [held_first, held_first + len(held)).
    held = numpy.zeros(0, dtype=numpy.float32)
    held_first = 0
    position = 0
    last_report = time.time()

    def cut_at(cut: int) -> int:
        left = max(start, cut - search)
        region = held[left - held_first : cut - held_first]
        boundary = left + _quietest(region, window)
        return min(max(boundary, start + 1), cut)

    def keep_from() -> int:
        # The earliest frame the NEXT cut's search can reach: its nominal cut
        # less the search, less the tail rule's pull-back. Monotonic in `start`.
        return max(start, start + max_len - search - min_tail)

    with soundfile.SoundFile(path) as source:
        for block in source.blocks(
            blocksize=READ_BLOCK_FRAMES, dtype="float32", always_2d=True
        ):
            magnitude = numpy.abs(block).mean(axis=1)
            held = numpy.concatenate((held, magnitude))
            position += block.shape[0]
            # Decide a cut only once the file is known to run at least
            # `min_tail` past it; otherwise the tail rule may move it.
            while position - start > max_len + min_tail:
                boundary = cut_at(start + max_len)
                spans.append((start, boundary))
                start = boundary
            # Never past what has been read: early in a piece the search window
            # is still ahead, and `held_first` must stay the frame `held[0]` is.
            drop = min(keep_from() - held_first, held.shape[0])
            if drop > 0:
                held = held[drop:]
                held_first += drop
            if on_progress is not None and time.time() - last_report >= CUT_REPORT_SECONDS:
                last_report = time.time()
                on_progress(position / float(rate))
    total = position
    while total - start > max_len:
        boundary = cut_at(min(start + max_len, total - min_tail))
        spans.append((start, boundary))
        start = boundary
    spans.append((start, total))
    return spans, total


def read_pieces(path: str, spans: list, total: int, pad: int):
    """Yield each span's audio with `pad` real frames each side, in order.

    A second streaming read. What is held is one piece plus its overlap, and
    each frame is read from the file once however much the pieces overlap.
    `(audio_first, audio_last, samples)` per span; the pad is clamped at the
    file's two ends.
    """
    with soundfile.SoundFile(path) as source:
        channels = source.channels
        held = numpy.zeros((0, channels), dtype=numpy.float32)
        held_first = 0
        for first, last in spans:
            audio_first = max(0, first - pad)
            audio_last = min(total, last + pad)
            need = audio_last - (held_first + held.shape[0])
            if need > 0:
                more = source.read(need, dtype="float32", always_2d=True)
                if more.shape[0] != need:
                    raise RuntimeError(
                        f"{path} ended {need - more.shape[0]} frame(s) earlier on "
                        "the second read than on the first; the input changed "
                        "while the job was reading it"
                    )
                held = numpy.concatenate((held, more))
            drop = audio_first - held_first
            if drop > 0:
                held = held[drop:]
                held_first = audio_first
            yield audio_first, audio_last, held[: audio_last - held_first].copy()


# ------------------------------------------------------------------ stitch


#: The two output choices a caller has (`jobs/rvc/__init__.py`, `RvcParams`).
OUTPUT_RATES = ("native", "input")
OUTPUT_CHANNELS = ("input", "mono")


def out_frame(frame: int, in_rate: int, out_rate: int) -> int:
    """Input frame `frame` as an output frame: `round(frame * out / in)`, half up.

    Integers only, so the answer is the same on every host. Monotonic, and
    `out_frame(total)` is the output's length, so spans that tile the input
    map to spans that tile the output. For a gap of `g` input frames the
    mapped gap is at least `g * out_rate // in_rate`, which is what the seam
    fades are clamped to (`Stitcher`).
    """
    return (2 * frame * out_rate + in_rate) // (2 * in_rate)


def output_rate(choice: str, in_rate: int, converted_rate: int) -> int:
    """The output's sample rate: never below the model's, unless asked.

    Owen, 2026-09-26: "i would like to keep 48 khz but if we cant then we
    cant". `native` is `max(input, what urvc wrote)`; `input` is the input's.
    """
    if choice == "input":
        return int(in_rate)
    return max(int(in_rate), int(converted_rate))


def _output_format(info, frames: int, channels: int = 1) -> tuple[str, str]:
    """The input's container and sample format, RF64 where WAV cannot hold it."""
    container, subtype = info.format, info.subtype
    if container == "WAV":
        size = frames * channels * SUBTYPE_BYTES.get(subtype, 8)
        if size > WAV_DATA_LIMIT_BYTES:
            container = "RF64"
    return container, subtype


def fit_piece(converted, converted_rate: int, rate: int, frames: int, where: str):
    """urvc's output for one piece, at the output `rate` and exactly `frames` long.

    Resampled with a rational polyphase filter (exact ratio, no drift), then
    trimmed or zero-padded AT THE END to the frames the piece was given. The pad
    lands in the overlap, which is discarded, while urvc's shortfall is smaller
    than the overlap. A length off by more than rounding is refused: it is not
    this piece's conversion.
    """
    if converted.ndim == 2:
        converted = converted.mean(axis=1)
    converted = converted.astype(numpy.float32, copy=False)
    if converted_rate != rate:
        common = math.gcd(int(converted_rate), int(rate))
        converted = resample_poly(
            converted, rate // common, converted_rate // common
        ).astype(numpy.float32, copy=False)
    tolerance = int(
        (LENGTH_TOLERANCE_SECONDS + LENGTH_TOLERANCE_FRACTION * frames / rate) * rate
    )
    if abs(converted.shape[0] - frames) > tolerance:
        raise RuntimeError(
            f"urvc returned {converted.shape[0] / rate:.3f} s for {where}, which "
            f"was {frames / rate:.3f} s; that is not a rounding difference"
        )
    if converted.shape[0] >= frames:
        return converted[:frames]
    return numpy.pad(converted, (0, frames - converted.shape[0]))


def seam_fades(spans: list, pad: int, crossfade: int) -> list[int]:
    """The fade, in frames, at the START of each span; 0 for the first.

    A fade needs converted audio from both neighbours on both sides of the seam,
    so it is clamped to twice the overlap and to either neighbour's own length.
    All three are in OUTPUT frames, and `pad` is the least the input's overlap
    maps to (`out_frame`), so the clamp never reaches past what was converted.
    """
    fades = [0]
    for (first_a, last_a), (first_b, last_b) in zip(spans, spans[1:]):
        fades.append(
            max(0, min(crossfade, 2 * pad, last_a - first_a, last_b - first_b))
        )
    return fades


class Stitcher:
    """One input's output file, written piece by piece, frame-exact.

    Everything here is in OUTPUT frames at `rate`: the input's spans, total and
    overlap are mapped through `out_frame` when the stitcher is made. Piece `i`
    owns `[first, last)` of the output. Around each seam `s` a fade of `F`
    frames runs over `[s - F//2, s + F - F//2)`, a raised cosine whose two
    weights sum to one; outside the fades each frame is its own piece's. So the
    frames written are exactly `out_frame(total)`, and `close` refuses to finish
    a file for which that is not true.

    The converted signal is one voice. With `channels` above 1 it is written,
    identical, to every channel: not a per-channel conversion.

    When the rates are not a whole multiple (44.1 kHz in, 48 kHz out) a piece's
    start maps to a fraction of an output frame and is rounded, so each piece
    sits up to HALF an output frame (10.4 us at 48 kHz) from where a single
    whole-file resample would put it. Checked with an identity fake urvc,
    2026-09-26: no lag of a whole frame anywhere, and a residual on a 3.1 kHz
    tone of the size that offset predicts. It is far below what two separate
    RVC conversions already differ by at a seam (the module docstring's
    crossfade note).
    """

    def __init__(
        self,
        path: str,
        info,
        spans: list,
        total: int,
        pad: int,
        crossfade_s: float,
        rate: int,
        channels: int,
    ):
        in_rate = int(info.samplerate)
        self.path = path
        self.in_rate = in_rate
        self.rate = int(rate)
        self.channels = int(channels)
        self.spans = [
            (out_frame(first, in_rate, self.rate), out_frame(last, in_rate, self.rate))
            for first, last in spans
        ]
        self.total = out_frame(total, in_rate, self.rate)
        self.fades = seam_fades(
            self.spans,
            pad * self.rate // in_rate,
            int(round(crossfade_s * self.rate)),
        )
        self.format, self.subtype = _output_format(info, self.total, self.channels)
        self.written = 0
        self.next_piece = 0
        self.tail = None
        self.sink = soundfile.SoundFile(
            path,
            "w",
            samplerate=self.rate,
            channels=self.channels,
            format=self.format,
            subtype=self.subtype,
        )

    def out(self, frame: int) -> int:
        """An input frame of this input, as a frame of this output."""
        return out_frame(frame, self.in_rate, self.rate)

    def _write(self, samples) -> None:
        # libsndfile does not clip float to PCM by default; a resampler's ripple
        # past full scale would wrap round instead of clipping.
        samples = numpy.clip(samples, -1.0, 1.0)
        if self.channels > 1:
            # One voice, the same in every channel (see the class docstring).
            samples = numpy.repeat(samples[:, None], self.channels, axis=1)
        self.sink.write(samples)
        self.written += samples.shape[0]

    def add(self, piece: int, audio_first: int, fitted) -> None:
        """Piece `piece`'s converted audio, covering `[audio_first, ...)`.

        `audio_first` is an OUTPUT frame (`out`), as `fitted` is output audio.
        """
        if piece != self.next_piece:
            raise RuntimeError(
                f"{self.path}: piece {piece} arrived where piece "
                f"{self.next_piece} was due"
            )
        first, last = self.spans[piece]
        fade_in = self.fades[piece]
        fade_out = self.fades[piece + 1] if piece + 1 < len(self.spans) else 0
        before_in, after_in = fade_in // 2, fade_in - fade_in // 2
        before_out, after_out = fade_out // 2, fade_out - fade_out // 2

        def span(a: int, b: int):
            return fitted[a - audio_first : b - audio_first]

        if fade_in:
            weight = 0.5 - 0.5 * numpy.cos(
                numpy.pi * (numpy.arange(fade_in, dtype=numpy.float64) + 0.5) / fade_in
            )
            incoming = span(first - before_in, first + after_in)
            self._write(
                (self.tail * (1.0 - weight) + incoming * weight).astype(numpy.float32)
            )
        self._write(span(first + after_in, last - before_out))
        self.tail = span(last - before_out, last + after_out) if fade_out else None
        self.next_piece += 1

    def close(self) -> None:
        self.sink.close()
        if self.next_piece != len(self.spans) or self.written != self.total:
            raise RuntimeError(
                f"{self.path}: wrote {self.written} frame(s) from "
                f"{self.next_piece} of {len(self.spans)} piece(s); the input has "
                f"{self.total}"
            )

    def abandon(self) -> None:
        try:
            self.sink.close()
        finally:
            try:
                os.remove(self.path)
            except OSError:
                pass


# ------------------------------------------------------------------ memory


def host_memory() -> tuple[int, int] | None:
    """`(available, total)` bytes of this host's memory, or None if it will not say.

    Linux (the cuda-linux guest): `/proc/meminfo`'s MemAvailable and MemTotal.
    Inside WSL that is the VM's memory, which is the pool urvc really draws on,
    not the Windows machine's. macOS: `vm_stat`'s free + inactive + speculative
    + purgeable pages and `hw.memsize`, the same reading of unified memory
    `crucible/accelerator.py` takes. Stdlib only, for this module's reason.
    """
    try:
        if sys.platform.startswith("linux"):
            fields: dict[str, int] = {}
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    key, _, value = line.partition(":")
                    parts = value.split()
                    if parts and parts[0].isdigit():
                        fields[key.strip()] = int(parts[0]) * 1024
            return fields["MemAvailable"], fields["MemTotal"]
        if sys.platform == "darwin":
            text = subprocess.run(
                ["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=30
            ).stdout
            page_match = re.search(r"page size of (\d+)", text)
            if page_match is None:
                return None
            counts: dict[str, int] = {}
            for line in text.splitlines():
                key, _, value = line.partition(":")
                digits = value.strip().rstrip(".")
                if digits.isdigit():
                    counts[key.strip()] = int(digits)
            wanted = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            available = sum(counts.get(key, 0) for key in wanted) * int(page_match.group(1))
            total = int(
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
            )
            return available, total
    except (OSError, KeyError, ValueError, subprocess.SubprocessError):
        return None
    return None


def process_memory(pid: int) -> int | None:
    """Bytes one urvc process holds, or None where this host cannot say.

    Linux: resident PLUS swapped (`VmRSS` + `VmSwap`). Swap counts because it
    is where kylies-pc's growth went on 2026-09-26 (Windows commit at 20.2 of
    23.7 GB a few files into one book): a leak that has been paged out has not
    stopped costing the host. macOS: `ps`'s RSS, which does NOT count Metal
    buffers the way Activity Monitor's footprint does, so it reads low there —
    which is why the seconds bound stands beside it (`memory_plan`).
    """
    try:
        if sys.platform.startswith("linux"):
            held = 0
            with open(f"/proc/{pid}/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith(("VmRSS:", "VmSwap:")):
                        held += int(line.split()[1]) * 1024
            return held
        if sys.platform == "darwin":
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            return int(out) * 1024 if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def memory_plan(request: dict) -> tuple[int, float, str]:
    """`(budget_bytes, batch_audio_s, basis)`: how much one urvc process may hold.

    The incident that set this, 2026-09-26: training-pc-1 sent one rvc job per
    book, 35 ten-minute files on the Mac and 44 on kylies-pc, and the worker
    recycled every 96 FILES, so each book ran in ONE urvc process. The Mac
    reached 28.7 of 29.7 GB of swap at file 26 (~88 GB for python in Activity
    Monitor); kylies-pc's commit reached 20.2 of 23.7 GB a few files in. Both
    were cancelled.

    So the process is recycled by MEMORY, twice over:

    - **A budget from the host**: `memory_fraction` of what was available when
      the job started (never more than that fraction of the total), so the
      number follows the machine — a 16 GB PC's WSL VM and a 64 GB Mac get
      different ones — and leaves the rest for the desktop, the server and the
      driver. A host that cannot say gets `fallback_budget_bytes`.
    - **A seconds bound derived from it**: the budget over the measured growth
      per second of audio (`leak_bytes_per_audio_s`, whose measurement
      `jobs/rvc/__init__.py` cites), capped at `max_batch_audio_s`. It holds
      where the live reading cannot be trusted (macOS, see `process_memory`).
    - **The live reading** (`_run_batch`): after every piece, the process's own
      memory against the budget, and a process past it is stopped and the
      pieces it had not reached go to the next one.

    Pieces are what is counted, never input files: the growth is per second of
    audio converted, whatever file it came from.
    """
    fraction = float(require(request, "memory_fraction", (int, float)))
    leak = float(require(request, "leak_bytes_per_audio_s", (int, float)))
    ceiling = float(require(request, "max_batch_audio_s", (int, float)))
    fallback = int(require(request, "fallback_budget_bytes", int))
    figures = host_memory()
    if figures is None:
        budget = fallback
        basis = (
            f"{fallback / 1e9:.1f} GB, because this host would not say how much "
            "memory it has"
        )
    else:
        available, total = figures
        budget = int(min(available, total) * fraction)
        basis = (
            f"{fraction:.0%} of the {available / 1e9:.1f} GB available of "
            f"{total / 1e9:.1f} GB when the job started"
        )
    seconds = max(1.0, min(ceiling, budget / leak))
    return budget, seconds, basis


# ------------------------------------------------------------------ engine


def _stop(process) -> None:
    """End a urvc process that has passed its budget: politely, then not."""
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_batch(
    argv: list[str],
    models_dir: str,
    tool_dirs: list[str],
    on_line,
    budget_bytes: int,
) -> tuple[int | None, int | None]:
    """One urvc process: `(pieces finished if it was stopped early, peak bytes)`.

    The first is None when urvc converted the whole batch and exited 0; a bad
    exit raises. When the process passes `budget_bytes` after a piece and
    before the batch's last, it is STOPPED and the count of pieces it finished
    comes back: urvc converts in `int(stem)` order and prints `[RVC] n/total`
    after the n-th output is written, so the first n slots are whole and
    anything after is not trusted. At least one piece is always finished per
    process, so a host whose budget is below urvc's own baseline still makes
    progress, one piece per process.

    The child inherits this worker's environment, which the SERVER set
    (`URVC_SKIP_INIT`, `HF_HUB_OFFLINE`, `KMP_DUPLICATE_LIB_OK`,
    `OMP_NUM_THREADS`) — see `jobs/rvc/__init__.py` for what each one is for. Two
    things are added here rather than there, because both depend on where this
    worker itself is running:

    - `URVC_MODELS_DIR`, the model root the server staged for this job.
    - **PATH: the ffmpeg and ffprobe the server checked, then this env's `bin`,
      then what was inherited.** urvc's `_add_ffmpeg_paths` asks PATH for
      ffmpeg and ffprobe and, finding none, reaches for `static_ffmpeg` — which
      the recipe no longer carries (2026-09-26, #25): the ffmpeg is Crucible's
      pinned build in `~/.crucible/tools/bin`, found by `hosttools` and named
      in the request. The env's `bin` still comes next, for `static-sox` and
      the env's own scripts, which are beside this interpreter and nowhere a
      bare `crucible serve` would have on its PATH (BookForge's
      `relocatableEnvBinDirs`, `electron/rvc-bridge.ts`).

    urvc's stderr goes to a file, not a pipe: a pipe read only at the end is a
    pipe that fills during a long batch and stops urvc mid-write.
    """
    environment = dict(os.environ)
    environment["URVC_MODELS_DIR"] = models_dir
    own_bin = os.path.dirname(os.path.abspath(sys.executable))
    first: list[str] = []
    for entry in [*tool_dirs, own_bin]:
        if entry and entry not in first:
            first.append(entry)
    environment["PATH"] = os.pathsep.join(
        [*first, environment.get("PATH", "")]
    ).rstrip(os.pathsep)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        tail: list[str] = []
        peak: int | None = None
        for line in process.stdout:
            stripped = line.rstrip("\n")
            tail.append(stripped)
            del tail[:-40]
            # urvc's stdout is READ, never forwarded: fd 1 here is the result stream.
            print(f"[urvc] {stripped}")
            on_line(stripped)
            match = PROGRESS_LINE.match(stripped.strip())
            if match is None:
                continue
            finished, of = int(match.group(1)), int(match.group(2))
            held = process_memory(process.pid)
            if held is not None:
                peak = held if peak is None else max(peak, held)
            if 0 < finished < of and held is not None and held > budget_bytes:
                print(
                    f"[rvc] urvc holds {held / 1e9:.2f} GB after {finished} of {of} "
                    f"piece(s), past this job's {budget_bytes / 1e9:.2f} GB budget; "
                    "recycling it"
                )
                _stop(process)
                return finished, peak
        code = process.wait()
        stderr.seek(0)
        errors = stderr.read()
    if code != 0:
        print(errors)
        detail = (errors.strip() or "\n".join(tail)).strip()[-1500:]
        raise RuntimeError(f"urvc convert-dir exited {code}: {detail}")
    return None, peak


def _batches(seconds: list[float], batch_size: int, batch_audio_s: float) -> list[list[int]]:
    """Piece indices per urvc process, in order — the PLAN, for `ready`.

    A batch closes at `batch_size` pieces or `batch_audio_s` seconds of audio,
    whichever comes first; a piece longer than the budget is a batch of its own.
    The live memory reading can end a batch sooner, so the count is an
    estimate from below.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    held = 0.0
    for number, length in enumerate(seconds):
        if current and (len(current) >= batch_size or held + length > batch_audio_s):
            batches.append(current)
            current, held = [], 0.0
        current.append(number)
        held += length
    if current:
        batches.append(current)
    return batches


def _describe_unreadable(name: str, exc: Exception) -> str:
    """Why an input cannot be read, and what to change — said about the BYTES.

    #45, 2026-09-26: the old refusal blamed the input's name. The name does not
    matter now; only what is in the file does.
    """
    has_extension = bool(os.path.splitext(name)[1])
    hint = (
        ""
        if has_extension
        else " (its name has no extension, and that is fine: the format is read "
        "from the bytes, not the name)"
    )
    return (
        f"input {name!r} is not audio this job can read{hint}: {exc}. Send it as "
        "WAV, FLAC, OGG, MP3 or AIFF"
    )


def main() -> int:
    line = sys.stdin.readline()
    if not line.strip():
        return fail("the rvc worker was given no request on stdin")
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return fail(f"the rvc request is not JSON: {exc}")
    if not isinstance(request, dict):
        return fail(f"the rvc request must be a JSON object, got {type(request).__name__}")

    try:
        models_dir = require(request, "models_dir", str)
        model_name = require(request, "model_name", str)
        input_dir = require(request, "input_dir", str)
        output_dir = require(request, "output_dir", str)
        inputs = require(request, "inputs", list)
        batch_size = require(request, "batch_size", int)
        piece_s = float(require(request, "piece_s", (int, float)))
        overlap_s = float(require(request, "overlap_s", (int, float)))
        crossfade_s = float(require(request, "crossfade_s", (int, float)))
        rate_choice = require(request, "output_rate", str)
        channel_choice = require(request, "output_channels", str)
        staging_root = require(request, "staging_dir", str)
        tool_dirs = [
            os.path.dirname(require(request, tool, str)) for tool in ("ffmpeg", "ffprobe")
        ]
        require(request, "index_rate", (int, float))
        require(request, "protect_rate", (int, float))
        require(request, "n_semitones", int)
        budget_bytes, batch_audio_s, budget_basis = memory_plan(request)
    except KeyError as exc:
        return fail(str(exc.args[0]))
    if not inputs:
        return fail("the rvc request lists no inputs")
    if batch_size < 1:
        return fail(f"batch_size must be at least 1, got {batch_size}")
    if rate_choice not in OUTPUT_RATES or channel_choice not in OUTPUT_CHANNELS:
        return fail(
            f"output_rate {rate_choice!r}, output_channels {channel_choice!r}: "
            "the server validates these, so this is a server bug"
        )
    if piece_s < 4 * MIN_TAIL_SECONDS or overlap_s < 0 or crossfade_s < 0:
        return fail(
            f"piece_s {piece_s}, overlap_s {overlap_s}, crossfade_s {crossfade_s}: "
            "the server validates these, so this is a server bug"
        )

    # ---- plan every input before the engine starts, so that an unreadable
    # file fails the job in seconds rather than after hours of conversion.
    plans = []
    unreadable = []
    for number, name in enumerate(inputs, start=1):
        path = os.path.join(input_dir, name)
        try:
            info = soundfile.info(path)
            container, subtype = _output_format(info, int(info.frames))
            if not soundfile.check_format(container, subtype):
                raise RuntimeError(
                    f"this build can read {info.format}/{info.subtype} and not "
                    "write it back"
                )

            def on_cut(seconds: float, number: int = number) -> None:
                send(
                    "progress",
                    stage="cutting",
                    processed=number - 1,
                    total=len(inputs),
                    at_s=round(seconds, 1),
                )

            spans, total = plan_input(path, piece_s, on_cut)
        except Exception as exc:  # noqa: BLE001 — libsndfile raises several kinds
            unreadable.append(_describe_unreadable(name, exc))
            continue
        if total == 0:
            unreadable.append(f"input {name!r} holds no audio (zero frames)")
            continue
        plans.append((name, path, info, spans, total))
        send("progress", stage="cutting", processed=number, total=len(inputs))
    if unreadable:
        return fail(
            f"{len(unreadable)} of {len(inputs)} input(s) cannot be converted, so "
            "nothing was: " + "; ".join(unreadable[:20])
            + (f" (and {len(unreadable) - 20} more)" if len(unreadable) > 20 else "")
        )

    # ---- every piece of every input, in order: (plan, piece, audio span).
    pieces = []
    for plan_number, (name, path, info, spans, total) in enumerate(plans):
        rate = int(info.samplerate)
        pad = int(round(overlap_s * rate))
        for piece_number, (first, last) in enumerate(spans):
            audio_first = max(0, first - pad)
            audio_last = min(total, last + pad)
            pieces.append(
                (plan_number, piece_number, audio_first, audio_last, rate)
            )
    durations = [(last - first) / float(rate) for _, _, first, last, rate in pieces]

    planned = len(_batches(durations, batch_size, batch_audio_s))
    os.makedirs(output_dir, exist_ok=True)
    send(
        "ready",
        files=len(inputs),
        pieces=len(pieces),
        batches=planned,
        batch_size=batch_size,
        batch_audio_s=round(batch_audio_s, 1),
        memory_budget_bytes=budget_bytes,
        memory_basis=budget_basis,
        model=model_name,
    )

    # One reader and one stitcher per input, opened when its first piece is
    # due and closed with its last, so a many-sentence job holds one of each.
    readers: dict[int, object] = {}
    stitchers: dict[int, Stitcher] = {}

    def reader_for(plan_number: int):
        if plan_number not in readers:
            name, path, info, spans, total = plans[plan_number]
            pad = int(round(overlap_s * int(info.samplerate)))
            readers[plan_number] = read_pieces(path, spans, total, pad)
        return readers[plan_number]

    def stitcher_for(plan_number: int, converted_rate: int) -> Stitcher:
        # Made when the input's FIRST piece comes back, because the output rate
        # is read from what urvc wrote (the module docstring's sample rate).
        if plan_number not in stitchers:
            name, path, info, spans, total = plans[plan_number]
            rate = int(info.samplerate)
            stitchers[plan_number] = Stitcher(
                os.path.join(output_dir, name),
                info,
                spans,
                total,
                int(round(overlap_s * rate)),
                crossfade_s,
                output_rate(rate_choice, rate, converted_rate),
                1 if channel_choice == "mono" else int(info.channels),
            )
        return stitchers[plan_number]

    # Pieces are staged into `holding`, and a piece a stopped process did not
    # reach waits there (`carried`) for the next one: its audio has already
    # been read, and the reader only goes forward.
    # Under the JOB's directory, not the system temp: a cancel's signal skips
    # the `finally` below, and the retention collector reaps job directories
    # and nothing else (2026-09-26, Owen: "we're supposed to have a garbage
    # collector that cleans up files older than 7 days").
    os.makedirs(staging_root, exist_ok=True)
    holding = tempfile.mkdtemp(prefix="crucible-rvc-", dir=staging_root)
    carried: list[tuple[int, str]] = []
    next_new = 0
    done_count = 0
    number = 0
    try:
        while done_count < len(pieces):
            number += 1
            # A fresh directory per batch, and a child that EXITS when it is
            # finished with it. That exit is the memory bound; see
            # `memory_plan` for what happens without it.
            staging = os.path.join(holding, f"batch-{number}")
            converted_dir = os.path.join(staging, "out")
            os.makedirs(converted_dir)
            batch: list[int] = []
            seconds = 0.0

            def fits(piece_index: int) -> bool:
                return not batch or (
                    len(batch) < batch_size
                    and seconds + durations[piece_index] <= batch_audio_s
                )

            while carried and fits(carried[0][0]):
                piece_index, waiting = carried.pop(0)
                os.replace(waiting, os.path.join(staging, f"{len(batch)}.wav"))
                batch.append(piece_index)
                seconds += durations[piece_index]
            while not carried and next_new < len(pieces) and fits(next_new):
                plan_number, _, audio_first, audio_last, rate = pieces[next_new]
                got_first, got_last, samples = next(reader_for(plan_number))
                assert (got_first, got_last) == (audio_first, audio_last)
                soundfile.write(
                    os.path.join(staging, f"{len(batch)}.wav"),
                    samples,
                    rate,
                    subtype="FLOAT",
                )
                batch.append(next_new)
                seconds += durations[next_new]
                next_new += 1

            def on_line(text: str, offset: int = done_count, batch_number: int = number) -> None:
                match = PROGRESS_LINE.match(text.strip())
                if match is None:
                    return
                send(
                    "progress",
                    stage="converting",
                    processed=offset + int(match.group(1)),
                    total=len(pieces),
                    batch=batch_number,
                    batches=max(planned, batch_number),
                )

            try:
                stopped_at, peak = _run_batch(
                    _convert_args(request, staging, converted_dir),
                    models_dir,
                    tool_dirs,
                    on_line,
                    budget_bytes,
                )
            except Exception as exc:
                # A batch that could not run at all is the whole job's
                # problem: every piece in it is missing, and the run cannot
                # tell which of them the engine would have managed. Say so
                # once, with the engine's own words. Inputs already finished
                # were published as they finished, and stay.
                return fail(
                    f"batch {number} failed: {type(exc).__name__}: {exc}"
                )
            finished = len(batch) if stopped_at is None else stopped_at
            if peak is not None:
                print(
                    f"[rvc] batch {number}: {finished} of {len(batch)} piece(s), "
                    f"urvc peaked at {peak / 1e9:.2f} GB of a "
                    f"{budget_bytes / 1e9:.2f} GB budget"
                )
            # What the stopped process did not reach waits for the next one,
            # ahead of anything still to be staged.
            carried = [
                (piece_index, _park(holding, staging, slot, piece_index))
                for slot, piece_index in enumerate(batch)
                if slot >= finished
            ] + carried

            for slot, piece_index in enumerate(batch[:finished]):
                plan_number, piece_number, audio_first, audio_last, rate = pieces[
                    piece_index
                ]
                name, _, _, spans, _ = plans[plan_number]
                first, last = spans[piece_number]
                where = (
                    f"piece {piece_number + 1} of {len(spans)} of {name!r} "
                    f"({first / rate:.2f}-{last / rate:.2f} s)"
                )
                produced = os.path.join(converted_dir, f"{slot}.wav")
                if not os.path.isfile(produced):
                    # EVERY INPUT MUST PRODUCE AN OUTPUT, and an input with a
                    # hole in it has not. Failing now saves the hours the
                    # rest of a master would take to end in the same refusal.
                    return fail(f"urvc wrote no output for {where}")
                converted, converted_rate = soundfile.read(produced, dtype="float32")
                os.remove(produced)
                stitcher = stitcher_for(plan_number, int(converted_rate))
                fitted = fit_piece(
                    converted, int(converted_rate), stitcher.rate,
                    stitcher.out(audio_last) - stitcher.out(audio_first), where,
                )
                stitcher.add(piece_number, stitcher.out(audio_first), fitted)
                if piece_number == len(spans) - 1:
                    stitcher.close()
                    del stitchers[plan_number]
                    readers.pop(plan_number, None)
                    # AS EACH INPUT FINISHES, not at the end (2026-09-26: a
                    # cancel at 26 of 35 lost all 26). Inputs finish in the
                    # order they arrived, so the Nth result is still the Nth
                    # input, and the server publishes it the moment it lands.
                    send(
                        "result",
                        bytes=os.path.getsize(stitcher.path),
                        frames=stitcher.written,
                        sample_rate=stitcher.rate,
                        channels=stitcher.channels,
                        format=stitcher.format,
                        subtype=stitcher.subtype,
                    )
            done_count += finished
            shutil.rmtree(staging, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 — a cut or stitch fault is the run's
        return fail(f"{type(exc).__name__}: {exc}")
    finally:
        for stitcher in stitchers.values():
            stitcher.abandon()
        shutil.rmtree(holding, ignore_errors=True)

    send("done")
    return 0


def _park(holding: str, staging: str, slot: int, piece_index: int) -> str:
    """Move a staged piece a stopped process did not reach out of its batch."""
    waiting = os.path.join(holding, f"waiting-{piece_index}.wav")
    os.replace(os.path.join(staging, f"{slot}.wav"), waiting)
    return waiting


if __name__ == "__main__":
    sys.exit(main())
