from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import json
import math
import re
import shutil
import subprocess
import tempfile
import time

from workerio import fail, send

import numpy
import soundfile
from scipy.signal import resample_poly

PROGRESS_LINE = re.compile(r"^\[RVC\]\s+(\d+)/(\d+)")

SEARCH_SECONDS = 10.0
ENERGY_WINDOW_MS = 100.0

MIN_TAIL_SECONDS = 1.0

READ_BLOCK_FRAMES = 1 << 18

LENGTH_TOLERANCE_SECONDS = 0.1
LENGTH_TOLERANCE_FRACTION = 0.005

WAV_DATA_LIMIT_BYTES = 0xFFFFFFFF - (1 << 20)

SUBTYPE_BYTES = {
    "PCM_S8": 1, "PCM_U8": 1, "ULAW": 1, "ALAW": 1,
    "PCM_16": 2, "PCM_24": 3, "PCM_32": 4, "FLOAT": 4, "DOUBLE": 8,
}

CUT_REPORT_SECONDS = 5.0


def require(request: dict, key: str, kind):
    return workerio.require(
        request,
        key,
        kind,
        "rvc",
        "every parameter except 'f0_method' and 'hop_length' is required, and "
        "those two are absent on purpose",
    )


def _convert_args(request: dict, input_dir: str, output_dir: str) -> list[str]:
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
        argv += ["--n-semitones", str(n_semitones)]
    return argv


def _quietest(magnitude, window: int) -> int:
    if magnitude.shape[0] <= window:
        return int(magnitude.shape[0])
    running = numpy.concatenate(([0.0], numpy.cumsum(magnitude, dtype=numpy.float64)))
    sums = running[window:] - running[:-window]
    return int(numpy.argmin(sums)) + window // 2


def _cut_lengths(rate: int, piece_s: float) -> tuple[int, int, int, int]:
    max_len = int(piece_s * rate)
    search = int(min(SEARCH_SECONDS, piece_s / 2) * rate)
    window = max(4, int(ENERGY_WINDOW_MS / 1000.0 * rate))
    min_tail = int(MIN_TAIL_SECONDS * rate)
    return max_len, search, window, min_tail


def plan_input(path: str, piece_s: float, on_progress=None) -> tuple[list, int]:
    info = soundfile.info(path)
    rate = int(info.samplerate)
    max_len, search, window, min_tail = _cut_lengths(rate, piece_s)

    spans: list[tuple[int, int]] = []
    start = 0
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
        return max(start, start + max_len - search - min_tail)

    with soundfile.SoundFile(path) as source:
        for block in source.blocks(
            blocksize=READ_BLOCK_FRAMES, dtype="float32", always_2d=True
        ):
            magnitude = numpy.abs(block).mean(axis=1)
            held = numpy.concatenate((held, magnitude))
            position += block.shape[0]
            while position - start > max_len + min_tail:
                boundary = cut_at(start + max_len)
                spans.append((start, boundary))
                start = boundary
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


OUTPUT_RATES = ("native", "input")
OUTPUT_CHANNELS = ("input", "mono")


def out_frame(frame: int, in_rate: int, out_rate: int) -> int:
    return (2 * frame * out_rate + in_rate) // (2 * in_rate)


def output_rate(choice: str, in_rate: int, converted_rate: int) -> int:
    if choice == "input":
        return int(in_rate)
    return max(int(in_rate), int(converted_rate))


def _output_format(info, frames: int, channels: int = 1) -> tuple[str, str]:
    container, subtype = info.format, info.subtype
    if container == "WAV":
        size = frames * channels * SUBTYPE_BYTES.get(subtype, 8)
        if size > WAV_DATA_LIMIT_BYTES:
            container = "RF64"
    return container, subtype


def fit_piece(converted, converted_rate: int, rate: int, frames: int, where: str):
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
    fades = [0]
    for (first_a, last_a), (first_b, last_b) in zip(spans, spans[1:]):
        fades.append(
            max(0, min(crossfade, 2 * pad, last_a - first_a, last_b - first_b))
        )
    return fades


class Stitcher:
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
        return out_frame(frame, self.in_rate, self.rate)

    def _write(self, samples) -> None:
        samples = numpy.clip(samples, -1.0, 1.0)
        if self.channels > 1:
            samples = numpy.repeat(samples[:, None], self.channels, axis=1)
        self.sink.write(samples)
        self.written += samples.shape[0]

    def add(self, piece: int, audio_first: int, fitted) -> None:
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


def host_memory() -> tuple[int, int] | None:
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


def _stop(process) -> None:
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
        except Exception as exc:
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

    readers: dict[int, object] = {}
    stitchers: dict[int, Stitcher] = {}

    def reader_for(plan_number: int):
        if plan_number not in readers:
            name, path, info, spans, total = plans[plan_number]
            pad = int(round(overlap_s * int(info.samplerate)))
            readers[plan_number] = read_pieces(path, spans, total, pad)
        return readers[plan_number]

    def stitcher_for(plan_number: int, converted_rate: int) -> Stitcher:
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

    os.makedirs(staging_root, exist_ok=True)
    holding = tempfile.mkdtemp(prefix="crucible-rvc-", dir=staging_root)
    carried: list[tuple[int, str]] = []
    next_new = 0
    done_count = 0
    number = 0
    try:
        while done_count < len(pieces):
            number += 1
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
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")
    finally:
        for stitcher in stitchers.values():
            stitcher.abandon()
        shutil.rmtree(holding, ignore_errors=True)

    send("done")
    return 0


def _park(holding: str, staging: str, slot: int, piece_index: int) -> str:
    waiting = os.path.join(holding, f"waiting-{piece_index}.wav")
    os.replace(os.path.join(staging, f"{slot}.wav"), waiting)
    return waiting


if __name__ == "__main__":
    sys.exit(main())
