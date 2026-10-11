"""The audio on either side of the separator: the client's file made into what the
separator reads, and each stem it writes made back into the client's timeline.

A separator reads WAV or FLAC at its own rate (44100 Hz for both shipped ones). A client
may send anything ffmpeg decodes at any rate - a YuE2 song is 48 kHz FLAC or mp3 - so
anything else is decoded with Crucible's own ffmpeg and resampled here, and every stem is
resampled back. A stem therefore always has the input's rate and exactly its frame count,
the promise a client slicing a stem at the input's sample offsets relies on (docs/API-CLI.md).

Runs inside the denoise worker (the rvc env: numpy, soundfile, soxr); a sibling of
worker.py, loaded through workerio.load_sibling.
"""

from __future__ import annotations

import json
import os
import subprocess

import workerio

# What the separator reads as sent, when it is at the separator's own rate: the file goes
# to it untouched, as every book client's 44.1 kHz WAV block always has.
NATIVE_CONTAINERS = ("WAV", "FLAC")

# soxr's very-high-quality setting, both ways: down to the separator's rate and back up.
RESAMPLE_QUALITY = "VHQ"

# Resampling a stem back from the separator's rate lands within a frame of the input's
# length (soxr rounds each way); the tail is trimmed or padded with silence to exactly
# the input's frames. More than this is not rounding, and is refused.
LENGTH_SLACK_FRAMES = 2

TOOL_TIMEOUT_SECONDS = 600


def _tail(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").strip()[-500:]


def probe(ffprobe: str, source: str) -> dict:
    """The input's container and its first audio stream's codec, for the record."""
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name:format=format_name",
                "-of", "json", source,
            ],
            capture_output=True,
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ffprobe could not run on {os.path.basename(source)}: {exc}") from None
    if completed.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(source)} could not be read as audio: ffprobe exited "
            f"{completed.returncode}: {_tail(completed.stderr)}"
        )
    answer = json.loads(completed.stdout.decode("utf-8", "replace") or "{}")
    streams = answer.get("streams") or []
    if not streams:
        raise RuntimeError(f"{os.path.basename(source)} has no audio stream")
    return {
        "codec": str(streams[0].get("codec_name")),
        "container": str((answer.get("format") or {}).get("format_name")),
    }


def decode(ffmpeg: str, source: str, target: str) -> None:
    """The input's first audio stream as 32-bit float WAV, at its own rate and channels."""
    try:
        completed = subprocess.run(
            [
                ffmpeg, "-nostdin", "-v", "error", "-y", "-i", source,
                "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "wav", target,
            ],
            capture_output=True,
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ffmpeg could not run on {os.path.basename(source)}: {exc}") from None
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not decode {os.path.basename(source)}: exited "
            f"{completed.returncode}: {_tail(completed.stderr)}"
        )


def _native(source: str):
    import soundfile

    try:
        return soundfile.info(source)
    except soundfile.LibsndfileError:
        # Not a container libsndfile reads (m4a, ogg/opus, ...): ffmpeg decodes it.
        return None


def prepare(ffmpeg: str, ffprobe: str, source: str, work_dir: str, separator_rate: int) -> dict:
    """What the separator is given, and the input's timeline every stem returns to."""
    import soundfile

    stream = probe(ffprobe, source)
    native = _native(source)
    if (
        native is not None
        and native.format in NATIVE_CONTAINERS
        and int(native.samplerate) == separator_rate
    ):
        return {
            **stream,
            "sample_rate": int(native.samplerate),
            "frames": int(native.frames),
            "channels": int(native.channels),
            "separator_input": source,
            "separator_rate": separator_rate,
            "separator_frames": int(native.frames),
            "decoded": False,
            "resampled": False,
        }

    # The separator names its stems after the file it reads, so what it reads keeps the
    # client's name: `song.mp3` comes back as `song_(vocals)_….<format>` either way.
    base = os.path.splitext(os.path.basename(source))[0]
    decoded = os.path.join(work_dir, base + ".decoded.wav")
    decode(ffmpeg, source, decoded)
    samples, rate = soundfile.read(decoded, dtype="float32", always_2d=True)
    os.remove(decoded)
    if samples.shape[0] == 0:
        raise RuntimeError(f"{os.path.basename(source)} decoded to no audio")
    resampled = int(rate) != separator_rate
    if resampled:
        import soxr

        prepared = soxr.resample(samples, int(rate), separator_rate, RESAMPLE_QUALITY)
    else:
        prepared = samples
    target = os.path.join(work_dir, base + ".wav")
    soundfile.write(target, prepared, separator_rate, format="WAV", subtype="FLOAT")
    return {
        **stream,
        "sample_rate": int(rate),
        "frames": int(samples.shape[0]),
        "channels": int(samples.shape[1]),
        "separator_input": target,
        "separator_rate": separator_rate,
        "separator_frames": int(prepared.shape[0]),
        "decoded": True,
        "resampled": resampled,
    }


def fit(samples, frames: int):
    """`samples` at exactly `frames`: the resampler's rounding trimmed or padded at the end."""
    import numpy

    have = int(samples.shape[0])
    if abs(have - frames) > LENGTH_SLACK_FRAMES:
        raise RuntimeError(
            f"a stem resampled back came to {have} frames against the input's {frames}; "
            f"rounding moves it at most {LENGTH_SLACK_FRAMES}, so this is not rounding"
        )
    if have >= frames:
        return samples[:frames]
    pad = numpy.zeros((frames - have,) + samples.shape[1:], dtype=samples.dtype)
    return numpy.concatenate([samples, pad])


def finish(raw: str, output_dir: str, plan: dict, audio_format: str) -> dict:
    """One separated stem, on the input's timeline and in the client's format."""
    import soundfile

    separated = soundfile.info(raw)
    name = os.path.splitext(os.path.basename(raw))[0] + "." + audio_format
    final = os.path.join(output_dir, name)
    if not plan["resampled"] and audio_format == "wav":
        # The separator's own WAV, byte for byte: what every denoise returned before formats.
        os.replace(raw, final)
    else:
        samples, rate = soundfile.read(raw, dtype="float32", always_2d=True)
        rate = int(rate)
        if plan["resampled"]:
            import soxr

            samples = soxr.resample(samples, rate, plan["sample_rate"], RESAMPLE_QUALITY)
            rate = plan["sample_rate"]
            # Only a stem the separator returned at its input's length is fitted: one it
            # changed keeps the change, and the job refuses it by name.
            if int(separated.frames) == plan["separator_frames"]:
                samples = fit(samples, plan["frames"])
        workerio.write_audio(final, samples, rate, audio_format)
        os.remove(raw)
    written = soundfile.info(final)
    return {
        "name": name,
        "sample_rate": int(written.samplerate),
        "frames": int(written.frames),
        "channels": int(written.channels),
        "bytes": os.path.getsize(final),
        "separated_rate": int(separated.samplerate),
        "separated_frames": int(separated.frames),
    }
