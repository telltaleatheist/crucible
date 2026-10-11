"""stemaudio with the real tools: ffmpeg decodes, soxr resamples, libsndfile encodes. The
separator itself is stood in for by a copy of what it was given (the identity "model"),
so what is measured is the round trip around it: rate, frames and container.

Needs the rvc env's numpy, soundfile and soxr and an ffmpeg; skipped where they are not
(the test env carries none of them; run it with the rvc env's python)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
soundfile = pytest.importorskip("soundfile")
pytest.importorskip("soxr")

JOBS_DIR = Path(__file__).resolve().parents[1] / "crucible" / "jobs"
sys.path.insert(0, str(JOBS_DIR))
import workerio  # noqa: E402

sys.path.pop(0)
stemaudio = workerio.load_sibling("stemaudio", str(JOBS_DIR / "denoise" / "worker.py"))


def _tool(name: str) -> str:
    home = Path(os.environ.get("CRUCIBLE_HOME", Path.home() / ".crucible"))
    placed = home / "tools" / "bin" / name
    found = str(placed) if placed.is_file() else shutil.which(name)
    if found is None:
        pytest.skip(f"no {name}")
    return found


SEPARATOR_RATE = 44100


def _tone(rate: int, frames: int):
    t = np.arange(frames) / rate
    return np.stack(
        [0.3 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 660 * t)], axis=1
    ).astype("float32")


def _separate(plan: dict, work: Path) -> Path:
    """The identity separator: one stem, its input, as audio-separator writes it (16-bit WAV)."""
    samples, rate = soundfile.read(plan["separator_input"], dtype="float32", always_2d=True)
    separated = work / "separated"
    separated.mkdir()
    base = Path(plan["separator_input"]).stem
    raw = separated / f"{base}_(other)_vocals_mel_band_roformer.wav"
    soundfile.write(raw, samples, rate, format="WAV", subtype="PCM_16")
    return raw


@pytest.mark.parametrize("container", ["flac", "mp3"])
@pytest.mark.parametrize("audio_format", ["wav", "flac", "mp3"])
def test_a_48k_song_comes_back_at_48k_and_exactly_its_frames(
    tmp_path: Path, container: str, audio_format: str
) -> None:
    frames = 48000 * 7 + 123
    source = tmp_path / f"song.{container}"
    workerio.write_audio(str(source), _tone(48000, frames), 48000, container)
    work = tmp_path / "work"
    out = tmp_path / "stems"
    work.mkdir()
    out.mkdir()

    plan = stemaudio.prepare(_tool("ffmpeg"), _tool("ffprobe"), str(source), str(work), SEPARATOR_RATE)
    assert (plan["sample_rate"], plan["frames"], plan["channels"]) == (48000, frames, 2)
    assert plan["codec"] == container and plan["decoded"] and plan["resampled"]
    assert soundfile.info(plan["separator_input"]).samplerate == SEPARATOR_RATE
    assert plan["separator_frames"] == soundfile.info(plan["separator_input"]).frames
    assert Path(plan["separator_input"]).name == "song.wav"

    stem = stemaudio.finish(str(_separate(plan, work)), str(out), plan, audio_format)
    assert stem["name"] == f"song_(other)_vocals_mel_band_roformer.{audio_format}"
    assert (stem["separated_rate"], stem["separated_frames"]) == (
        SEPARATOR_RATE, plan["separator_frames"]
    )
    assert (stem["sample_rate"], stem["frames"]) == (48000, frames)
    written = soundfile.info(str(out / stem["name"]))
    assert written.format == {"wav": "WAV", "flac": "FLAC", "mp3": "MP3"}[audio_format]


def test_a_44k_wav_goes_to_the_separator_as_sent_and_its_stem_untouched(tmp_path: Path) -> None:
    frames = SEPARATOR_RATE * 3
    source = tmp_path / "block_00.wav"
    soundfile.write(source, _tone(SEPARATOR_RATE, frames), SEPARATOR_RATE, subtype="PCM_16")
    work = tmp_path / "work"
    out = tmp_path / "stems"
    work.mkdir()
    out.mkdir()
    plan = stemaudio.prepare(_tool("ffmpeg"), _tool("ffprobe"), str(source), str(work), SEPARATOR_RATE)
    assert plan["separator_input"] == str(source)
    assert (plan["decoded"], plan["resampled"], plan["codec"]) == (False, False, "pcm_s16le")
    raw = _separate(plan, work)
    before = raw.read_bytes()
    stem = stemaudio.finish(str(raw), str(out), plan, "wav")
    assert (out / stem["name"]).read_bytes() == before
    assert (stem["sample_rate"], stem["frames"]) == (SEPARATOR_RATE, frames)


def test_fit_trims_or_pads_rounding_and_refuses_more(tmp_path: Path) -> None:
    samples = np.ones((100, 2), dtype="float32")
    assert stemaudio.fit(samples, 99).shape == (99, 2)
    padded = stemaudio.fit(samples, 101)
    assert padded.shape == (101, 2) and padded[-1].tolist() == [0.0, 0.0]
    with pytest.raises(RuntimeError, match="not rounding"):
        stemaudio.fit(samples, 110)


def test_something_that_is_not_audio_is_refused_by_name(tmp_path: Path) -> None:
    source = tmp_path / "song.flac"
    source.write_bytes(b"not audio at all")
    # ffprobe or ffmpeg, whichever gives up first, says so naming the file.
    with pytest.raises(RuntimeError, match=r"could not (be read as audio|decode song\.flac)"):
        stemaudio.prepare(_tool("ffmpeg"), _tool("ffprobe"), str(source), str(tmp_path), SEPARATOR_RATE)
