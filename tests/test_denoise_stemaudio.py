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


class _StandInSeparator:
    """What audio-separator 0.31.1's MDXC separate does around each stem, on its logger:
    "Saving <stem> stem to <file>..." and then the write, which for a near-silent stem is
    a warning and no file (common_separator.write_audio_pydub)."""

    def __init__(self, logger, directory: Path, frames: int, near_silent_warns: bool) -> None:
        self.logger, self.directory, self.frames = logger, directory, frames
        self.near_silent_warns = near_silent_warns

    def separate(self, base: str) -> None:
        for stem, silent in (("other", False), ("vocals", True)):
            name = f"{base}_({stem})_vocals_mel_band_roformer.wav"
            self.logger.info(f"Saving {stem} stem to {name}...")
            if not silent:
                soundfile.write(
                    self.directory / name, _tone(SEPARATOR_RATE, self.frames),
                    SEPARATOR_RATE, subtype="PCM_16",
                )
            elif self.near_silent_warns:
                self.logger.warning("Warning: stem_source array is near-silent or empty.")


def _separate_instrumental(tmp_path: Path, near_silent_warns: bool):
    import logging

    frames = 48000 * 5
    source = tmp_path / "song.flac"
    workerio.write_audio(str(source), _tone(48000, frames), 48000, "flac")
    work = tmp_path / "work"
    out = tmp_path / "stems"
    work.mkdir()
    out.mkdir()
    plan = stemaudio.prepare(_tool("ffmpeg"), _tool("ffprobe"), str(source), str(work), SEPARATOR_RATE)
    separated = work / "separated"
    separated.mkdir()
    logger = logging.getLogger(f"stand-in-separator-{near_silent_warns}")
    logger.setLevel(logging.INFO)
    watch = stemaudio.NearSilentWatch()
    logger.addHandler(watch)
    try:
        _StandInSeparator(logger, separated, plan["separator_frames"], near_silent_warns).separate("song")
    finally:
        logger.removeHandler(watch)
    filled = stemaudio.fill_silent(str(separated), watch.silent, plan)
    stems = [
        stemaudio.finish(str(separated / name), str(out), plan, "mp3")
        for name in sorted(os.listdir(separated))
    ]
    return plan, filled, stems, out


def test_a_stem_the_library_skipped_as_near_silent_comes_back_as_silence(tmp_path: Path) -> None:
    plan, filled, stems, out = _separate_instrumental(tmp_path, near_silent_warns=True)
    assert filled == ["vocals"]
    names = [stem["name"] for stem in stems]
    assert names == [
        "song_(other)_vocals_mel_band_roformer.mp3",
        "song_(vocals)_vocals_mel_band_roformer.mp3",
    ]
    vocals = stems[1]
    assert (vocals["sample_rate"], vocals["frames"]) == (48000, plan["frames"])
    assert (vocals["separated_rate"], vocals["separated_frames"]) == (
        SEPARATOR_RATE, plan["separator_frames"]
    )
    samples, _ = soundfile.read(str(out / vocals["name"]), dtype="float32")
    assert float(np.abs(samples).max()) == 0.0


def test_a_stem_missing_with_no_near_silent_warning_is_not_filled(tmp_path: Path) -> None:
    _plan, filled, stems, _out = _separate_instrumental(tmp_path, near_silent_warns=False)
    assert filled == []
    assert [stem["name"] for stem in stems] == ["song_(other)_vocals_mel_band_roformer.mp3"]


def test_the_real_library_writer_warns_as_the_watch_reads_it(tmp_path: Path) -> None:
    """audio-separator's own pydub writer, given a silent stem: it warns and writes
    nothing, and the watch pins that to the stem being saved."""
    import logging
    from types import SimpleNamespace

    common = pytest.importorskip("audio_separator.separator.common_separator")
    logger = logging.getLogger("stand-in-library")
    logger.setLevel(logging.INFO)
    watch = stemaudio.NearSilentWatch()
    logger.addHandler(watch)
    instance = SimpleNamespace(
        logger=logger, normalization_threshold=0.9, amplification_threshold=0.0,
        output_dir=str(tmp_path), sample_rate=SEPARATOR_RATE, output_bitrate=None,
    )
    name = "song_(vocals)_vocals_mel_band_roformer.wav"
    try:
        logger.info(f"Saving vocals stem to {name}...")
        common.CommonSeparator.write_audio_pydub(
            instance, name, np.zeros((SEPARATOR_RATE, 2), dtype="float32")
        )
    finally:
        logger.removeHandler(watch)
    assert not (tmp_path / name).exists()
    assert watch.silent == [{"stem": "vocals", "file": name}]
