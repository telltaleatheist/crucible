"""`crucible/jobs/asr/qwen_worker.py`'s engine-free parts, in process.

The worker's engines (vLLM, mlx-audio) cannot run here, and the end-to-end API
tests use a fake in their place (`tests/test_asr_qwen.py`). What CAN be tested
on any machine is everything the two engines share, and it is where a wrong
number would move every timestamp: where the pieces are cut, what the wav
files hold, what prompt vLLM is given, and that a transcribe request is decoded
in batches of the manifest's size and answered in order.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy
import pytest

from crucible.jobs.asr import qwen_worker as worker

RATE = worker.SAMPLE_RATE


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """The worker's fd 1, as a buffer a test can read."""
    buffer = io.StringIO()
    monkeypatch.setattr(worker, "_RESULTS", buffer)
    return buffer


def messages(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def _speech(seconds: float, quiet: list[tuple[float, float]]) -> numpy.ndarray:
    """A loud signal with silent stretches at the stated places."""
    samples = numpy.full(int(seconds * RATE), 0.5, dtype=numpy.float32)
    samples[::2] = -0.5
    for start, end in quiet:
        samples[int(start * RATE) : int(end * RATE)] = 0.0
    return samples


def test_a_short_input_is_one_piece() -> None:
    assert worker.split_points(_speech(30.0, []), 180) == [(0, 30 * RATE)]


def test_the_cut_lands_in_the_quiet_before_the_limit_and_never_past_it() -> None:
    """The boundary is found in the 10 s BEFORE the nominal cut, so no piece is
    longer than the limit the aligner, the budget and the context were sized
    for."""
    wav = _speech(400.0, [(174.0, 174.5), (355.0, 355.4)])
    spans = worker.split_points(wav, 180)
    assert spans[0][0] == 0 and spans[-1][1] == wav.shape[0]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start  # no gap, no overlap
    first_cut = spans[0][1] / RATE
    assert 174.0 <= first_cut <= 174.5
    assert all((end - start) / RATE <= 180 for start, end in spans)


def test_loud_audio_with_no_quiet_still_cuts_within_the_limit() -> None:
    spans = worker.split_points(_speech(500.0, []), 60)
    assert all((end - start) / RATE <= 60 for start, end in spans)
    assert sum(end - start for start, end in spans) == 500 * RATE


def test_a_piece_wav_round_trips_within_one_sixteen_bit_step(tmp_path: Path) -> None:
    samples = numpy.linspace(-1.0, 1.0, RATE, dtype=numpy.float32)
    path = tmp_path / "p.wav"
    worker.write_wav(str(path), samples)
    back = worker.read_wav(str(path))
    assert back.shape == samples.shape
    assert float(numpy.max(numpy.abs(back - samples))) <= 2.0 / 32768


def test_a_wav_the_worker_did_not_write_is_refused(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(b"\x00\x00" * 4)
    with pytest.raises(ValueError) as caught:
        worker.read_wav(str(path))
    assert "(1, 2, 16000)" in str(caught.value)


def test_the_vllm_prompt_is_the_repos_own_chat_template_with_the_language_forced() -> None:
    """`chat_template.json` at the pinned revision: a system turn always, the
    context verbatim before `<|im_end|>`, then `language English<asr_text>` as
    qwen_asr 0.0.6's `_build_text_prompt` forces it."""
    assert worker.official_prompt("Verbatim. um, uh.", "English") == (
        "<|im_start|>system\nVerbatim. um, uh.<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage English<asr_text>"
    )
    assert worker.official_prompt(None, "German").startswith(
        "<|im_start|>system\n<|im_end|>\n"
    )


def test_split_writes_every_piece_and_reports_offsets_in_order(
    tmp_path: Path, wire: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = _speech(130.0, [(55.0, 55.3)])
    monkeypatch.setattr(worker, "decode", lambda ffmpeg, source, progress: wav)
    worker.split(
        {
            "ffmpeg": "/usr/bin/ffmpeg",
            "source": str(tmp_path / "stream.m4a"),
            "max_piece_s": 60,
            "out_dir": str(tmp_path / "pieces"),
        }
    )
    sent = messages(wire)
    assert sent[0] == {"type": "ready", "duration_s": 130.0, "pieces": 3}
    results = [m for m in sent if m["type"] == "result"]
    assert [round(r["offset_s"], 1) for r in results][0] == 0.0
    assert 55.0 <= results[1]["offset_s"] <= 55.3
    assert sum(r["duration_s"] for r in results) == pytest.approx(130.0)
    for result in results:
        assert Path(result["wav"]).is_file()
        assert Path(result["wav"]).name.startswith("stream.")
    assert sent[-1] == {"type": "done"}


def test_transcribe_decodes_in_batches_of_the_manifests_size_in_order(
    wire: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_engine(batch: list) -> list:
        seen.append([piece["wav"] for piece in batch])
        return [
            {"text": f"text of {p['wav']}", "tokens": 3, "hit_token_limit": False}
            for p in batch
        ]

    monkeypatch.setattr(worker, "transcribe_vllm", fake_engine)
    monkeypatch.setitem(worker._STATE, "engine", "vllm")
    monkeypatch.setitem(worker._STATE, "model", object())
    monkeypatch.setitem(worker._STATE, "max_batch", 2)
    worker.transcribe(
        {"pieces": [{"wav": f"p{i}.wav", "max_tokens": 100} for i in range(5)]}
    )
    assert seen == [["p0.wav", "p1.wav"], ["p2.wav", "p3.wav"], ["p4.wav"]]
    sent = messages(wire)
    texts = [m["text"] for m in sent if m["type"] == "result"]
    assert texts == [f"text of p{i}.wav" for i in range(5)]
    progress = [m["processed"] for m in sent if m["type"] == "progress"]
    assert progress == [2, 4, 5]


def test_a_transcribe_before_a_load_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(worker._STATE, "model", None)
    with pytest.raises(RuntimeError) as caught:
        worker.transcribe({"pieces": [{"wav": "x", "max_tokens": 1}]})
    assert "before a load" in str(caught.value)


def test_a_piece_without_its_budget_is_refused_by_name(
    wire: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(worker._STATE, "model", object())
    with pytest.raises(KeyError) as caught:
        worker.transcribe({"pieces": [{"wav": "x"}]})
    assert "'max_tokens'" in str(caught.value.args[0])
