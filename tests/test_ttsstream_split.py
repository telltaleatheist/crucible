from __future__ import annotations

import asyncio
import base64

import pytest

from crucible import ttsstream
from crucible.errors import ApiError
from crucible.ttsstream import decode, log, validate

PUBLIC_NAMES = {
    "BATCH_COALESCE_SECONDS", "GRACE_SECONDS", "STREAM_BATCH_WIDTH", "StreamManager",
    "StreamSession",
}


def test_the_package_answers_its_public_surface_and_no_more() -> None:
    assert set(ttsstream.__all__) == PUBLIC_NAMES
    assert all(hasattr(ttsstream, name) for name in PUBLIC_NAMES)


def test_the_batch_width_is_read_through_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ttsstream, "STREAM_BATCH_WIDTH", {"only-this": 4})
    assert validate.batch_width_for("only-this") == 4
    with pytest.raises(ApiError) as refused:
        validate.batch_width_for("higgs-v3")
    assert refused.value.code == "unknown_narrator_engine"


def _log_with(frames: int, monkeypatch: pytest.MonkeyPatch, grace: float) -> log.EventLog:
    monkeypatch.setattr(ttsstream, "GRACE_SECONDS", grace)
    loop = asyncio.new_event_loop()
    try:
        events = log.EventLog("s1", loop)
        for index in range(frames):
            events._append("audio", {"seq": index})
    finally:
        loop.close()
    return events


def test_a_log_nobody_reads_forgets_frames_older_than_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _log_with(3, monkeypatch, grace=-1.0)
    assert events.frames_after(0) == []
    with pytest.raises(ApiError) as refused:
        events.check_replayable(0)
    assert refused.value.code == "replay_unavailable"


def test_a_log_keeps_what_an_attached_reader_has_not_had(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ttsstream, "GRACE_SECONDS", -1.0)
    loop = asyncio.new_event_loop()
    try:
        events = log.EventLog("s1", loop)
        reader = events.attach(0)
        events._append("ready", {})
        events._append("audio", {})
    finally:
        loop.close()
    assert [frame.id for frame in events.frames_after(reader.delivered)] == [1, 2]
    assert events.ever_attached
    events.detach(reader)
    assert events.grace_expired(float("inf"))


def _chunk(pcm: bytes, **overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": decode.CHUNK, "i": 0, "format": "pcm16", "sampleRate": 1000,
        "data": base64.b64encode(pcm).decode(), "duration": len(pcm) / 2 / 1000,
    }
    message.update(overrides)
    return message


def test_a_chunk_is_measured_not_believed() -> None:
    pcm = b"\x00\x01" * 100
    assert decode.pcm_of(_chunk(pcm), "r1", sample_rate=1000, voice="v") == (pcm, 0.1)
    with pytest.raises(decode.RowFailure, match="0.500s for a chunk of row r1"):
        decode.pcm_of(_chunk(pcm, duration=0.5), "r1", sample_rate=1000, voice="v")
    with pytest.raises(decode.RowFailure, match="refuses rather than resamples"):
        decode.pcm_of(_chunk(pcm, sampleRate=24000), "r1", sample_rate=1000, voice="v")


def test_an_item_end_says_which_of_its_four_outcomes_it_is() -> None:
    assert decode.item_end({"message": "boom"}, "r1", 1.0).failure == "boom"
    assert decode.item_end({"cancelled": True}, "r1", 1.0).failure == "cancelled"
    assert "0.200s" in (decode.item_end({"duration": 0.2}, "r1", 1.0).protocol_error or "")
    missing_gap = decode.item_end({"capped": True}, "r1", 1.0)
    assert missing_gap.capped is True and "gapSec None" in (missing_gap.protocol_error or "")
    assert decode.item_end({"gapSec": 0.3, "capped": "x"}, "r1", 1.0) == decode.ItemEnd(
        gap_sec=0.3
    )
