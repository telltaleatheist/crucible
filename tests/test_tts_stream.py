from __future__ import annotations

import base64
import json
import shutil
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest

from crucible import ttsstream
from crucible.settle import SETTLEMENT_HOLDER
from crucible.ttsstream.validate import batch_width_for

from . import fake_narrator_engine
from .conftest import a_clearance_to_hold
from .live_server import run_job, serve
from .test_residency import STUBBORN_PID, a_process_that_will_not_stop
from .test_tts_api import (
    fake_env,
    fake_weights,
    idle_card,
    tts_recipes,
)

VOICE = "deathstalker"
OTHER_VOICE = "thirdreich"

CHARS_PER_SEC = 15.0

SAMPLE_RATE = 24_000

LONG_TEXT = (
    "He had been walking for some time, and the road did not appear to end, "
    "not that day and not the next, and the rain did not stop either."
)

WAIT = 20.0

DROP_UNWIND_SECONDS = 10.0


@pytest.fixture(autouse=True)
def quick_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def streaming_server(
    make_app: Callable[..., Any],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:

    @contextmanager
    def start(**fake_options: Any) -> Iterator[str]:
        fake_narrator_engine.install(monkeypatch)
        if fake_options:
            fake_narrator_engine.steer(monkeypatch, **fake_options)
        app = make_app(enable_tts=True, enable_echo=False)
        fake_weights(VOICE)
        fake_weights(OTHER_VOICE)
        with serve(app) as base:
            run_job(base, auth, type="load-voice", model=VOICE)
            yield base

    return start


def open_session(base: str, auth: dict[str, str], **body: Any) -> httpx.Response:
    payload = {"voice": VOICE, "language": "en"}
    payload.update(body)
    return httpx.post(f"{base}/v1/tts/stream", headers=auth, json=payload, timeout=30.0)


def opened(base: str, auth: dict[str, str], **body: Any) -> dict[str, Any]:
    response = open_session(base, auth, **body)
    assert response.status_code == 201, response.text
    return response.json()


def post_op(
    base: str, auth: dict[str, str], session_id: str, **op: Any
) -> httpx.Response:
    return httpx.post(
        f"{base}/v1/tts/stream/{session_id}", headers=auth, json=op, timeout=30.0
    )


def say(base: str, auth: dict[str, str], session_id: str, row: str, text: str) -> None:
    say_at_take(base, auth, session_id, row, text, 0)


def say_at_take(
    base: str, auth: dict[str, str], session_id: str, row: str, text: str, take: int
) -> None:
    response = post_op(base, auth, session_id, op="say", id=row, text=text, take=take)
    assert response.status_code == 202, response.text
    assert response.json() == {"id": row}


class Listener:

    def __init__(self, response: httpx.Response) -> None:
        self.frames: list[dict[str, Any]] = []
        self.ended = threading.Event()
        self.keepalives = 0
        self._lock = threading.Lock()
        self._response = response
        self._socket = response.extensions["network_stream"].get_extra_info("socket")
        self._dropped = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def drop(self) -> None:
        if self._dropped:
            return
        self._dropped = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if not self.ended.wait(DROP_UNWIND_SECONDS):
            raise AssertionError(
                "the event stream's reader did not unwind "
                f"{DROP_UNWIND_SECONDS:.0f}s after the socket was shut down, so "
                "this drop did not happen and nothing after it means anything"
            )

    def _pump(self) -> None:
        current: dict[str, Any] = {}
        try:
            for line in self._response.iter_lines():
                if line == "":
                    if current:
                        with self._lock:
                            self.frames.append(current)
                        current = {}
                    continue
                if line.startswith(":"):
                    with self._lock:
                        self.keepalives += 1
                    continue
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "id":
                    current["id"] = int(value)
                elif field == "event":
                    current["event"] = value
                elif field == "data":
                    current["data"] = json.loads(value)
        except (httpx.HTTPError, OSError, RuntimeError, ValueError):
            pass
        finally:
            self.ended.set()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.frames)

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [frame["data"] for frame in self.snapshot() if frame["event"] == kind]

    def audio_for(self, row: str) -> list[dict[str, Any]]:
        return [data for data in self.of("audio") if data["id"] == row]

    def last_id(self) -> int:
        frames = self.snapshot()
        return frames[-1]["id"] if frames else 0

    def wait_for(
        self, predicate: Callable[["Listener"], bool], what: str, timeout: float = WAIT
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self):
                return
            time.sleep(0.01)
        raise AssertionError(
            f"waited {timeout:.0f}s for {what}; the stream held "
            f"{[frame['event'] for frame in self.snapshot()]}"
        )


@contextmanager
def listen(
    base: str, auth: dict[str, str], session_id: str, after: int | None = None
) -> Iterator[Listener]:
    headers = dict(auth)
    headers["Accept"] = "text/event-stream"
    if after is not None:
        headers["Last-Event-ID"] = str(after)
    with httpx.stream(
        "GET",
        f"{base}/v1/tts/stream/{session_id}/events",
        headers=headers,
        timeout=60.0,
    ) as response:
        assert response.status_code == 200, response.read()
        listener = Listener(response)
        try:
            yield listener
        finally:
            listener.drop()


def pcm_of(listener: Listener, row: str) -> bytes:
    chunks = sorted(listener.audio_for(row), key=lambda data: data["seq"])
    assert [data["seq"] for data in chunks] == list(range(len(chunks))), chunks
    return b"".join(base64.b64decode(data["pcm_base64"]) for data in chunks)


def seconds_of(pcm: bytes) -> float:
    return len(pcm) / 2 / SAMPLE_RATE


def test_opening_a_session_answers_the_identity_of_what_will_speak(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        body = opened(base, auth)
        assert body["voice"] == VOICE
        assert body["sample_rate"] == SAMPLE_RATE
        assert body["backend"] == "cuda-linux"
        assert body["fingerprint"].startswith(f"{VOICE}@")
        assert len(body["session_id"]) == 32


def test_a_voice_that_is_not_resident_is_loaded_inside_the_stream_s_queue_session(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        body = opened(base, auth, voice=OTHER_VOICE)
        assert body["voice"] == OTHER_VOICE
        activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
        assert activity["resident"]["id"] == OTHER_VOICE
        held = activity["session"]
        assert held["session_id"] == body["queue_session_id"]
        assert held["items_run"] == 2, "the load-voice job and the stream are its items"
        assert held["stream_session"]["session_id"] == body["session_id"]


def test_a_second_session_is_refused_by_name(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        first = opened(base, auth)
        response = open_session(base, auth)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "stream_session_open"
        assert first["session_id"] in error["message"]


def test_a_session_that_cannot_be_built_does_not_keep_the_card(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with streaming_server() as base:
        measured = dict(ttsstream.STREAM_BATCH_WIDTH)
        monkeypatch.setattr(ttsstream, "STREAM_BATCH_WIDTH", {})
        refused = open_session(base, auth)
        assert refused.status_code == 500, refused.text
        error = refused.json()["error"]
        assert error["code"] == "unknown_narrator_engine"
        assert "higgs-v3" in error["message"]

        activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0)
        assert activity.json()["claim"] is None, activity.text

        monkeypatch.setattr(ttsstream, "STREAM_BATCH_WIDTH", measured)
        assert opened(base, auth)["voice"] == VOICE


def test_an_unknown_session_is_a_named_404(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        response = post_op(base, auth, "nosuchsession", op="cancel_all")
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "unknown_session"


def test_the_door_is_shut_when_tts_is_off(
    make_app: Callable[..., Any], auth: dict[str, str]
) -> None:
    with serve(make_app()) as base:
        response = open_session(base, auth)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "job_type_disabled"


def test_a_session_streams_sub_sentence_audio_and_retires_the_row(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    text = "He had been walking for some time."
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            ready = stream.of("ready")[0]
            assert ready["voice"] == VOICE
            assert ready["fingerprint"] == session["fingerprint"]

            say(base, auth, session["session_id"], "r1", text)
            stream.wait_for(lambda s: s.of("done"), "r1 to retire")

            chunks = stream.audio_for("r1")
            assert len(chunks) > 1, chunks

            pcm = pcm_of(stream, "r1")
            expected = len(text) / CHARS_PER_SEC
            assert abs(seconds_of(pcm) - expected) < 0.01

            done = stream.of("done")[0]
            assert done["id"] == "r1"
            assert done["cancelled"] is False
            assert done["chars"] == len(text)
            assert abs(done["seconds"] - expected) < 0.01
            assert abs(done["chars_per_sec"] - CHARS_PER_SEC) < 0.1


def test_say_answers_with_the_row_id_and_not_the_audio(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            response = post_op(
                base, auth, session["session_id"], op="say", id="r1", text="Rain.",
                take=0,
            )
            assert response.status_code == 202
            assert response.json() == {"id": "r1"}
            assert "pcm_base64" not in response.text


def test_a_client_that_never_opened_the_stream_is_refused_by_name(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        response = post_op(
            base, auth, session["session_id"], op="say", id="r1", text="Rain.", take=0
        )
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "stream_not_attached"
        assert "/events" in error["message"]


def test_two_rows_cannot_share_an_id(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=40) as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, session["session_id"], "r1", LONG_TEXT)
            response = post_op(
                base, auth, session["session_id"], op="say", id="r1", text="Rain.",
                take=0,
            )
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "duplicate_row_id"


def test_a_row_longer_than_the_cap_is_said_rather_than_refused(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            response = post_op(
                base, auth, session["session_id"], op="say", id="r1",
                text="x" * 801, take=0,
            )
            assert response.status_code == 202, response.text
            assert response.json()["id"] == "r1"
            stream.wait_for(
                lambda s: [d for d in s.of("done") if d["id"] == "r1"],
                "r1's done",
            )


def test_a_take_above_the_ladder_renders_in_its_own_lane_and_is_never_clamped(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            response = post_op(
                base, auth, session["session_id"], op="say", id="r1", text="Rain.",
                take=3,
            )
            assert response.status_code == 202, response.text
            stream.wait_for(
                lambda s: [d for d in s.of("done") if d["id"] == "r1"],
                "r1's done",
            )


def test_a_row_at_take_one_carries_that_rungs_numbers_and_take_zero_carries_none(
    streaming_server: Callable[..., Any], auth: dict[str, str], tmp_path: Path
) -> None:
    log = tmp_path / "sampling.jsonl"
    with streaming_server(sampling_log=str(log)) as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            for row, take in (("r0", 0), ("r1", 1)):
                response = post_op(
                    base, auth, session["session_id"], op="say", id=row,
                    text="Rain fell on the road.", take=take,
                )
                assert response.status_code == 202, response.text
            stream.wait_for(
                lambda s: len({data["id"] for data in s.of("done")}) == 2,
                "both rows to retire",
            )
    rows = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines() if line
    ]
    assert {row["i"]: row["sampling"] for row in rows} == {
        0: None, 1: {"temperature": 0.7},
    }
    assert {row["i"]: row["take"] for row in rows} == {0: 0, 1: 1}


def test_a_rung_narrator_cannot_honour_fails_that_row_by_name(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(sampling_levers="topP") as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say_at_take(base, auth, session["session_id"], "r1", "Rain.", 1)
            stream.wait_for(lambda s: s.of("error"), "the row's error frame")
            error = stream.of("error")[0]
    assert error["id"] == "r1"
    assert "sampling_not_supported:" in error["message"]


def test_a_narrator_without_the_channel_refuses_a_rung_and_still_says_take_zero(
    streaming_server: Callable[..., Any], auth: dict[str, str], tmp_path: Path
) -> None:
    log = tmp_path / "sampling.jsonl"
    with streaming_server(sampling_log=str(log), no_item_take=1) as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")

            refused = post_op(
                base, auth, session["session_id"], op="say", id="r1",
                text="Rain fell on the road.", take=1,
            )
            assert refused.status_code == 409, refused.text
            error = refused.json()["error"]
            assert error["code"] == "sampling_not_wired"
            assert "did not announce `itemTake`" in error["message"]

            accepted = post_op(
                base, auth, session["session_id"], op="say", id="r0",
                text="Rain fell on the road.", take=0,
            )
            assert accepted.status_code == 202, accepted.text
            stream.wait_for(lambda s: s.of("done"), "the take-0 row to retire")

    rows = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines() if line
    ]
    assert [row["sampling"] for row in rows] == [None]
    assert [row["take"] for row in rows] == [0]


def test_say_has_no_default_take_on_the_wire(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        response = post_op(
            base, auth, session["session_id"], op="say", id="r1", text="Rain."
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"


def test_a_blank_row_is_refused_before_it_reaches_narrator(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        response = post_op(
            base, auth, session["session_id"], op="say", id="r1", text="   ", take=0
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"


def test_a_stream_s_queue_session_holds_the_card_against_every_job_that_wants_it(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    jobs = (
        {"type": "tts", "model": VOICE,
         "params": {"language": "en", "take": 0,
                    "chunks": [{"index": 0, "text": "Rain."}]}},
        {"type": "load-voice", "model": OTHER_VOICE, "params": {}},
        {"type": "unload-voice", "model": VOICE, "params": {}},
    )
    other = {**auth, "X-Crucible-Client": "briefcase"}
    with streaming_server() as base:
        stream = opened(base, auth)
        for body in jobs:
            response = httpx.post(f"{base}/v1/jobs", headers=other, json=body, timeout=30.0)
            assert response.status_code == 409, (body["type"], response.text)
            error = response.json()["error"]
            assert error["code"] == "server_busy", body["type"]
            assert error["details"]["session_id"] == stream["queue_session_id"]

        own = httpx.post(f"{base}/v1/jobs", headers=auth, json=jobs[2], timeout=30.0)
        assert own.status_code == 202, own.text
        assert own.json()["queued"] is True, (
            "the stream's own client's job is an item of its session, and waits while "
            "the stream holds narrator's one conversation"
        )
        httpx.delete(f"{base}/v1/tts/stream/{stream['session_id']}", headers=auth,
                     timeout=30.0)
        state = httpx.get(f"{base}/v1/jobs/{own.json()['job_id']}", headers=auth,
                          timeout=30.0).json()
        assert state["status"] == "removed"
        assert state["removal"]["reason"] == "session_closed"


def test_a_session_opened_during_a_clearance_waits_it_out(
    make_app: Callable[..., Any],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_narrator_engine.install(monkeypatch)
    app = make_app(enable_tts=True, enable_echo=False)
    fake_weights(VOICE)
    residency = app.state.residency
    with serve(app) as base:
        run_job(base, auth, type="load-voice", model=VOICE)
        reached, release = a_clearance_to_hold(residency.voice_engine)
        waiting = threading.Event()
        wait = residency.await_settled

        def instrumented(what: str, *, timeout: float) -> None:
            waiting.set()
            wait(what, timeout=timeout)

        monkeypatch.setattr(residency, "await_settled", instrumented)
        clearing = threading.Thread(
            target=app.state.settlement.settle_quietly,
            args=("the render job finished",),
            daemon=True,
        )
        clearing.start()
        try:
            assert reached.wait(timeout=WAIT), "the settlement never reached narrator"
            assert residency.claimed_by == SETTLEMENT_HOLDER
            answers: list[httpx.Response] = []
            opener = threading.Thread(
                target=lambda: answers.append(open_session(base, auth)), daemon=True
            )
            opener.start()
            assert waiting.wait(timeout=WAIT), "the door never waited"
            assert not answers, "the door answered before the clearance finished"
        finally:
            release.set()
            clearing.join(timeout=WAIT)
        opener.join(timeout=WAIT)
        assert answers, "the door is still waiting after the clearance"
        response = answers[0]
        assert response.status_code == 201, (
            "the clearance took the voice off; the stream's queue session loaded it "
            f"again rather than open on nothing: {response.text}"
        )
        assert response.json()["voice"] == VOICE
        assert residency.claimed_by is not None, "the stream holds narrator's conversation"


def test_the_card_is_free_again_once_the_session_closes(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
        closed = httpx.delete(
            f"{base}/v1/tts/stream/{session['session_id']}", headers=auth, timeout=30.0
        )
        assert closed.status_code == 200, closed.text
        assert closed.json()["closed"] is True

        render = {
            "type": "tts",
            "model": VOICE,
            "params": {"language": "en", "take": 0,
                       "chunks": [{"index": 0, "text": "Rain."}]},
        }
        if shutil.which("ffmpeg") is not None:
            run_job(base, auth, **render)
        else:
            refused = httpx.post(
                f"{base}/v1/jobs", headers=auth, json=render, timeout=30.0
            )
            assert refused.status_code == 409, refused.text
            assert refused.json()["error"]["code"] == "ffmpeg_missing", refused.text

        run_job(base, auth, type="load-voice", model=OTHER_VOICE)


def test_a_row_cancelled_before_it_starts_costs_nothing(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=40) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            say(base, auth, sid, "r2", LONG_TEXT)
            stream.wait_for(lambda s: s.audio_for("r1"), "r1 to start")
            response = post_op(base, auth, sid, op="cancel", id="r2")
            assert response.status_code == 202, response.text
            assert response.json()["outcome"] == "dropped"

            stream.wait_for(
                lambda s: len(s.of("done")) == 2, "both rows to retire"
            )
            by_id = {data["id"]: data for data in stream.of("done")}
            assert by_id["r2"]["cancelled"] is True
            assert by_id["r2"]["seconds"] == 0.0
            assert by_id["r2"]["chars_per_sec"] is None
            assert stream.audio_for("r2") == []
            assert by_id["r1"]["cancelled"] is False
            assert not stream.of("restart")


def test_cancelling_the_row_in_flight_stops_it_where_it_is(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=60) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            stream.wait_for(
                lambda s: len(s.audio_for("r1")) >= 2, "r1 to be under way"
            )
            response = post_op(base, auth, sid, op="cancel", id="r1")
            assert response.status_code == 202, response.text
            assert response.json()["outcome"] == "aborting_batch"

            stream.wait_for(lambda s: s.of("done"), "r1 to retire")
            done = stream.of("done")[0]
            assert done["id"] == "r1"
            assert done["cancelled"] is True
            assert done["seconds"] < len(LONG_TEXT) / CHARS_PER_SEC
            assert abs(done["seconds"] - seconds_of(pcm_of(stream, "r1"))) < 0.01
            assert not stream.of("restart")


def test_a_cancel_costs_its_batch_and_the_survivors_are_restarted(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(ttsstream.STREAM_BATCH_WIDTH, "higgs-v3", 3)
    monkeypatch.setattr(ttsstream, "BATCH_COALESCE_SECONDS", 2.0)
    with streaming_server(chunk_delay_ms=40) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            for row in ("r1", "r2", "r3"):
                say(base, auth, sid, row, LONG_TEXT)
            stream.wait_for(
                lambda s: all(s.audio_for(row) for row in ("r1", "r2", "r3")),
                "all three rows to be generating at once",
            )
            assert post_op(base, auth, sid, op="cancel", id="r2").status_code == 202

            stream.wait_for(
                lambda s: len(s.of("done")) == 3, "every row to retire"
            )
            by_id = {data["id"]: data for data in stream.of("done")}
            assert by_id["r2"]["cancelled"] is True

            restarted = {data["id"]: data for data in stream.of("restart")}
            assert sorted(restarted) == ["r1", "r3"]
            full = len(LONG_TEXT) / CHARS_PER_SEC
            for row in ("r1", "r3"):
                assert restarted[row]["from_seq"] > 0
                assert by_id[row]["cancelled"] is False
                assert abs(by_id[row]["seconds"] - full) < 0.01
                after = [
                    data for data in stream.audio_for(row)
                    if data["seq"] >= restarted[row]["from_seq"]
                ]
                assert abs(sum(d["seconds"] for d in after) - full) < 0.01


def test_cancel_all_stops_everything_and_restarts_nothing(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=40) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            for row in ("r1", "r2"):
                say(base, auth, sid, row, LONG_TEXT)
            stream.wait_for(lambda s: s.audio_for("r1"), "r1 to start")
            response = post_op(base, auth, sid, op="cancel_all")
            assert response.status_code == 202, response.text
            assert response.json()["cancelled"] == 2

            stream.wait_for(lambda s: len(s.of("done")) == 2, "both rows to retire")
            assert all(data["cancelled"] is True for data in stream.of("done"))
            assert not stream.of("restart")


def test_cancelling_a_row_that_has_already_retired_is_not_an_error(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", "Rain.")
            stream.wait_for(lambda s: s.of("done"), "r1 to retire")
            response = post_op(base, auth, sid, op="cancel", id="r1")
            assert response.status_code == 202, response.text
            assert response.json()["outcome"] == "already_finished"


def test_cancelling_a_row_nobody_said_is_refused(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        response = post_op(base, auth, session["session_id"], op="cancel", id="ghost")
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "unknown_row"


def test_a_dropped_stream_reattaches_and_is_replayed_what_it_missed(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=60) as base:
        session = opened(base, auth)
        sid = session["session_id"]

        with listen(base, auth, sid) as first:
            first.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            first.wait_for(
                lambda s: len(s.audio_for("r1")) >= 2, "r1 to be under way"
            )
            before = first.snapshot()
            delivered = first.last_id()
            first.drop()

        with listen(base, auth, sid, after=delivered) as second:
            second.wait_for(lambda s: s.of("done"), "r1 to retire on the new stream")
            after = second.snapshot()

            assert after[0]["id"] == delivered + 1
            assert [frame["id"] for frame in after] == list(
                range(delivered + 1, delivered + 1 + len(after))
            )

            seqs = [data["seq"] for data in first.audio_for("r1")] + [
                data["seq"] for data in second.audio_for("r1")
            ]
            assert seqs == sorted(seqs)
            assert len(seqs) == len(set(seqs)), seqs

            whole = b"".join(
                base64.b64decode(data["pcm_base64"])
                for data in sorted(
                    first.audio_for("r1") + second.audio_for("r1"),
                    key=lambda data: data["seq"],
                )
            )
            assert abs(
                seconds_of(whole) - len(LONG_TEXT) / CHARS_PER_SEC
            ) < 0.01
            assert second.of("done")[0]["cancelled"] is False
        assert [frame["event"] for frame in before].count("closed") == 0


def test_the_session_closes_when_nobody_comes_back(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ttsstream, "GRACE_SECONDS", 0.4)
    with streaming_server(chunk_delay_ms=60) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            stream.wait_for(lambda s: s.audio_for("r1"), "r1 to start")
            stream.drop()

        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            response = post_op(base, auth, sid, op="cancel_all")
            if response.status_code == 404:
                break
            time.sleep(0.05)
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "unknown_session"

        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            health = httpx.get(f"{base}/v1/health", headers=auth, timeout=30.0)
            if health.json()["resident_kind"] is None:
                break
            time.sleep(0.05)
        assert health.json()["resident_kind"] is None, health.text
        held = httpx.get(
            f"{base}/v1/queue/sessions/{session['queue_session_id']}", headers=auth,
            timeout=30.0,
        ).json()
        assert held["status"] == "closed" and held["reason"] == "client", held
        assert sid in held["message"]


def test_a_resume_the_session_can_no_longer_serve_is_refused_not_skipped(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ttsstream, "GRACE_SECONDS", 0.3)
    with streaming_server() as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            stream.wait_for(lambda s: s.of("done"), "r1 to retire")
            time.sleep(0.5)
            say(base, auth, sid, "r2", "Rain.")
            stream.wait_for(lambda s: len(s.of("done")) == 2, "r2 to retire")

            headers = dict(auth)
            headers["Last-Event-ID"] = "1"
            response = httpx.get(
                f"{base}/v1/tts/stream/{sid}/events", headers=headers, timeout=30.0
            )
            assert response.status_code == 409, response.text
            error = response.json()["error"]
            assert error["code"] == "replay_unavailable"
            assert error["details"]["oldest"] > 1


def test_the_stream_keeps_itself_alive_while_nothing_is_happening(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.api.sse.KEEPALIVE_SECONDS", 0.1)
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.keepalives >= 2, "two keepalive comments")


def test_closing_ends_the_stream_with_a_closed_frame(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            response = post_op(base, auth, sid, op="close")
            assert response.status_code == 202, response.text
            assert response.json()["closed"] is True
            stream.wait_for(lambda s: s.of("closed"), "the closed frame")
            assert stream.of("closed")[0]["reason"]
            stream.ended.wait(WAIT)
            assert stream.ended.is_set()
        assert post_op(base, auth, sid, op="cancel_all").status_code == 404


def test_closing_cancels_every_row_still_in_flight(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=60) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            say(base, auth, sid, "r2", LONG_TEXT)
            stream.wait_for(lambda s: s.audio_for("r1"), "r1 to start")
            assert httpx.delete(
                f"{base}/v1/tts/stream/{sid}", headers=auth, timeout=30.0
            ).status_code == 200
            stream.wait_for(lambda s: s.of("closed"), "the closed frame")
            done = {data["id"]: data for data in stream.of("done")}
            assert set(done) == {"r1", "r2"}
            assert all(data["cancelled"] is True for data in done.values())


def test_a_narrator_row_that_fails_on_its_own_is_reported_not_restarted(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(fail_row=0) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", "Rain.")
            stream.wait_for(lambda s: s.of("error"), "r1 to be reported failed")
            error = stream.of("error")[0]
            assert error["id"] == "r1"
            assert error["code"] == "row_failed"
            assert not stream.of("restart")
            assert not stream.of("done")


def test_a_retiring_row_carries_the_gap_the_player_must_insert(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(gap_sec=1.25) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", "He had been walking for some time.")
            stream.wait_for(lambda s: s.of("done"), "r1 to retire")
            done = stream.of("done")[0]
            assert done["gap_sec"] == 1.25
            pcm = pcm_of(stream, "r1")
            assert abs(seconds_of(pcm) - done["seconds"]) < 0.01


def test_a_row_narrator_retires_without_a_gap_is_refused_by_name(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(omit_gap=1) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", "Rain.")
            stream.wait_for(lambda s: s.of("error"), "r1 to be refused")
            error = stream.of("error")[0]
            assert error["id"] == "r1"
            assert error["code"] == "narrator_protocol"
            assert "gapSec" in error["message"]
            assert not stream.of("done")


def test_a_cancelled_row_has_no_gap_to_keep(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server(chunk_delay_ms=60) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            stream.wait_for(lambda s: len(s.audio_for("r1")) >= 2, "r1 to be in flight")
            response = post_op(base, auth, sid, op="cancel", id="r1")
            assert response.status_code == 202, response.text
            stream.wait_for(lambda s: s.of("done"), "r1 to retire cancelled")
            done = stream.of("done")[0]
            assert done["cancelled"] is True
            assert done["gap_sec"] is None


def test_the_batch_width_has_no_default_for_an_unmeasured_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert batch_width_for("higgs-v3") == 1
    with pytest.raises(Exception) as caught:
        batch_width_for("some-engine-nobody-measured")
    assert getattr(caught.value, "code", None) == "unknown_narrator_engine"


def test_the_bench_does_not_show_an_idle_machine_while_a_session_runs(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        body = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()

        assert body["slots"]["accelerated"]["busy"] == 0
        assert body["running"] == []
        assert body["queued"] == []

        assert body["slots"]["accelerated"]["accepts_work"] is False
        assert body["claim"] is not None
        assert "tts stream" in body["claim"]["held_by"]

        streaming = body["streaming"]
        assert streaming is not None
        assert streaming["session_id"] == session["session_id"]
        assert streaming["voice"] == VOICE
        assert streaming["since"]

        assert "progress" in streaming
        assert streaming["progress"] is None
        assert streaming["said"] == 0
        assert streaming["finished"] == 0
        assert streaming["in_flight"] == 0

        assert body["session"]["session_id"] == session["queue_session_id"]
        assert streaming["queue_session_id"] == session["queue_session_id"]
        refused = httpx.post(
            f"{base}/v1/jobs",
            headers={**auth, "X-Crucible-Client": "briefcase"},
            json={"type": "tts", "model": VOICE,
                  "params": {"language": "en", "take": 0,
                             "chunks": [{"index": 0, "text": "Rain."}]}},
            timeout=30.0,
        )
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["details"]["session_id"] == session["queue_session_id"]


def test_the_bench_counts_what_a_session_has_actually_said(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", "Rain fell on the roof.")
            stream.wait_for(lambda s: s.of("done"), "the row to retire")

        streaming = httpx.get(
            f"{base}/v1/activity", headers=auth, timeout=30.0
        ).json()["streaming"]
        assert streaming["said"] == 1
        assert streaming["finished"] == 1
        assert streaming["in_flight"] == 0
        assert streaming["chars"] == len("Rain fell on the roof.")
        assert streaming["seconds"] > 0.0
        assert streaming["progress"] is None


def test_the_bench_names_the_client_that_opened_the_session_or_says_it_did_not(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        named = httpx.post(
            f"{base}/v1/tts/stream",
            headers={**auth, "User-Agent": "bookforge/owens-pc crucible-client/0.4.0"},
            json={"voice": VOICE, "language": "en"},
            timeout=30.0,
        )
        assert named.status_code == 201, named.text
        body = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
        assert body["streaming"]["client"] == "bookforge/owens-pc crucible-client/0.4.0"
        httpx.delete(
            f"{base}/v1/tts/stream/{body['streaming']['session_id']}",
            headers=auth,
            timeout=30.0,
        )


def test_the_door_names_the_dying_process_before_it_names_the_voice(
    make_client: Callable[..., Any], auth: dict[str, str]
) -> None:
    with make_client(enable_tts=True, enable_echo=False) as client:
        with a_process_that_will_not_stop(client.app.state.residency):
            response = client.post(
                "/v1/tts/stream",
                json={"voice": VOICE, "language": "en"},
                headers=auth,
            )
            assert response.status_code == 409, response.text
            error = response.json()["error"]
            assert error["code"] == "engine_still_stopping"
            assert str(STUBBORN_PID) in error["message"]
            assert (
                client.get("/v1/activity", headers=auth).json()["streaming"] is None
            )
