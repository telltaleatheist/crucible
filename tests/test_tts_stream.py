"""The streaming door — `POST /v1/tts/stream` and its three companions.

PHASE3-TTS.md section 7. Every test here runs the **real app under a real
uvicorn on a real socket** (tests/live_server.py) rather than through
`TestClient`, and that is not a preference:

- The centrepiece of this door is what happens when a listener's connection
  drops mid-row and comes back with `Last-Event-ID` inside the grace window.
  `TestClient`'s `receive` answers `http.disconnect` only once the app has
  finished responding, so a caller who walks away from a stream is a state it
  cannot reach at all.
- The ops are separate requests that have to arrive **while** the event stream
  is open and a row is generating. A cancel that could only be posted after the
  stream was fully consumed would be testing a queue, not a cancel.

What is faked is the env (a stamped directory), the weights (a stamped
directory), the card (monkeypatched nvidia-smi probes) and narrator itself
(tests/fake_narrator.py, driven through the real
`crucible/engines/narrator.py` and its real pipes). Nothing about the server's
own logic is: the session, the replay buffer, the grace window, the per-row
cancel and the residency claim are exactly what will run on the PC.
"""

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

from . import fake_narrator_engine
from .live_server import run_job, serve
from .test_tts_api import (  # noqa: F401 — imported to be used as fixtures
    fake_env,
    fake_weights,
    idle_card,
    tts_recipes,
)

VOICE = "deathstalker"
OTHER_VOICE = "thirdreich"

#: The fake worker's default: 15.0 characters of text per second of audio, which
#: is Higgs's configured pace. Every duration assertion below is arithmetic on
#: this number rather than a tolerance, because the fake makes it exact.
CHARS_PER_SEC = 15.0

#: The rate every voice in this build declares and the fake renders at.
SAMPLE_RATE = 24_000

#: Long enough that, at the delay the slow fixtures set, a row is still
#: generating several chunks after the test has read the first one.
LONG_TEXT = (
    "He had been walking for some time, and the road did not appear to end, "
    "not that day and not the next, and the rain did not stop either."
)

#: How long a test waits for a frame it expects. Generous against a fake that
#: answers in microseconds; it is a wedge detector, not a budget.
WAIT = 20.0

#: How long a dropped listener's reader thread may take to unwind. It is end of
#: file on a socket that has just been shut down, so this is a wedge detector
#: too — and one that earns its name: when the drop was `close()` rather than
#: `shutdown()`, every test in this file silently paid this in full on macOS
#: (measured 2026-09-14: 20.3 s each, against well under a second on Linux).
DROP_UNWIND_SECONDS = 10.0


# ------------------------------------------------------------------ fixtures


@pytest.fixture(autouse=True)
def quick_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def streaming_server(
    make_app: Callable[..., Any],
    auth: dict[str, str],
    fake_env: Path,  # noqa: F811
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:
    """A live server with `deathstalker` resident, ready to be streamed from.

    The voice is put on the card by a real `load-voice` job over HTTP, because
    the streaming door never loads one — it refuses `voice_not_resident` and
    names what is resident instead, exactly as chat does (section 7).
    """

    @contextmanager
    def start(**fake_options: Any) -> Iterator[str]:
        fake_narrator_engine.install(monkeypatch)
        if fake_options:
            fake_narrator_engine.steer(monkeypatch, **fake_options)
        fake_weights(VOICE)
        fake_weights(OTHER_VOICE)
        with serve(make_app(enable_tts=True, enable_echo=False)) as base:
            run_job(base, auth, type="load-voice", model=VOICE)
            yield base

    return start


# ------------------------------------------------------------------ the wire


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
    """One attached event stream, read on a thread so a test can post while it runs.

    The frames accumulate in arrival order and `wait_for` blocks until the
    predicate is satisfied, which is what lets a test say "once two chunks of
    r1 have landed, drop the connection" without a sleep standing in for it.
    """

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
        """Half-close the TCP socket under the reader. **Measured, not assumed.**

        `httpx.Response.close()` from this thread while the pump thread is
        blocked inside `iter_lines()` does **not** close the socket, and the
        server goes on holding the connection until its next write fails — 15 s
        later, at the keepalive. A test that called it and then reconnected was
        not testing a reattach at all; it was opening a second reader beside a
        first that had never left. Measured 2026-09-13 by timing the server's
        shutdown, which waits for its open connections.

        So the drop is a real `shutdown(SHUT_RDWR)`, which is what a tunnel
        collapsing does: the peer gets a FIN, the reader gets end of file.

        **And it is `shutdown` ALONE — never `close()`.** The first version of
        this helper closed the socket straight afterwards, which is closing a
        file descriptor another thread is blocked reading, and that is undefined
        behaviour rather than a strong way to hang up. Linux and Windows
        tolerate it; macOS does not, and on 2026-09-14 it was the whole of why
        this file failed on both macOS CI jobs. Measured there rather than
        guessed at: `faulthandler` put the reader in `httpcore`'s `recv`, and a
        standalone probe on the same box gave

            shutdown : reader exited after 0.00s
            close    : reader STILL BLOCKED after 10.01s
            both     : reader exited after 0.00s

        — so `shutdown` is what wakes it, `close` is what never does, and "both"
        only looks safe: `close()` frees the descriptor number, uvicorn and
        httpx are opening sockets constantly in this process, and a reader that
        had not yet been scheduled woke up armed on somebody else's connection.
        Which is worse than a hang. It reproduced only sometimes, and only on
        macOS, for exactly that reason.

        The consequence for the rest of the file is that the socket is closed by
        `httpx.stream`'s own context manager, after the reader has unwound —
        which is the only thread allowed to be reading it.

        Waiting for that unwind is deliberate and is checked rather than hoped
        for: a test reads `last_id()` immediately after a drop, and a reader
        still appending frames would make that cursor a moving target.
        """
        if self._dropped:
            return
        self._dropped = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            # Already gone — the server hung up first. Nothing to half-close,
            # and the reader is on its way out for the same reason.
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
            # The test shut the socket down under this thread. A dropped stream
            # is the subject of half this file, so it is an outcome and not an
            # error. (After a clean `shutdown` the usual arrival is end of file
            # rather than an exception; both end the same way.)
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
    """Attach an event stream, and close the socket for real on the way out."""
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
    """Every chunk of one row's audio, concatenated in `seq` order.

    The fake threads the sine's phase through consecutive chunks on purpose, so
    a discontinuity here is a chunk Crucible dropped or reordered.
    """
    chunks = sorted(listener.audio_for(row), key=lambda data: data["seq"])
    assert [data["seq"] for data in chunks] == list(range(len(chunks))), chunks
    return b"".join(base64.b64decode(data["pcm_base64"]) for data in chunks)


def seconds_of(pcm: bytes) -> float:
    return len(pcm) / 2 / SAMPLE_RATE


# ------------------------------------------------------------------- opening


def test_opening_a_session_answers_the_identity_of_what_will_speak(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with streaming_server() as base:
        body = opened(base, auth)
        assert body["voice"] == VOICE
        assert body["sample_rate"] == SAMPLE_RATE
        assert body["backend"] == "cuda-linux"
        # The fingerprint binds the audio to a merge rather than to a name: two
        # merges of one fine-tune are two narrators.
        assert body["fingerprint"].startswith(f"{VOICE}@")
        assert len(body["session_id"]) == 32


def test_the_streaming_door_never_loads_a_voice(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """A connection behaves like chat, not like a render job (section 6 and 7)."""
    with streaming_server() as base:
        response = open_session(base, auth, voice=OTHER_VOICE)
        assert response.status_code == 409, response.text
        error = response.json()["error"]
        assert error["code"] == "voice_not_resident"
        # Naming what IS resident is the whole point: "not resident" alone
        # leaves a client with nothing to do about it.
        assert VOICE in error["message"]
        assert "never loads" in error["message"]


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


# ----------------------------------------------------------------- speaking


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

            # More than one audio frame for one row is the whole difference
            # between this door and the render door: sub-sentence chunks, emitted
            # while the row is still generating.
            chunks = stream.audio_for("r1")
            assert len(chunks) > 1, chunks

            pcm = pcm_of(stream, "r1")
            expected = len(text) / CHARS_PER_SEC
            assert abs(seconds_of(pcm) - expected) < 0.01

            done = stream.of("done")[0]
            assert done["id"] == "r1"
            assert done["cancelled"] is False
            # The server's own count of the text it sent, never a number read
            # back off a reply.
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
    """Generating into nothing is worse than a refusal (section 7)."""
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


def test_a_row_longer_than_the_cap_is_refused_not_re_split(
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
            assert response.status_code == 400, response.text
            error = response.json()["error"]
            assert error["code"] == "chunk_too_long"
            assert error["details"]["max_chars"] == 800


def test_a_take_above_the_ladder_is_never_clamped(
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
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "unknown_take"


def test_a_row_at_take_one_carries_that_rungs_numbers_and_take_zero_carries_none(
    streaming_server: Callable[..., Any], auth: dict[str, str], tmp_path: Path
) -> None:
    """The ladder, one row at a time. deathstalker's rung 1 is one line,
    `temperature = 0.7`; take 0 sends no `sampling` key at all, because absent
    means "the loaded voice's own sampling", which IS take 0.

    **A batch here may MIX rungs** — rows arrive one `say` at a time and each
    carries its own, which is what Correct Sentences spreading N candidates
    across the ladder looks like on this door — so both rows go into one
    session and the log is keyed by the row's slot."""
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
    # Slot 0 is the first `say` and slot 1 the second: the session allocates
    # one per row and never reuses one.
    assert {row["i"]: row["sampling"] for row in rows} == {
        0: None, 1: {"temperature": 0.7},
    }


def test_a_rung_narrator_cannot_honour_fails_that_row_by_name(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """narrator's per-row refusal, carried across as this row's `error` frame
    by name and not interpreted. `sampling_not_supported` is the engine
    saying it has no such lever — the MLX arm has no repetition penalty — and
    it is a different fact from `sampling_malformed`, which is a typo."""
    with streaming_server(sampling_levers="topP") as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say_at_take(base, auth, session["session_id"], "r1", "Rain.", 1)
            stream.wait_for(lambda s: s.of("error"), "the row's error frame")
            error = stream.of("error")[0]
    assert error["id"] == "r1"
    assert "sampling_not_supported:" in error["message"]


def test_say_has_no_default_take_on_the_wire(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """A default here would be the server choosing a take nobody asked for."""
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
    """narrator answers an empty generate with a WHOLE-REQUEST error."""
    with streaming_server() as base:
        session = opened(base, auth)
        response = post_op(
            base, auth, session["session_id"], op="say", id="r1", text="   ", take=0
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"


# ------------------------------------------------------------- the residency


def test_a_session_holds_the_card_against_every_job_that_wants_it(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """narrator has one stdin, so a render and a session cannot both have it.

    Refused **before the job is queued**, so the client is told by name rather
    than watching a job fail in the lane a minute later.
    """
    with streaming_server() as base:
        opened(base, auth)
        for body in (
            {"type": "tts", "model": VOICE,
             "params": {"language": "en", "take": 0,
                        "chunks": [{"index": 0, "text": "Rain."}]}},
            {"type": "load-voice", "model": OTHER_VOICE, "params": {}},
            {"type": "unload-voice", "model": VOICE, "params": {}},
        ):
            response = httpx.post(
                f"{base}/v1/jobs", headers=auth, json=body, timeout=30.0
            )
            assert response.status_code == 409, (body["type"], response.text)
            error = response.json()["error"]
            assert error["code"] == "engine_in_use", body["type"]
            assert "tts stream" in error["details"]["held_by"]


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

        # The render door is no longer refused for the CLAIM, which is the
        # proof — and it is available on every host, including the ones with no
        # ffmpeg. `TtsJobType.preflight` asks `refuse_if_claimed` BEFORE it
        # probes for ffmpeg, so a refusal that has moved from `engine_in_use` to
        # `ffmpeg_missing` has got past the claim and says so by name. Where
        # ffmpeg is there, the job simply runs, which is better still.
        #
        # This is not a concession to the runner. The first version ran the
        # render unconditionally and failed on both macOS CI jobs for a second
        # reason entirely — macOS runners carry no ffmpeg by design
        # (.github/workflows/ci.yml), which is why `test_tts_render.py` skips
        # there wholesale. Skipping this test there would have thrown away the
        # residency claim's release, which has nothing to do with ffmpeg.
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

        # And the job that was refused `engine_in_use` a moment ago in the test
        # above now runs to completion. No ffmpeg anywhere in it.
        run_job(base, auth, type="load-voice", model=OTHER_VOICE)


# ------------------------------------------------------------ per-row cancel


def test_a_row_cancelled_before_it_starts_costs_nothing(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """The exact half of per-row cancel: it never reaches narrator at all."""
    with streaming_server(chunk_delay_ms=40) as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            say(base, auth, sid, "r2", LONG_TEXT)
            # r2 is still pending: the width for higgs-v3 is 1, so r1 is the
            # whole of the batch in flight.
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
            # Its neighbour was never touched.
            assert by_id["r1"]["cancelled"] is False
            assert not stream.of("restart")


def test_cancelling_the_row_in_flight_stops_it_where_it_is(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """On higgs-v3 the in-flight row IS the batch, so this is exact and free.

    `HIGGS_STREAM_BATCH_WIDTH = 1` (CLIENT-SURFACES.md section 3.3, measured
    worthless above one at 2.0x realtime), which is why nothing is restarted
    here — there are no survivors to restart.
    """
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
            # Stopped short, and honest about it: the row's full duration is
            # never reported for audio that was never sent.
            assert done["seconds"] < len(LONG_TEXT) / CHARS_PER_SEC
            assert abs(done["seconds"] - seconds_of(pcm_of(stream, "r1"))) < 0.01
            assert not stream.of("restart")


def test_a_cancel_costs_its_batch_and_the_survivors_are_restarted(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What per-row cancel costs on a wide engine, and what the client is told.

    **This test patches the batch width**, and that is deliberate rather than
    convenient. Every voice this build ships declares `narrator_engine =
    "higgs-v3"`, the only engine this build names and the one whose measured
    width is 1, so the survivor branch is unreachable through a manifest today
    — and it is exactly the branch that will run the day a wider engine lands.
    narrator has no per-row cancel: its `cancel` aborts everything in flight. So
    a cancel of one in-flight row costs its whole batch, and the rows nobody
    cancelled are resubmitted with a `restart` frame saying the audio already
    sent for them is void.
    """
    monkeypatch.setitem(ttsstream.STREAM_BATCH_WIDTH, "higgs-v3", 3)
    # And the coalescing window with it. In production it is 25 ms, BookForge's
    # own measured flush window; three `say` posts over a loopback socket do not
    # reliably land inside that, and a test that sometimes put r1 in a batch of
    # its own would sometimes pass for the wrong reason. Widening it here is
    # widening the same knob, not disabling one.
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

            # The survivors. Each was told its earlier audio is void, and each
            # then rendered in full — which is the resubmission, not a partial.
            restarted = {data["id"]: data for data in stream.of("restart")}
            assert sorted(restarted) == ["r1", "r3"]
            full = len(LONG_TEXT) / CHARS_PER_SEC
            for row in ("r1", "r3"):
                assert restarted[row]["from_seq"] > 0
                assert by_id[row]["cancelled"] is False
                assert abs(by_id[row]["seconds"] - full) < 0.01
                # `seq` never restarts, so the void is a prefix and the good
                # audio is everything at or after `from_seq`.
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
    """The ordinary race on a live connection deserves an answer, not a refusal."""
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


# ------------------------------------------------------- the grace window


def test_a_dropped_stream_reattaches_and_is_replayed_what_it_missed(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """**The centrepiece.** A tunnel costs a reconnect, not a sentence.

    This is the one behaviour a WebSocket could not have given for free, and it
    is the reason PHASE3-TTS.md section 7 chose SSE: `Last-Event-ID` already
    works on this server's streams. The row keeps generating across the drop,
    the session is not cancelled, and the frames that landed while nobody was
    listening are still there when somebody is.

    The proof is arithmetic rather than a feeling: the two halves of the stream,
    concatenated in `seq` order, are the row's whole audio — no gap, no repeat,
    and no seq seen twice.
    """
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
            # The socket goes down HERE, for real, while r1 is still generating
            # — not at the end of the block, so the drop and the reattach are
            # two things this test does rather than one it hopes for. `drop()`
            # itself refuses to return until the reader has unwound, so the
            # cursor just read is final and every drop in this file is checked
            # rather than only this one.
            first.drop()

        # Inside the grace window, and carrying the id of the last frame seen.
        with listen(base, auth, sid, after=delivered) as second:
            second.wait_for(lambda s: s.of("done"), "r1 to retire on the new stream")
            after = second.snapshot()

            # Nothing is replayed twice and nothing is skipped: the ids pick up
            # exactly where the first stream stopped.
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
    """Work nobody is waiting for is time stolen from the next job."""
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

        # And the card is not merely free, it is EMPTY. A session closing is the
        # last holder letting go, so Owen's ruling clears the voice behind it
        # (2026-09-14, crucible/settle.py) — which is a stronger version of the
        # point of closing it at all. The streaming door never loads, so the
        # honest answer to the next `open` is that nothing is resident.
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            health = httpx.get(f"{base}/v1/health", headers=auth, timeout=30.0)
            if health.json()["resident_kind"] is None:
                break
            time.sleep(0.05)
        assert health.json()["resident_kind"] is None, health.text
        refused = open_session(base, auth)
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "voice_not_resident"


def test_a_resume_the_session_can_no_longer_serve_is_refused_not_skipped(
    streaming_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio with a hole in it and nothing saying so is the failure to avoid."""
    monkeypatch.setattr(ttsstream, "GRACE_SECONDS", 0.3)
    with streaming_server() as base:
        session = opened(base, auth)
        sid = session["session_id"]
        with listen(base, auth, sid) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            say(base, auth, sid, "r1", LONG_TEXT)
            stream.wait_for(lambda s: s.of("done"), "r1 to retire")
            # Everything so far is delivered and older than the window, so the
            # next frame prunes it.
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
    """The 15 s keepalive the job streams already send, on the same clock."""
    monkeypatch.setattr("crucible.api.KEEPALIVE_SECONDS", 0.1)
    with streaming_server() as base:
        session = opened(base, auth)
        with listen(base, auth, session["session_id"]) as stream:
            stream.wait_for(lambda s: s.keepalives >= 2, "two keepalive comments")


# ---------------------------------------------------------------- closing


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
            # The stream ends after it, the way a job's does at its terminal
            # event: a client never has to guess whether more is coming.
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
    """A genuine per-row failure is not collateral, and is never resubmitted."""
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


def test_the_batch_width_has_no_default_for_an_unmeasured_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guess is wrong in both directions: too low halves throughput, too
    high multiplies the cost of a cancel. The table has one row and the
    refusal is what keeps the second engine from arriving without a
    measurement."""
    assert ttsstream.batch_width_for("higgs-v3") == 1
    with pytest.raises(Exception) as caught:
        ttsstream.batch_width_for("some-engine-nobody-measured")
    assert getattr(caught.value, "code", None) == "unknown_narrator_engine"


# --------------------------------------- the bench, while a session is running


def test_the_bench_does_not_show_an_idle_machine_while_a_session_runs(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """The defect this was written for, and the direction it failed in.

    `/v1/activity` drew its whole answer from the JobStore, and a streaming
    session does not occupy the lane. So a bench polling for a free machine read
    `busy: 0`, `running: []` and `queued: []` **while the browser extension was
    streaming from that very server**, submitted, and was refused
    `engine_in_use` after the round trip. The refusal was right; the display was
    a lie, and it lied in the one direction that matters.

    Owen, 2026-09-13, naming the three surfaces that do this — the streaming
    page, the correct-sentences/re-roll page and the browser extension:
    *"those places are independent of a queue but claim a server while they
    run... that means crucible wont always have a percent complete to hand
    back."*
    """
    with streaming_server() as base:
        session = opened(base, auth)
        body = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()

        # The lane really is free, and still says so. `busy` counts the lane and
        # nothing else — redefining it to mean "the card" would make it a second
        # owner of the claim.
        assert body["slots"]["accelerated"]["busy"] == 0
        assert body["running"] == []
        assert body["queued"] == []

        # But the machine will not take work, and now says so in one read.
        assert body["slots"]["accelerated"]["accepts_work"] is False
        assert body["claim"] is not None
        assert "tts stream" in body["claim"]["held_by"]

        streaming = body["streaming"]
        assert streaming is not None
        assert streaming["session_id"] == session["session_id"]
        assert streaming["voice"] == VOICE
        assert streaming["since"]

        # THE POINT OF THE WHOLE FIELD. A session has no denominator: rows arrive
        # one `say` at a time, indefinitely, so any percentage would be a
        # percentage of the work that happens to have arrived — a number that
        # goes DOWN when more arrives. The key is present and null, which is this
        # server's one spelling of "did not say"; what it offers instead is
        # counts.
        assert "progress" in streaming
        assert streaming["progress"] is None
        assert streaming["said"] == 0
        assert streaming["finished"] == 0
        assert streaming["in_flight"] == 0

        # And the refusal a client gets if it submits anyway agrees with the
        # bench about who has it. One fact.
        refused = httpx.post(
            f"{base}/v1/jobs",
            headers=auth,
            json={"type": "tts", "model": VOICE,
                  "params": {"language": "en", "take": 0,
                             "chunks": [{"index": 0, "text": "Rain."}]}},
            timeout=30.0,
        )
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["details"]["held_by"] == body["claim"]["held_by"]


def test_the_bench_counts_what_a_session_has_actually_said(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """Counts rather than a percentage — the honest half of the same answer."""
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
        # Still no percentage, and there never will be one.
        assert streaming["progress"] is None


def test_the_bench_names_the_client_that_opened_the_session_or_says_it_did_not(
    streaming_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """null means "it did not say" — never a name this server invented.

    A bench that guessed would be confidently wrong about who is on the card,
    which is the rule `Job.client` already follows (PHASE7-LANES.md section 5).
    """
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
