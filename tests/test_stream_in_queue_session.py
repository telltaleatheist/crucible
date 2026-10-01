"""A TTS stream session runs inside a queue session, with no priority of its own.

The fake narrator from tests/fake_narrator_engine.py stands in for the engine; nothing
here touches a GPU or a real voice.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest

from crucible.protocol import CLIENT_HEADER, SESSION_HEADER

from . import fake_narrator_engine
from .live_server import run_job, serve
from .test_tts_api import (  # noqa: F401 - fixtures this module uses
    fake_env,
    fake_weights,
    idle_card,
    tts_recipes,
)
from .test_tts_stream import VOICE, WAIT, listen, quick_engine  # noqa: F401

EXTENSION = "bookforge-extension"
OTHER = "briefcase"


@pytest.fixture
def stream_server(
    make_app: Callable[..., Any],
    auth: dict[str, str],
    fake_env: Path,  # noqa: F811
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Any]:

    @contextmanager
    def start() -> Iterator[tuple[str, Any]]:
        fake_narrator_engine.install(monkeypatch)
        app = make_app(enable_tts=True, enable_echo=True)
        fake_weights(VOICE)
        with serve(app) as base:
            run_job(base, auth, type="load-voice", model=VOICE)
            yield base, app

    return start


def as_client(auth: dict[str, str], name: str) -> dict[str, str]:
    return {**auth, CLIENT_HEADER: name}


def open_stream(base: str, headers: dict[str, str], **extra: Any) -> httpx.Response:
    return httpx.post(f"{base}/v1/tts/stream", headers=headers,
                      json={"voice": VOICE, "language": "en", **extra}, timeout=60.0)


def queue_session(base: str, headers: dict[str, str], session_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue/sessions/{session_id}", headers=headers,
                     timeout=30.0).json()


def wait_until(condition: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError(f"never saw {what}")


def test_a_stream_with_no_session_opens_one_and_closing_the_stream_closes_it(
    stream_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    mine = as_client(auth, EXTENSION)
    with stream_server() as (base, _app):
        answer = open_stream(base, mine)
        assert answer.status_code == 201, answer.text
        body = answer.json()
        assert body["queue_session_opened_for_stream"] is True
        held = queue_session(base, auth, body["queue_session_id"])
        assert (held["status"], held["act"], held["client"]) == ("open", "tts", EXTENSION)
        assert held["idle_s"] == 900, "the extension's fifteen minutes is the default"
        assert held["stream_session"]["session_id"] == body["session_id"]

        closed = httpx.delete(f"{base}/v1/tts/stream/{body['session_id']}",
                              headers=mine, timeout=30.0)
        assert closed.status_code == 200 and closed.json()["closed"] is True
        ended = queue_session(base, auth, body["queue_session_id"])
        assert (ended["status"], ended["reason"]) == ("closed", "client")
        activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
        assert activity["session"] is None and activity["streaming"] is None
        assert activity["resident"] is None, "the settlement cleared the card after"


def test_a_stream_waits_behind_another_client_s_session_with_no_priority(
    stream_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    mine, theirs = as_client(auth, EXTENSION), as_client(auth, OTHER)
    with stream_server() as (base, _app):
        other = httpx.post(f"{base}/v1/queue/sessions", headers=theirs,
                           json={"act": "analysis"}, timeout=30.0).json()
        assert other["status"] == "open"

        refused = open_stream(base, mine, queue=False)
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "session_open"
        assert OTHER in refused.json()["error"]["message"]

        out: list[httpx.Response] = []
        opener = threading.Thread(target=lambda: out.append(open_stream(base, mine)),
                                  daemon=True)
        opener.start()
        wait_until(
            lambda: [row["kind"] for row in httpx.get(
                f"{base}/v1/queue", headers=auth, timeout=30.0).json()["items"]
            ] == ["session"],
            "the stream's session waiting in the line",
        )
        time.sleep(0.5)
        assert not out, "the stream did not jump the other client's session"

        httpx.delete(f"{base}/v1/queue/sessions/{other['session_id']}", headers=theirs,
                     timeout=30.0)
        opener.join(WAIT)
        assert out and out[0].status_code == 201, out
        assert out[0].json()["voice"] == VOICE, "its session loaded the voice again"


def test_idle_closes_the_stream_s_session_and_the_stream_is_told_why(
    stream_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    mine = as_client(auth, EXTENSION)
    with stream_server() as (base, app):
        body = open_stream(base, mine, idle_s=10).json()
        held = app.state.sessions.get(body["queue_session_id"])
        with listen(base, mine, body["session_id"]) as stream:
            stream.wait_for(lambda s: s.of("ready"), "the ready frame")
            seen = held.seen
            ops = httpx.post(f"{base}/v1/tts/stream/{body['session_id']}", headers=mine,
                             json={"op": "cancel_all"}, timeout=30.0)
            assert ops.status_code == 202
            assert held.seen > seen, "every stream op is activity of its session"
            held.idle_s = 1
            stream.wait_for(lambda s: s.of("closed"), "the stream to close")
            closed = stream.of("closed")[0]
        assert closed["code"] == "session_closed"
        assert closed["session_reason"] == "idle"
        assert closed["queue_session_id"] == body["queue_session_id"]
        ended = queue_session(base, auth, body["queue_session_id"])
        assert (ended["status"], ended["reason"]) == ("closed", "idle")


def test_a_stream_inside_an_explicit_session_leaves_it_open_when_it_closes(
    stream_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    mine = as_client(auth, EXTENSION)
    with stream_server() as (base, _app):
        session = httpx.post(f"{base}/v1/queue/sessions", headers=mine,
                             json={"act": "tts"}, timeout=30.0).json()
        body = open_stream(base, {**mine, SESSION_HEADER: session["session_id"]}).json()
        assert body["queue_session_id"] == session["session_id"]
        assert body["queue_session_opened_for_stream"] is False

        httpx.delete(f"{base}/v1/tts/stream/{body['session_id']}", headers=mine,
                     timeout=30.0)
        still = queue_session(base, auth, session["session_id"])
        assert still["status"] == "open", "the client's own session outlives its stream"
        assert still["stream_session"] is None
        activity = httpx.get(f"{base}/v1/activity", headers=auth, timeout=30.0).json()
        assert activity["resident"]["id"] == VOICE, "the session still holds the voice"

        again = open_stream(base, mine)
        assert again.status_code == 201, "the same client's stream is an implicit item"
        assert again.json()["queue_session_id"] == session["session_id"]
        closed = httpx.delete(f"{base}/v1/queue/sessions/{session['session_id']}",
                              headers=mine, timeout=30.0)
        assert closed.json()["reason"] == "client"
        assert httpx.get(f"{base}/v1/activity", headers=auth,
                         timeout=30.0).json()["streaming"] is None, (
            "closing the session closed the stream inside it"
        )


def test_another_client_cannot_open_a_stream_in_a_session_that_is_not_its_own(
    stream_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    with stream_server() as (base, _app):
        session = httpx.post(f"{base}/v1/queue/sessions",
                             headers=as_client(auth, EXTENSION),
                             json={"act": "tts"}, timeout=30.0).json()
        stolen = open_stream(base, {**as_client(auth, OTHER),
                                    SESSION_HEADER: session["session_id"]})
        assert stolen.status_code == 409
        assert stolen.json()["error"]["code"] == "session_not_yours"
