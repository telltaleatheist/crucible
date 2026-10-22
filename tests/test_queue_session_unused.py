"""A queue session that sends no work within FIRST_WORK_S of opening closes `unused`,
whatever idle_s it asked for (Owen, 2026-10-11): an app that asked and then crashed or
restarted does not hold the card for up to a day.

The TestClient half drives echo jobs (no engine); the live half runs the fake engine
behind a real uvicorn server for a load that is held mid-way. The clock is moved by
handing `due()` a time, or by moving the session's own `opened_at` back, never by
sleeping. Nothing here touches a GPU or a real model.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import clock
from crucible.queuesessions import FIRST_WORK_S, UNUSED

from .live_server import serve
from .test_chat_queue import MODEL, _as, _wait_for
from .test_queue import as_client, body, free_the_lane, occupy_the_lane, status, wait_for
from .test_queue_sessions import item, open_session, session_events, sessions_of, state, submit


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def test_an_unused_session_closes_at_the_window_and_says_why_everywhere(
    client: TestClient, auth: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    mine, theirs = as_client(auth, "b-sides"), as_client(auth, "foundry")
    session_id = open_session(client, mine, idle_s=86_400)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)

    opened = state(client, auth, session_id)
    assert _at(opened["first_work_deadline"]) == held.opened_at + timedelta(seconds=FIRST_WORK_S)
    assert sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S - 1)) is None
    due = sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S))
    assert due is not None and due[1] == UNUSED, "a day's idle_s does not keep an unused session"
    assert f"sent no work and no touch within {FIRST_WORK_S} s of opening" in due[2]

    waiting = submit(client, theirs, **body())
    assert waiting.status_code == 202 and waiting.json()["queued"] is True
    waiting_id = waiting.json()["job_id"]

    held.opened_at -= timedelta(seconds=FIRST_WORK_S + 1)
    wait_for(lambda: state(client, auth, session_id)["status"] == "closed",
             "the pump to close the unused session")
    closed = state(client, auth, session_id)
    assert closed["reason"] == UNUSED
    assert f"sent no work and no touch within {FIRST_WORK_S} s of opening" in closed["message"]
    assert closed["first_work_deadline"] is None and closed["items_run"] == 0

    last = session_events(client, auth, session_id)[-1]
    assert (last["event"], last["data"]["reason"]) == ("closed", UNUSED)
    assert last["data"]["message"] == closed["message"]
    hub = client.app.state.events
    published = [event for topic, event in hub._history if topic == "session"][-1]
    assert (published["event"], published["data"]["reason"]) == ("session.closed", UNUSED)
    assert published["data"]["client"] == "b-sides"
    assert f"closed session {session_id} (unused)" in capsys.readouterr().err

    late = submit(client, item(mine, session_id), **body())
    assert late.status_code == 409
    error = late.json()["error"]
    assert error["code"] == "session_closed" and error["details"]["reason"] == UNUSED
    assert f"closed (unused): it sent no work and no touch within {FIRST_WORK_S} s of opening" in (
        error["message"]
    )
    touched = client.post(f"/v1/queue/sessions/{session_id}/touch", headers=mine)
    assert touched.status_code == 409 and touched.json()["error"]["details"]["reason"] == UNUSED

    wait_for(lambda: status(client, auth, waiting_id) == "done",
             "the next client's job once the unused session let the server go")


def test_a_touch_in_the_window_keeps_it_open_and_idle_s_runs_from_the_last_touch(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Foundry's page reader, 2026-10-22: it opens its session so dots-ocr loads while it
    renders the book, and a 968-page render took ~90 s. It touched at 10:14:30 and was
    closed `unused` at 10:15:02. A touch now ends the window: the client is present, and
    only its idle_s (from the last touch) closes it."""
    mine = as_client(auth, "foundry")
    session_id = open_session(client, mine, idle_s=300)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)

    touched = client.post(f"/v1/queue/sessions/{session_id}/touch", headers=mine)
    assert touched.status_code == 200
    opened = state(client, auth, session_id)
    assert opened["first_work_deadline"] is None, "a touch ends the unused window"
    assert held.first_touch_at is not None and held.first_work_at is None
    last_touch = held.seen
    assert last_touch is not None

    past_window = held.opened_at + timedelta(seconds=FIRST_WORK_S + 30)
    assert sessions.due(past_window) is None, "a touched session is not unused"
    assert sessions.due(last_touch + timedelta(seconds=299)) is None
    due = sessions.due(last_touch + timedelta(seconds=300))
    assert due is not None and due[1] == "idle", "an abandoned client still lets go, at idle_s"

    done = submit(client, mine, **body()).json()["job_id"]
    wait_for(lambda: status(client, auth, done) == "done", "its first item after the render")
    assert held.first_work_at is not None and state(client, auth, session_id)["status"] == "open"


def test_a_session_that_sends_work_in_time_keeps_its_own_idle_s(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine, idle_s=600)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)

    done = submit(client, mine, **body()).json()["job_id"]
    wait_for(lambda: status(client, auth, done) == "done", "the implicit item")
    assert held.first_work_at is not None
    assert state(client, auth, session_id)["first_work_deadline"] is None
    assert sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S + 1)) is None, (
        "after its first item only idle_s decides"
    )
    due = sessions.due(held.seen + timedelta(seconds=601))
    assert due is not None and due[1] == "idle"
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_an_item_the_server_refuses_still_counts_as_work(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine)["session_id"]
    refused = submit(client, item(mine, session_id), type="load-model", model="no-such-model")
    assert 400 <= refused.status_code < 500, refused.text
    assert state(client, auth, session_id)["first_work_deadline"] is None, (
        "the app is plainly using its session even when an item is refused"
    )
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_a_short_idle_s_still_closes_idle_before_the_window(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine, idle_s=10)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)
    due = sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S + 5))
    assert due is not None and due[1] == "idle", "the earlier deadline names the close"
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_work_in_flight_at_the_window_is_work_and_a_long_item_never_trips_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine, idle_s=60)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)
    running = submit(client, item(mine, session_id), **body(delay_ms=60_000)).json()["job_id"]
    wait_for(lambda: status(client, auth, running) == "running", "the long item to run")
    held.first_work_at = None  # as if only its being in flight showed it
    assert sessions.due(held.opened_at + timedelta(hours=6)) is None
    assert held.first_work_at is not None, "something of its in flight is its first work"
    client.delete(f"/v1/jobs/{running}", headers=mine)
    wait_for(lambda: status(client, auth, running) == "cancelled", "the item to end")
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_a_queued_session_s_window_starts_when_it_opens(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    mine = as_client(auth, "b-sides")
    session_id = open_session(client, mine)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)
    waited = state(client, auth, session_id)
    assert waited["status"] == "queued" and waited["first_work_deadline"] is None
    assert sessions.due(clock.now() + timedelta(days=1)) is None, "a queued session is not open"

    held.created -= timedelta(minutes=30)  # it waited half an hour in the line
    free_the_lane(client, auth, holder)
    wait_for(lambda: state(client, auth, session_id)["status"] == "open", "it to open")
    opened = state(client, auth, session_id)
    assert _at(opened["first_work_deadline"]) == held.opened_at + timedelta(seconds=FIRST_WORK_S)
    assert sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S - 1)) is None
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


@pytest.fixture
def live(
    make_app: Callable[..., Any], fake_model_ready: None, idle_card: None,
    engine_factory: Callable[..., Any],
) -> Callable[..., Any]:
    def start(**engine_options: Any) -> tuple[Any, Any]:
        engine_factory(**engine_options)
        app = make_app(enable_llm=True)
        return app, serve(app)

    return start


@pytest.fixture
def fake_model_ready(fake_env: Path, fake_weights: Callable[[str], Path]) -> None:
    fake_weights(MODEL)


def _state(base: str, headers: dict[str, str], session_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue/sessions/{session_id}", headers=headers,
                     timeout=30.0).json()


def _job(base: str, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/jobs/{job_id}", headers=headers, timeout=30.0).json()


def test_the_session_s_own_load_runs_before_its_window_starts(
    live: Callable[..., Any], auth: dict[str, str]
) -> None:
    hold = threading.Event()
    app, server = live(hold=hold)
    sessions = app.state.sessions
    mine = _as(auth, "b-sides")
    with server as base:
        answer = httpx.post(f"{base}/v1/queue/sessions", headers=mine, timeout=30.0,
                            json={"act": "generate", "model": MODEL})
        session_id = answer.json()["session_id"]
        try:
            _wait_for(lambda: _state(base, auth, session_id)["load_job"] is not None,
                      "the session's own load to start")
            loading = _state(base, auth, session_id)
            assert loading["status"] == "queued" and loading["first_work_deadline"] is None
            assert sessions.due(clock.now() + timedelta(hours=1)) is None, (
                "a load that takes minutes runs before the session opens, so the window "
                "cannot run out during it"
            )
        finally:
            hold.set()
        _wait_for(lambda: _state(base, auth, session_id)["status"] == "open",
                  "the session to open on its loaded model")
        opened = _state(base, auth, session_id)
        load = _job(base, auth, opened["load_job"])
        assert _at(opened["opened_at"]) >= _at(load["finished"])
        assert _at(opened["first_work_deadline"]) == (
            _at(opened["opened_at"]) + timedelta(seconds=FIRST_WORK_S)
        )
        httpx.delete(f"{base}/v1/queue/sessions/{session_id}", headers=mine, timeout=30.0)


def test_an_app_whose_first_act_is_a_long_load_keeps_its_session(
    live: Callable[..., Any], auth: dict[str, str]
) -> None:
    hold = threading.Event()
    app, server = live(hold=hold)
    sessions = app.state.sessions
    mine = _as(auth, "b-sides")
    with server as base:
        answer = httpx.post(f"{base}/v1/queue/sessions", headers=mine, timeout=30.0,
                            json={"act": "generate", "idle_s": 180})
        session_id = answer.json()["session_id"]
        assert answer.json()["status"] == "open"
        held = sessions.get(session_id)
        try:
            load = httpx.post(f"{base}/v1/jobs", timeout=30.0,
                              headers={**mine, "X-Crucible-Session": session_id},
                              json={"type": "load-model", "model": MODEL})
            assert load.status_code == 202, load.text
            load_id = load.json()["job_id"]
            _wait_for(lambda: _job(base, auth, load_id)["status"] == "running",
                      "the app's own load to run")
            assert _state(base, auth, session_id)["first_work_deadline"] is None
            assert sessions.due(held.opened_at + timedelta(minutes=20)) is None, (
                "a 27B load that takes minutes is the session's work, never its absence"
            )
        finally:
            hold.set()
        _wait_for(lambda: _job(base, auth, load_id)["status"] == "done", "the load to finish")
        assert sessions.due(held.opened_at + timedelta(seconds=FIRST_WORK_S + 1)) is None
        httpx.delete(f"{base}/v1/queue/sessions/{session_id}", headers=mine, timeout=30.0)
