"""Queue sessions: one client holding the server for a run of requests.

The TestClient half drives echo jobs (no engine at all); the live half runs the fake
engine from tests/fake_engine.py behind a real uvicorn server. Nothing here touches a
GPU or a real model.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import clock
from crucible import engines as engines_module
from crucible.config import load_config
from crucible.engines import EngineError
from crucible.errors import ConfigError
from crucible.protocol import SESSION_HEADER

from .conftest import parse_sse
from .live_server import run_job
from .test_chat_queue import (
    MODEL,
    SEEN_TIMEOUT,
    _as,
    _chat,
    _in_background,
    _post_chat,
    _until,
    _wait_for,
    chat_server,  # noqa: F401 - a fixture this module uses
    one_slot,  # noqa: F401 - a fixture this module uses
)
from .test_decide_api import EXAMPLE
from .test_queue import (
    as_client,
    body,
    free_the_lane,
    occupy_the_lane,
    status,
    wait_for,
)


def open_session(
    client: TestClient, headers: dict[str, str], **extra: Any
) -> dict[str, Any]:
    answer = client.post("/v1/queue/sessions", json={"act": "analysis", **extra}, headers=headers)
    assert answer.status_code == 202, answer.text
    return answer.json()


def state(client: TestClient, headers: dict[str, str], session_id: str) -> dict[str, Any]:
    answer = client.get(f"/v1/queue/sessions/{session_id}", headers=headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


def session_events(
    client: TestClient, headers: dict[str, str], session_id: str
) -> list[dict[str, Any]]:
    with client.stream(
        "GET", f"/v1/queue/sessions/{session_id}/events", headers=headers
    ) as response:
        assert response.status_code == 200
        return parse_sse(response.iter_lines())


def submit(client: TestClient, headers: dict[str, str], **document: Any) -> Any:
    return client.post("/v1/jobs", json=document, headers=headers)


def item(headers: dict[str, str], session_id: str) -> dict[str, str]:
    return {**headers, SESSION_HEADER: session_id}


def sessions_of(client: TestClient) -> Any:
    return client.app.state.sessions


def test_on_an_idle_server_a_session_opens_at_once_and_its_client_closes_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    ticket = open_session(client, mine)
    assert ticket["session_id"].startswith("ses-")
    assert ticket["status"] == "open"

    opened = state(client, auth, ticket["session_id"])
    assert opened["status"] == "open" and opened["client"] == "briefcase"
    assert opened["act"] == "analysis" and opened["idle_s"] == 300
    assert opened["opened_at"] is not None and opened["idle_deadline"] is not None
    assert opened["max_hold_deadline"] is None, "there is no maximum hold by default"
    activity = client.get("/v1/activity", headers=auth).json()
    assert activity["session"]["session_id"] == ticket["session_id"]
    assert activity["slots"]["accelerated"]["accepts_work"] is False

    closed = client.delete(f"/v1/queue/sessions/{ticket['session_id']}", headers=mine)
    assert closed.status_code == 200, closed.text
    assert (closed.json()["status"], closed.json()["reason"]) == ("closed", "client")
    names = [e["event"] for e in session_events(client, auth, ticket["session_id"])]
    assert names == ["queued", "opened", "closed"]
    assert client.get("/v1/activity", headers=auth).json()["session"] is None

    again = client.delete(f"/v1/queue/sessions/{ticket['session_id']}", headers=mine)
    assert again.status_code == 200 and again.json()["reason"] == "client"


def test_a_session_waits_in_the_line_behind_a_busy_lane(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    mine = as_client(auth, "briefcase")
    ticket = open_session(client, mine)
    assert (ticket["status"], ticket["position"]) == ("queued", 1)
    rows = client.get("/v1/queue", headers=auth).json()["items"]
    assert [(row["kind"], row["job_id"]) for row in rows] == [
        ("session", ticket["session_id"])
    ]
    assert state(client, auth, ticket["session_id"])["status"] == "queued"
    refused = submit(client, item(mine, ticket["session_id"]), **body())
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "session_not_open"

    free_the_lane(client, auth, holder)
    wait_for(
        lambda: state(client, auth, ticket["session_id"])["status"] == "open",
        "the session to open once the lane was free",
    )
    assert client.get("/v1/queue", headers=auth).json()["depth"] == 0
    client.delete(f"/v1/queue/sessions/{ticket['session_id']}", headers=mine)


def test_while_a_session_is_open_its_items_run_and_nobody_else_s_do(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine, theirs = as_client(auth, "briefcase"), as_client(auth, "bookforge")
    session_id = open_session(client, mine)["session_id"]

    plain = submit(client, theirs, **body(queue=False))
    assert plain.status_code == 409
    error = plain.json()["error"]
    assert error["code"] == "server_busy"
    assert error["details"]["session_id"] == session_id
    assert error["details"]["door"] == "session"
    assert "briefcase" in error["message"]

    waiting = submit(client, theirs, **body())
    assert waiting.status_code == 202 and waiting.json()["queued"] is True
    waiting_id = waiting.json()["job_id"]

    explicit = submit(client, item(mine, session_id), **body())
    assert explicit.status_code == 202, explicit.text
    wait_for(lambda: status(client, auth, explicit.json()["job_id"]) == "done",
             "the session's own job")
    implicit = submit(client, mine, **body())
    assert implicit.status_code == 202, implicit.text
    wait_for(lambda: status(client, auth, implicit.json()["job_id"]) == "done",
             "the same client's job, sent without the header")
    assert status(client, auth, waiting_id) == "queued", "nothing from anyone else ran"
    assert state(client, auth, session_id)["items_run"] == 2

    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)
    wait_for(lambda: status(client, auth, waiting_id) == "done",
             "the other client's job once the session closed")


def test_a_session_s_jobs_wait_inside_the_session_ahead_of_everyone(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine, theirs = as_client(auth, "briefcase"), as_client(auth, "bookforge")
    session_id = open_session(client, mine)["session_id"]
    first = submit(client, item(mine, session_id), **body(delay_ms=60_000)).json()["job_id"]
    wait_for(lambda: status(client, auth, first) == "running", "the first item to run")

    other = submit(client, theirs, **body()).json()["job_id"]
    second = submit(client, item(mine, session_id), **body())
    assert second.status_code == 202, second.text
    assert (second.json()["queued"], second.json()["position"]) == (True, 1)
    rows = client.get("/v1/queue", headers=auth).json()["items"]
    assert [row["job_id"] for row in rows] == [second.json()["job_id"], other]
    assert rows[0]["session"] == session_id and rows[1]["session"] is None
    in_flight = state(client, auth, session_id)["in_flight"]
    assert {entry["id"] for entry in in_flight} == {first, second.json()["job_id"]}

    client.delete(f"/v1/jobs/{first}", headers=mine)
    wait_for(lambda: status(client, auth, second.json()["job_id"]) == "done",
             "the session's second job")
    assert status(client, auth, other) == "queued"
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)
    wait_for(lambda: status(client, auth, other) == "done", "the other job, after")


def test_a_header_naming_a_session_that_is_not_the_caller_s_open_one_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine, theirs = as_client(auth, "briefcase"), as_client(auth, "bookforge")
    session_id = open_session(client, mine)["session_id"]

    stolen = submit(client, item(theirs, session_id), **body())
    assert stolen.status_code == 409
    assert stolen.json()["error"]["code"] == "session_not_yours"
    assert stolen.json()["error"]["details"]["client"] == "briefcase"

    unknown = submit(client, item(mine, "ses-nope"), **body())
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_queue_session"

    not_theirs = client.delete(f"/v1/queue/sessions/{session_id}", headers=theirs)
    assert not_theirs.status_code == 409
    assert not_theirs.json()["error"]["code"] == "session_not_yours"

    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)
    late = submit(client, item(mine, session_id), **body())
    assert late.status_code == 409
    assert late.json()["error"]["code"] == "session_closed"
    assert late.json()["error"]["details"]["reason"] == "client"
    touched = client.post(f"/v1/queue/sessions/{session_id}/touch", headers=mine)
    assert touched.status_code == 409
    assert touched.json()["error"]["code"] == "session_closed"


def test_idle_closes_a_session_and_anything_in_flight_or_a_touch_keeps_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine, idle_s=10)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)

    running = submit(client, mine, **body(delay_ms=60_000)).json()["job_id"]
    wait_for(lambda: status(client, auth, running) == "running", "the item to run")
    a_day_later = clock.now() + timedelta(days=1)
    assert sessions.due(a_day_later) is None, "a running job is never idleness"
    assert state(client, auth, session_id)["idle_deadline"] is None
    client.delete(f"/v1/jobs/{running}", headers=mine)
    wait_for(lambda: status(client, auth, running) == "cancelled", "the job to end")

    touched = client.post(f"/v1/queue/sessions/{session_id}/touch", headers=mine)
    assert touched.status_code == 200 and touched.json()["status"] == "open"
    seen = held.seen
    assert sessions.due(seen + timedelta(seconds=9)) is None
    due = sessions.due(seen + timedelta(seconds=11))
    assert due is not None and due[1] == "idle"

    held.idle_s = 1
    wait_for(lambda: state(client, auth, session_id)["status"] == "closed",
             "the pump to close the idle session")
    closed = state(client, auth, session_id)
    assert closed["reason"] == "idle"
    assert session_events(client, auth, session_id)[-1]["data"]["reason"] == "idle"


def test_max_hold_closes_a_session_only_when_the_config_sets_one(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine)["session_id"]
    sessions = sessions_of(client)
    held = sessions.get(session_id)
    running = submit(client, mine, **body(delay_ms=60_000)).json()["job_id"]
    wait_for(lambda: status(client, auth, running) == "running", "the item to run")
    assert sessions.due(held.opened_at + timedelta(days=2)) is None

    config = client.app.state.config
    object.__setattr__(config, "max_session_hold_s", 60)
    assert state(client, auth, session_id)["max_hold_deadline"] is not None
    assert sessions.due(held.opened_at + timedelta(seconds=59)) is None
    due = sessions.due(held.opened_at + timedelta(seconds=61))
    assert due is not None and due[1] == "max_hold", "it holds even with work in flight"
    assert "max_session_hold_s" in due[2]
    client.delete(f"/v1/jobs/{running}", headers=mine)
    client.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_the_maximum_hold_is_read_from_config_and_absent_is_no_limit(home: Path) -> None:
    from .conftest import configure_box

    configure_box(home)
    assert load_config(home).max_session_hold_s == 0
    path = home / "config.toml"
    original = path.read_bytes()
    path.write_bytes(original + b"\n[queue]\nmax_session_hold_s = 86400\n")
    assert load_config(home).max_session_hold_s == 86400
    path.write_bytes(original + b"\n[queue]\nmax_session_hold_s = -1\n")
    with pytest.raises(ConfigError, match="max_session_hold_s"):
        load_config(home)


def test_an_operator_ends_an_open_session_and_its_waiting_items_leave(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine = as_client(auth, "briefcase")
    session_id = open_session(client, mine)["session_id"]
    first = submit(client, mine, **body(delay_ms=60_000)).json()["job_id"]
    wait_for(lambda: status(client, auth, first) == "running", "the first item")
    second = submit(client, mine, **body()).json()["job_id"]
    assert status(client, auth, second) == "queued"

    ended = client.delete(f"/v1/queue/{session_id}", headers=auth)
    assert ended.status_code == 200, ended.text
    assert ended.json() == {"job_id": session_id, "status": "closed", "reason": "operator"}
    assert state(client, auth, session_id)["reason"] == "operator"
    removal = client.get(f"/v1/jobs/{second}", headers=auth).json()["removal"]
    assert removal["reason"] == "session_closed"
    assert session_id in removal["message"]
    assert status(client, auth, first) == "running", "what was running finishes"
    free_the_lane(client, auth, first)


def test_a_queued_session_is_removed_by_its_client_or_an_operator(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    mine = as_client(auth, "briefcase")
    first = open_session(client, mine)["session_id"]
    second = open_session(client, as_client(auth, "foundry"))["session_id"]

    withdrawn = client.delete(f"/v1/queue/sessions/{first}", headers=mine)
    assert withdrawn.status_code == 200
    assert (withdrawn.json()["status"], withdrawn.json()["reason"]) == ("closed", "client")
    assert [e["event"] for e in session_events(client, auth, first)] == ["queued", "removed"]

    removed = client.delete(f"/v1/queue/{second}", headers=auth)
    assert removed.status_code == 200, removed.text
    assert removed.json()["status"] == "closed"
    final = session_events(client, auth, second)
    assert [e["event"] for e in final] == ["queued", "moved", "removed"]
    assert final[-1]["data"]["reason"] == "operator"
    assert client.get("/v1/queue", headers=auth).json()["depth"] == 0
    again = client.delete(f"/v1/queue/{second}", headers=auth)
    assert again.status_code == 409 and again.json()["error"]["code"] == "not_queued"
    free_the_lane(client, auth, holder)


def test_one_session_is_open_at_a_time_and_the_next_waits_its_turn(
    client: TestClient, auth: dict[str, str]
) -> None:
    mine, theirs = as_client(auth, "briefcase"), as_client(auth, "contentstudio")
    first = open_session(client, mine)["session_id"]
    second = open_session(client, theirs)
    assert (second["status"], second["position"]) == ("queued", 1)
    client.delete(f"/v1/queue/sessions/{first}", headers=mine)
    wait_for(lambda: state(client, auth, second["session_id"])["status"] == "open",
             "the second session to open once the first closed")
    client.delete(f"/v1/queue/sessions/{second['session_id']}", headers=theirs)


def test_a_restart_closes_the_open_session(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client() as first:
        session_id = open_session(first, as_client(auth, "briefcase"))["session_id"]
        sessions = sessions_of(first)
    assert sessions.get(session_id).reason == "server_restart"
    assert sessions.current() is None


def test_a_session_asking_for_a_model_that_does_not_exist_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    refused = client.post(
        "/v1/queue/sessions", json={"act": "analysis", "model": "no-such-model"}, headers=auth
    )
    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "unknown_model"
    upstream = client.post(
        "/v1/queue/sessions", json={"act": "analysis", "model": "openai/gpt-x"}, headers=auth
    )
    assert upstream.status_code == 409
    assert upstream.json()["error"]["code"] == "upstream_never_resident"
    bad_act = client.post("/v1/queue/sessions", json={"act": "sorcery"}, headers=auth)
    assert bad_act.status_code == 400 and bad_act.json()["error"]["code"] == "unknown_act"
    for idle in (5, 86_401):
        out = client.post("/v1/queue/sessions", json={"act": "analysis", "idle_s": idle},
                          headers=auth)
        assert out.status_code == 400, idle
    assert client.post("/v1/queue/sessions", json={"act": "analysis", "idle_s": 86_400},
                       headers=as_client(auth, "nas-copy")).json()["status"] == "open"


def _session(base: str, headers: dict[str, str], **extra: Any) -> dict[str, Any]:
    answer = httpx.post(f"{base}/v1/queue/sessions", headers=headers,
                        json={"act": "analysis", **extra}, timeout=30.0)
    assert answer.status_code == 202, answer.text
    return answer.json()


def _state(base: str, headers: dict[str, str], session_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue/sessions/{session_id}", headers=headers,
                     timeout=30.0).json()


def _activity(base: str, headers: dict[str, str]) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/activity", headers=headers, timeout=30.0).json()


def test_a_session_opens_with_its_model_loaded_and_holds_it_until_it_closes(
    chat_server: Callable[..., Any], auth: dict[str, str]  # noqa: F811
) -> None:
    engines, server = chat_server()
    mine = _as(auth, "briefcase")
    with server as base:
        ticket = _session(base, mine, model=MODEL)
        _wait_for(lambda: _state(base, auth, ticket["session_id"])["status"] == "open",
                  "the session to open on its model")
        opened = _state(base, auth, ticket["session_id"])
        assert opened["load_job"] is not None
        load = httpx.get(f"{base}/v1/jobs/{opened['load_job']}", headers=auth,
                         timeout=30.0).json()
        assert load["status"] == "done" and load["type"] == "load-model"
        assert load["client_ref"] == f"opening session {ticket['session_id']}"
        assert len(engines) == 1

        for _ in range(2):
            answered = _post_chat(base, mine, _chat())
            assert answered.status_code == 200, answered.text
            resident = _activity(base, auth)["resident"]
            assert resident is not None and resident["id"] == MODEL, (
                "the session holds the model between its items"
            )
            assert resident["held_by"]["fact"] == "a session"
        assert _state(base, auth, ticket["session_id"])["items_run"] == 2

        closed = httpx.delete(f"{base}/v1/queue/sessions/{ticket['session_id']}",
                              headers=mine, timeout=30.0)
        assert closed.status_code == 200 and closed.json()["reason"] == "client"
        assert _activity(base, auth)["resident"] is None, "the settlement cleared the card"


def test_a_session_whose_model_will_not_load_never_opens(
    chat_server: Callable[..., Any], auth: dict[str, str], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    _, server = chat_server()

    def refuses(engine_name: str, python: Path, log_path: Path) -> Any:
        raise EngineError("this test's engine refuses to start")

    monkeypatch.setattr(engines_module, "build_engine", refuses)
    with server as base:
        ticket = _session(base, _as(auth, "briefcase"), model=MODEL)
        _wait_for(lambda: _state(base, auth, ticket["session_id"])["status"] == "closed",
                  "the session to end when its load failed")
        ended = _state(base, auth, ticket["session_id"])
        assert ended["opened_at"] is None
        assert ended["reason"] == "load_failed"
        assert ended["error"]["code"] == "session_load_failed"
        assert ended["load_job"] in ended["error"]["message"]
        assert _activity(base, auth)["session"] is None


def test_other_clients_chats_wait_or_are_refused_while_a_session_is_open(
    chat_server: Callable[..., Any], auth: dict[str, str]  # noqa: F811
) -> None:
    _, server = chat_server()
    mine, theirs = _as(auth, "contentstudio"), _as(auth, "bookforge")
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        session_id = _session(base, mine)["session_id"]

        refused = _post_chat(base, theirs, _chat(queue=False))
        assert refused.status_code == 409
        error = refused.json()["error"]
        assert error["code"] == "session_open"
        assert "contentstudio" in error["message"] and session_id in error["message"]
        decided = httpx.post(f"{base}/v1/decide", headers=theirs, timeout=30.0,
                             json={**EXAMPLE, "queue": False})
        assert decided.status_code == 409
        assert decided.json()["error"]["code"] == "session_open"

        waiting, out = _in_background(lambda: _post_chat(base, theirs, _chat()))
        _wait_for(lambda: httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0)
                  .json()["depth"] == 1, "the other client's chat in the line")

        own = _post_chat(base, mine, _chat())
        assert own.status_code == 200, "the holder's standalone chat is an implicit item"
        explicit = _post_chat(base, {**mine, SESSION_HEADER: session_id}, _chat())
        assert explicit.status_code == 200
        assert waiting.is_alive(), "the other client's chat is still waiting"

        httpx.delete(f"{base}/v1/queue/sessions/{session_id}", headers=mine, timeout=30.0)
        waiting.join(SEEN_TIMEOUT)
        assert out and out[0].status_code == 200, out


def test_a_session_chat_waits_ahead_of_the_line_for_a_slot(
    chat_server: Callable[..., Any], auth: dict[str, str], one_slot: None  # noqa: F811
) -> None:
    release = threading.Event()
    _, server = chat_server(delay_for=_until(release))
    mine = _as(auth, "briefcase")
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        session_id = _session(base, mine)["session_id"]
        busy, busy_out = _in_background(lambda: _post_chat(base, mine, _chat()))
        _wait_for(lambda: _activity(base, auth)["chat"]["in_flight"] == 1,
                  "the session's first chat in flight")
        theirs, theirs_out = _in_background(
            lambda: _post_chat(base, _as(auth, "bookforge"), _chat())
        )
        _wait_for(lambda: httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0)
                  .json()["depth"] == 1, "the other client's chat waiting")
        second, second_out = _in_background(lambda: _post_chat(base, mine, _chat()))
        _wait_for(lambda: httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0)
                  .json()["depth"] == 2, "the session's second chat waiting")
        rows = httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0).json()["items"]
        assert [row["session"] for row in rows] == [session_id, None]
        state_now = _state(base, auth, session_id)
        assert {entry["kind"] for entry in state_now["in_flight"]} == {"chat", "call"}

        release.set()
        busy.join(SEEN_TIMEOUT)
        second.join(SEEN_TIMEOUT)
        assert busy_out[0].status_code == 200 and second_out[0].status_code == 200
        assert theirs.is_alive(), "the other chat waits for the session to close"
        httpx.delete(f"{base}/v1/queue/sessions/{session_id}", headers=mine, timeout=30.0)
        theirs.join(SEEN_TIMEOUT)
        assert theirs_out[0].status_code == 200


def test_every_session_change_reaches_the_server_wide_event_stream(
    client: TestClient, auth: dict[str, str]
) -> None:
    hub = client.app.state.events
    mine = as_client(auth, "briefcase")
    ticket = open_session(client, mine)
    client.delete(f"/v1/queue/sessions/{ticket['session_id']}", headers=mine)
    sessions = [event for topic, event in hub._history if topic == "session"]
    assert [event["event"] for event in sessions] == [
        "session.queued", "session.opened", "session.closed",
    ]
    assert all(event["data"]["session_id"] == ticket["session_id"] for event in sessions)
    assert sessions[0]["data"]["client"] == "briefcase"
    assert sessions[-1]["data"]["reason"] == "client"


def test_a_session_removed_while_its_model_loads_takes_the_load_with_it(
    chat_server: Callable[..., Any], auth: dict[str, str]  # noqa: F811
) -> None:
    hold = threading.Event()
    _, server = chat_server(hold=hold)
    mine = _as(auth, "fleet-loser")
    with server as base:
        ticket = _session(base, mine, model=MODEL)
        session_id = ticket["session_id"]
        _wait_for(lambda: _state(base, auth, session_id)["load_job"] is not None,
                  "the session's load to start")
        load_job = _state(base, auth, session_id)["load_job"]
        try:
            gone = httpx.delete(f"{base}/v1/queue/sessions/{session_id}", headers=mine,
                                timeout=30.0)
            assert gone.status_code == 200, gone.text
            assert gone.json()["reason"] == "client"
        finally:
            hold.set()

        def load_status() -> str:
            return httpx.get(f"{base}/v1/jobs/{load_job}", headers=auth,
                             timeout=30.0).json()["status"]

        _wait_for(lambda: load_status() in ("done", "failed", "cancelled"),
                  "the abandoned load to end")
        assert load_status() == "cancelled"
        _wait_for(lambda: _activity(base, auth)["resident"] is None
                  and _activity(base, auth)["warming"] is None,
                  "the card to be left empty")
