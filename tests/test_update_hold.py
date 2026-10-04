"""The update hold: a deploy stops this server admitting work BEFORE it asks whether
anything is working, so nothing is admitted in the gap and then killed by the restart
(B-Side's phone song, 2026-10-04: admitted 8 s after an idle answer, killed by 1.0.102).
"""
from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from crucible.errors import ApiError
from crucible.updating import MAX_HOLD_S, RETRY_AFTER_S, SERVER_UPDATING, UpdateHold

from .test_queue import body, free_the_lane, occupy_the_lane, status, wait_for

HOLD = "/v1/server/updating"


def test_an_idle_server_is_held_and_refuses_new_work_until_let_go(
    client: TestClient, auth: dict[str, str]
) -> None:
    held = client.post(HOLD, headers=auth, json={"release": "9.9.9", "seconds": 60})
    assert held.status_code == 200, held.text
    assert held.json()["holding"] is True
    assert held.json()["release"] == "9.9.9"
    assert client.get("/v1/activity", headers=auth).json()["updating"]["release"] == "9.9.9"

    refused = client.post("/v1/jobs", headers=auth, json=body())
    assert refused.status_code == 503
    assert refused.json()["error"]["code"] == SERVER_UPDATING
    assert refused.headers["retry-after"] == str(RETRY_AFTER_S)
    assert "9.9.9" in refused.json()["error"]["message"]
    session = client.post("/v1/queue/sessions", headers=auth, json={"act": "clean"})
    assert session.status_code == 503
    assert session.json()["error"]["code"] == SERVER_UPDATING
    with pytest.raises(ApiError) as chat:
        client.app.state.inflight.open(act="clean", model="m", client=None)
    assert chat.value.code == SERVER_UPDATING
    # Nothing was admitted, so nothing is there for the restart to kill.
    assert client.get("/v1/activity", headers=auth).json()["running"] == []

    let_go = client.delete(HOLD, headers=auth)
    assert let_go.json()["holding"] is False
    assert let_go.json()["released"]["release"] == "9.9.9"
    assert client.get("/v1/activity", headers=auth).json()["updating"] is None
    admitted = client.post("/v1/jobs", headers=auth, json=body())
    assert admitted.status_code == 202, admitted.text
    job_id = admitted.json()["job_id"]
    wait_for(lambda: status(client, auth, job_id) == "done", "the job admitted after the hold")


def test_a_working_server_is_not_held_and_names_the_work(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    refused = client.post(HOLD, headers=auth, json={"release": "9.9.9"})
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "server_working"
    assert any(holder in line for line in refused.json()["error"]["details"]["working"])
    # The hold was let go in the same breath: work is still admitted.
    assert client.get("/v1/activity", headers=auth).json()["updating"] is None
    queued = client.post("/v1/jobs", headers=auth, json=body())
    assert queued.status_code == 202, queued.text
    free_the_lane(client, auth, holder)
    free_the_lane(client, auth, queued.json()["job_id"])


def test_the_check_is_where_work_is_created_not_at_the_door(
    client: TestClient, auth: dict[str, str]
) -> None:
    """An admission can await between its route and the moment its job exists; one that
    passed the door before the hold must still be refused when it creates the job."""
    client.post(HOLD, headers=auth, json={"seconds": 60})
    with pytest.raises(ApiError) as created:
        client.app.state.store.create("echo", None, {})
    assert created.value.code == SERVER_UPDATING
    client.delete(HOLD, headers=auth)


def test_every_owner_that_creates_work_shares_the_one_hold(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client() as client:
        state = client.app.state
        hold = state.updating
        for owner in (state.store, state.inflight, state.sessions, state.streams, state.tasks):
            assert owner.updating is hold, type(owner).__name__


def test_a_hold_nobody_ends_lapses_by_itself() -> None:
    now = [100.0]
    hold = UpdateHold(monotonic=lambda: now[0])
    hold.begin(seconds=30, release=None, by="deploy")
    with pytest.raises(ApiError):
        hold.refuse_if_holding("a job")
    now[0] += 30
    assert hold.current() is None
    hold.refuse_if_holding("a job")


def test_a_hold_is_bounded(client: TestClient, auth: dict[str, str]) -> None:
    too_long = client.post(HOLD, headers=auth, json={"seconds": MAX_HOLD_S + 1})
    assert too_long.status_code == 400
    assert client.get("/v1/activity", headers=auth).json()["updating"] is None
