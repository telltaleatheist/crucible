from __future__ import annotations

import base64
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import clock
from crucible.api import sse
from crucible.errors import ApiError
from crucible.jobs import line as line_module
from crucible.protocol import CLIENT_HEADER

from .conftest import parse_sse

PAYLOAD = base64.b64encode(b"queued bytes").decode("ascii")


def body(delay_ms: int = 0, queue: Any = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "type": "echo",
        "params": {"delay_ms": delay_ms},
        "inputs": {"x.bin": {"inline_base64": PAYLOAD}},
    }
    if queue is not None:
        document["queue"] = queue
    return document


def as_client(auth: dict[str, str], name: str) -> dict[str, str]:
    return {**auth, CLIENT_HEADER: name}


def occupy_the_lane(client: TestClient, auth: dict[str, str]) -> str:
    answer = client.post("/v1/jobs", json=body(delay_ms=60_000), headers=auth)
    assert answer.status_code == 202, answer.text
    job_id = answer.json()["job_id"]
    wait_for(lambda: status(client, auth, job_id) == "running", "the lane job to run")
    return job_id


def status(client: TestClient, auth: dict[str, str], job_id: str) -> str:
    return client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"]


def wait_for(condition: Callable[[], bool], what: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def queue_up(client: TestClient, headers: dict[str, str], **queue: Any) -> dict[str, Any]:
    answer = client.post("/v1/jobs", json=body(queue=queue or None), headers=headers)
    assert answer.status_code == 202, answer.text
    return answer.json()


def events(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as response:
        assert response.status_code == 200
        return parse_sse(response.iter_lines())


def free_the_lane(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_with_queue_false_a_busy_server_refuses_at_once(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    refused = client.post("/v1/jobs", json=body(queue=False), headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "server_busy"
    assert client.get("/v1/queue", headers=auth).json()["depth"] == 0
    free_the_lane(client, auth, holder)


def test_a_plain_submit_waits_by_default_when_the_server_is_busy(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    answer = client.post("/v1/jobs", json=body(), headers=auth)
    assert answer.status_code == 202, answer.text
    receipt = answer.json()
    assert (receipt["queued"], receipt["position"]) == (True, 1)
    listed = client.get("/v1/queue", headers=auth).json()["items"]
    assert [(row["job_id"], row["max_wait_s"]) for row in listed] == [
        (receipt["job_id"], line_module.DEFAULT_MAX_WAIT_S)
    ]
    free_the_lane(client, auth, holder)
    assert events(client, auth, receipt["job_id"])[-1]["event"] == "done"


@pytest.mark.parametrize("queue", [{}, True, None, 0, "yes"])
def test_a_queue_that_is_neither_false_nor_a_wait_is_refused_by_name(
    client: TestClient, auth: dict[str, str], queue: Any
) -> None:
    document = {**body(), "queue": queue}
    refused = client.post("/v1/jobs", json=document, headers=auth)
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["code"] == "invalid_request"
    assert '"queue" is false' in error["message"]


def test_an_idle_server_admits_a_queued_submit_at_once(
    client: TestClient, auth: dict[str, str]
) -> None:
    receipt = queue_up(client, auth)
    assert receipt["queued"] is False
    seen = events(client, auth, receipt["job_id"])
    names = [event["event"] for event in seen]
    assert names[:2] == ["queued", "started"]
    assert names[-1] == "done"


def test_a_busy_server_queues_first_come_first_served(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    first = queue_up(client, as_client(auth, "bookforge"))
    second = queue_up(client, as_client(auth, "briefcase"), max_wait_s=600)
    assert (first["queued"], first["position"]) == (True, 1)
    assert (second["queued"], second["position"]) == (True, 2)

    listed = client.get("/v1/queue", headers=auth).json()
    assert [row["job_id"] for row in listed["items"]] == [first["job_id"], second["job_id"]]
    assert [row["position"] for row in listed["items"]] == [1, 2]
    assert listed["items"][0]["client"] == "bookforge"
    assert listed["items"][1]["max_wait_s"] == 600
    assert listed["limits"]["per_client"] == line_module.PER_CLIENT_LIMIT

    activity = client.get("/v1/activity", headers=auth).json()
    assert [row["job_id"] for row in activity["queued"]] == [
        first["job_id"], second["job_id"]
    ]
    assert activity["slots"]["accelerated"]["queue_depth"] == 3
    assert activity["queued"][0]["max_wait_s"] == line_module.DEFAULT_MAX_WAIT_S
    state = client.get(f"/v1/jobs/{second['job_id']}", headers=auth).json()
    assert (state["status"], state["position"]) == ("queued", 2)

    free_the_lane(client, auth, holder)
    first_events = events(client, auth, first["job_id"])
    second_events = events(client, auth, second["job_id"])
    assert [e["event"] for e in first_events][:2] == ["queued", "started"]
    assert first_events[-1]["event"] == "done"
    moved = [e["data"]["position"] for e in second_events if e["event"] == "queued"]
    assert moved == [2, 1]
    assert second_events[-1]["event"] == "done"
    assert client.get("/v1/queue", headers=auth).json()["depth"] == 0


def test_an_operator_and_a_client_can_each_remove_a_waiting_job(
    client: TestClient, auth: dict[str, str]
) -> None:
    holder = occupy_the_lane(client, auth)
    mine = queue_up(client, auth)["job_id"]
    theirs = queue_up(client, auth)["job_id"]

    removed = client.delete(f"/v1/queue/{theirs}", headers=auth)
    assert removed.status_code == 200, removed.text
    assert removed.json() == {"job_id": theirs, "status": "removed", "reason": "operator"}
    cancelled = client.delete(f"/v1/jobs/{mine}", headers=auth)
    assert cancelled.json() == {"job_id": mine, "status": "removed"}

    for job_id, reason in ((theirs, "operator"), (mine, "client")):
        state = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        assert state["status"] == "removed"
        assert state["error"] is None
        assert state["removal"]["reason"] == reason
        final = events(client, auth, job_id)[-1]
        assert final["event"] == "removed"
        assert final["data"]["reason"] == reason

    again = client.delete(f"/v1/queue/{theirs}", headers=auth)
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "not_queued"
    assert client.delete("/v1/queue/nope", headers=auth).status_code == 404
    free_the_lane(client, auth, holder)


def test_a_submit_that_will_not_wait_does_not_jump_a_waiting_line(
    client: TestClient, auth: dict[str, str]
) -> None:
    pump = client.app.state.queue_pump

    async def stalled() -> None:
        return None

    pump.step = stalled
    holder = occupy_the_lane(client, auth)
    waiting = queue_up(client, auth)["job_id"]
    free_the_lane(client, auth, holder)
    wait_for(lambda: client.app.state.store.lane_free, "the lane to go idle")
    refused = client.post("/v1/jobs", json=body(queue=False), headers=auth)
    assert refused.status_code == 409
    details = refused.json()["error"]["details"]
    assert refused.json()["error"]["code"] == "server_busy"
    assert (details["job_id"], details["queue_depth"]) == (waiting, 1)
    activity = client.get("/v1/activity", headers=auth).json()
    assert activity["slots"]["accelerated"]["accepts_work"] is False


def test_a_refusal_at_the_front_fails_the_job_by_name(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = client.app.state.store
    holder = occupy_the_lane(client, auth)
    doomed = queue_up(client, auth)["job_id"]

    def no_room(model: str | None, params: dict[str, Any]) -> None:
        raise ApiError(409, "insufficient_memory", "no room on the card for this")

    monkeypatch.setattr(store.registry["echo"], "preflight", no_room)
    free_the_lane(client, auth, holder)
    final = events(client, auth, doomed)[-1]
    assert final["event"] == "failed"
    assert final["data"]["error"]["code"] == "insufficient_memory"
    assert status(client, auth, doomed) == "failed"


def test_a_busy_refusal_at_the_front_keeps_the_job_waiting(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = client.app.state.store
    echo = store.registry["echo"]
    holder = occupy_the_lane(client, auth)
    patient = queue_up(client, auth)["job_id"]
    asked: list[int] = []

    def claimed(model: str | None, params: dict[str, Any]) -> None:
        asked.append(1)
        raise ApiError(409, "engine_in_use", "a streaming session holds the engine")

    monkeypatch.setattr(echo, "preflight", claimed)
    free_the_lane(client, auth, holder)
    wait_for(lambda: len(asked) >= 2, "the pump to try twice")
    assert status(client, auth, patient) == "queued"
    monkeypatch.setattr(echo, "preflight", lambda model, params: None)
    wait_for(lambda: status(client, auth, patient) == "done", "the job to run")


def test_an_item_expires_at_max_wait_and_when_abandoned(
    client: TestClient, auth: dict[str, str]
) -> None:
    line = client.app.state.line
    holder = occupy_the_lane(client, auth)
    short = queue_up(client, as_client(auth, "a"), max_wait_s=10)["job_id"]
    long = queue_up(client, as_client(auth, "b"))["job_id"]
    beat = client.post(f"/v1/queue/{long}/heartbeat", headers=auth)
    assert beat.status_code == 200 and beat.json()["position"] == 2

    gone = line.expire(clock.now() + timedelta(seconds=11))
    assert [item.job.id for item in gone] == [short]
    assert client.get(f"/v1/jobs/{short}", headers=auth).json()["removal"]["reason"] == (
        "expired"
    )
    assert line.expire(clock.now() + timedelta(seconds=60)) == []
    later = clock.now() + timedelta(seconds=line_module.ABANDON_AFTER_S + 1)
    gone = line.expire(later)
    assert [item.job.id for item in gone] == [long]
    assert "nobody followed it" in client.get(
        f"/v1/jobs/{long}", headers=auth
    ).json()["removal"]["message"]
    free_the_lane(client, auth, holder)


def test_a_followed_item_is_not_abandoned(
    client: TestClient, auth: dict[str, str]
) -> None:
    line, store = client.app.state.line, client.app.state.store
    holder = occupy_the_lane(client, auth)
    waiting = queue_up(client, as_client(auth, "bookforge"))["job_id"]
    waiter = store.subscribe(store.get(holder))
    store.get(holder).client = "bookforge"
    try:
        later = clock.now() + timedelta(seconds=line_module.ABANDON_AFTER_S + 1)
        assert line.expire(later) == []
    finally:
        store.unsubscribe(store.get(holder), waiter)
    assert status(client, auth, waiting) == "queued"
    free_the_lane(client, auth, holder)


def test_the_caps_refuse_by_name(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "PER_CLIENT_LIMIT", 1)
    monkeypatch.setattr(line_module, "TOTAL_LIMIT", 2)
    holder = occupy_the_lane(client, auth)
    queue_up(client, as_client(auth, "a"))
    refused = client.post("/v1/jobs", json=body(), headers=as_client(auth, "a"))
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "queue_full"
    assert refused.json()["error"]["details"]["scope"] == "client"
    queue_up(client, as_client(auth, "b"))
    refused = client.post("/v1/jobs", json=body(), headers=as_client(auth, "c"))
    assert refused.json()["error"]["details"]["scope"] == "server"
    free_the_lane(client, auth, holder)


@pytest.mark.parametrize("max_wait_s", [0, 5, 86_401])
def test_max_wait_out_of_range_is_refused(
    client: TestClient, auth: dict[str, str], max_wait_s: int
) -> None:
    refused = client.post(
        "/v1/jobs", json=body(queue={"max_wait_s": max_wait_s}), headers=auth
    )
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_request"


def test_a_restart_ends_waiting_jobs_removed_and_says_so_after(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client() as first:
        holder = occupy_the_lane(first, auth)
        waiting = queue_up(first, auth)["job_id"]
        store = first.app.state.store
    job = store.get(waiting)
    assert job.status == "removed"
    assert job.removal["reason"] == "server_restart"
    record = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert (record["status"], record["waiting"]) == ("removed", None)
    assert store.get(holder).status == "interrupted"

    with make_client() as second:
        state = second.get(f"/v1/jobs/{waiting}", headers=auth).json()
        assert state["status"] == "removed"
        assert state["removal"]["reason"] == "server_restart"
        assert events(second, auth, waiting)[-1]["event"] == "removed"


def test_a_crash_leaves_waiting_records_that_come_back_removed(tmp_path: Path) -> None:
    from crucible.jobs.queue import JobStore

    class Config:
        jobs_dir = tmp_path
        retention_days = 7

    directory = tmp_path / "abc123"
    (directory / "inputs").mkdir(parents=True)
    (directory / "inputs" / "x.bin").write_bytes(b"x")
    (directory / "job.json").write_text(json.dumps({
        "job_id": "abc123", "type": "echo", "status": "queued",
        "created": "2026-09-30T10:00:00+00:00",
        "waiting": {"max_wait_s": 3600, "submitted": "2026-09-30T10:00:00+00:00"},
    }), encoding="utf-8")
    store = JobStore(Config(), backend=None, registry={})
    store.restore()
    job = store.get("abc123")
    assert job.status == "removed"
    assert job.removal["reason"] == "server_restart"
    assert job.error is None
    assert not (directory / "inputs").exists()
    assert job.events[-1]["event"] == "removed"


def test_the_queue_stream_opens_with_a_snapshot_then_says_every_change(
    client: TestClient, auth: dict[str, str]
) -> None:
    line = client.app.state.line
    holder = occupy_the_lane(client, auth)
    first = queue_up(client, auth)["job_id"]
    feed = client.portal.call(_open_feed, line)
    opening = feed.after(0)
    assert opening[0][1]["event"] == "snapshot"
    assert [row["job_id"] for row in opening[0][1]["data"]["items"]] == [first]
    cursor = opening[-1][0]
    second = queue_up(client, auth)["job_id"]
    client.delete(f"/v1/queue/{first}", headers=auth)
    later = [event for _, event in feed.after(cursor)]
    assert [(e["event"], e["data"]["job_id"]) for e in later] == [
        ("added", second), ("removed", first), ("moved", second)
    ]
    assert later[1]["data"]["reason"] == "operator"
    assert later[2]["data"]["position"] == 1
    feed.close()
    free_the_lane(client, auth, holder)


async def _open_feed(line: Any) -> Any:
    return sse._queue_feed(line)
