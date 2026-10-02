from __future__ import annotations

import base64
import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import API_VERSION, VERSION

from .test_residency import STUBBORN_PID, a_process_that_will_not_stop
from .test_tts_api import VOICE

PAYLOAD = {"x.bin": base64.b64encode(b"activity").decode("ascii")}


def job_body(**params: Any) -> dict[str, Any]:
    return {
        "type": "echo",
        "params": params,
        "inputs": {name: {"inline_base64": data} for name, data in PAYLOAD.items()},
    }


def submit(client: TestClient, auth: dict[str, str], **params: Any) -> str:
    response = client.post("/v1/jobs", json=job_body(**params), headers=auth)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def activity(client: TestClient, auth: dict[str, str], **query: Any) -> dict[str, Any]:
    response = client.get("/v1/activity", params=query, headers=auth)
    assert response.status_code == 200, response.text
    return response.json()


def wait_until(
    client: TestClient,
    auth: dict[str, str],
    predicate: Callable[[dict[str, Any]], bool],
    what: str,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = activity(client, auth)
        if predicate(last):
            return last
        time.sleep(0.02)
    pytest.fail(f"never saw {what} within {timeout_s}s; last activity was {last}")


def test_an_idle_server_reports_itself_and_nothing_else(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = activity(client, auth)

    assert body["server"]["name"] == "crucible@test"
    assert body["server"]["version"] == VERSION
    assert body["server"]["api_version"] == API_VERSION
    assert body["server"]["uptime_s"] >= 0.0

    assert body["resident"] is None
    assert body["warming"] is None
    assert body["running"] == []
    assert body["queued"] == []
    assert body["claim"] is None
    assert body["streaming"] is None
    assert body["chat"] == {
        "in_flight": 0,
        "max_in_flight": None,
        "max_in_flight_basis": None,
        "rows": [],
    }
    assert body["slots"]["accelerated"] == {
        "busy": 0,
        "of": 1,
        "queue_depth": 0,
        "accepts_work": True,
    }


def test_the_probe_is_off_unless_it_is_asked_for(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert "accelerator" not in activity(client, auth)
    assert "accelerator" in activity(client, auth, accelerator_probe=True)


def test_it_needs_the_token_and_the_version_header_like_every_private_route(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert client.get("/v1/activity").status_code == 401
    assert (
        client.get(
            "/v1/activity", headers={"Authorization": auth["Authorization"]}
        ).status_code
        == 426
    )


def test_a_running_job_fills_the_slot_and_says_what_it_is_doing(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=4_000)
    try:
        body = wait_until(
            client, auth, lambda a: a["running"], f"job {job_id} running"
        )

        assert body["slots"]["accelerated"]["busy"] == 1
        assert body["slots"]["accelerated"]["of"] == 1

        row = body["running"][0]
        assert row["job_id"] == job_id
        assert row["type"] == "echo"
        assert row["status"] == "running"
        assert row["position"] == 0
        assert row["started"] is not None
        assert isinstance(row["message"], str) and row["message"]
    finally:
        client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_the_bench_shows_one_job_and_nothing_waiting_behind_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = submit(client, auth, delay_ms=4_000)
    try:
        body = wait_until(client, auth, lambda a: a["running"], f"job {first} running")
        assert [row["job_id"] for row in body["running"]] == [first]
        assert body["queued"] == []
        assert body["slots"]["accelerated"] == {
            "busy": 1,
            "of": 1,
            "queue_depth": 1,
            "accepts_work": False,
        }

        refused = client.post(
            "/v1/jobs", json={**job_body(delay_ms=0), "queue": False}, headers=auth
        )
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["details"]["job_id"] == first

        after = activity(client, auth)
        assert after["queued"] == []
        assert after["slots"]["accelerated"]["queue_depth"] == 1
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_a_finished_job_leaves_the_bench(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    body = wait_until(
        client,
        auth,
        lambda a: not a["running"] and not a["queued"],
        f"job {job_id} to leave the bench",
    )
    assert body["slots"]["accelerated"]["busy"] == 0
    assert client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "done"


def test_the_job_records_who_submitted_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    headers = {**auth, "User-Agent": "bookforge/owens-pc crucible-client/0.4.0"}
    response = client.post("/v1/jobs", json=job_body(delay_ms=4_000), headers=headers)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    try:
        body = wait_until(client, auth, lambda a: a["running"], "the job running")
        assert body["running"][0]["client"] == "bookforge/owens-pc crucible-client/0.4.0"
    finally:
        client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_a_client_that_sends_no_user_agent_is_null_and_not_a_guess(
    make_app: Callable[..., Any], auth: dict[str, str]
) -> None:
    with TestClient(make_app()) as client:
        response = client.post(
            "/v1/jobs", json=job_body(delay_ms=4_000), headers={**auth, "User-Agent": ""}
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        try:
            body = wait_until(client, auth, lambda a: a["running"], "the job running")
            assert body["running"][0]["client"] is None
        finally:
            client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_it_never_publishes_a_jobs_params(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=4_000)
    try:
        body = wait_until(client, auth, lambda a: a["running"], "the job running")
        assert "params" not in body["running"][0]
        assert "delay_ms" not in repr(body["running"][0])
    finally:
        client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_a_stubborn_engine_shows_on_the_bench_read(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert activity(client, auth)["stopping"] is None

    with a_process_that_will_not_stop(client.app.state.residency):
        body = activity(client, auth)
        assert body["resident"] is None
        assert body["stopping"] == {
            "kind": "tts",
            "id": VOICE,
            "since": body["stopping"]["since"],
            "pids": [STUBBORN_PID],
        }
        assert body["stopping"]["since"].startswith("20")
        assert body["slots"]["accelerated"]["accepts_work"] is True


def test_health_says_it_too_and_the_two_reads_cannot_differ(
    client: TestClient, auth: dict[str, str]
) -> None:
    health = client.get("/v1/health", headers=auth).json()
    assert health["stopping"] is None

    with a_process_that_will_not_stop(client.app.state.residency):
        health = client.get("/v1/health", headers=auth).json()
        assert health["stopping"] == activity(client, auth)["stopping"]
        assert health["stopping"]["pids"] == [STUBBORN_PID]
        assert health["status"] == "ok"
