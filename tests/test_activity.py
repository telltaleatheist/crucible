"""`GET /v1/activity` — the whole-server read. PHASE7-LANES.md section 5.

THE TIMING PROBLEM, AND HOW THESE TESTS AVOID IT. A job's life is running →
done, and a test that submits and then reads is racing the lane. Every test here
that needs to see a job IN FLIGHT uses a long `delay_ms` and polls for the state
it wants with a deadline, rather than sleeping a guessed amount and asserting:
a sleep that is too short fails on a loaded machine, and a sleep that is long
enough to be safe makes the suite slow for everybody. The deadline is generous
and the failure message says what it was waiting for.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import API_VERSION, VERSION

from .test_residency import STUBBORN_PID, a_process_that_will_not_stop
from .test_tts_api import VOICE


#: `echo` refuses `no_inputs`, so every submission here carries one. It is not
#: what any of these tests are about — they are about what the bench sees — but
#: a job that fails in its first millisecond is never observed running, which is
#: exactly the shape of failure this cost an hour to.
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


# --------------------------------------------------------------- an idle server


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
    # Nobody holds narrator's wire and nobody is streaming. Both keys are PRESENT
    # and null: a key that is absent would mean "this build does not speak the
    # field", which is a different piece of news from "nothing is happening".
    assert body["claim"] is None
    assert body["streaming"] is None
    assert body["chat"] == {"in_flight": 0, "rows": []}
    assert body["slots"]["accelerated"] == {
        "busy": 0,
        "of": 1,
        "queue_depth": 0,
        "accepts_work": True,
    }


def test_the_probe_is_off_unless_it_is_asked_for(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The whole reason this route is cheap enough to poll (section 5)."""
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


# ------------------------------------------------------------- a job in flight


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
        # The last `progress` line, kept on the job so this route need not
        # replay an event log.
        assert isinstance(row["message"], str) and row["message"]
    finally:
        client.delete(f"/v1/jobs/{job_id}", headers=auth)


def test_the_bench_shows_one_job_and_nothing_waiting_behind_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Nothing queues, so the bench never draws a line.

    This test used to submit two jobs and assert the second appeared in `queued`
    at `position: 1` with `queue_depth: 2`. Owen ruled that policy out on
    2026-09-13 (ARCHITECTURE.md section 3): the second submission is refused
    `server_busy`, so there is no second row for this route to draw and the
    intent behind the old assertion — *the bench shows contention honestly* — is
    now served by the running row plus the refusal the other client got.

    `queued` and `position` are NOT removed and are still rendered: the window
    between admission and the lane picking a job up is real, if sub-millisecond.
    It is exercised where it can be held still, on a store with no lane running
    (`tests/test_admission.py`).
    """
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

        refused = client.post("/v1/jobs", json=job_body(delay_ms=0), headers=auth)
        assert refused.status_code == 409, refused.text
        # The bench and the refusal agree about who has the card — one fact, and
        # the refusal is what a client that never polls this route still learns.
        assert refused.json()["error"]["details"]["job_id"] == first

        after = activity(client, auth)
        assert after["queued"] == []
        assert after["slots"]["accelerated"]["queue_depth"] == 1
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_a_finished_job_leaves_the_bench(
    client: TestClient, auth: dict[str, str]
) -> None:
    """It reports what is HAPPENING. History is `GET /v1/jobs/{id}`."""
    job_id = submit(client, auth, delay_ms=0)
    body = wait_until(
        client,
        auth,
        lambda a: not a["running"] and not a["queued"],
        f"job {job_id} to leave the bench",
    )
    assert body["slots"]["accelerated"]["busy"] == 0
    assert client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "done"


# -------------------------------------------------------------------- the client


def test_the_job_records_who_submitted_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Two BookForge instances can point at one server, and a bench that drew a
    foreign render as its own would offer a cancel button for the other
    machine's chapter. PHASE7-LANES.md section 5."""
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
    """`None` means "it did not say". Inventing a name here would make the bench
    confidently wrong about whose render is on the card."""
    # TestClient sets a User-Agent of its own, so it is cleared explicitly —
    # httpx drops a header whose value is None.
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
    """A chat prompt or a chapter of somebody's book is not something a
    whole-server read should hand to anyone holding the token."""
    job_id = submit(client, auth, delay_ms=4_000)
    try:
        body = wait_until(client, auth, lambda a: a["running"], "the job running")
        assert "params" not in body["running"][0]
        assert "delay_ms" not in repr(body["running"][0])
    finally:
        client.delete(f"/v1/jobs/{job_id}", headers=auth)


# ------------------------------------------ what was told to go and has not


def test_a_stubborn_engine_shows_on_the_bench_read(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Ledger R13. `resident` goes null the moment the stop is asked for, so a
    bench that read only that drew an idle machine which refuses everything —
    and nothing in Crucible ever clears the state, because a killed CUDA
    process wedges WSL2 until Windows reboots. The pids are published because
    ending it is a thing a human does with them."""
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
        # The lane is genuinely free and still says so. `accepts_work` was
        # never the question this field answers.
        assert body["slots"]["accelerated"]["accepts_work"] is True


def test_health_says_it_too_and_the_two_reads_cannot_differ(
    client: TestClient, auth: dict[str, str]
) -> None:
    """One object (`DyingResident.to_dict`) behind both routes."""
    health = client.get("/v1/health", headers=auth).json()
    assert health["stopping"] is None

    with a_process_that_will_not_stop(client.app.state.residency):
        health = client.get("/v1/health", headers=auth).json()
        assert health["stopping"] == activity(client, auth)["stopping"]
        assert health["stopping"]["pids"] == [STUBBORN_PID]
        # `status` reports the LANE and is untouched: `ok` there has always
        # meant "no job is running", never "the card is free".
        assert health["status"] == "ok"
