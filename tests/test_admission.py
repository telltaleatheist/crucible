from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.api.routes.jobs import _job_state
from crucible.errors import ApiError
from crucible.jobs.queue import JobStore

PAYLOAD = {"x.bin": {"inline_base64": base64.b64encode(b"admission").decode("ascii")}}

HELD_MS = 4_000


def job_body(**params: Any) -> dict[str, Any]:
    return {"type": "echo", "params": params, "inputs": dict(PAYLOAD)}


def submit(
    client: TestClient, auth: dict[str, str], **params: Any
) -> Any:
    return client.post("/v1/jobs", json=job_body(**params), headers=auth)


def admitted(client: TestClient, auth: dict[str, str], **params: Any) -> str:
    response = submit(client, auth, **params)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def wait_until_running(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        state = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if state["status"] == "running":
            return
        if state["status"] in ("done", "failed", "cancelled"):
            pytest.fail(f"job {job_id} finished before it could be observed: {state}")
        time.sleep(0.01)
    pytest.fail(f"job {job_id} never started running")


def wait_for_terminal(client: TestClient, auth: dict[str, str], job_id: str) -> str:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        status = client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"]
        if status in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never finished")


def test_a_second_submission_is_refused_and_not_queued(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)

        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "server_busy"

        activity = client.get("/v1/activity", headers=auth).json()
        assert activity["queued"] == []
        assert activity["slots"]["accelerated"]["queue_depth"] == 1
        assert client.get("/v1/health", headers=auth).json()["status"] == "busy"
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_refusal_names_the_holder_and_what_it_is_doing(
    client: TestClient, auth: dict[str, str]
) -> None:
    headers = {**auth, "User-Agent": "bookforge/owens-pc crucible-client/0.4.0"}
    response = client.post("/v1/jobs", json=job_body(delay_ms=HELD_MS), headers=headers)
    assert response.status_code == 202, response.text
    first = response.json()["job_id"]
    try:
        wait_until_running(client, auth, first)
        state = client.get(f"/v1/jobs/{first}", headers=auth).json()

        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409, refused.text
        error = refused.json()["error"]
        assert error["code"] == "server_busy"

        details = error["details"]
        assert details["door"] == "job"
        assert details["holder"] == "bookforge/owens-pc crucible-client/0.4.0"
        assert details["job_id"] == first
        assert details["type"] == "echo"
        assert details["model"] is None
        assert details["status"] == "running"
        assert details["since"] == state["started"]
        assert isinstance(details["progress"], float)
        assert isinstance(details["message"], str) and details["message"]

        assert first in error["message"]
        assert "bookforge/owens-pc" in error["message"]
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_holder_is_null_when_the_client_did_not_say(
    make_app: Callable[..., Any], auth: dict[str, str]
) -> None:
    with TestClient(make_app()) as client:
        response = client.post(
            "/v1/jobs",
            json=job_body(delay_ms=HELD_MS),
            headers={**auth, "User-Agent": ""},
        )
        assert response.status_code == 202, response.text
        first = response.json()["job_id"]
        try:
            wait_until_running(client, auth, first)
            refused = client.post(
                "/v1/jobs", json=job_body(delay_ms=0), headers={**auth, "User-Agent": ""}
            )
            assert refused.status_code == 409, refused.text
            error = refused.json()["error"]
            assert error["details"]["holder"] is None
            assert "an unnamed client" in error["message"]
        finally:
            client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_refusal_never_publishes_the_holders_params(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409
        assert "params" not in refused.json()["error"]["details"]
        assert "delay_ms" not in refused.text
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_echo_refuses_too_because_the_policy_is_the_lane_not_the_card(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409
        assert refused.json()["error"]["details"]["type"] == "echo"
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_an_unknown_type_is_still_refused_as_unknown_while_busy(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        response = client.post("/v1/jobs", json={"type": "summon"}, headers=auth)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "unknown_job_type"
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_lane_takes_the_next_job_the_moment_it_is_free(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=0)
    assert wait_for_terminal(client, auth, first) == "done"
    second = admitted(client, auth, delay_ms=0)
    assert wait_for_terminal(client, auth, second) == "done"


def test_a_cancelled_job_frees_the_lane(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = admitted(client, auth, delay_ms=HELD_MS)
    wait_until_running(client, auth, first)
    assert submit(client, auth, delay_ms=0).status_code == 409

    assert client.delete(f"/v1/jobs/{first}", headers=auth).status_code == 200
    assert wait_for_terminal(client, auth, first) == "cancelled"
    assert wait_for_terminal(client, auth, admitted(client, auth, delay_ms=0)) == "done"


def test_a_failed_job_frees_the_lane(
    client: TestClient, auth: dict[str, str]
) -> None:
    failed = client.post(
        "/v1/jobs", json={"type": "echo", "params": {}, "inputs": {}}, headers=auth
    )
    assert failed.status_code == 202, failed.text
    assert wait_for_terminal(client, auth, failed.json()["job_id"]) == "failed"
    assert wait_for_terminal(client, auth, admitted(client, auth, delay_ms=0)) == "done"


def test_a_refused_submission_leaves_nothing_behind(
    client: TestClient, auth: dict[str, str]
) -> None:
    jobs_dir = Path(client.app.state.config.jobs_dir)
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        before = sorted(p.name for p in jobs_dir.iterdir())

        for _ in range(5):
            assert submit(client, auth, delay_ms=0).status_code == 409

        assert sorted(p.name for p in jobs_dir.iterdir()) == before
        store: JobStore = client.app.state.store
        assert len(store._jobs) == 1
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_an_admitted_job_the_lane_has_not_reached_yet_still_refuses(
    client: TestClient, auth: dict[str, str]
) -> None:
    store = JobStore(
        client.app.state.config,
        client.app.state.backend,
        client.app.state.store.registry,
    )
    first = store.create("echo", None, {}, client="foundry/owens-pc")
    store.enqueue(first)

    assert store.running_id is None, "nothing is running: the lane was never started"
    assert first.status == "queued"
    assert store.position(first) == 1
    assert store.queue_depth == 1
    assert [job.id for job in store.queued()] == [first.id]

    with pytest.raises(ApiError) as caught:
        store.refuse_if_busy()
    assert caught.value.status_code == 409
    assert caught.value.code == "server_busy"
    details = caught.value.details
    assert details is not None
    assert details["door"] == "job"
    assert details["job_id"] == first.id
    assert details["status"] == "queued"
    assert details["holder"] == "foundry/owens-pc"
    assert details["since"] == first.created
    assert details["progress"] == 0.0


def test_a_job_type_s_done_extra_cannot_overwrite_the_hold_fields(
    client: TestClient,
) -> None:
    store: JobStore = client.app.state.store
    job = store.create("echo", None, {}, client="foundry/owens-pc")
    job.status = "done"
    job.held_by = "foundry/owens-pc"
    job.held_since = job.created
    job.done_extra = {
        "held_by": "somebody else",
        "held_since": "never",
        "resident": "the type's own news",
    }
    state = _job_state(store, job)
    assert state["held_by"] == "foundry/owens-pc"
    assert state["held_since"] == job.created
    assert state["resident"] == "the type's own news"


def test_enqueue_is_the_authority_and_refuses_on_its_own(
    client: TestClient, auth: dict[str, str]
) -> None:
    store = JobStore(
        client.app.state.config,
        client.app.state.backend,
        client.app.state.store.registry,
    )
    store.enqueue(store.create("echo", None, {}))
    second = store.create("echo", None, {})
    with pytest.raises(ApiError) as caught:
        store.enqueue(second)
    assert caught.value.code == "server_busy"
    assert store.queue_depth == 1, "the refused job must not be on the lane"


def test_a_discarded_job_is_forgotten_completely(
    client: TestClient, auth: dict[str, str]
) -> None:
    store = JobStore(
        client.app.state.config,
        client.app.state.backend,
        client.app.state.store.registry,
    )
    job = store.create("echo", None, {})
    assert job.dir.is_dir()
    store.discard(job)
    assert not job.dir.exists()
    with pytest.raises(ApiError) as caught:
        store.get(job.id)
    assert caught.value.code == "unknown_job"
