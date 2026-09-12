"""The job framework, end to end, through the echo job type."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import VERSION

from .conftest import parse_sse

ALPHA = b"the quick brown fox\x00\x01\x02"
BETA = b"jumped over the lazy dog" * 64


def submit(
    client: TestClient, auth: dict[str, str], inputs: dict[str, bytes], **params: Any
) -> str:
    body = {
        "type": "echo",
        "params": params,
        "inputs": {
            name: {"inline_base64": base64.b64encode(data).decode("ascii")}
            for name, data in inputs.items()
        },
    }
    response = client.post("/v1/jobs", json=body, headers=auth)
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def stream_events(
    client: TestClient, auth: dict[str, str], job_id: str, **headers: str
) -> list[dict[str, Any]]:
    merged = dict(auth)
    merged.update(headers)
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=merged) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return parse_sse(response.iter_lines())


def wait_for_terminal(
    client: TestClient, auth: dict[str, str], job_id: str, timeout: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if state["status"] in ("done", "failed", "cancelled"):
            return state
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_echo_end_to_end(client: TestClient, auth: dict[str, str]) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA, "beta.bin": BETA}, delay_ms=80)

    events = stream_events(client, auth, job_id)
    kinds = [event["event"] for event in events]

    # Order is fixed: queued, the worker's start, then per input a progress and its
    # artifact, then the final progress, then done.
    assert kinds == [
        "queued",
        "progress",
        "progress",
        "artifact",
        "progress",
        "artifact",
        "progress",
        "done",
    ]
    assert [event["id"] for event in events] == list(range(1, len(events) + 1))
    assert [e["data"]["name"] for e in events if e["event"] == "artifact"] == [
        "alpha.bin",
        "beta.bin",
    ]
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert events[-1]["data"]["artifacts"] == ["alpha.bin", "beta.bin"]

    state = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert state["status"] == "done"
    assert state["progress"] == 1.0
    assert state["position"] is None
    assert state["error"] is None
    assert state["artifacts"] == ["alpha.bin", "beta.bin"]
    assert state["started"] is not None and state["finished"] is not None

    for name, expected in (("alpha.bin", ALPHA), ("beta.bin", BETA)):
        fetched = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
        assert fetched.status_code == 200
        assert fetched.content == expected

        sidecar = client.get(
            f"/v1/jobs/{job_id}/artifacts/{name}.provenance.json", headers=auth
        )
        assert sidecar.status_code == 200
        provenance = json.loads(sidecar.content)
        assert provenance["server"] == {"name": "crucible@test", "version": VERSION}
        assert provenance["backend"] == "cuda-linux"
        assert provenance["job_type"] == "echo"
        assert provenance["model"] is None
        assert provenance["params"] == {"delay_ms": 80}
        assert provenance["started"] == state["started"]
        assert provenance["finished"] == state["finished"]


def test_events_replay_from_last_event_id(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA}, delay_ms=0)
    wait_for_terminal(client, auth, job_id)

    everything = stream_events(client, auth, job_id)
    assert len(everything) >= 4

    resumed = stream_events(client, auth, job_id, **{"Last-Event-ID": "3"})
    assert [event["id"] for event in resumed] == [
        event["id"] for event in everything if event["id"] > 3
    ]
    assert resumed[-1]["event"] == "done"


def test_bad_last_event_id_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA}, delay_ms=0)
    headers = dict(auth)
    headers["Last-Event-ID"] = "banana"
    response = client.get(f"/v1/jobs/{job_id}/events", headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_last_event_id"


def test_cancel_a_running_job(client: TestClient, auth: dict[str, str]) -> None:
    inputs = {f"chunk{index}.bin": ALPHA for index in range(8)}
    job_id = submit(client, auth, inputs, delay_ms=400)

    # Wait until the lane has actually picked it up, then cancel.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the job never started running")

    cancelled = client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in ("cancelling", "cancelled")

    state = wait_for_terminal(client, auth, job_id)
    assert state["status"] == "cancelled"
    assert len(state["artifacts"]) < len(inputs)

    events = stream_events(client, auth, job_id)
    assert events[-1]["event"] == "cancelled"


def test_cancel_a_finished_job_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA}, delay_ms=0)
    wait_for_terminal(client, auth, job_id)
    response = client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_cancellable"


def test_upload_then_reference_the_blob(
    client: TestClient, auth: dict[str, str]
) -> None:
    uploaded = client.post(
        "/v1/uploads", files={"file": ("beta.bin", BETA)}, headers=auth
    )
    assert uploaded.status_code == 201
    blob = uploaded.json()
    assert blob["bytes"] == len(BETA)
    assert len(blob["sha256"]) == 64

    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"beta.bin": {"blob_id": blob["blob_id"]}},
        },
        headers=auth,
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert wait_for_terminal(client, auth, job_id)["status"] == "done"
    fetched = client.get(f"/v1/jobs/{job_id}/artifacts/beta.bin", headers=auth)
    assert fetched.content == BETA


def test_unknown_blob_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/v1/jobs",
        json={"type": "echo", "inputs": {"x.bin": {"blob_id": "deadbeef"}}},
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_blob"


def test_input_with_both_sources_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo",
            "inputs": {"x.bin": {"blob_id": "a", "inline_base64": "AA=="}},
        },
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_unknown_job_type_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post("/v1/jobs", json={"type": "summon"}, headers=auth)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_job_type"
    assert "summon" in error["message"]


def test_echo_disabled_is_refused_by_name(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_echo=False) as client:
        response = client.post("/v1/jobs", json={"type": "echo"}, headers=auth)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "job_type_disabled"
        assert "enable_echo" in error["message"]


def test_model_on_a_modelless_type_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/jobs", json={"type": "echo", "model": "qwen3.5:9b"}, headers=auth
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert "qwen3.5:9b" in error["message"]


def test_unknown_echo_param_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA}, delya_ms=5)
    state = wait_for_terminal(client, auth, job_id)
    assert state["status"] == "failed"
    assert "delya_ms" in state["error"]["message"]


def test_echo_without_inputs_fails_by_name(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, {})
    state = wait_for_terminal(client, auth, job_id)
    assert state["status"] == "failed"
    assert state["error"]["code"] == "no_inputs"


def test_unknown_job_is_404(client: TestClient, auth: dict[str, str]) -> None:
    response = client.get("/v1/jobs/nope", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_job"


def test_artifact_traversal_is_refused(client: TestClient, auth: dict[str, str]) -> None:
    job_id = submit(client, auth, {"alpha.bin": ALPHA}, delay_ms=0)
    wait_for_terminal(client, auth, job_id)
    # %2E%2E survives httpx's URL normalisation and reaches the route as "..".
    response = client.get(f"/v1/jobs/{job_id}/artifacts/%2E%2E", headers=auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_artifact_name"


def test_queue_runs_one_at_a_time(client: TestClient, auth: dict[str, str]) -> None:
    first = submit(client, auth, {"alpha.bin": ALPHA}, delay_ms=200)
    second = submit(client, auth, {"beta.bin": BETA}, delay_ms=0)

    # While the first runs, the second must be queued behind it with a position.
    deadline = time.monotonic() + 10.0
    saw_queued_behind = False
    while time.monotonic() < deadline:
        second_state = client.get(f"/v1/jobs/{second}", headers=auth).json()
        if second_state["status"] == "queued" and second_state["position"] == 1:
            saw_queued_behind = True
        if second_state["status"] != "queued":
            break
        time.sleep(0.01)
    assert saw_queued_behind, "the second job never queued behind the first"

    assert wait_for_terminal(client, auth, first)["status"] == "done"
    assert wait_for_terminal(client, auth, second)["status"] == "done"


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden"])
def test_bad_input_names_are_refused(
    client: TestClient, auth: dict[str, str], name: str
) -> None:
    response = client.post(
        "/v1/jobs",
        json={"type": "echo", "inputs": {name: {"inline_base64": "AA=="}}},
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input_name"
