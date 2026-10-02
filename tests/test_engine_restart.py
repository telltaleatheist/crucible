from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible import tasks
from crucible.platform.paths import HOST_DOOR_ENV
from crucible.tasks import hostdoor

from .test_engine_task import (  # noqa: F401 - a fixture this module uses
    FakeDoor,
    door,
    events,
    wait_for,
    windows_client,
)


def restart(client: TestClient, auth: dict[str, str]) -> Any:
    return client.post("/v1/tasks", headers=auth, json={"type": "engine-restart"})


def test_a_server_no_orchestrator_started_is_refused_by_its_OWN_name(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(HOST_DOOR_ENV, raising=False)
    answer = restart(client, auth)
    assert answer.status_code == 409
    error = answer.json()["error"]
    assert error["code"] == hostdoor.ENGINE_RESTART_NEEDS_ORCHESTRATOR
    assert error["details"]["env"] == HOST_DOOR_ENV
    assert "move" not in error["message"].lower()


def test_a_restart_takes_no_fields_at_all(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOST_DOOR_ENV, "http://127.0.0.1:7101")
    refused = client.post(
        "/v1/tasks", headers=auth, json={"type": "engine-restart", "target": "wsl"}
    )
    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "invalid_request"


def test_any_backend_may_be_restarted_unlike_the_move(
    client: TestClient, auth: dict[str, str], door: FakeDoor  # noqa: F811
) -> None:
    import os

    door.script = [("done", {"engine": "http://127.0.0.1:7100"})]
    os.environ[HOST_DOOR_ENV] = door.url
    try:
        answer = restart(client, auth)
        assert answer.status_code == 202, answer.text
        task = wait_for(client, auth, answer.json()["task_id"])
    finally:
        os.environ.pop(HOST_DOOR_ENV, None)
    assert task["state"] == "done", task


def test_the_restart_goes_to_slash_restart_with_the_engines_bearer_and_no_body_fields(
    client: TestClient, auth: dict[str, str], door: FakeDoor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    from .conftest import TOKEN

    door.script = [("done", {})]
    monkeypatch.setenv(HOST_DOOR_ENV, door.url)
    answer = restart(client, auth)
    wait_for(client, auth, answer.json()["task_id"])
    assert len(door.seen) == 1
    assert door.seen[0]["path"] == hostdoor.HOST_DOOR_RESTART_PATH
    assert door.seen[0]["authorization"] == f"Bearer {TOKEN}"
    assert door.seen[0]["body"] == {}


def test_the_orchestrators_events_arrive_UNALTERED_under_this_tasks_id(
    client: TestClient, auth: dict[str, str], door: FakeDoor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    door.script = [
        ("step", {"name": "restart the guest's unit", "index": 1, "total": 2}),
        ("step", {"name": "wait for /v1/ping", "index": 2, "total": 2}),
        ("done", {"engine": "http://127.0.0.1:7100"}),
    ]
    monkeypatch.setenv(HOST_DOOR_ENV, door.url)
    answer = restart(client, auth)
    task_id = answer.json()["task_id"]
    wait_for(client, auth, task_id)
    seen = events(client, auth, task_id)
    steps = [e["data"]["name"] for e in seen if e["event"] == "step"]
    assert steps == [
        "hand the restart to the orchestrator",
        "restart the guest's unit",
        "wait for /v1/ping",
    ]
    assert seen[-1]["event"] == "done"


def test_engine_not_ours_comes_back_with_the_orchestrators_OWN_code(
    client: TestClient, auth: dict[str, str], door: FakeDoor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    door.refusal = (409, "engine_not_ours", "watched and never acted on")
    monkeypatch.setenv(HOST_DOOR_ENV, door.url)
    answer = restart(client, auth)
    task = wait_for(client, auth, answer.json()["task_id"])
    assert task["state"] == "failed"
    assert task["error"]["code"] == "engine_not_ours"


def test_a_failed_event_from_the_orchestrator_is_NOT_described_twice(
    client: TestClient, auth: dict[str, str], door: FakeDoor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    door.script = [
        ("step", {"name": "restart the guest's unit", "index": 1, "total": 2}),
        ("failed", {"code": "engine_did_not_return", "message": "nothing answered"}),
    ]
    monkeypatch.setenv(HOST_DOOR_ENV, door.url)
    answer = restart(client, auth)
    task_id = answer.json()["task_id"]
    task = wait_for(client, auth, task_id)
    assert task["state"] == "failed"
    failures = [e for e in events(client, auth, task_id) if e["event"] == "failed"]
    assert len(failures) == 1, "two sentences about one failure in one place"
    assert failures[0]["data"]["code"] == "engine_did_not_return"


def test_a_stream_that_just_STOPS_is_the_expected_shape_and_says_to_read_info(
    client: TestClient, auth: dict[str, str], door: FakeDoor, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    door.script = [("step", {"name": "restart the guest's unit", "index": 1, "total": 2})]
    door.cut_after = 1
    monkeypatch.setenv(HOST_DOOR_ENV, door.url)
    answer = restart(client, auth)
    task = wait_for(client, auth, answer.json()["task_id"])
    assert task["state"] == "failed"
    assert task["error"]["code"] == hostdoor.HOST_INSTALL_FAILED
    assert "/v1/info" in task["error"]["message"]


def test_an_orchestrator_whose_door_is_dead_is_host_unreachable(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOST_DOOR_ENV, "http://127.0.0.1:1")
    answer = restart(client, auth)
    assert answer.status_code == 202, answer.text
    task = wait_for(client, auth, answer.json()["task_id"])
    assert task["error"]["code"] == hostdoor.HOST_UNREACHABLE


def test_engine_restart_is_a_task_type_and_the_vocabulary_is_closed(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert "engine-restart" in tasks.TASK_TYPES
    refused = client.post("/v1/tasks", headers=auth, json={"type": "engine-reboot"})
    assert refused.status_code == 400, refused.text
