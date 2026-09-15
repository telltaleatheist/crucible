"""`POST /v1/tasks {"type": "engine", "target": "wsl"}` — PHASE15-HOST.md 4.7.

THE SERVER DOES NOT RUN THE MOVE. Only the host can run `wsl.exe`, prompt for
administrator and survive the reboot, so the Windows server hands the task to
the host's loopback door and RELAYS the host's events under its own task id.
A server doing it itself would stop halfway through and take its own event
stream with it.

So what is tested here is the seam and nothing else: the three refusals, the
relay being a relay (the host's event shapes arrive unaltered, because
`crucible/host/installer.py`'s `Event` was written to be this module's
shape), and the two ways a stream can end badly. The door is a fake one — a
`http.server` in this process that speaks the real ndjson — because the real
one is a tray process on Windows and this suite runs in WSL.

NOTHING HERE STARTS A DISTRO, A SERVER OR A MODEL.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import tasks

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, TOKEN


# ------------------------------------------------------------ the fake door


class FakeDoor:
    """The host's `POST /install`, as far as the server can tell.

    It speaks the real wire: a bearer, a `{"target": ...}` body, and
    newline-delimited JSON out, flushed per line. What it fakes is the
    sequence behind it, which is the host's and is tested in
    `tests/test_host.py`.
    """

    def __init__(self) -> None:
        #: The lines this door will emit, as `(event, data)`.
        self.script: list[tuple[str, dict[str, Any]]] = []
        #: Refuse instead, as `(status, code, message)`.
        self.refusal: tuple[int, str, str] | None = None
        #: Stop the stream after this many lines, without a terminal event.
        self.cut_after: int | None = None
        #: What arrived, so a test can see the bearer and the body.
        self.seen: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeDoor":
        door = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                door.seen.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(raw.decode("utf-8") or "{}"),
                    }
                )
                if door.refusal is not None:
                    status, code, message = door.refusal
                    body = json.dumps(
                        {"error": {"code": code, "message": message}}
                    ).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for index, (event, data) in enumerate(door.script, start=1):
                    if door.cut_after is not None and index > door.cut_after:
                        break
                    line = json.dumps({"id": index, "event": event, "data": data})
                    self.wfile.write(line.encode("utf-8") + b"\n")
                    self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self._server.serve_forever, daemon=True, name="fake-host-door"
        ).start()
        return self

    def __exit__(self, *_exc: object) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def door() -> Iterator[FakeDoor]:
    with FakeDoor() as fake:
        yield fake


def windows_client(
    make_client: Callable[..., TestClient], monkeypatch: pytest.MonkeyPatch, door_url: str
) -> TestClient:
    """A `llama-windows` server that a host started."""
    from dataclasses import replace

    windows = replace(FAKE_BACKEND, kind="llama-windows", platform="windows")
    monkeypatch.setenv(tasks.HOST_DOOR_ENV, door_url)
    return make_client(backend=windows)


def wait_for(client: TestClient, auth: dict[str, str], task_id: str) -> dict[str, Any]:
    """Read the task until it leaves `running`. The relay is a thread."""
    import time

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        body = client.get(f"/v1/tasks/{task_id}", headers=auth).json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never finished: {body}")


def events(client: TestClient, auth: dict[str, str], task_id: str) -> list[dict[str, Any]]:
    stream = client.get(f"/v1/tasks/{task_id}/events", headers=auth)
    assert stream.status_code == 200, stream.text
    found: list[dict[str, Any]] = []
    name: str | None = None
    for line in stream.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :].strip()
        elif line.startswith("data: ") and name is not None:
            found.append({"event": name, "data": json.loads(line[len("data: ") :])})
    return found


# ------------------------------------------------------------- the refusals


def test_off_win32_it_is_engine_move_not_here(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cuda-linux server has nothing to move FROM."""
    monkeypatch.setenv(tasks.HOST_DOOR_ENV, "http://127.0.0.1:7101")
    response = client.post(
        "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "engine_move_not_here"
    assert error["details"]["backend"] == FAKE_BACKEND.kind


def test_the_mac_gets_the_same_refusal(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tasks.HOST_DOOR_ENV, "http://127.0.0.1:7101")
    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        response = mac.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
    assert response.json()["error"]["code"] == "engine_move_not_here"


def test_a_server_no_host_started_is_engine_move_needs_host(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4.7, verbatim: a developer running `crucible serve` by hand.

    The fact is STATED by the host (`CRUCIBLE_HOST_DOOR`) and never probed
    for: 127.0.0.1:7101 can be answered by something that is not a host, and
    a host restarting its own door is still the host.
    """
    from dataclasses import replace

    monkeypatch.delenv(tasks.HOST_DOOR_ENV, raising=False)
    windows = replace(FAKE_BACKEND, kind="llama-windows", platform="windows")
    with make_client(backend=windows) as server:
        response = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "engine_move_needs_host"
    assert error["details"]["env"] == "CRUCIBLE_HOST_DOOR"


def test_the_reverse_move_is_refused_by_name_rather_than_half_done(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    with windows_client(make_client, monkeypatch, door.url) as server:
        response = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "windows"}
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "engine_target_unknown"
    assert error["details"]["targets"] == ["wsl"]


def test_the_request_is_checked_before_the_machine(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An impossible target is impossible on every machine."""
    monkeypatch.delenv(tasks.HOST_DOOR_ENV, raising=False)
    response = client.post(
        "/v1/tasks", headers=auth, json={"type": "engine", "target": "amiga"}
    )
    assert response.json()["error"]["code"] == "engine_target_unknown"


def test_an_engine_task_may_not_carry_another_type_s_fields(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/tasks",
        headers=auth,
        json={"type": "engine", "target": "wsl", "job_type": "llm"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# ----------------------------------------------------------------- the relay


def test_the_host_s_events_arrive_UNALTERED_under_this_task_s_id(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    """*"A relay that reshapes is a second owner of the shape."*"""
    door.script = [
        ("state", {"code": "distro_absent", "message": "importing the guest"}),
        ("step", {"name": "server pack", "index": 2, "total": 9}),
        ("progress", {"bytes_done": 10, "bytes_total": 100, "file": "pack"}),
        ("done", {"steps": 9}),
    ]
    with windows_client(make_client, monkeypatch, door.url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        assert accepted.status_code == 202, accepted.text
        task_id = accepted.json()["task_id"]
        finished = wait_for(server, auth, task_id)
        seen = events(server, auth, task_id)

    assert finished["state"] == "done"
    names = [row["event"] for row in seen]
    # `started` is this server's own, then the host's four, then `done`.
    assert names[0] == "started"
    assert names[1] == "step"  # "hand the move to the host"
    assert "state" in names and "progress" in names
    relayed = [row for row in seen if row["event"] == "progress"][0]
    assert relayed["data"] == {"bytes_done": 10, "bytes_total": 100, "file": "pack"}
    state_row = [row for row in seen if row["event"] == "state"][0]
    assert state_row["data"]["code"] == "distro_absent"


def test_the_door_is_given_the_ENGINE_s_token_and_the_target(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    """The bearer is the engine's own, which the server already holds.

    A second copy in an environment variable would be a secret with two
    owners and one more place for it to be stale.
    """
    door.script = [("done", {})]
    with windows_client(make_client, monkeypatch, door.url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        wait_for(server, auth, accepted.json()["task_id"])
    assert len(door.seen) == 1
    call = door.seen[0]
    assert call["path"] == "/install"
    assert call["authorization"] == f"Bearer {TOKEN}"
    assert call["body"] == {"target": "wsl"}


def test_a_host_failure_fails_the_task_and_says_it_ONCE(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    """The host's own `failed` event is the sentence a person reads.

    A second description of one failure written by the relay would put two
    accounts of it in one stream.
    """
    door.script = [
        ("step", {"name": "wsl --install", "index": 1, "total": 9}),
        (
            "failed",
            {"code": "wsl_needs_reboot", "message": "reboot, then Crucible continues"},
        ),
    ]
    with windows_client(make_client, monkeypatch, door.url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        task_id = accepted.json()["task_id"]
        finished = wait_for(server, auth, task_id)
        seen = events(server, auth, task_id)

    assert finished["state"] == "failed"
    failures = [row for row in seen if row["event"] == "failed"]
    assert len(failures) == 1, "one failure, one sentence"
    assert failures[0]["data"]["code"] == "wsl_needs_reboot"


def test_a_stream_that_ends_with_no_terminal_event_is_a_FAILURE(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    """Nothing here can tell a completed install from an abandoned one."""
    door.script = [
        ("step", {"name": "import the distro", "index": 1, "total": 9}),
        ("progress", {"line": "extracting"}),
        ("done", {}),
    ]
    door.cut_after = 2
    with windows_client(make_client, monkeypatch, door.url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "failed"
    # NOT `engine_move_needs_host`: a host answered. The door's other caller
    # (`sdk/bootstrap/src/hostdoor.ts`) already calls this ending
    # `host_install_failed`, and one door has one set of names.
    assert finished["error"]["code"] == "host_install_failed"
    assert "without saying whether" in finished["error"]["message"]


def test_the_host_s_own_refusal_code_travels_rather_than_being_flattened(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    """`host_install_running` is what a client acts on; "it failed" is not."""
    door.refusal = (
        409,
        "host_install_running",
        "an engine move is already running on this machine",
    )
    with windows_client(make_client, monkeypatch, door.url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "failed"
    assert finished["error"]["code"] == "host_install_running"


def test_a_door_that_does_not_answer_is_host_unreachable(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variable says a host is there; the connection says it is not.

    **T10, 2026-09-14.** The POST is ACCEPTED — with the door set, 4.7 hands
    the move to the host and relays its events, so there is nothing to refuse
    at submit time and the failure is named IN THE TASK. Which is a different
    failure from `engine_move_needs_host`, and now says so: *start your
    host's door again*, not *start a host*.
    """
    with FakeDoor() as dead:
        url = dead.url
    with windows_client(make_client, monkeypatch, url) as server:
        accepted = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        # The submit itself is a 202 and says nothing about the door: reading
        # the POST body for the refusal is what T10's stage got wrong.
        assert accepted.status_code == 202
        assert "error" not in accepted.json()
        finished = wait_for(server, auth, accepted.json()["task_id"])
    assert finished["state"] == "failed"
    assert finished["error"]["code"] == "host_unreachable"
    assert "not answering" in finished["error"]["message"]
    assert url.rstrip("/") in finished["error"]["message"]


def test_it_is_one_task_at_a_time_like_every_other(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    door: FakeDoor,
) -> None:
    door.script = [("done", {})]
    with windows_client(make_client, monkeypatch, door.url) as server:
        first = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        assert first.status_code == 202
        wait_for(server, auth, first.json()["task_id"])
        # And the lane is free again afterwards, which is what makes a
        # failed move retryable from the page.
        again = server.post(
            "/v1/tasks", headers=auth, json={"type": "engine", "target": "wsl"}
        )
        assert again.status_code == 202
        wait_for(server, auth, again.json()["task_id"])


# ------------------------------------------- the two sides name the same thing


def test_the_env_var_the_host_sets_is_the_one_the_server_reads() -> None:
    """One name, two files, tied by a check rather than by a comment.

    `crucible/host/app.py` puts it on the child's environment and
    `crucible/tasks.py` reads it; they are in different packages and neither
    imports the other for anything else.
    """
    from crucible.host.app import server_environment
    from crucible.host.paths import door_url

    environment = server_environment({})
    assert tasks.HOST_DOOR_ENV in environment
    assert environment[tasks.HOST_DOOR_ENV] == door_url("")
    assert environment[tasks.HOST_DOOR_ENV] == "http://127.0.0.1:7101"
    # And the path the server POSTs to is the path the door serves.
    from crucible.host.door import INSTALL_PATH

    assert tasks.HOST_DOOR_PATH == INSTALL_PATH


def test_the_token_is_NOT_in_the_child_s_environment() -> None:
    """A secret with two owners is one more place for it to be stale."""
    from crucible.host.app import server_environment

    environment = server_environment({})
    assert not any("TOKEN" in name.upper() for name in environment)
