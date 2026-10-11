"""An install a queue session's item is waiting on is the session's work in flight.

B-Sides opens a session, asks for a model the server has not pulled, and is refused
`409 installing` while the server downloads it (multi-GB on Victoria's 8 GiB laptop).
Until that install ends, success or failure, the session is not idle, whatever its
idle_s; from its end the session's idle_s runs again. The pull here is a fake that
waits on an event, so nothing is downloaded and no GPU is touched.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import clock
from crucible.errors import ApiError

from .test_llm_api import MODEL
from .test_queue import as_client, wait_for
from .test_queue_sessions import item, open_session, sessions_of, state, submit


class Pulls:
    """Every pull the task store starts waits until the test lets it end."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.fail = False

    async def pull(self, store: Any, task: Any, subject: Any) -> None:
        await asyncio.to_thread(self.release.wait, 30)
        if self.fail:
            raise ApiError(500, "pull_failed", f"the test failed the pull of {subject.id}")


@pytest.fixture
def pulls(monkeypatch: pytest.MonkeyPatch) -> Iterator[Pulls]:
    held = Pulls()

    async def pull(self: Any, task: Any, subject: Any) -> None:
        await held.pull(self, task, subject)

    monkeypatch.setattr("crucible.tasks.TaskStore._pull", pull)
    yield held
    held.release.set()


@pytest.fixture
def llm(make_client: Callable[..., TestClient], fake_env: Path) -> Iterator[TestClient]:
    with make_client(enable_llm=True) as client:
        yield client


def _refused_installing(client: TestClient, headers: dict[str, str], session_id: str) -> str:
    refused = submit(client, item(headers, session_id), type="load-model", model=MODEL)
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "installing", error
    return str(error["details"]["task_id"])


def _task(client: TestClient, task_id: str) -> Any:
    return client.app.state.tasks.get(task_id)


def test_a_long_install_keeps_the_session_open_past_its_idle_s(
    llm: TestClient, auth: dict[str, str], pulls: Pulls
) -> None:
    mine = as_client(auth, "b-sides")
    session_id = open_session(llm, mine, idle_s=10)["session_id"]
    task_id = _refused_installing(llm, mine, session_id)

    in_flight = state(llm, auth, session_id)["in_flight"]
    assert [(row["kind"], row["id"]) for row in in_flight] == [("install", task_id)]
    assert state(llm, auth, session_id)["idle_deadline"] is None
    assert sessions_of(llm).due(clock.now() + timedelta(hours=2)) is None, (
        "a two-hour download is the session's work, never its idleness"
    )
    again = submit(llm, item(mine, session_id), type="load-model", model=MODEL)
    assert again.json()["error"]["details"]["task_id"] == task_id
    assert len(sessions_of(llm).get(session_id).installs) == 1, "asking again adds no hold"

    pulls.release.set()
    wait_for(lambda: _task(llm, task_id).state == "done", "the install to finish")
    assert state(llm, auth, session_id)["in_flight"] == []
    llm.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_idle_starts_counting_when_the_install_ends(
    llm: TestClient, auth: dict[str, str], pulls: Pulls
) -> None:
    mine = as_client(auth, "b-sides")
    session_id = open_session(llm, mine, idle_s=180)["session_id"]
    task_id = _refused_installing(llm, mine, session_id)
    pulls.release.set()
    wait_for(lambda: _task(llm, task_id).state == "done", "the install to finish")

    ended = datetime.fromisoformat(_task(llm, task_id).finished)
    after = state(llm, auth, session_id)
    assert after["status"] == "open" and after["in_flight"] == []
    assert datetime.fromisoformat(after["idle_deadline"]) == ended + timedelta(seconds=180)
    sessions = sessions_of(llm)
    assert sessions.due(ended + timedelta(seconds=179)) is None
    due = sessions.due(ended + timedelta(seconds=181))
    assert due is not None and due[1] == "idle"
    llm.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_a_failed_install_ends_the_hold(
    llm: TestClient, auth: dict[str, str], pulls: Pulls
) -> None:
    mine = as_client(auth, "b-sides")
    session_id = open_session(llm, mine, idle_s=60)["session_id"]
    task_id = _refused_installing(llm, mine, session_id)
    assert state(llm, auth, session_id)["in_flight"][0]["kind"] == "install"

    pulls.fail = True
    pulls.release.set()
    wait_for(lambda: _task(llm, task_id).state == "failed", "the install to fail")
    ended = datetime.fromisoformat(_task(llm, task_id).finished)
    after = state(llm, auth, session_id)
    assert after["in_flight"] == [] and sessions_of(llm).get(session_id).installs == []
    assert datetime.fromisoformat(after["idle_deadline"]) == ended + timedelta(seconds=60)
    due = sessions_of(llm).due(ended + timedelta(seconds=61))
    assert due is not None and due[1] == "idle", "a failed install holds nothing"
    llm.delete(f"/v1/queue/sessions/{session_id}", headers=mine)


def test_an_install_refused_outside_a_session_is_held_by_no_session(
    llm: TestClient, auth: dict[str, str], pulls: Pulls
) -> None:
    mine, theirs = as_client(auth, "b-sides"), as_client(auth, "foundry")
    refused = submit(llm, theirs, type="load-model", model=MODEL)
    assert refused.json()["error"]["code"] == "installing"
    session_id = open_session(llm, mine, idle_s=60)["session_id"]
    assert state(llm, auth, session_id)["in_flight"] == []
    llm.delete(f"/v1/queue/sessions/{session_id}", headers=mine)
