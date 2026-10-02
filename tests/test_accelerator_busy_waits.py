"""A card held by a process Crucible does not own is weather: whatever may wait, waits.

`accelerator_busy` (crucible/accelerator.py) means memory on the card is held by another
program, or by an engine an earlier Crucible left running. It frees when that process
lets go, so a job, a queue session opening on its model, and a queued chat keep their
place at the front of the line, say who holds the card, and are checked again on a pace
(crucible/jobs/line.py ``CARD_RECHECK_S``) until their own ``max_wait_s`` runs out. A
request sent with ``"queue": false`` is still refused at once, and the guard's
misconfiguration refusals (no room, a missing feature, larger than the host) still end
the job by name. Nothing here touches a GPU: echo jobs, the fake engine, and the
nvidia-smi probes monkeypatched.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, clock
from crucible.accelerator import WAITS_FOR_THE_CARD, ComputeApp
from crucible.admission import BUSY, KEEPS_WAITING
from crucible.errors import ApiError
from crucible.jobs import line as line_module
from crucible.jobs import llm as llm_module
from crucible.memorybudget import GIB

from .fake_engine import FakeEngine
from .live_server import serve
from .test_chat_queue import MODEL, _as, _chat, _in_background, _post_chat, _wait_for
from .test_queue import (
    as_client,
    body,
    events,
    free_the_lane,
    occupy_the_lane,
    status,
    wait_for,
)

HOLDER = (
    "cannot load 'echo': the accelerator is held by pid 4242 (python, 10.0 GiB). "
    "Crucible never evicts another process."
)


def held_card(asked: list[int]) -> Callable[[str | None, dict[str, Any]], None]:
    def preflight(model: str | None, params: dict[str, Any]) -> None:
        asked.append(1)
        raise ApiError(409, "accelerator_busy", HOLDER,
                       {"model": "echo", "processes": [{"pid": 4242}]})

    return preflight


def due_now(line: Any, item_id: str) -> None:
    """Bring the item's next card check forward, so a test need not sleep the pace."""
    item = line.get(item_id)
    assert item is not None and item.card_wait is not None
    item.card_wait = replace(item.card_wait, next_check=clock.now())


def test_only_a_held_card_waits_and_misconfiguration_does_not() -> None:
    assert WAITS_FOR_THE_CARD == {"accelerator_busy"}
    assert KEEPS_WAITING == BUSY | WAITS_FOR_THE_CARD
    for named in ("insufficient_memory", "card_lacks_feature", "accelerator_unreadable",
                  "insufficient_kv_cache"):
        assert named not in KEEPS_WAITING


def test_a_job_at_the_front_waits_for_a_held_card_and_says_who_holds_it(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    store, line = client.app.state.store, client.app.state.line
    echo = store.registry["echo"]
    holder = occupy_the_lane(client, auth)
    patient = client.post("/v1/jobs", json=body(), headers=auth).json()["job_id"]
    asked: list[int] = []
    monkeypatch.setattr(echo, "preflight", held_card(asked))
    free_the_lane(client, auth, holder)

    wait_for(lambda: line.get(patient) is not None
             and line.get(patient).card_wait is not None, "the job to wait for the card")
    time.sleep(2.5)
    assert len(asked) == 1, "the card is checked again on its pace, not every tick"
    assert status(client, auth, patient) == "queued"
    assert store.lane_free, "a job waiting for the card does not hold the lane"

    row = client.get("/v1/queue", headers=auth).json()["items"][0]
    assert row["job_id"] == patient and row["position"] == 1
    assert row["waiting_for"]["code"] == "accelerator_busy"
    assert "pid 4242" in row["waiting_for"]["message"]
    queued = client.get("/v1/activity", headers=auth).json()["queued"]
    assert queued[0]["waiting_for"]["message"] == HOLDER

    monkeypatch.setattr(echo, "preflight", lambda model, params: None)
    due_now(line, patient)
    wait_for(lambda: status(client, auth, patient) == "done", "the job to run")
    said = [event for event in events(client, auth, patient) if event["event"] == "waiting"]
    assert len(said) == 1, "said once while the holder stayed the same"
    assert said[0]["data"]["code"] == "accelerator_busy"
    assert "pid 4242" in said[0]["data"]["message"]
    assert "waiting for the accelerator" in said[0]["data"]["message"]


def test_a_long_card_wait_is_said_again_every_minute_with_its_next_check(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    store, line = client.app.state.store, client.app.state.line
    echo = store.registry["echo"]
    holder = occupy_the_lane(client, auth)
    patient = client.post("/v1/jobs", json=body(), headers=auth).json()["job_id"]
    asked: list[int] = []
    monkeypatch.setattr(echo, "preflight", held_card(asked))
    free_the_lane(client, auth, holder)
    wait_for(lambda: line.get(patient) is not None
             and line.get(patient).card_wait is not None, "the job to wait for the card")

    def said() -> list[dict[str, Any]]:
        # the job's own event log, read while it still waits (its SSE would not end)
        return [e for e in store.get(patient).events if e["event"] == "waiting"]

    first = said()
    assert len(first) == 1
    assert set(first[0]["data"]) == {"code", "message", "details", "since", "next_check_at"}

    due_now(line, patient)
    wait_for(lambda: len(asked) >= 2, "a second check inside the minute")
    assert len(said()) == 1, "the same holder inside the minute is not said again"

    item = line.get(patient)
    item.card_wait = replace(
        item.card_wait,
        said_at=clock.now() - timedelta(seconds=line_module.CARD_WAIT_REPEAT_S + 1),
        next_check=clock.now(),
    )
    wait_for(lambda: len(said()) == 2, "the wait to be said again after a minute")
    again = said()[1]["data"]
    assert again["since"] == first[0]["data"]["since"], "the same wait, not a new one"
    assert again["code"] == "accelerator_busy"

    monkeypatch.setattr(echo, "preflight", lambda model, params: None)
    due_now(line, patient)
    wait_for(lambda: status(client, auth, patient) == "done", "the job to run")


def test_a_fresh_submit_to_a_held_card_queues_and_queue_false_is_refused(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    store, line = client.app.state.store, client.app.state.line
    echo = store.registry["echo"]
    asked: list[int] = []
    monkeypatch.setattr(echo, "preflight", held_card(asked))

    refused = client.post("/v1/jobs", json=body(queue=False), headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "accelerator_busy"
    assert len(line) == 0

    receipt = client.post("/v1/jobs", json=body(), headers=auth)
    assert receipt.status_code == 202, receipt.text
    assert receipt.json()["queued"] is True and receipt.json()["position"] == 1
    job_id = receipt.json()["job_id"]
    assert line.get(job_id).card_wait is not None, "it waits for the card from the start"
    time.sleep(1.5)
    assert len(asked) == 2, "one check at submit; the next waits for the pace"

    monkeypatch.setattr(echo, "preflight", lambda model, params: None)
    due_now(line, job_id)
    wait_for(lambda: status(client, auth, job_id) == "done", "the job to run")


def test_a_job_whose_wait_for_the_card_runs_out_is_removed_naming_the_holder(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    store, line = client.app.state.store, client.app.state.line
    monkeypatch.setattr(store.registry["echo"], "preflight", held_card([]))
    job_id = client.post(
        "/v1/jobs", json=body(queue={"max_wait_s": 10}), headers=as_client(auth, "briefcase")
    ).json()["job_id"]
    gone = line.expire(clock.now() + timedelta(seconds=11))
    assert [item.job.id for item in gone] == [job_id]
    removal = client.get(f"/v1/jobs/{job_id}", headers=auth).json()["removal"]
    assert removal["reason"] == "expired"
    assert "waiting for the accelerator" in removal["message"]
    assert "pid 4242" in removal["message"]


def test_a_job_behind_the_one_waiting_for_the_card_keeps_its_place(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """First come, first served holds: the line does not reorder around a held card."""
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    store, line = client.app.state.store, client.app.state.line
    echo = store.registry["echo"]
    monkeypatch.setattr(echo, "preflight", held_card([]))
    first = client.post("/v1/jobs", json=body(), headers=as_client(auth, "a")).json()
    monkeypatch.setattr(echo, "preflight", lambda model, params: None)
    second = client.post("/v1/jobs", json=body(), headers=as_client(auth, "b")).json()
    assert second["queued"] is True and second["position"] == 2
    time.sleep(1.5)
    assert status(client, auth, second["job_id"]) == "queued"
    due_now(line, first["job_id"])
    wait_for(lambda: status(client, auth, second["job_id"]) == "done", "both to run")
    assert status(client, auth, first["job_id"]) == "done"


@pytest.fixture
def llm_server(
    make_app: Callable[..., Any],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
):
    engines = engine_factory()
    fake_weights(MODEL)
    app = make_app(enable_llm=True)
    return engines, app


def hold_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        accelerator, "probe_compute_apps",
        lambda: [ComputeApp(pid=4242, name="python", used_bytes=10 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (12 * GIB, 24 * GIB))


def let_go(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


def _queue(base: str, auth: dict[str, str]) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue", headers=auth, timeout=30.0).json()


def _session_state(base: str, auth: dict[str, str], session_id: str) -> dict[str, Any]:
    return httpx.get(f"{base}/v1/queue/sessions/{session_id}", headers=auth,
                     timeout=30.0).json()


def test_a_session_opening_on_a_held_card_waits_then_opens(
    llm_server: Any, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    engines, app = llm_server
    hold_the_card(monkeypatch)
    with serve(app) as base:
        ticket = httpx.post(f"{base}/v1/queue/sessions", headers=_as(auth, "briefcase"),
                            json={"act": "analysis", "model": MODEL}, timeout=30.0).json()
        session_id = ticket["session_id"]
        _wait_for(lambda: (_queue(base, auth)["items"] or [{}])[0].get("waiting_for")
                  is not None, "the session to wait for the card")
        row = _queue(base, auth)["items"][0]
        assert row["kind"] == "session" and row["job_id"] == session_id
        assert "pid 4242" in row["waiting_for"]["message"]
        state = _session_state(base, auth, session_id)
        assert state["status"] == "queued" and state["load_job"] is None
        assert engines == [], "nothing was started on a held card"

        let_go(monkeypatch)
        due_now(app.state.line, session_id)
        _wait_for(lambda: _session_state(base, auth, session_id)["status"] == "open",
                  "the session to open once the card was free")
        said = [event["event"] for event in app.state.sessions.get(session_id).events]
        assert said[:3] == ["queued", "waiting", "opened"]
        httpx.delete(f"{base}/v1/queue/sessions/{session_id}",
                     headers=_as(auth, "briefcase"), timeout=30.0)


def test_a_session_whose_load_meets_a_holder_on_the_lane_waits_rather_than_failing(
    llm_server: Any, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission's check passed, then the load's own guard on the lane met the holder:
    the load job ends failed, and the session it was opening keeps waiting."""
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    engines, app = llm_server
    store = app.state.store
    real = llm_module.card_guard
    tripped: list[int] = []

    def on_the_lane_once(config: Any, **kwargs: Any) -> Any:
        if store.running_id is not None and not tripped:
            tripped.append(1)
            raise ApiError(409, "accelerator_busy", HOLDER)
        return real(config, **kwargs)

    monkeypatch.setattr(llm_module, "card_guard", on_the_lane_once)
    with serve(app) as base:
        ticket = httpx.post(f"{base}/v1/queue/sessions", headers=_as(auth, "briefcase"),
                            json={"act": "analysis", "model": MODEL}, timeout=30.0).json()
        session_id = ticket["session_id"]
        _wait_for(lambda: app.state.line.get(session_id) is not None
                  and app.state.line.get(session_id).card_wait is not None,
                  "the session to wait after its load met the holder")
        assert _session_state(base, auth, session_id)["status"] == "queued"
        due_now(app.state.line, session_id)
        _wait_for(lambda: _session_state(base, auth, session_id)["status"] == "open",
                  "the session to open on the second load")
        assert len(engines) == 1
        httpx.delete(f"{base}/v1/queue/sessions/{session_id}",
                     headers=_as(auth, "briefcase"), timeout=30.0)


def test_a_queued_chat_whose_model_load_meets_a_held_card_waits_then_is_answered(
    llm_server: Any, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(line_module, "CARD_RECHECK_S", 3600.0)
    engines, app = llm_server
    hold_the_card(monkeypatch)
    with serve(app) as base:
        refused = _post_chat(base, auth, _chat(queue=False))
        assert refused.status_code == 409
        thread, out = _in_background(lambda: _post_chat(base, _as(auth, "briefcase"), _chat()))
        _wait_for(lambda: (_queue(base, auth)["items"] or [{}])[0].get("waiting_for")
                  is not None, "the chat to wait for the card")
        row = _queue(base, auth)["items"][0]
        assert row["kind"] == "call" and "pid 4242" in row["waiting_for"]["message"]
        assert engines == []

        let_go(monkeypatch)
        due_now(app.state.line, row["job_id"])
        thread.join(timeout=60.0)
        assert out and out[0].status_code == 200, out
        assert len(engines) == 1
