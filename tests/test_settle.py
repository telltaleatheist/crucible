from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible.jobs import ALL_JOB_TYPES
from crucible.settle import LEAVES_IT_RESIDENT, Settlement

from .fake_engine import FakeEngine

MODEL = "qwen3.5-9b"


@pytest.fixture
def engines(engine_factory: Callable[..., list[FakeEngine]]) -> list[FakeEngine]:
    return engine_factory()


@pytest.fixture
def resident(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> Iterator[TestClient]:
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:
        finish(client, auth, submit_echo(client, auth))
        run_load(client, auth)
        assert is_resident(client, auth)
        yield client


def run_load(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/v1/jobs", headers=auth, json={"type": "load-model", "model": MODEL}
    )
    assert response.status_code == 202, response.text
    finish(client, auth, response.json()["job_id"])


def submit_echo(
    client: TestClient, auth: dict[str, str], delay_ms: int = 0
) -> str:
    response = client.post(
        "/v1/jobs",
        headers=auth,
        json={
            "type": "echo",
            "params": {"delay_ms": delay_ms},
            "inputs": {
                "x.bin": {"inline_base64": base64.b64encode(b"poke").decode("ascii")}
            },
        },
    )
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def finish(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict]:
    from .conftest import parse_sse

    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def echoed(client: TestClient, auth: dict[str, str], delay_ms: int = 0) -> list[dict]:
    return finish(client, auth, submit_echo(client, auth, delay_ms))


def is_resident(client: TestClient, auth: dict[str, str]) -> bool:
    body = client.get("/v1/activity", headers=auth).json()
    return body["resident"] is not None


def a_session(client: TestClient, auth: dict[str, str]) -> str:
    response = client.post("/v1/queue/sessions", headers=auth, json={"act": "clean"})
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "open", response.text
    return str(response.json()["session_id"])


def close(client: TestClient, auth: dict[str, str], session_id: str) -> None:
    response = client.delete(f"/v1/queue/sessions/{session_id}", headers=auth)
    assert response.status_code == 200, response.text


def a_chat(client: TestClient, auth: dict[str, str], **extra: Any) -> Any:
    return client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra},
    )


def settlement_of(client: TestClient) -> Settlement:
    return client.app.state.settlement


def test_a_job_on_the_lane_holds_it_and_its_end_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit_echo(resident, auth, delay_ms=200)
    held = settlement_of(resident).holder()
    assert held is not None
    assert held.fact == "a job"
    assert job_id in held.who
    assert settlement_of(resident).settle("a test asked") is None
    assert is_resident(resident, auth)

    finish(resident, auth, job_id)
    assert not is_resident(resident, auth)


def test_an_open_session_holds_it_and_closing_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    session_id = a_session(resident, auth)
    echoed(resident, auth)
    held = settlement_of(resident).holder()
    assert held is not None and held.fact == "a session"
    assert held.details["session_id"] == session_id
    assert is_resident(resident, auth), "a job that ends inside a session keeps it"

    close(resident, auth, session_id)
    assert not is_resident(resident, auth)


def test_a_streaming_claim_holds_it_and_the_release_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    residency = resident.app.state.residency
    residency.claim("tts stream abc123", may_mutate=False)
    echoed(resident, auth)
    held = settlement_of(resident).holder()
    assert held is not None and held.fact == "the claim"
    assert held.who == "tts stream abc123"
    assert is_resident(resident, auth)

    residency.release("tts stream abc123")
    assert settlement_of(resident).settle("the session closed") is not None
    assert not is_resident(resident, auth)


def test_a_chat_in_flight_holds_it_and_the_last_one_returning_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    inflight = resident.app.state.inflight
    with inflight.tracked(act="clean", model=MODEL, client=None):
        echoed(resident, auth)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a chat"
        assert is_resident(resident, auth)

    assert settlement_of(resident).settle("the last chat finished") is not None
    assert not is_resident(resident, auth)


def test_a_real_chat_run_outside_a_session_reloads_its_model(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    first = a_chat(resident, auth)
    assert first.status_code == 200, first.text
    assert not is_resident(resident, auth)

    refused = a_chat(resident, auth, queue=False)
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "model_not_resident"
    assert len(engines) == 1

    second = a_chat(resident, auth)
    assert second.status_code == 200, second.text
    assert len(engines) == 2, "the chat waited while its model was loaded again"


def test_a_session_turns_two_jobs_back_to_back_into_one_load(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    session_id = a_session(resident, auth)
    echoed(resident, auth)
    echoed(resident, auth)
    assert is_resident(resident, auth)
    assert len(engines) == 1

    close(resident, auth, session_id)
    assert not is_resident(resident, auth)
    assert len(engines) == 1


def test_the_unload_is_said_on_the_job_that_triggered_it_and_in_the_log(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    events = echoed(resident, auth)
    notes = [row for row in events if row["event"] == "note"]
    assert len(notes) == 1, events
    said = notes[0]["data"]
    assert said["unloaded"] == MODEL
    assert said["kind"] == "llm"
    assert "nothing holds it" in said["message"]
    assert "(echo) finished" in said["trigger"]

    kinds = [row["event"] for row in events]
    assert kinds.index("note") < kinds.index("done")

    assert f"unloaded {MODEL}" in capfd.readouterr().err


def test_a_settlement_with_no_job_behind_it_still_says_so_in_the_log(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    session_id = a_session(resident, auth)
    capfd.readouterr()
    close(resident, auth, session_id)
    said = capfd.readouterr().err
    assert f"unloaded {MODEL}" in said
    assert f"session {session_id} closed (client)" in said


def test_a_load_is_not_a_holder_letting_go(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    assert is_resident(resident, auth)
    run_load(resident, auth)
    assert is_resident(resident, auth)
    assert len(engines) == 2


def test_a_cancelled_load_is_settled_like_any_other_job_and_leaves_no_card(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    hold = threading.Event()
    engines = engine_factory(hold=hold)
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:
        response = client.post(
            "/v1/jobs", headers=auth, json={"type": "load-model", "model": MODEL}
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        try:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if engines and engines[0].warming_started.wait(timeout=0.1):
                    break
                time.sleep(0.05)
            assert engines and engines[0].warming_started.is_set(), (
                "the lane never reached the engine"
            )
            cancelling = client.delete(f"/v1/jobs/{job_id}", headers=auth)
            assert cancelling.status_code == 200, cancelling.text
            assert cancelling.json()["status"] == "cancelling"
        finally:
            hold.set()

        events = finish(client, auth, job_id)
        assert events[-1]["event"] == "cancelled", events[-1]
        assert not is_resident(client, auth)
        assert engines[0].stopped is True
        notes = [row for row in events if row["event"] == "note"]
        assert [note["data"]["unloaded"] for note in notes] == [MODEL], events

        run_load(client, auth)
        assert is_resident(client, auth)
        assert engines[1].stopped is False


def test_every_exempt_name_is_a_job_type_this_build_knows() -> None:
    assert LEAVES_IT_RESIDENT <= set(ALL_JOB_TYPES)


def test_the_exempt_names_are_the_ones_whose_whole_content_is_being_resident() -> None:
    from crucible.jobtypes import CARD_EFFECTS

    for name in LEAVES_IT_RESIDENT:
        assert CARD_EFFECTS[name].makes_resident is not None, name
    assert {"tts", "align"} & LEAVES_IT_RESIDENT == set()


def test_settling_an_empty_card_is_a_no_op_rather_than_a_refusal(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert settlement_of(client).holder() is None
    assert settlement_of(client).settle("a test asked") is None
    echoed(client, auth)


def test_an_unload_job_and_the_settlement_do_not_fight_over_one_engine(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    response = resident.post(
        "/v1/jobs", headers=auth, json={"type": "unload-model", "model": MODEL}
    )
    assert response.status_code == 202, response.text
    events = finish(resident, auth, response.json()["job_id"])
    assert events[-1]["event"] == "done", events[-1]
    assert [row for row in events if row["event"] == "note"] == []
    assert not is_resident(resident, auth)
    assert engines[0].pids == frozenset()


def test_the_settlement_never_clears_a_card_it_could_not_claim(
    resident: TestClient, auth: dict[str, str]
) -> None:
    residency = resident.app.state.residency
    residency.claim("a render", may_mutate=False)
    try:
        assert settlement_of(resident).settle("a test asked") is None
        assert is_resident(resident, auth)
    finally:
        residency.release("a render")


def test_a_settlement_in_progress_is_visible_as_the_claim(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = resident.app.state.residency
    seen: list[str | None] = []

    original = residency.unload

    def watched(subject_id: str) -> Any:
        seen.append(residency.claimed_by)
        return original(subject_id)

    monkeypatch.setattr(residency, "unload", watched)
    assert settlement_of(resident).settle("a test asked") is not None
    assert seen == ["the settlement clearing the card"]
    assert residency.claimed_by is None


def test_a_card_that_will_not_be_cleared_does_not_fail_the_job(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    residency = resident.app.state.residency
    original = residency.unload

    def refuses(subject_id: str) -> Any:
        raise RuntimeError("this engine will not stop")

    residency.unload = refuses
    try:
        events = echoed(resident, auth)
    finally:
        residency.unload = original
    assert events[-1]["event"] == "done", events[-1]
    notes = [row["data"]["message"] for row in events if row["event"] == "note"]
    assert any("could not clear the card" in note for note in notes), notes
    assert "could not clear the card" in capfd.readouterr().err


def test_a_session_that_goes_idle_clears_the_card_when_it_closes(
    resident: TestClient, auth: dict[str, str]
) -> None:
    session_id = a_session(resident, auth)
    sessions = resident.app.state.sessions
    echoed(resident, auth)
    assert is_resident(resident, auth)
    sessions.get(session_id).idle_s = 1
    deadline = time.monotonic() + 20.0
    while is_resident(resident, auth) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not is_resident(resident, auth), "the idle close settled the card"
    assert sessions.get(session_id).reason == "idle"


def test_the_facts_are_read_in_a_fixed_order_so_a_refusal_names_the_same_one(
    resident: TestClient, auth: dict[str, str]
) -> None:
    residency = resident.app.state.residency
    a_session(resident, auth)
    residency.claim("tts stream abc123", may_mutate=False)
    with resident.app.state.inflight.tracked(act="clean", model=MODEL, client=None):
        job_id = submit_echo(resident, auth, delay_ms=200)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a job"
        finish(resident, auth, job_id)
        assert settlement_of(resident).holder().fact == "a session"
    residency.release("tts stream abc123")
    assert settlement_of(resident).holder().fact == "a session"
    assert is_resident(resident, auth)


def test_nothing_about_activity_grew_a_second_owner_of_residency(
    resident: TestClient, auth: dict[str, str]
) -> None:
    body = resident.get("/v1/activity", headers=auth).json()
    assert body["resident"]["id"] == MODEL
    assert "unloaded" not in body
    assert "settlement" not in body
    echoed(resident, auth)
    after = resident.get("/v1/activity", headers=auth).json()
    assert after["resident"] is None
    assert set(after) == set(body)


def test_the_lane_reports_a_queued_job_as_a_holder_too(
    resident: TestClient, auth: dict[str, str]
) -> None:
    store = resident.app.state.store
    job = store.create("echo", None, {})
    try:
        store.admitted.admit(job.id)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a job"
        assert job.id in held.who
        assert settlement_of(resident).settle("a test asked") is None
    finally:
        store.admitted.release(job.id)
        store.discard(job)

    assert store.occupied_by_anything_but(None) is None


def test_a_job_the_lane_is_handing_over_is_never_out_of_the_settlement_s_sight(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = resident.app.state.store
    persist = store._persist
    seen: list[Any] = []

    def persist_and_look(job: Any) -> None:
        if job.status == "running" and not seen:
            seen.append(settlement_of(resident).holder())
        persist(job)

    monkeypatch.setattr(store, "_persist", persist_and_look)
    events = echoed(resident, auth)
    assert events[-1]["event"] == "done"
    assert seen and seen[0] is not None, "the handover hid the job from holder()"
    assert seen[0].fact == "a job"


def test_the_server_still_tears_the_residency_down_on_the_way_out(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:
        run_load(client, auth)
        assert is_resident(client, auth)
    assert engines[0].pids == frozenset()


def test_nothing_settles_until_a_settlement_is_attached() -> None:
    from crucible.jobs.queue import JobStore

    store = JobStore(object(), object(), {})
    assert store.occupied_by_anything_but(None) is None
    assert store._settlement is None


def test_the_grace_of_a_slow_stop_does_not_delay_the_answer(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = resident.app.state.residency
    original = residency.unload
    started: list[float] = []

    def slow(subject_id: str) -> Any:
        started.append(time.monotonic())
        return original(subject_id)

    monkeypatch.setattr(residency, "unload", slow)
    before = time.monotonic()
    response = a_chat(resident, auth)
    assert response.status_code == 200
    assert started, "the card was never cleared"
    assert started[0] >= before
    assert not is_resident(resident, auth)


def _echo_that_reports_the_card(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    from crucible.jobs.echo import EchoJobType

    residency = client.app.state.residency
    original = EchoJobType.run

    def run(self: Any, job: Any, ctx: Any) -> None:
        original(self, job, ctx)
        ctx.done_extra(resident=residency.resident_id)

    monkeypatch.setattr(EchoJobType, "run", run)


def test_the_done_event_says_what_is_resident_after_the_settlement_cleared_it(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _echo_that_reports_the_card(monkeypatch, resident)
    events = echoed(resident, auth)
    note = next(row["data"] for row in events if row["event"] == "note")
    done = next(row["data"] for row in events if row["event"] == "done")
    assert note["unloaded"] == MODEL
    assert done["resident"] is None, (
        "the note said the model came off the card, so the done event after it "
        "must not say it is still there"
    )
    assert not is_resident(resident, auth)


def test_the_done_event_keeps_the_resident_when_a_session_holds_the_card(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _echo_that_reports_the_card(monkeypatch, resident)
    session_id = a_session(resident, auth)
    events = echoed(resident, auth)
    done = next(row["data"] for row in events if row["event"] == "done")
    assert done["resident"] == MODEL
    assert not [row for row in events if row["event"] == "note"]
    close(resident, auth, session_id)


def test_the_done_event_follows_the_card_when_the_settlement_fails_after_the_unload(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The settlement took the model off the residency, then stopping its process
    # raised: the note says the clearing failed, and the done event must still say
    # what the residency holds rather than what the job read before the settlement.
    _echo_that_reports_the_card(monkeypatch, resident)
    residency = resident.app.state.residency

    stop = residency._stop_the_dying

    def stop_fails() -> None:
        raise RuntimeError("the engine would not stop")

    monkeypatch.setattr(residency, "_stop_the_dying", stop_fails)
    events = echoed(resident, auth)
    monkeypatch.setattr(residency, "_stop_the_dying", stop)
    note = next(row["data"] for row in events if row["event"] == "note")
    done = next(row["data"] for row in events if row["event"] == "done")
    assert "could not clear the card" in note["message"]
    assert residency.resident_id is None
    assert done["resident"] is None
