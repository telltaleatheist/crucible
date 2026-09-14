"""Owen's unload ruling: when the last holder lets go, the card is cleared.

> **Owen, 2026-09-14:** *"Models should always be unloaded when we're done with
> them. Every time."*

`crucible/settle.py` is the rule and this is the proof of it. The shape of this
file is the shape of the ruling: there are **four facts**, each one of them on
its own keeps the resident thing on the card, and the moment the last of them
goes false the thing is unloaded and says so.

NOTHING HERE SLEEPS THROUGH A WINDOW, because there is no window to sleep
through. That is the whole point of the ruling over the keep-warm timer it
replaces: every assertion below is about a state, not about a duration.
"""

from __future__ import annotations

import base64
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible.jobs import ALL_JOB_TYPES
from crucible.settle import LEAVES_IT_RESIDENT, Settlement

from .fake_engine import FakeEngine

MODEL = "qwen3.5-9b"
TTL = 60


# ------------------------------------------------------------------ fixtures


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
    """A server with `qwen3.5-9b` actually resident, and `echo` to poke it with.

    `load-model` is the one job whose end does NOT clear the card — a load's
    whole content is *"be resident"* — so this fixture is the ruling's own
    statement that a load is the start of a resident thing's life, not the end.
    """
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
    """Read the job's whole event stream, which ends at its terminal event."""
    from .conftest import parse_sse

    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def echoed(client: TestClient, auth: dict[str, str], delay_ms: int = 0) -> list[dict]:
    return finish(client, auth, submit_echo(client, auth, delay_ms))


def is_resident(client: TestClient, auth: dict[str, str]) -> bool:
    body = client.get("/v1/activity", headers=auth).json()
    return body["resident"] is not None


def a_lease(client: TestClient, auth: dict[str, str]) -> str:
    response = client.post(
        f"/v1/models/{MODEL}/lease",
        headers=auth,
        json={"act": "clean", "ttl_seconds": TTL},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["lease_id"])


def a_chat(client: TestClient, auth: dict[str, str]) -> Any:
    return client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )


def settlement_of(client: TestClient) -> Settlement:
    return client.app.state.settlement


# --------------------------------------------- the four facts, one at a time
#
# Each of these holds the card ALONE: in every one of them the other three are
# false, so what keeps the model resident is the single fact named in the test.


def test_a_job_on_the_lane_holds_it_and_its_end_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """Fact 1. And the job's own end is the moment the card goes."""
    job_id = submit_echo(resident, auth, delay_ms=200)
    # Asked from outside while the lane is occupied: the job is the holder, and
    # a settlement that ignored it would clear the card under work in progress.
    held = settlement_of(resident).holder()
    assert held is not None
    assert held.fact == "a job"
    assert job_id in held.who
    assert settlement_of(resident).settle("a test asked") is None
    assert is_resident(resident, auth)

    finish(resident, auth, job_id)
    assert not is_resident(resident, auth)


def test_an_open_lease_holds_it_and_releasing_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """Fact 2 — and the one a client can state deliberately."""
    lease_id = a_lease(resident, auth)
    echoed(resident, auth)
    held = settlement_of(resident).holder()
    assert held is not None and held.fact == "a lease"
    assert is_resident(resident, auth), "a job that ends under a lease keeps it"

    assert resident.delete(f"/v1/leases/{lease_id}", headers=auth).status_code == 204
    assert not is_resident(resident, auth)


def test_a_streaming_claim_holds_it_and_the_release_is_what_clears_it(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """Fact 3. The claim is narrator's wire, held for a session's lifetime."""
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
    """Fact 4 — the one that holds nothing else, which is why it is here.

    A chat takes no lane, makes no job and reserves nothing. That is right while
    it runs and is exactly why its END is the moment worth asking at.
    """
    inflight = resident.app.state.inflight
    with inflight.tracked(act="clean", model=MODEL, client=None):
        echoed(resident, auth)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a chat"
        assert is_resident(resident, auth)

    assert settlement_of(resident).settle("the last chat finished") is not None
    assert not is_resident(resident, auth)


def test_a_real_chat_run_without_a_lease_reloads_its_model(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    """THE BILL FOR NOT STATING AN INTENTION, stated rather than discovered.

    BookForge's `crucible` provider does not lease yet, so this is what its
    cleanup run meets today: the completion arrives, the card is cleared behind
    it, and the next request is refused by name until something loads again.
    The fix is a lease at BookForge's door, never an exception here.
    """
    first = a_chat(resident, auth)
    assert first.status_code == 200, first.text
    assert not is_resident(resident, auth)

    second = a_chat(resident, auth)
    assert second.status_code == 409, second.text
    assert second.json()["error"]["code"] == "model_not_resident"
    # And nothing was loaded behind the client's back to answer it.
    assert len(engines) == 1


def test_a_lease_turns_two_jobs_back_to_back_into_one_load(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    """What the lease buys, measured: one engine for the whole run.

    This is the difference between Owen's ruling being safe and being a
    44-second reload between every book (FROM-FOUNDRY-WSL-VLLM.md section 3).
    """
    lease_id = a_lease(resident, auth)
    echoed(resident, auth)
    echoed(resident, auth)
    assert is_resident(resident, auth)
    assert len(engines) == 1

    resident.delete(f"/v1/leases/{lease_id}", headers=auth)
    assert not is_resident(resident, auth)
    assert len(engines) == 1


# --------------------------------------------------------------- it is SAID


def test_the_unload_is_said_on_the_job_that_triggered_it_and_in_the_log(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    """A reader must never have to guess why the next request paid a load."""
    events = echoed(resident, auth)
    notes = [row for row in events if row["event"] == "note"]
    assert len(notes) == 1, events
    said = notes[0]["data"]
    assert said["unloaded"] == MODEL
    assert said["kind"] == "llm"
    assert "nothing holds it" in said["message"]
    assert "(echo) finished" in said["trigger"]

    # BEFORE the terminal event, because `_event_stream` returns at the terminal
    # event: a note appended after `done` is a note nobody is told.
    kinds = [row["event"] for row in events]
    assert kinds.index("note") < kinds.index("done")

    assert f"unloaded {MODEL}" in capfd.readouterr().err


def test_a_settlement_with_no_job_behind_it_still_says_so_in_the_log(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    """A lease- or chat-triggered unload has no job to carry an event."""
    lease_id = a_lease(resident, auth)
    capfd.readouterr()
    resident.delete(f"/v1/leases/{lease_id}", headers=auth)
    said = capfd.readouterr().err
    assert f"unloaded {MODEL}" in said
    assert "the lease was released" in said


# ------------------------------------------------------- what it does NOT do


def test_a_load_is_not_a_holder_letting_go(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    """`load-model` exists to make something resident and nothing else.

    Its own completion cannot be the moment the card is cleared, or the model
    would be gone before the operator's next request and neither the chat door
    nor the streaming door ever loads. See the RULING OWED in
    `crucible/settle.py`: the way out is a lease at the load door.
    """
    assert is_resident(resident, auth)
    run_load(resident, auth)
    assert is_resident(resident, auth)
    # Loaded twice on purpose: the second load evicts the first, which is the
    # residency's own rule and not a settlement.
    assert len(engines) == 2


def test_every_exempt_name_is_a_job_type_this_build_knows() -> None:
    """A rename must be a failing test, never a loader that silently settles."""
    assert LEAVES_IT_RESIDENT <= set(ALL_JOB_TYPES)


def test_the_exempt_names_are_the_ones_whose_whole_content_is_being_resident() -> None:
    """`LEAVES_IT_RESIDENT` is listed, and the list is tied to a second fact.

    Every exempt name must be one `CARD_EFFECTS` agrees MAKES something resident
    — an exemption for a job that loads nothing would be an exemption for
    nothing. The converse is deliberately not asserted: `tts` and `align` make
    something resident too and are NOT exempt, because making it resident is not
    the whole of what they do, and a render that left its voice on the card
    would strand it exactly as an unused load does. What holds a voice across
    twenty chapters is a lease on that voice, not a second name here.
    """
    from crucible.leases import CARD_EFFECTS

    for name in LEAVES_IT_RESIDENT:
        assert CARD_EFFECTS[name].makes_resident is not None, name
    assert {"tts", "align"} & LEAVES_IT_RESIDENT == set()


def test_settling_an_empty_card_is_a_no_op_rather_than_a_refusal(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Nothing resident is one of the three honest ways to answer None."""
    assert settlement_of(client).holder() is None
    assert settlement_of(client).settle("a test asked") is None
    echoed(client, auth)  # and a job on an empty card settles nothing either


def test_an_unload_job_and_the_settlement_do_not_fight_over_one_engine(
    resident: TestClient, auth: dict[str, str], engines: list[FakeEngine]
) -> None:
    """`unload-model` takes it off; the settlement that follows finds nothing."""
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
    """Somebody claiming between the read and the claim is somebody using it.

    The settlement runs off the event loop — stopping an engine can wait three
    minutes on a SIGTERM — so admission can move underneath it. It takes the
    same exclusive claim a render does, and a claim it cannot get is the answer
    to the question it was asking.
    """
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
    """It holds the card by name, so a job that races it is refused by name."""
    residency = resident.app.state.residency
    seen: list[str | None] = []

    original = residency.unload

    def watched(subject_id: str) -> Any:
        seen.append(residency.claimed_by)
        return original(subject_id)

    monkeypatch.setattr(residency, "unload", watched)
    assert settlement_of(resident).settle("a test asked") is not None
    assert seen == ["the settlement clearing the card"]
    # And it is given back, whatever happened: a claim left standing would make
    # every later load refuse `engine_in_use` forever.
    assert residency.claimed_by is None


def test_a_card_that_will_not_be_cleared_does_not_fail_the_job(
    resident: TestClient, auth: dict[str, str], capfd: pytest.CaptureFixture[str]
) -> None:
    """A cleanup failure is not an operation failure.

    An engine that will not stop is a real fact and is said in both places a
    reader looks. It does not turn a job that did its work into a failed job.
    """
    residency = resident.app.state.residency
    original = residency.unload

    def refuses(subject_id: str) -> Any:
        raise RuntimeError("this engine will not stop")

    # Restored by hand rather than by `monkeypatch`: the engine fixtures this
    # server is built on request `monkeypatch` first, so its teardown runs AFTER
    # the client's lifespan — and a shutdown that met this stub would fail the
    # test in teardown for a reason that has nothing to do with the ruling.
    residency.unload = refuses  # type: ignore[method-assign]
    try:
        events = echoed(resident, auth)
    finally:
        residency.unload = original  # type: ignore[method-assign]
    assert events[-1]["event"] == "done", events[-1]
    notes = [row["data"]["message"] for row in events if row["event"] == "note"]
    assert any("could not clear the card" in note for note in notes), notes
    assert "could not clear the card" in capfd.readouterr().err


def test_a_lease_that_expires_unheld_clears_the_card_on_its_own_deadline(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """The fifth moment, which has no edge of its own.

    A lease is READ against the clock and never swept (`crucible/leases.py`), so
    a client that crashed mid-run stops holding the card at an instant nothing is
    watching. The one-shot armed at the lease's OWN `expires_at` is what turns
    that into an edge — the client's number, never an interval tuned here.
    """
    resident.post(
        f"/v1/models/{MODEL}/lease",
        headers=auth,
        json={"act": "clean", "ttl_seconds": 30},
    )
    settlement = settlement_of(resident)
    # Reached by moving the lease's deadline into the past rather than by
    # sleeping through a ttl, then firing the one-shot the way the loop does.
    leases = resident.app.state.leases
    # The same lease, already expired.
    leases._lease = replace(leases._lease, expires_at=leases._lease.since)
    assert leases.current() is None, "the lease is past its deadline"
    assert settlement.holder() is None
    assert settlement.settle("the lease expired and was not renewed") is not None
    assert not is_resident(resident, auth)


def test_the_deadline_is_rearmed_by_a_heartbeat_rather_than_fixed_at_the_open(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """A live client always has its whole ttl left, and so does the one-shot."""
    lease_id = a_lease(resident, auth)
    settlement = settlement_of(resident)
    first = settlement._deadline
    assert first is not None
    beat = resident.post(f"/v1/leases/{lease_id}/heartbeat", headers=auth)
    assert beat.status_code == 200, beat.text
    assert settlement._deadline is not None
    assert settlement._deadline is not first

    assert resident.delete(f"/v1/leases/{lease_id}", headers=auth).status_code == 204
    # Released: there is no deadline left to watch.
    assert settlement._deadline is None


def test_the_facts_are_read_in_a_fixed_order_so_a_refusal_names_the_same_one(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """Four holders at once is one answer, and it is always the same one.

    Not cosmetic: `holder()` is what the log and every test above read, and a
    holder that changed with the wind would make the same state produce two
    different sentences.
    """
    residency = resident.app.state.residency
    a_lease(resident, auth)
    residency.claim("tts stream abc123", may_mutate=False)
    with resident.app.state.inflight.tracked(act="clean", model=MODEL, client=None):
        job_id = submit_echo(resident, auth, delay_ms=200)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a job"
        finish(resident, auth, job_id)
        assert settlement_of(resident).holder().fact == "a lease"
    residency.release("tts stream abc123")
    assert settlement_of(resident).holder().fact == "a lease"
    assert is_resident(resident, auth)


def test_nothing_about_activity_grew_a_second_owner_of_residency(
    resident: TestClient, auth: dict[str, str]
) -> None:
    """One fact, one owner (R1): the unload adds no row to `/v1/activity`.

    What is resident is already `resident`; why the last thing went away is the
    log's and the triggering job's, because it is history rather than state.
    """
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
    """Admitted and not yet picked up is as much a hold as already running.

    Clearing the card out from under it would cost it a reload it never asked
    for — `refuse_if_busy`'s reason, applied to the other end of a job's life.
    """
    store = resident.app.state.store
    job = store.create("echo", None, {})
    try:
        store._pending.append(job.id)
        held = settlement_of(resident).holder()
        assert held is not None and held.fact == "a job"
        assert job.id in held.who
        assert settlement_of(resident).settle("a test asked") is None
    finally:
        store._pending.remove(job.id)
        store.discard(job)

    # And it does not report ITSELF: a job asking at its own end must not find
    # its own row, or nothing would ever be unloaded.
    assert store.occupied_by_anything_but(None) is None


def test_the_server_still_tears_the_residency_down_on_the_way_out(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The ruling adds a door; it does not replace the one that was there.

    A load with nothing after it is the case the RULING OWED names, and until
    the load door can lease, shutdown is what catches it.
    """
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:
        run_load(client, auth)
        assert is_resident(client, auth)
    assert engines[0].pids == frozenset()


def test_nothing_settles_until_a_settlement_is_attached() -> None:
    """`crucible doctor` builds a store with no server around it and no card."""
    from crucible.jobs.queue import JobStore

    store = JobStore(object(), object(), {})
    assert store.occupied_by_anything_but(None) is None
    assert store._settlement is None


def test_the_grace_of_a_slow_stop_does_not_delay_the_answer(
    resident: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat's completion is written before the card is cleared behind it.

    Starlette runs a response's background task once the body has gone out, so a
    settlement that waits on a SIGTERM cannot hold somebody's answer hostage.
    """
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
