"""Admission: one job at a time, refused rather than queued.

Owen, 2026-09-13 (ARCHITECTURE.md section 3): *"i think all queuing logic should
exist in the clients, not the server. if the server is busy, it cant receive a
new job. if its not busy, it receives the next job requested."*

What these tests hold the server to is not "it says 409". It is that **the
refusal is worth having**: a bare "busy" would force clients to poll, and polling
is a worse queue than FIFO — the winner becomes whoever polls at the luckiest
moment rather than whoever asked first. So most of what is asserted here is the
*body*: who has the card, what they are running, since when, and how far along.

THE TIMING PROBLEM, handled the way `tests/test_activity.py` handles it: a job's
life is running → done, so every test that needs one in flight submits with a
long `delay_ms` and polls with a deadline rather than sleeping a guessed amount.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.errors import ApiError
from crucible.jobs.queue import JobStore

#: `echo` refuses `no_inputs`, so every submission carries one. A job that fails
#: in its first millisecond is never observed holding the lane.
PAYLOAD = {"x.bin": {"inline_base64": base64.b64encode(b"admission").decode("ascii")}}

#: Long enough that the lane is unambiguously occupied for the whole of a test,
#: short enough that a forgotten cancel does not stall the suite.
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


# ------------------------------------------------------------------ the refusal


def test_a_second_submission_is_refused_and_not_queued(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The ruling, in one assertion pair: 409, and the deque did not grow."""
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)

        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "server_busy"

        # The point of the ruling: nothing is waiting behind the running job,
        # because the client still owns its own queue.
        activity = client.get("/v1/activity", headers=auth).json()
        assert activity["queued"] == []
        assert activity["slots"]["accelerated"]["queue_depth"] == 1
        assert client.get("/v1/health", headers=auth).json()["status"] == "busy"
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_refusal_names_the_holder_and_what_it_is_doing(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A bare "busy" would make clients poll, and polling is a worse queue.

    This is also exactly the *"GPU busy: foundry"* line BookForge draws, which is
    why `holder` is asserted to be the User-Agent verbatim rather than merely
    present.
    """
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

        # These KEYS are the wire, not just this test's expectations. The
        # TypeScript SDK reads every one of them into `CrucibleBusy`
        # (sdk/ts/src/errors.ts) and answers a missing one with a
        # CrucibleProtocolError rather than a quiet downgrade — so renaming one
        # here without renaming it there breaks every bench's "GPU busy" line.
        # This assertion block is what goes red first when that happens.
        details = error["details"]
        assert details["holder"] == "bookforge/owens-pc crucible-client/0.4.0"
        assert details["job_id"] == first
        assert details["type"] == "echo"
        assert details["model"] is None
        assert details["status"] == "running"
        # `since` is the holder's own `started`, so a client can say how long it
        # has been waiting rather than guessing from when it happened to ask.
        assert details["since"] == state["started"]
        assert isinstance(details["progress"], float)
        # The holder's last progress line — "busy" turned into "copying x.bin".
        assert isinstance(details["message"], str) and details["message"]

        # The message is readable on its own, because it is what ends up in a log
        # on a machine nobody is looking at.
        assert first in error["message"]
        assert "bookforge/owens-pc" in error["message"]
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


def test_the_holder_is_null_when_the_client_did_not_say(
    make_app: Callable[..., Any], auth: dict[str, str]
) -> None:
    """`None` means "it did not say". A name invented here would be a bench
    confidently wrong about whose render is on the card."""
    # TestClient sends a User-Agent of its own; httpx drops an empty one.
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
    """`/v1/activity` withholds them and so must this: a chat prompt or a chapter
    of somebody's book is not something a refusal should hand back."""
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        refused = submit(client, auth, delay_ms=0)
        assert refused.status_code == 409
        assert "params" not in refused.json()["error"]["details"]
        assert "delay_ms" not in refused.text
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


# ------------------------------------------------------- the policy is the LANE


def test_echo_refuses_too_because_the_policy_is_the_lane_not_the_card(
    client: TestClient, auth: dict[str, str]
) -> None:
    """`echo` takes no accelerator, and is still refused. That is deliberate.

    The alternative — exempting the types that need no card — would put them
    straight back on a deque, because the lane is exclusive whatever a job wants
    from it. The server would then be queueing again for exactly the jobs it
    claimed not to queue for. So admission is about THE LANE: anything that
    occupies it is refused while it is occupied, and anything occupying it makes
    the server busy for everything else.

    This test is the symmetry: an `echo` holder refuses an `echo` submission, and
    the holder's type is on the refusal so a client is never left guessing what
    it is waiting for.
    """
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
    """Check order, asserted rather than assumed.

    A client with a typo must be told about the typo. `unknown_job_type` is true
    whether or not the lane is occupied, so it is answered first; `server_busy`
    is a fact about right now and comes after.
    """
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        response = client.post("/v1/jobs", json={"type": "summon"}, headers=auth)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "unknown_job_type"
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


# ------------------------------------------------------------ the lane frees up


def test_the_lane_takes_the_next_job_the_moment_it_is_free(
    client: TestClient, auth: dict[str, str]
) -> None:
    """*"if its not busy, it receives the next job requested."* The other half of
    the ruling, and the one that would be quietly broken by a `_running_id` left
    set on some path."""
    first = admitted(client, auth, delay_ms=0)
    assert wait_for_terminal(client, auth, first) == "done"
    second = admitted(client, auth, delay_ms=0)
    assert wait_for_terminal(client, auth, second) == "done"


def test_a_cancelled_job_frees_the_lane(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Cancel still works and still ends the occupancy — the queue was not
    removed, only its admission policy changed."""
    first = admitted(client, auth, delay_ms=HELD_MS)
    wait_until_running(client, auth, first)
    assert submit(client, auth, delay_ms=0).status_code == 409

    assert client.delete(f"/v1/jobs/{first}", headers=auth).status_code == 200
    assert wait_for_terminal(client, auth, first) == "cancelled"
    assert wait_for_terminal(client, auth, admitted(client, auth, delay_ms=0)) == "done"


def test_a_failed_job_frees_the_lane(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A job that fails releases the lane as surely as one that succeeds. Worth
    its own test: a `_running_id` that survived a failure would wedge the server
    permanently, and under the old policy the symptom was a growing queue rather
    than a server that refuses everything forever."""
    failed = client.post(
        "/v1/jobs", json={"type": "echo", "params": {}, "inputs": {}}, headers=auth
    )
    assert failed.status_code == 202, failed.text
    assert wait_for_terminal(client, auth, failed.json()["job_id"]) == "failed"
    assert wait_for_terminal(client, auth, admitted(client, auth, delay_ms=0)) == "done"


# ------------------------------------------------------- a refusal costs nothing


def test_a_refused_submission_leaves_nothing_behind(
    client: TestClient, auth: dict[str, str]
) -> None:
    """No job record, no scratch directory, no inputs written and deleted.

    Admission is asked BEFORE `store.create` precisely so that a client retrying
    in a loop cannot thrash the disk of a server that is busy rendering a book.
    """
    jobs_dir = Path(client.app.state.config.jobs_dir)
    first = admitted(client, auth, delay_ms=HELD_MS)
    try:
        wait_until_running(client, auth, first)
        before = sorted(p.name for p in jobs_dir.iterdir())

        for _ in range(5):
            assert submit(client, auth, delay_ms=0).status_code == 409

        assert sorted(p.name for p in jobs_dir.iterdir()) == before
        # And the store did not gain five jobs nobody can ever reach.
        store: JobStore = client.app.state.store
        assert len(store._jobs) == 1
    finally:
        client.delete(f"/v1/jobs/{first}", headers=auth)


# -------------------------------------------------- the window the lane owns


def test_an_admitted_job_the_lane_has_not_reached_yet_still_refuses(
    client: TestClient, auth: dict[str, str]
) -> None:
    """THE RACE, and why admission reads `_pending` and not only `_running_id`.

    `enqueue` appends and sets `_wake`; the lane is a task on the same event loop
    and does not resume until the current one yields. So there is a window in
    which a job is admitted and nothing is running yet — and a check that asked
    only "is something running" would let a second job in, `_pending` would reach
    2, and the server would be queueing again. Two clients a millisecond apart is
    all it takes.

    The window is sub-millisecond over HTTP, so it is reproduced here on a store
    whose lane was never started: that is the same state, held still. The job is
    genuinely admitted — `queued`, `position` 1, `queue_depth` 1 — which is also
    the one place those still-supported values are exercised now that no client
    can observe them through the API.
    """
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
    assert details["job_id"] == first.id
    assert details["status"] == "queued"
    assert details["holder"] == "foundry/owens-pc"
    # `created`, not a null `started`: "busy since never" is not an answer.
    assert details["since"] == first.created
    assert details["progress"] == 0.0


def test_enqueue_is_the_authority_and_refuses_on_its_own(
    client: TestClient, auth: dict[str, str]
) -> None:
    """`POST /v1/jobs` asks first to avoid building a job it will throw away, but
    the append is where the decision has to be final. A future handler that grew
    an `await` between the two must not be able to put a second job on the lane.
    """
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
    """A job exists from `create()`, before its inputs are written and before it
    is admitted. Anything refused in between leaves no record: one left at
    `queued` without being on the lane would make `position()` raise off
    `deque.index` for anyone who asked about it."""
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
