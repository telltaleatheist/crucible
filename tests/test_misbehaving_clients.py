"""The server under clients that do the wrong thing. Seeded from real failures.

Owen, 2026-09-20: *"the system is entirely too fragile. harden it so it works
correctly."* Every row below is a client behaviour that actually happened on
2026-09-20 and took a book with it (`docs/BUG-HUNT-2026-09-20.md` §0, S1-S14).
The question these ask is not "does the door refuse correctly" — the other suites
ask that — but **"is the server still whole afterwards"**: no leaked slot, no
poisoned statistic, no state a later well-behaved client would trip over.

That distinction matters because a refusal is a code path that runs far more
often than a success once something upstream is misconfigured, and it is the path
nobody watches. S10 is exactly that shape: Crucible 1.0.10 correctly advertised
`chat.max_in_flight: 2` and correctly refused a client that kept sending 4; the
pass still died. The refusals were right and the outcome was still a dead book.

WHAT S10 WOULD HAVE COST IF THE REFUSAL PATH LEAKED
---------------------------------------------------
`_chat_queue_full` returns BEFORE `inflight.open()`, so a refusal opens no record
and closes none. Two things depend on that and neither is obvious:

* **A refused request must not consume a slot.** If it did, every refusal would
  make the next one likelier, and a door at its limit would wedge shut for good
  under a client that retries — which is precisely what a client with a hardcoded
  pool of 4 does against a limit of 2.
* **A refused request must not enter the duration record.** A refusal takes
  microseconds. If refusals were counted as completions, the median behind
  `Retry-After` would collapse toward the 1 s floor, and the server would answer
  a storm by telling every caller to come straight back — turning contention into
  a spin at exactly the moment it is least affordable.

Both are properties of where one `return` sits, which is the kind of thing an
innocent refactor moves. Hence the keepers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.engines import ENGINES
from crucible.inflight import InFlight

from .fake_engine import FakeEngine

# The fixtures live where the llm door's own tests are. Imported by name rather
# than copied, so this file cannot drift from the setup those tests prove works —
# `fake_env` comes with them deliberately: `test_llm_api` overrides conftest's,
# and resolving to the wrong one would give this module a different server.
from .test_llm_api import (  # noqa: F401
    MODEL,
    engines,
    fake_env,
    llm_client,
    run_job,
    submit,
)


@pytest.fixture
def serial_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every engine states a concurrency of 1, as mlx-lm really does.

    All of them rather than the resident one: which engine the fake manifest
    names is not what any of this is about.
    """
    for cls in ENGINES.values():
        monkeypatch.setattr(cls, "chat_concurrency_flag", None, raising=False)
        monkeypatch.setattr(cls, "chat_concurrency", 1, raising=False)
        monkeypatch.setattr(
            cls, "chat_concurrency_basis", "one generation thread", raising=False
        )


def _load_and_wait(client: TestClient, auth: dict[str, str]) -> str:
    """Run a `load-model` to completion and return its id.

    `run_job` hands back the event list, and these two tests need the id to
    cancel it afterwards — which is the whole of S14.
    """
    response = submit(client, auth, type="load-model", model=MODEL)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        for _ in stream.iter_lines():
            pass
    return job_id


def _chat(client: TestClient, auth: dict[str, str]) -> Any:
    return client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )


# ---------------------------------------------------------------------------
# S10 · a client whose pool is larger than the server's stated admission
# ---------------------------------------------------------------------------


def test_a_storm_of_refusals_leaks_no_slot(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    serial_engines: None,
) -> None:
    """S10. The door must not wedge shut under a client that will not stop.

    A pool of four against an admission of two is a misconfiguration the server
    cannot fix — but it must survive it, and the two requests that ARE admissible
    must keep being admitted. If a refusal consumed a slot, the sixth refusal
    would be inevitable rather than unlucky.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    held = [inflight.open(act=None, model=MODEL, client="a-test") for _ in range(2)]
    try:
        for _ in range(20):
            assert _chat(llm_client, auth).status_code == 503
            # The count never moves. A refusal is not a completion.
            assert len(inflight) == 2
    finally:
        for entry in held:
            inflight.close(entry)

    # And the door is not sulking: with the slots free it answers again.
    assert len(inflight) == 0
    assert _chat(llm_client, auth).status_code == 200


def test_refusals_do_not_poison_the_retry_after(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    serial_engines: None,
) -> None:
    """THE SUBTLE HALF OF S10, and the one that would make a storm worse.

    A refusal takes microseconds. Counted as a completion it would drag the
    median behind `Retry-After` to the 1 s floor, so the server would answer a
    pile-up by inviting everyone straight back. The record must only ever hold
    work that actually ran.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    # ONE HELD RECORD THROUGHOUT, and it is not scaffolding. A chat that
    # completes on a card nothing holds settles it — `settle.py`'s rule that a
    # run of chats without a lease reloads its model — and the next request
    # would be `model_not_resident`, not the refusal under test. A held record
    # is the cheapest stand-in for the lease a real client of a run would take.
    keeper = inflight.open(act=None, model=MODEL, client="a-test")
    try:
        # One real completion, so there IS a measurement to poison. One in
        # flight against a limit of two, so it is admitted.
        assert _chat(llm_client, auth).status_code == 200
        measured = len(inflight._recent)  # noqa: SLF001
        assert measured == 1

        second = inflight.open(act=None, model=MODEL, client="a-test")
        try:
            for _ in range(15):
                assert _chat(llm_client, auth).status_code == 503
        finally:
            inflight.close(second)
    finally:
        inflight.close(keeper)

    # Fifteen refusals added NOTHING. The two held records each added one when
    # they closed, and nothing else did.
    assert len(inflight._recent) == measured + 2  # noqa: SLF001


def test_the_refusal_states_the_limit_the_client_should_have_read(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    serial_engines: None,
) -> None:
    """A client with a hardcoded pool has to be able to learn the real one from
    the refusal itself, not only from a poll it may never make."""
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    held = [inflight.open(act=None, model=MODEL, client="a-test") for _ in range(2)]
    try:
        response = _chat(llm_client, auth)
    finally:
        for entry in held:
            inflight.close(entry)

    details = response.json()["error"]["details"]
    assert details["max_in_flight"] == 2
    assert details["max_in_flight_basis"] == "one generation thread"
    # And the same number is on the bench read, so the two cannot disagree.
    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["chat"]["max_in_flight"] == details["max_in_flight"]


# ---------------------------------------------------------------------------
# S14 · a client that cancels a job which has already finished
# ---------------------------------------------------------------------------


def test_cancelling_a_finished_load_is_refused_and_changes_nothing(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """S14, as it happened on the PC at 18:29:38Z.

    The Stop landed one second after the load reached `done`, so the cancel door
    saw a terminal job and refused it. What this pins is that the refusal is
    INERT — it must not half-cancel a job whose work is finished, and the model
    the load was asked for must still be there, because a client that reads
    `409 already done` and then unloads is relying on exactly that.

    What SHOULD happen instead of an inert refusal is §F.8 and is Owen's ruling.
    This test will need rewriting when he makes it, and that is the point: it
    says what today does, so a change to it is visible.
    """
    fake_weights(MODEL)
    job_id = _load_and_wait(llm_client, auth)

    refused = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "job_not_cancellable"

    after = llm_client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert after["status"] == "done"

    # THE STRANDED CARD. The model is still resident and, as of 1.0.11, the
    # server says out loud that nothing holds it — which is what lets a client
    # reconcile instead of guessing.
    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["resident"] is not None
    assert activity["resident"]["id"] == MODEL
    assert activity["resident"]["held_by"] is None
    assert activity["resident"]["unclaimed_since"] is not None


def test_a_second_cancel_of_the_same_finished_job_is_the_same_answer(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """A client that retries its Stop must not get a different story each time."""
    fake_weights(MODEL)
    job_id = _load_and_wait(llm_client, auth)

    first = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    second = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert first.status_code == second.status_code == 409
    assert first.json()["error"]["code"] == second.json()["error"]["code"]


# ---------------------------------------------------------------------------
# The registry itself, under abuse
# ---------------------------------------------------------------------------


def test_closing_an_entry_that_was_never_opened_is_not_an_error() -> None:
    """The streamed chat door really does close twice on some paths, and a
    relay that outlives its handler can close after a reset."""
    flight = InFlight()
    entry = flight.open(act=None, model="q", client=None)
    flight.close(entry)
    flight.close(entry)
    flight.close(entry)
    assert len(flight) == 0
    assert len(flight._recent) == 1  # noqa: SLF001


def test_the_count_survives_many_open_close_cycles() -> None:
    """A long clean pass is thousands of these; a count that drifted by one per
    thousand would close the door silently after a few books."""
    flight = InFlight()
    for _ in range(2000):
        flight.close(flight.open(act=None, model="q", client=None))
    assert len(flight) == 0


# ---------------------------------------------------------------------------
# A load that fails in a way nobody wrote an `except` for
# ---------------------------------------------------------------------------


def test_a_load_that_fails_any_way_at_all_takes_its_engine_down() -> None:
    """ASKED BY THE BookForge HUNT: can a half-started engine outlive its load?

    `Residency._start` tidies up a half-started engine — it always did, for
    `EngineError`, which is a `ready()` timeout and narrator's own refusals. What
    it did not cover was every OTHER way the block can end: `NarratorEngine.load`
    raises `JobCancelled`, and a transport answering nonsense raises whatever
    `json` raises.

    That gap does not leak an object, it leaks A LIVE GPU PROCESS NOTHING CAN
    SEE. `self._engine` and `self._resident` are assigned only after `_start`
    returns, so an engine orphaned inside it sits in no slot at all: the card
    reports `resident: null`, the settlement has nothing to unload, and
    `owned_pids()` — which reads those same slots — omits its pid, so the
    accelerator guard calls Crucible's own child a foreign process. The
    `held_by`/`unclaimed_since` fields cannot show it either, because nothing is
    resident.

    Each exception below is one a real caller can produce.
    """

    class _Recording:
        """The engine's whole surface as `_start` uses it."""

        def __init__(self) -> None:
            self.stopped = False

        def start(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def ready(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def stop(self) -> None:
            self.stopped = True

    from crucible.errors import JobCancelled
    from crucible.residency import Residency

    for boom in (
        JobCancelled("narrator was cancelled mid-request"),
        KeyError("loaded"),
        ValueError("narrator answered something that is not JSON"),
        KeyboardInterrupt(),
    ):
        engine = _Recording()

        def confirm(_boom: BaseException = boom) -> None:
            raise _boom

        with pytest.raises(type(boom)):
            Residency._start(  # noqa: SLF001
                engine,  # type: ignore[arg-type]
                Path("weights"),
                "served",
                1,
                [],
                lambda _message: None,
                1.0,
                confirm=confirm,
            )
        assert engine.stopped is True, (
            f"a load that ended in {type(boom).__name__} left its engine "
            "running, in no slot, invisible to /v1/activity and to owned_pids()"
        )


def test_the_original_failure_is_what_the_caller_is_told() -> None:
    """The teardown must not become the story. A caller debugging a failed load
    needs the reason it failed, not the fact that tidying up worked."""

    class _Fine:
        def start(self, *_a: Any, **_k: Any) -> None:
            return None

        def ready(self, *_a: Any, **_k: Any) -> None:
            return None

        def stop(self) -> None:
            return None

    from crucible.residency import Residency

    def confirm() -> None:
        raise ValueError("the sample rate disagreed")

    with pytest.raises(ValueError, match="the sample rate disagreed"):
        Residency._start(  # noqa: SLF001
            _Fine(),  # type: ignore[arg-type]
            Path("weights"),
            "served",
            1,
            [],
            lambda _message: None,
            1.0,
            confirm=confirm,
        )
