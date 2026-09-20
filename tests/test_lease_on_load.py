"""A load can hold what it made resident, from the instant it exists.

`settle.py`'s "half that is still open", closed 2026-09-20. A load that succeeds
is exempt from settling — its whole content is "be resident", so its own end
cannot clear the card — which leaves the window between its `done` and its
client's `POST /v1/models/{id}/lease` held by NOTHING. A client that dies in that
window strands the card for ever, because a quiet hold has no end. That is
exactly how a 21 GB model sat on the PC on 2026-09-20: the load completed, the
runner was stopped one second later, and it walked away before leasing.

WHY THIS REOPENS A RULING `settle.py` ALREADY MADE
---------------------------------------------------
That file rejected lease-on-load, in as many words: *"a lease is another
holder... A load that took one would hold the card for its whole ttl AND refuse
everybody else meanwhile, which is strictly worse than a load that holds it
quietly."*

True of a HUMAN typing `crucible load` and walking away — and backwards for a
programmatic client, because **a quiet hold never expires and a lease does**. A
runner that dies mid-book strands the card for ever today; with a lease it
self-heals when the ttl runs out. The two cases separate without anyone guessing
which is which: by whether the request asks for one. An operator asks for
nothing and keeps today's behaviour exactly.

The other half of that is `Settlement.settle_for_lapsed_lease` — a lease is only
a self-healing hold if something acts when it lapses, and until 2026-09-20
nothing did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from .fake_engine import FakeEngine
from .test_llm_api import (  # noqa: F401
    MODEL,
    engines,
    fake_env,
    llm_client,
    run_job,
    submit,
)


def _events(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    return run_job(client, auth, **body)


def _done(events: list[dict]) -> dict[str, Any]:
    terminal = [event for event in events if event["event"] == "done"]
    assert terminal, [
        (event["event"], event["data"]) for event in events
    ]
    return terminal[-1]["data"]


def test_a_load_without_a_lease_is_exactly_what_it_was(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """THE OPERATOR CASE, untouched. `crucible load` and walk away still holds
    the card quietly, because nothing asked for a lease."""
    fake_weights(MODEL)
    done = _done(_events(llm_client, auth, type="load-model", model=MODEL))

    assert done["resident"] == MODEL
    # STATED, AND NULL. "this load held nothing" and "this server does not speak
    # leases on a load" are different pieces of news, and an absent key would be
    # the second one. A client cannot tell them apart from a hole.
    assert done["lease_id"] is None
    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["lease"] is None
    # And this is the stranded card the 1.0.11 fields exist to say out loud.
    assert activity["resident"]["held_by"] is None


def test_a_load_that_asks_for_a_lease_is_held_the_moment_it_is_resident(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """THE WINDOW, CLOSED. There is no moment between `done` and the hold."""
    fake_weights(MODEL)
    done = _done(
        _events(
            llm_client,
            auth,
            type="load-model",
            model=MODEL,
            params={"lease": {"act": "clean", "ttl_seconds": 120}},
        )
    )

    assert done["resident"] == MODEL
    lease_id = done["lease_id"]
    assert isinstance(lease_id, str) and lease_id

    activity = llm_client.get("/v1/activity", headers=auth).json()
    # `lease_id`, not `id`, and NO `subject`: a lease is only ever on the
    # resident thing, which `resident.id` already owns (crucible/leases.py's
    # `to_dict` — one fact, one owner, in one document).
    assert activity["lease"]["lease_id"] == lease_id
    assert activity["lease"]["act"] == "clean"
    assert activity["resident"]["id"] == MODEL
    # HELD, so the card is nobody's to reclaim and there is nothing to date.
    assert activity["resident"]["held_by"]["fact"] == "a lease"
    assert activity["resident"]["unclaimed_since"] is None


def test_the_lease_id_is_readable_after_the_stream_is_gone(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """THE RECEIPT IS NOT THE FACT. A client that lost the events stream must be
    able to read its own lease id back — a `lease_id` nobody can recover is a
    hold nobody can release, which is the stranding this release exists to end.
    """
    fake_weights(MODEL)
    response = submit(
        llm_client,
        auth,
        type="load-model",
        model=MODEL,
        params={"lease": {"act": "clean", "ttl_seconds": 120}},
    )
    job_id = response.json()["job_id"]
    frames: list[str] = []
    with llm_client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        for line in stream.iter_lines():
            frames.append(line)
    import json as _json

    payloads = [
        _json.loads(line[len("data: ") :])
        for line in frames
        if line.startswith("data: ")
    ]
    done_frame_lease_id = next(
        payload["lease_id"] for payload in payloads if "lease_id" in payload
    )

    record = llm_client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert record["status"] == "done"
    assert record["lease_id"]
    assert record["lease_id"] == done_frame_lease_id
    assert record["resident"] == MODEL
    # And the job record's own keys are still its own.
    assert record["job_id"] == job_id
    assert record["type"] == "load-model"


def test_a_released_lease_clears_the_card(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """THE WHOLE POINT, end to end: Stop is now a release, and the settlement
    clears the card through the one existing unload door.

    This is what makes §F.8 unnecessary — `DELETE /v1/jobs/{id}` never had to
    learn a second meaning.
    """
    fake_weights(MODEL)
    done = _done(
        _events(
            llm_client,
            auth,
            type="load-model",
            model=MODEL,
            params={"lease": {"act": "clean", "ttl_seconds": 120}},
        )
    )
    assert llm_client.get("/v1/activity", headers=auth).json()["resident"] is not None

    released = llm_client.delete(
        f"/v1/leases/{done['lease_id']}", headers=auth
    )
    assert released.status_code in (200, 204), released.text

    assert llm_client.get("/v1/activity", headers=auth).json()["resident"] is None


def test_an_invalid_act_is_refused_in_the_lease_doors_own_words(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """One vocabulary. A load's lease and `POST /v1/models/{id}/lease` must not
    come to disagree about what a valid act is, so the load uses that door's own
    validator rather than a second spelling."""
    fake_weights(MODEL)
    events = _events(
        llm_client,
        auth,
        type="load-model",
        model=MODEL,
        params={"lease": {"act": "not-a-capability", "ttl_seconds": 120}},
    )
    failed = [event for event in events if event["event"] == "failed"]
    assert failed, events
    assert failed[-1]["data"]["error"]["code"] == "unknown_act"


def test_an_out_of_range_ttl_is_refused_by_the_same_rule(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    events = _events(
        llm_client,
        auth,
        type="load-model",
        model=MODEL,
        params={"lease": {"act": "clean", "ttl_seconds": 5}},
    )
    failed = [event for event in events if event["event"] == "failed"]
    assert failed, events
    assert "ttl" in failed[-1]["data"]["error"]["message"].lower()


def test_an_unknown_key_in_the_lease_block_is_refused_not_ignored(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """`extra="forbid"`, so a typo is a refusal and not a lease that quietly
    does something other than what was asked."""
    fake_weights(MODEL)
    response = submit(
        llm_client,
        auth,
        type="load-model",
        model=MODEL,
        params={"lease": {"act": "clean", "ttl_seconds": 120, "hold": "lane"}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
