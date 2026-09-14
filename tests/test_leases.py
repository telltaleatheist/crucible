"""The model lease — a client says it intends a run. PHASE7-LANES.md section 5.2.

THE DEFECT THESE TESTS DESCRIBE. Foundry translates a book as two thousand chat
completions against a resident 27B. A chat holds nothing — no lane, no job, no
claim — so between any two blocks this server reports itself idle, and BookForge
submitting a `load-voice` at block 400 took the translator off the card with
nothing having gone wrong anywhere. A lease is the client saying the run exists.

NOTHING HERE SLEEPS. Expiry is decided at read time from an injected clock, so
every deadline in this file is reached by advancing a number. A test that slept
through a 30-second ttl would be a test nobody runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible.jobs import ALL_JOB_TYPES
from crucible.leases import (
    EVICTS_THE_RESIDENT_MODEL,
    KEEPS_THE_RESIDENT_MODEL,
    MAX_TTL_SECONDS,
    MIN_TTL_SECONDS,
    Leases,
)

from .conftest import parse_sse
from .fake_engine import FakeEngine

MODEL = "qwen3.5-9b"
OTHER_MODEL = "qwen3.8-27b"
TTL = 60


# ------------------------------------------------------------------- fixtures


class Clock:
    """A clock a test moves by hand. `Leases` asks it; nothing else does."""

    def __init__(self, start: datetime) -> None:
        self.moment = start

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> None:
        self.moment = self.moment + timedelta(seconds=seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock(datetime(2026, 9, 14, 3, 0, 0, tzinfo=timezone.utc))


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    """Submit a job and read its whole event stream. Fails loudly if refused."""
    response = client.post("/v1/jobs", headers=auth, json=body)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


@pytest.fixture
def resident_client(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    clock: Clock,
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> Iterator[TestClient]:
    """A server with a model actually resident, and a clock a test can move.

    `enable_tts` as well as `enable_llm`, because the thing a lease has to refuse
    is a voice loader, and the point of the test is that it is refused BEFORE
    anything about tts is checked.
    """
    engine_factory()
    fake_weights(MODEL)
    with make_client(enable_llm=True, enable_tts=True) as client:
        client.app.state.leases = Leases(now=clock)
        run_job(client, auth, type="load-model", model=MODEL)
        yield client


def take(
    client: TestClient,
    auth: dict[str, str],
    *,
    model: str = MODEL,
    act: str = "translate",
    ttl_seconds: int = TTL,
) -> Any:
    return client.post(
        f"/v1/models/{model}/lease",
        headers=auth,
        json={"act": act, "ttl_seconds": ttl_seconds},
    )


def a_lease(client: TestClient, auth: dict[str, str], **options: Any) -> dict[str, Any]:
    response = take(client, auth, **options)
    assert response.status_code == 201, response.text
    return response.json()


def activity(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    response = client.get("/v1/activity", headers=auth)
    assert response.status_code == 200, response.text
    return response.json()


def a_voice(client: TestClient, auth: dict[str, str]) -> str:
    """A voice id this build really has, read off the server rather than typed."""
    rows = client.get("/v1/voices", headers=auth).json()
    assert rows, "the tts fixture serves no voices, so nothing can be load-voiced"
    return str(rows[0]["id"])


# ------------------------------------------------------------- the happy path


def test_take_heartbeat_release_is_the_whole_life_of_a_lease(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    headers = {**auth, "User-Agent": "foundry/owens-pc crucible-client/0.5.0"}
    opened = resident_client.post(
        f"/v1/models/{MODEL}/lease",
        headers=headers,
        json={"act": "translate", "ttl_seconds": TTL},
    )
    assert opened.status_code == 201, opened.text
    lease = opened.json()
    assert lease["model"] == MODEL
    assert lease["act"] == "translate"
    # The same name `/v1/activity` reports for a job's holder, from the same
    # reader: one column on a bench, one way of filling it.
    assert lease["client"] == "foundry/owens-pc crucible-client/0.5.0"
    assert lease["since"] == clock.moment.isoformat()
    assert lease["expires_at"] == (clock.moment + timedelta(seconds=TTL)).isoformat()

    # Half a ttl later the run is still going, and says so.
    clock.advance(TTL / 2)
    beat = resident_client.post(
        f"/v1/leases/{lease['lease_id']}/heartbeat", headers=headers
    )
    assert beat.status_code == 200, beat.text
    # Extended from NOW by the ttl it was opened with, not from the old deadline:
    # a client that heartbeats always has its full ttl left.
    assert beat.json() == {
        "expires_at": (clock.moment + timedelta(seconds=TTL)).isoformat()
    }

    released = resident_client.delete(
        f"/v1/leases/{lease['lease_id']}", headers=headers
    )
    assert released.status_code == 204
    assert released.content == b""
    assert activity(resident_client, auth)["lease"] is None


def test_activity_reports_the_open_lease_and_not_the_model_twice(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    """`resident.id` already owns which model it is (R1)."""
    lease = a_lease(resident_client, auth, act="clean")
    body = activity(resident_client, auth)

    assert body["lease"] == {
        "lease_id": lease["lease_id"],
        "client": body["lease"]["client"],
        "act": "clean",
        "since": clock.moment.isoformat(),
        "expires_at": (clock.moment + timedelta(seconds=TTL)).isoformat(),
    }
    assert "model" not in body["lease"]
    assert body["resident"]["id"] == MODEL

    # A lease is not a reservation. The lane is free and this server will still
    # take work that leaves the card's contents alone.
    assert body["slots"]["accelerated"]["accepts_work"] is True


def test_an_idle_server_says_null_rather_than_leaving_the_key_out(
    client: TestClient, auth: dict[str, str]
) -> None:
    """A present null is a statement; an absent key is a different build."""
    body = activity(client, auth)
    assert "lease" in body
    assert body["lease"] is None


def test_a_restart_forgets_every_lease(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    """In memory, deliberately: a restarted server holds no model to protect."""
    engine_factory()
    fake_weights(MODEL)
    with make_client(enable_llm=True) as first:
        run_job(first, auth, type="load-model", model=MODEL)
        lease = a_lease(first, auth)
    with make_client(enable_llm=True) as second:
        assert activity(second, auth)["lease"] is None
        gone = second.post(f"/v1/leases/{lease['lease_id']}/heartbeat", headers=auth)
        assert gone.status_code == 404
        assert gone.json()["error"]["code"] == "unknown_lease"


# ---------------------------------------------------------------- the refusals


def test_a_model_that_is_not_resident_cannot_be_leased(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """A lease promises not to move what is there; it never loads anything."""
    response = take(resident_client, auth, model=OTHER_MODEL)
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert error["details"] == {"requested": OTHER_MODEL, "resident": MODEL}


def test_leasing_an_empty_card_says_no_model_is_resident(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    with make_client(enable_llm=True) as client:
        response = take(client, auth)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert error["details"]["resident"] is None


def test_an_act_that_is_not_a_capability_class_is_refused_by_name(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """The same vocabulary and the same refusal `X-Crucible-Act` gets.

    A lease that recorded `translat` would put a name nothing knows on a bench,
    which is the defect Owen ruled out on 2026-09-13 — *"they can't lie to the
    user and say a translate job is running when it's actually a simplify job."*
    """
    response = take(resident_client, auth, act="translat")
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "unknown_act"
    assert "'translat'" in error["message"]
    assert "a lease's `act`" in error["message"]
    assert "translate" in error["message"] and "simplify" in error["message"]
    assert activity(resident_client, auth)["lease"] is None


def test_a_ttl_outside_the_range_is_refused_at_both_ends_with_the_range(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    for ttl in (MIN_TTL_SECONDS - 1, MAX_TTL_SECONDS + 1, 0, -60):
        response = take(resident_client, auth, ttl_seconds=ttl)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_ttl"
        # The range is stated rather than implied: a client refused without one
        # has to guess, and guessing is how 3600 becomes 3601 twice.
        assert error["details"]["min_ttl_seconds"] == MIN_TTL_SECONDS
        assert error["details"]["max_ttl_seconds"] == MAX_TTL_SECONDS
        assert str(MIN_TTL_SECONDS) in error["message"]
        assert str(MAX_TTL_SECONDS) in error["message"]

    # Both ends of the range itself are fine. The model is loaded again between
    # the two, because releasing the first lease is a holder letting go and the
    # card is cleared behind it (Owen's ruling, crucible/settle.py) — so the
    # second `POST .../lease` would otherwise be answered `model_not_resident`,
    # truthfully.
    for ttl in (MIN_TTL_SECONDS, MAX_TTL_SECONDS):
        if activity(resident_client, auth)["resident"] is None:
            run_job(resident_client, auth, type="load-model", model=MODEL)
        lease = a_lease(resident_client, auth, ttl_seconds=ttl)
        assert (
            resident_client.delete(
                f"/v1/leases/{lease['lease_id']}", headers=auth
            ).status_code
            == 204
        )


def test_one_lease_at_a_time_and_the_second_is_told_who_has_it(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    first = a_lease(
        resident_client,
        auth,
        act="translate",
    )
    second = take(resident_client, auth, act="clean")
    assert second.status_code == 409, second.text
    error = second.json()["error"]
    assert error["code"] == "model_leased"
    assert error["details"] == {
        "lease_id": first["lease_id"],
        "client": first["client"],
        "act": "translate",
        "since": first["since"],
        "expires_at": first["expires_at"],
    }
    assert MODEL in error["message"] and "translate" in error["message"]


def test_a_heartbeat_or_a_release_of_a_lease_nobody_has_is_a_404(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    for path in ("/v1/leases/nosuchlease/heartbeat",):
        response = resident_client.post(path, headers=auth)
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "unknown_lease"
    gone = resident_client.delete("/v1/leases/nosuchlease", headers=auth)
    assert gone.status_code == 404
    assert gone.json()["error"]["code"] == "unknown_lease"


def test_a_released_lease_says_it_was_released(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    lease = a_lease(resident_client, auth)
    assert (
        resident_client.delete(
            f"/v1/leases/{lease['lease_id']}", headers=auth
        ).status_code
        == 204
    )
    again = resident_client.post(
        f"/v1/leases/{lease['lease_id']}/heartbeat", headers=auth
    )
    assert again.status_code == 404, again.text
    error = again.json()["error"]
    assert error["code"] == "unknown_lease"
    assert "released" in error["details"]["reason"]
    # Releasing twice is the same 404 and not a 204: a client that thinks it
    # still holds a lease has to be told it does not.
    assert (
        resident_client.delete(
            f"/v1/leases/{lease['lease_id']}", headers=auth
        ).status_code
        == 404
    )


def test_the_lease_doors_need_the_token_like_every_private_route(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    assert (
        resident_client.post(
            f"/v1/models/{MODEL}/lease", json={"act": "clean", "ttl_seconds": TTL}
        ).status_code
        == 401
    )
    assert resident_client.post("/v1/leases/x/heartbeat").status_code == 401
    assert resident_client.delete("/v1/leases/x").status_code == 401


# --------------------------------------------------------------------- expiry


def test_a_lease_past_its_deadline_is_simply_not_open(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    """No sweeper, no background task: the clock decides at read time.

    So a client that dies mid-run stops blocking the card whether or not
    anything was watching — which is the only reason a ttl exists at all.
    """
    lease = a_lease(resident_client, auth)
    voice = a_voice(resident_client, auth)

    clock.advance(TTL + 1)

    assert activity(resident_client, auth)["lease"] is None
    refused = resident_client.post(
        "/v1/jobs", headers=auth, json={"type": "load-voice", "model": voice}
    )
    assert refused.json().get("error", {}).get("code") != "model_leased"

    expired = resident_client.post(
        f"/v1/leases/{lease['lease_id']}/heartbeat", headers=auth
    )
    assert expired.status_code == 404, expired.text
    assert "expired" in expired.json()["error"]["details"]["reason"]


def test_a_client_may_lease_again_after_its_own_lease_expired(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    first = a_lease(resident_client, auth)
    clock.advance(TTL + 1)
    second = a_lease(resident_client, auth)
    assert second["lease_id"] != first["lease_id"]
    assert second["since"] == clock.moment.isoformat()


def test_a_heartbeat_keeps_it_open_past_the_original_deadline(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    lease = a_lease(resident_client, auth)
    for _ in range(4):
        clock.advance(TTL - 5)
        beat = resident_client.post(
            f"/v1/leases/{lease['lease_id']}/heartbeat", headers=auth
        )
        assert beat.status_code == 200, beat.text
    # Four ttls past where it would have died, still the same lease.
    assert activity(resident_client, auth)["lease"]["lease_id"] == lease["lease_id"]


# ------------------------------------------------------ what it does and does not


def test_a_lease_refuses_every_loader_that_would_move_the_model(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """Including the holder's own. A run that wants the card to change releases
    first — an exemption would make "who asked" part of the answer, and the
    server has no way to tell two runs from one client apart."""
    lease = a_lease(resident_client, auth)
    voice = a_voice(resident_client, auth)
    for body in (
        {"type": "load-voice", "model": voice},
        {"type": "load-model", "model": OTHER_MODEL},
        {"type": "unload-model", "model": MODEL},
    ):
        response = resident_client.post("/v1/jobs", headers=auth, json=body)
        assert response.status_code == 409, (body, response.text)
        error = response.json()["error"]
        assert error["code"] == "model_leased", body
        assert error["details"]["lease_id"] == lease["lease_id"]
        assert error["details"]["act"] == "translate"


def test_a_lease_does_not_stop_a_chat_because_chats_are_what_it_protects(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    a_lease(resident_client, auth)
    response = resident_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=json.dumps(
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8"),
    )
    assert response.status_code == 200, response.text


def test_a_lease_does_not_stop_work_that_leaves_the_card_alone(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """`echo` never touches the accelerator, so a lease has nothing to say."""
    a_lease(resident_client, auth)
    response = resident_client.post(
        "/v1/jobs",
        headers=auth,
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"x.bin": {"inline_base64": "YQ=="}},
        },
    )
    assert response.status_code == 202, response.text


def test_the_refusal_lands_before_the_lane_and_before_preflight(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """A `load-voice` for a voice whose env was never installed is refused for
    the LEASE, not for the env: the durable reason is the one a client needs, and
    it is reached without shelling out to ffmpeg or probing the card."""
    a_lease(resident_client, auth)
    voice = a_voice(resident_client, auth)
    response = resident_client.post(
        "/v1/jobs", headers=auth, json={"type": "load-voice", "model": voice}
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "model_leased"


def test_a_typo_in_the_job_type_still_wins_over_the_lease(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """The job door's existing rule: a client with a typo is told about the typo
    rather than about somebody else's lease."""
    a_lease(resident_client, auth)
    response = resident_client.post(
        "/v1/jobs", headers=auth, json={"type": "load-voic", "model": "whatever"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "unknown_job_type"


# --------------------------------------------------------------- the drift guard


def test_every_job_type_this_build_knows_is_ruled_on(
    resident_client: TestClient, auth: dict[str, str]
) -> None:
    """A new job type must say whether it can take the model off the card.

    This is the check that makes `EVICTS_THE_RESIDENT_MODEL` a fact with an owner
    rather than a list somebody hoped was complete (R1). Without it, a job type
    added next month that loads something would reopen exactly the hole the
    lease was built to close, silently.
    """
    ruled = EVICTS_THE_RESIDENT_MODEL | KEEPS_THE_RESIDENT_MODEL
    assert set(ALL_JOB_TYPES) == ruled, (
        "every job type must be in exactly one of crucible/leases.py's two sets; "
        f"unruled: {sorted(set(ALL_JOB_TYPES) - ruled)}, "
        f"stale: {sorted(ruled - set(ALL_JOB_TYPES))}"
    )
    assert not (EVICTS_THE_RESIDENT_MODEL & KEEPS_THE_RESIDENT_MODEL)


def test_an_unruled_job_type_is_named_rather_than_guessed_at(
    resident_client: TestClient, auth: dict[str, str], clock: Clock
) -> None:
    """The runtime half of the guard above, for the day it is edited away.

    While nothing is leased there is nothing to protect and nothing to decide,
    so an unknown type passes; with a lease open, the server says which ruling
    is missing instead of picking one.
    """
    leases: Leases = resident_client.app.state.leases
    leases.refuse_if_leased("some-new-job-type")  # no lease: no question

    a_lease(resident_client, auth)
    with pytest.raises(Exception) as raised:
        leases.refuse_if_leased("some-new-job-type")
    assert getattr(raised.value, "code", None) == "lease_scope_unknown"
