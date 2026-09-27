from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.engines import ENGINES
from crucible.inflight import InFlight

from .fake_engine import FakeEngine

from .test_llm_api import (
    MODEL,
    engines,
    fake_env,
    llm_client,
    run_job,
    submit,
)


@pytest.fixture
def serial_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in ENGINES.values():
        monkeypatch.setattr(cls, "chat_concurrency_flag", None, raising=False)
        monkeypatch.setattr(cls, "chat_concurrency", 1, raising=False)
        monkeypatch.setattr(
            cls, "chat_concurrency_basis", "one generation thread", raising=False
        )


def _load_and_wait(client: TestClient, auth: dict[str, str]) -> str:
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


def test_a_storm_of_refusals_leaks_no_slot(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    serial_engines: None,
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    held = [inflight.open(act=None, model=MODEL, client="a-test") for _ in range(2)]
    try:
        for _ in range(20):
            assert _chat(llm_client, auth).status_code == 503
            assert len(inflight) == 2
    finally:
        for entry in held:
            inflight.close(entry)

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
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    keeper = inflight.open(act=None, model=MODEL, client="a-test")
    try:
        assert _chat(llm_client, auth).status_code == 200
        measured = len(inflight._recent)
        assert measured == 1

        second = inflight.open(act=None, model=MODEL, client="a-test")
        try:
            for _ in range(15):
                assert _chat(llm_client, auth).status_code == 503
        finally:
            inflight.close(second)
    finally:
        inflight.close(keeper)

    assert len(inflight._recent) == measured + 2


def test_the_refusal_states_the_limit_the_client_should_have_read(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    serial_engines: None,
) -> None:
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
    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["chat"]["max_in_flight"] == details["max_in_flight"]


def test_cancelling_a_finished_load_is_refused_and_changes_nothing(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    job_id = _load_and_wait(llm_client, auth)

    refused = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "job_not_cancellable"

    after = llm_client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert after["status"] == "done"

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
    fake_weights(MODEL)
    job_id = _load_and_wait(llm_client, auth)

    first = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    second = llm_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert first.status_code == second.status_code == 409
    assert first.json()["error"]["code"] == second.json()["error"]["code"]


def test_closing_an_entry_that_was_never_opened_is_not_an_error() -> None:
    flight = InFlight()
    entry = flight.open(act=None, model="q", client=None)
    flight.close(entry)
    flight.close(entry)
    flight.close(entry)
    assert len(flight) == 0
    assert len(flight._recent) == 1


def test_the_count_survives_many_open_close_cycles() -> None:
    flight = InFlight()
    for _ in range(2000):
        flight.close(flight.open(act=None, model="q", client=None))
    assert len(flight) == 0


def test_a_load_that_fails_any_way_at_all_takes_its_engine_down() -> None:

    class _Recording:

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
            Residency._start(
                engine,
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
        Residency._start(
            _Fine(),
            Path("weights"),
            "served",
            1,
            [],
            lambda _message: None,
            1.0,
            confirm=confirm,
        )
