from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from .fake_engine import FakeEngine
from .test_llm_api import (
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
    fake_weights(MODEL)
    done = _done(_events(llm_client, auth, type="load-model", model=MODEL))

    assert done["resident"] == MODEL
    assert done["lease_id"] is None
    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["lease"] is None
    assert activity["resident"]["held_by"] is None


def test_a_load_that_asks_for_a_lease_is_held_the_moment_it_is_resident(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
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
    assert activity["lease"]["lease_id"] == lease_id
    assert activity["lease"]["act"] == "clean"
    assert activity["resident"]["id"] == MODEL
    assert activity["resident"]["held_by"]["fact"] == "a lease"
    assert activity["resident"]["unclaimed_since"] is None


def test_the_lease_id_is_readable_after_the_stream_is_gone(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
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
    assert record["job_id"] == job_id
    assert record["type"] == "load-model"


def test_a_released_lease_clears_the_card(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
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
