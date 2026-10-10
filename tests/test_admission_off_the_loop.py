"""A job type's preflight, and the routes that read an env, run off the event loop.

Every route, SSE stream and chat proxy shares one asyncio loop (crucible/loopwatch.py).
A preflight reads the card: `nvidia-smi`, which under WSL2 takes over a second (Victoria's
laptop, 2026-10-09: "the event loop has not run for 1.0 s" on every load, from
`accelerator.guard -> read_state -> _nvidia_state -> probe_compute_apps`). A model or
voice row reads its env, which is a `pip list` the first time. Nothing here touches a
GPU: the preflights and row readers are replaced by ones that say which thread ran them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.api.routes import catalog as catalog_routes
from crucible.api.routes import info as info_routes
from crucible.errors import ApiError

from .test_admission import job_body, wait_for_terminal


def on_the_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def recording(seen: list[bool], then: Callable[[], None] = lambda: None) -> Any:
    def preflight(model: str | None, params: dict[str, Any]) -> None:
        seen.append(on_the_loop())
        then()

    return preflight


def test_a_preflight_runs_off_the_event_loop(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    echo = client.app.state.store.registry["echo"]
    seen: list[bool] = []
    monkeypatch.setattr(echo, "preflight", recording(seen))
    response = client.post("/v1/jobs", json=job_body(), headers=auth)
    assert response.status_code == 202, response.text
    assert seen == [False], "the preflight ran on the event loop's thread"
    assert wait_for_terminal(client, auth, response.json()["job_id"]) == "done"


def test_a_preflight_that_reads_the_residency_lock_does_not_wait_on_the_loop(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``settled_for`` holds the residency's claim lock on the loop's thread. A preflight
    run on another thread while it was held would wait on that lock forever (the unload
    preflight takes it, through ``being_cleared``)."""
    residency = client.app.state.residency
    echo = client.app.state.store.registry["echo"]
    seen: list[bool] = []
    monkeypatch.setattr(
        echo, "preflight", recording(seen, lambda: residency.being_cleared("echo"))
    )
    response = client.post("/v1/jobs", json=job_body(), headers=auth)
    assert response.status_code == 202, response.text
    assert seen == [False]


def test_a_preflight_refusal_still_reaches_the_client(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    echo = client.app.state.store.registry["echo"]

    def refuse(model: str | None, params: dict[str, Any]) -> None:
        raise ApiError(409, "insufficient_memory", "cannot load 'echo': no room")

    monkeypatch.setattr(echo, "preflight", refuse)
    response = client.post("/v1/jobs", json=job_body(), headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "insufficient_memory"
    assert len(client.app.state.store._jobs) == 0


def test_the_lane_is_checked_again_after_the_preflight(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preflight runs outside the settled section, so the lane can be taken while it
    reads the card. The checks that decide a place on the lane are made again after it,
    and a job is never put onto a lane something else took in the meantime."""
    store = client.app.state.store
    echo = store.registry["echo"]
    taken: list[bool] = []
    honest = store.refuse_if_busy

    def busy_once_taken() -> None:
        if taken:
            raise ApiError(409, "server_busy", "this server is running job j-other")
        honest()

    monkeypatch.setattr(store, "refuse_if_busy", busy_once_taken)
    monkeypatch.setattr(echo, "preflight", recording([], lambda: taken.append(True)))
    response = client.post("/v1/jobs", json=job_body(), headers=auth)
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "server_busy"
    assert len(client.app.state.store._jobs) == 0


def test_the_model_rows_are_read_off_the_event_loop(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def rows(where: str) -> Callable[..., list[dict[str, Any]]]:
        def read(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            seen.append(f"{where} {'on' if on_the_loop() else 'off'} the loop")
            return []

        return read

    monkeypatch.setattr(catalog_routes, "model_rows", rows("/v1/models"))
    monkeypatch.setattr(info_routes, "model_rows", rows("/v1/info"))
    with make_client(enable_llm=True) as client:
        assert client.get("/v1/models", headers=auth).status_code == 200
        assert client.get("/v1/info", headers=auth).status_code == 200
    assert seen == ["/v1/models off the loop", "/v1/info off the loop"]


def test_the_setup_addresses_are_read_off_the_event_loop(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[bool] = []

    def urls(host: str, port: int, advertise: Any = ()) -> list[str]:
        seen.append(on_the_loop())
        return [f"http://{host}:{port}"]

    monkeypatch.setattr(info_routes.pairing, "reachable_urls", urls)
    assert client.get("/v1/setup", headers=auth).status_code == 200
    assert seen and not any(seen), "an address was read on the event loop's thread"
