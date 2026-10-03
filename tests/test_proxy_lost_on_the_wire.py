from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible.api import app as api_module
from crucible.api.proxy import (
    LOST_ON_THE_WIRE,
    PROXY_KEEPALIVE_EXPIRY,
    WIRE_ATTEMPTS,
    WIRE_BACKOFF_SECONDS,
)

from .fake_engine import FakeEngine
from .live_server import run_job, serve

MODEL = "qwen3.5-9b"

VLLM_HTTP_TIMEOUT_KEEP_ALIVE = 5


@pytest.fixture
def resident_server(
    make_app: Callable[..., Any],
    auth: dict[str, str],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
):

    def start(**engine_options: Any):
        engines = engine_factory(**engine_options)
        fake_weights(MODEL)
        return engines, serve(make_app(enable_llm=True))

    return start


def _chat(**extra: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Read me page thirty-two."}],
        **extra,
    }


def test_a_request_the_wire_loses_is_sent_again_and_answered(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(drop_requests=1)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        response = httpx.post(
            f"{base}/v1/openai/chat/completions", headers=auth, json=_chat(), timeout=30.0
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        assert len(engines[0]._handler.requests) == 2


def test_a_streamed_request_the_wire_loses_is_opened_again(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(drop_requests=1)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        with httpx.stream(
            "POST",
            f"{base}/v1/openai/chat/completions",
            headers=auth,
            json=_chat(stream=True),
            timeout=30.0,
        ) as response:
            assert response.status_code == 200
            frames = b"".join(response.iter_bytes())
        assert b"data: [DONE]" in frames
        assert len(engines[0]._handler.requests) == 2


def test_the_budget_is_stated_attempts_and_the_502_says_so(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(drop_requests=WIRE_ATTEMPTS + 1)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        response = httpx.post(
            f"{base}/v1/openai/chat/completions", headers=auth, json=_chat(), timeout=30.0
        )
        assert response.status_code == 502, response.text
        error = response.json()["error"]
        assert error["code"] == "engine_unreachable"
        assert f"on {WIRE_ATTEMPTS} attempts" in error["message"]
        assert any(kind.__name__ in error["message"] for kind in LOST_ON_THE_WIRE), error
        assert len(engines[0]._handler.requests) == WIRE_ATTEMPTS


def test_a_timeout_is_not_a_lost_socket_and_is_tried_once(
    resident_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "PROXY_READ_TIMEOUT", 0.5)
    engines, server = resident_server(answer_delay=3.0)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        response = httpx.post(
            f"{base}/v1/openai/chat/completions", headers=auth, json=_chat(), timeout=30.0
        )
        assert response.status_code == 502, response.text
        error = response.json()["error"]
        assert error["code"] == "engine_unreachable"
        assert "ReadTimeout" in error["message"]
        assert "attempts" not in error["message"]
        assert len(engines[0]._handler.requests) == 1


def test_every_retry_waits_and_the_waits_are_a_budget_not_a_spin() -> None:
    assert len(WIRE_BACKOFF_SECONDS) == WIRE_ATTEMPTS - 1
    assert all(b > a for a, b in zip(WIRE_BACKOFF_SECONDS, WIRE_BACKOFF_SECONDS[1:]))
    assert 5.0 <= sum(WIRE_BACKOFF_SECONDS) <= 15.0


def test_the_proxy_lets_a_socket_go_before_the_engine_would(
    make_client: Callable[..., TestClient],
) -> None:
    assert PROXY_KEEPALIVE_EXPIRY < VLLM_HTTP_TIMEOUT_KEEP_ALIVE
    assert httpx.Limits().keepalive_expiry == 5.0, (
        "httpx changed its default keep-alive; re-read the reasoning on PROXY_KEEPALIVE_EXPIRY"
    )
    with make_client(enable_llm=True) as client:
        pool = client.app.state.http._transport._pool
        assert pool._keepalive_expiry == PROXY_KEEPALIVE_EXPIRY


def test_timeouts_are_not_in_the_retry_set() -> None:
    assert not any(
        issubclass(httpx.TimeoutException, kind) or issubclass(kind, httpx.TimeoutException)
        for kind in LOST_ON_THE_WIRE
    )
    assert issubclass(httpx.ReadError, LOST_ON_THE_WIRE)
    assert issubclass(httpx.ConnectError, LOST_ON_THE_WIRE)
    assert issubclass(httpx.RemoteProtocolError, LOST_ON_THE_WIRE)
    assert not issubclass(httpx.ReadTimeout, LOST_ON_THE_WIRE)
