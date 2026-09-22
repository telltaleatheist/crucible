"""The wire loses a request, and the proxy sends it once more on a fresh socket.

Measured 2026-09-21 22:03:20 on the PC: one `POST /openai/v1/chat/completions`
answered 502 `engine_unreachable` with `ReadError: ` (an empty message) while the
dots-ocr engine was answering nine other completions in the same ten seconds.
The client took the 502 as fatal, released its lease and closed its other eleven
requests, and the settlement cleared the card exactly as ruled. The socket was
the fault: vLLM's uvicorn closes an idle keep-alive connection after 5 s
(`VLLM_HTTP_TIMEOUT_KEEP_ALIVE`, vllm 0.29.0 `vllm/envs.py:109`), httpx keeps
one for the same 5.0 s, and at that boundary the proxy offers a request to a
socket the engine has just closed.

Two changes, pinned here: the proxy's pool lets a socket go BEFORE the engine
does (`PROXY_KEEPALIVE_EXPIRY`), and a request the wire loses is sent once more
on a fresh socket (`LOST_ON_THE_WIRE`, `WIRE_ATTEMPTS`) — while a timeout, which
means a wedged engine and not a dead socket, is still tried exactly once.

Like tests/test_proxy_disconnect.py these run the real app under a real uvicorn
on a real socket: the fault being reproduced is a TCP reset between the proxy
and its engine, and `TestClient` has no wire to reset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import api as api_module
from crucible.api import LOST_ON_THE_WIRE, PROXY_KEEPALIVE_EXPIRY, WIRE_ATTEMPTS

from .fake_engine import FakeEngine
from .live_server import run_job, serve

MODEL = "qwen3.5-9b"

#: vLLM's default keep-alive, seconds — `VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5`
#: at vllm 0.29.0 `vllm/envs.py:109`, applied at
#: `vllm/entrypoints/launchers/api_server/entry.py:151`. Restated here as the
#: number the proxy must stay under; if vLLM ever lowers it, this is the line
#: that says so.
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
    """A live server with a model resident and an engine that drops what it is told to."""

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
    """The incident, with the fix in: the first socket dies, the caller gets a 200."""
    engines, server = resident_server(drop_requests=1)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        response = httpx.post(
            f"{base}/v1/openai/chat/completions", headers=auth, json=_chat(), timeout=30.0
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        # The engine saw the request twice: the one it reset, and the one it answered.
        assert len(engines[0]._handler.requests) == 2


def test_a_streamed_request_the_wire_loses_is_opened_again(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """Opening the stream is before the first byte, so the same rule holds there."""
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


def test_the_budget_is_two_attempts_and_the_502_says_so(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """A socket that dies twice is a 502 that names both the fault and the count."""
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
        # The fault is named by its httpx type: a reset reads as `ReadError`, and
        # when the FIN outruns the RST as `RemoteProtocolError`. Both are in the set.
        assert any(kind.__name__ in error["message"] for kind in LOST_ON_THE_WIRE), error
        # Exactly the budget, never a third try.
        assert len(engines[0]._handler.requests) == WIRE_ATTEMPTS


def test_a_timeout_is_not_a_lost_socket_and_is_tried_once(
    resident_server: Callable[..., Any],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine that does not answer in time is wedged; repeating it doubles the wait.

    The read ceiling is shrunk to half a second so the test sees it inside a
    minute; the engine takes three. The proxy must answer 502 after ONE attempt,
    with the timeout named and no attempt count claimed.
    """
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


def test_the_proxy_lets_a_socket_go_before_the_engine_would(
    make_client: Callable[..., TestClient],
) -> None:
    """The pool's expiry is below vLLM's keep-alive, and the live client carries it.

    The first half is the number; the second reaches into httpx's pool because
    a constant nobody applied would pin nothing.
    """
    assert PROXY_KEEPALIVE_EXPIRY < VLLM_HTTP_TIMEOUT_KEEP_ALIVE
    assert httpx.Limits().keepalive_expiry == 5.0, (
        "httpx changed its default keep-alive; re-read the reasoning on PROXY_KEEPALIVE_EXPIRY"
    )
    with make_client(enable_llm=True) as client:
        pool = client.app.state.http._transport._pool
        assert pool._keepalive_expiry == PROXY_KEEPALIVE_EXPIRY


def test_timeouts_are_not_in_the_retry_set() -> None:
    """The set is network faults and a peer that hung up mid-protocol, nothing timed."""
    assert not any(
        issubclass(httpx.TimeoutException, kind) or issubclass(kind, httpx.TimeoutException)
        for kind in LOST_ON_THE_WIRE
    )
    assert issubclass(httpx.ReadError, LOST_ON_THE_WIRE)
    assert issubclass(httpx.ConnectError, LOST_ON_THE_WIRE)
    assert issubclass(httpx.RemoteProtocolError, LOST_ON_THE_WIRE)
    assert not issubclass(httpx.ReadTimeout, LOST_ON_THE_WIRE)
