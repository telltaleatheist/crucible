from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from crucible.api.routes import decide as api_module

from .conftest import FAKE_BACKEND
from .fake_engine import FakeEngine
from .live_server import run_job, serve

MODEL = "qwen3.5-9b"

NOTICE_TIMEOUT = 15.0


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
        "messages": [{"role": "user", "content": "Write me something long."}],
        **extra,
    }


def test_dropping_a_streamed_completion_closes_the_engine_s_stream(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(stream_forever=True, answer_delay=0.05)
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
            for line in response.iter_lines():
                if line.startswith("data: "):
                    break
        assert engines[0].aborted.wait(NOTICE_TIMEOUT), (
            "the engine was still generating for a caller that had gone"
        )


def test_dropping_a_non_streamed_completion_cancels_the_engine_request(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(answer_delay=30.0)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        started = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                f"{base}/v1/openai/chat/completions",
                headers=auth,
                json=_chat(),
                timeout=httpx.Timeout(connect=10.0, read=0.5, write=10.0, pool=10.0),
            )
        assert time.monotonic() - started < 10.0, "the client's own timeout did not fire"
        assert engines[0].aborted.wait(NOTICE_TIMEOUT), (
            "the engine was still answering a request nobody was waiting for"
        )


def test_a_completion_that_is_waited_for_is_not_cancelled(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = resident_server(answer_delay=1.5)
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        response = httpx.post(
            f"{base}/v1/openai/chat/completions",
            headers=auth,
            json=_chat(),
            timeout=30.0,
        )
        assert response.status_code == 200
        assert response.json()["model"] == MODEL
        assert response.json()["choices"][0]["finish_reason"] == "stop"
        assert not engines[0].aborted.is_set()


def test_the_backend_kind_these_tests_run_against_is_the_real_one(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    _, server = resident_server()
    with server as base:
        info = httpx.get(f"{base}/v1/info", headers=auth, timeout=30.0).json()
        assert info["host"]["backend"] == FAKE_BACKEND.kind == "cuda-linux"


def test_no_middleware_stands_between_a_route_and_its_caller(
    make_app: Callable[..., Any],
) -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    app = make_app(enable_llm=True)
    wrapping = [m.cls for m in app.user_middleware if m.cls is BaseHTTPMiddleware]
    assert wrapping == [], (
        "a BaseHTTPMiddleware wraps every route's `receive`; a caller who hangs up "
        "would look present to the chat and decision doors again"
    )


QUESTIONS = 6
GATE = 2

PROMPTLY = 2.0


def _decision() -> dict[str, Any]:
    return {
        "model": MODEL,
        "state": "A long page the model is deciding about.",
        "questions": {
            f"q{n}": {"type": "yesno", "instructions": f"Statement number {n} holds"}
            for n in range(1, QUESTIONS + 1)
        },
    }


def _is_question(body: dict[str, Any]) -> bool:
    return "Statement number" in json.dumps(body.get("messages", []))


def _yes_mostly(messages: list[dict[str, Any]]) -> dict[str, float]:
    return {"A": 0.7, "B": 0.2}


@pytest.fixture
def deciding_server(
    make_app: Callable[..., Any],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        api_module, "chat_admission", lambda engine, args: (GATE, "the test's")
    )
    engines = engine_factory(
        probs_for=_yes_mostly,
        delay_for=lambda body: 30.0 if _is_question(body) else 0.0,
    )
    fake_weights(MODEL)
    app = make_app(enable_llm=True)
    return engines, app, serve(app)


def _wait_for(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_dropping_a_decision_takes_down_its_requests_and_sends_no_more(
    deciding_server: Any, auth: dict[str, str]
) -> None:
    engines, app, server = deciding_server
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        engine = engines[0]
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(
                f"{base}/v1/decide",
                headers=auth,
                json=_decision(),
                timeout=httpx.Timeout(connect=10.0, read=1.0, write=10.0, pool=10.0),
            )
        gave_up = time.monotonic()
        assert len(engine.requests) == 1 + GATE, engine.requests

        assert _wait_for(lambda: engine.aborts == GATE, PROMPTLY), (
            f"{engine.aborts} of the {GATE} questions at the engine noticed their "
            "caller leave; the rest were still being answered for nobody"
        )
        assert _wait_for(lambda: engine.stopped, PROMPTLY), (
            "the card was still held after the only caller had gone"
        )
        assert time.monotonic() - gave_up < PROMPTLY + 0.5
        assert len(app.state.inflight) == 0
        assert len(engine.requests) == 1 + GATE, (
            f"{len(engine.requests) - 1 - GATE} question(s) were sent after the "
            "caller had gone"
        )


def test_a_lease_released_mid_decision_is_answered_at_once(
    deciding_server: Any, auth: dict[str, str]
) -> None:
    engines, app, server = deciding_server
    with server as base:
        run_job(base, auth, type="load-model", model=MODEL)
        engine = engines[0]
        lease = httpx.post(
            f"{base}/v1/models/{MODEL}/lease",
            headers=auth,
            json={"act": "clean", "ttl_seconds": 600},
            timeout=30.0,
        )
        assert lease.status_code == 201, lease.text

        outcome: dict[str, Any] = {}

        def decide() -> None:
            try:
                httpx.post(
                    f"{base}/v1/decide",
                    headers=auth,
                    json=_decision(),
                    timeout=httpx.Timeout(connect=10.0, read=4.0, write=10.0, pool=10.0),
                )
            except httpx.ReadTimeout as exc:
                outcome["timed_out"] = exc
            outcome["gave_up"] = time.monotonic()

        caller = threading.Thread(target=decide, name="decision-caller")
        caller.start()
        try:
            assert _wait_for(lambda: len(engine.requests) == 1 + GATE, 10.0), (
                "the decision never reached its questions"
            )

            started = time.monotonic()
            released = httpx.delete(
                f"{base}/v1/leases/{lease.json()['lease_id']}",
                headers=auth,
                timeout=30.0,
            )
            took = time.monotonic() - started
            assert released.status_code == 204, released.text
            assert took < 1.0, f"the lease's release took {took:.2f}s to answer"
            assert not engine.stopped
            assert len(app.state.inflight) == 1
        finally:
            caller.join(timeout=30.0)
        assert "timed_out" in outcome, "the decision was answered; nothing was dropped"

        assert _wait_for(lambda: engine.aborts == GATE, PROMPTLY)
        assert _wait_for(lambda: engine.stopped, PROMPTLY), (
            "the lease was gone and the caller was gone, and the card was still held"
        )
        assert time.monotonic() - outcome["gave_up"] < PROMPTLY + 0.5
        assert len(engine.requests) == 1 + GATE
