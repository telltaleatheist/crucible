"""The caller goes away, and the engine stops working for nobody.

Neither app can cancel a chat any other way. Foundry's `Transport` has no abort
member at all and BookForge chains an `AbortSignal` to the fetch, so for both of
them the only cancel is dropping the connection (CLIENT-SURFACES.md, closing
section). On a server that runs one job at a time, tokens generated for a caller
who has hung up are not merely wasted — they are time taken from the next job.

These tests run the real app under a real uvicorn on a real socket
(tests/live_server.py) and then close the socket, because `TestClient` cannot
reach the state being tested: its `receive` answers `http.disconnect` only once
the app has finished responding.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from crucible import api as api_module

from .conftest import FAKE_BACKEND
from .fake_engine import FakeEngine
from .live_server import run_job, serve

MODEL = "qwen3.5-9b"

#: How long to give the engine to notice. The engine is polling its own socket
#: on a 20 ms tick and the non-streamed watcher wakes four times a second, so
#: this is many multiples of the expected latency, not a hopeful sleep.
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
    """A live server with a model resident and an engine configured to be slow."""

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
    """Read one frame, walk away, and the engine's stream ends with you.

    The fake engine here never sends `[DONE]` — it emits frames until somebody
    hangs up, the way a real one is mid-generation when a caller loses patience.
    So the only thing that can end this stream is the disconnect being carried
    through the proxy to the engine's socket.
    """
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
                    break  # one frame is all this caller wanted
        # Leaving the `with` closes the socket, which is the whole cancel.
        assert engines[0].aborted.wait(NOTICE_TIMEOUT), (
            "the engine was still generating for a caller that had gone"
        )


def test_dropping_a_non_streamed_completion_cancels_the_engine_request(
    resident_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    """A plain `await client.post(...)` would sit there until the engine answered.

    The caller here gives up after half a second on an engine that takes thirty;
    httpx closes the connection on the timeout, and the proxy has to notice and
    take its own request down with it.
    """
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
    """The other half of the rule, and the one that would break silently.

    A disconnect watcher that mistook a slow answer for a dead client would
    cancel every request that took longer than one poll. This caller waits the
    engine out and gets its completion.
    """
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
    """A guard on the fixture, not on the proxy.

    Everything above depends on the engine's name and Crucible's id being the
    same string, which is the vLLM shape — and on `cuda-linux` being the backend,
    because that is what the fixtures configure. If either changed, the tests
    above would still pass while testing something else.
    """
    _, server = resident_server()
    with server as base:
        info = httpx.get(f"{base}/v1/info", headers=auth, timeout=30.0).json()
        assert info["host"]["backend"] == FAKE_BACKEND.kind == "cuda-linux"


def test_no_middleware_stands_between_a_route_and_its_caller(
    make_app: Callable[..., Any],
) -> None:
    """The regression's owner, held in place.

    `@app.middleware("http")` is starlette's `BaseHTTPMiddleware`, which hands
    every route a `receive` wrapped in a task group of its own. One was added in
    1.0.18 to follow config.toml, and from then on no non-streamed completion
    and no decision ever learned its caller had gone: the live tests above fail
    under it. The config step is a plain ASGI step now (`BeforeEveryRequest`),
    and this says so before a live socket has to.
    """
    from starlette.middleware.base import BaseHTTPMiddleware

    app = make_app(enable_llm=True)
    wrapping = [m.cls for m in app.user_middleware if m.cls is BaseHTTPMiddleware]
    assert wrapping == [], (
        "a BaseHTTPMiddleware wraps every route's `receive`; a caller who hangs up "
        "would look present to the chat and decision doors again"
    )


# ------------------------------------------------------------------ decisions
#
# The Mac, 2026-09-24: Owen stopped a Foundry clean-triage run, Foundry's engine
# was dead within 0.3 s, and mlx-lm went on prefilling and answering its
# decisions for 45 s - `POST /v1/decide 200` for callers that no longer existed,
# the card held until the last of them finished. A decision is several engine
# requests behind one caller, so "the caller left" has two halves here: the
# requests already at the engine are taken down, AND the questions still waiting
# at the gate are never sent.

#: How many questions the decision below asks, and how many the door lets out
#: at once. More questions than the gate is wide, so some are WAITING when the
#: caller leaves - the half a per-request cancel would not reach.
QUESTIONS = 6
GATE = 2

#: The contract: the caller's connection closes and Crucible has its engine
#: requests down within about a second. Measured from the moment the client's
#: own timeout fired; the slack over one second is for a loaded test machine,
#: not for a poll.
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
    """A question, not the prime: only a question carries its statement."""
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
    """A live server whose engine answers a decision's prime at once and then
    sits on every question for thirty seconds, watching its socket the way an
    engine mid-prefill has a connection that can close under it.

    The gate is narrowed to `GATE` by stating the engine's concurrency, which is
    the door's own input (`chat_admission`); the fake's `vllm` name states none,
    and sixteen at once would leave nothing waiting to be held back.
    """
    monkeypatch.setattr(api_module, "chat_admission", lambda engine: (GATE, "the test's"))
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
    """The caller gives up mid-decision; the engine stops, and the card goes.

    Four things, each its own failure: the questions at the engine are aborted
    (their sockets closed under them), the questions still at the gate are never
    sent, the `InFlight` row is gone, and the settlement - which only sees a
    chat's end once that row closes - takes the card promptly rather than when
    the engine would have finished.
    """
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
        # The prime and a gate's width of questions reached the engine before
        # the caller left; that is the state being tested, not an assumption.
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
        # And nothing more was sent: the waiting questions were taken back at
        # the gate, not released onto an engine nobody was listening to.
        assert len(engine.requests) == 1 + GATE, (
            f"{len(engine.requests) - 1 - GATE} question(s) were sent after the "
            "caller had gone"
        )


def test_a_lease_released_mid_decision_is_answered_at_once(
    deciding_server: Any, auth: dict[str, str]
) -> None:
    """The DELETE is its own request and waits on nobody's decision.

    Foundry's Stop is two acts: its engine's connections close, and the lease
    is released. The release must answer while decisions are still in flight -
    and must NOT take the card from under them, because they hold it - and the
    card must then go the moment the decision's caller does.
    """
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
            # The decision still holds the card: releasing a lease is a holder
            # letting go, not a clearance over the ones still working.
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
