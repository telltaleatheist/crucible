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

import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

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
