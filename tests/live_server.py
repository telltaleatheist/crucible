"""Run the real app under uvicorn, on a real socket, for the length of a test.

Almost everything the API does is provable through `TestClient`, which drives the
ASGI app directly and is faster and quieter for it. One thing is not: **a client
that goes away**. `TestClient`'s `receive` only ever answers `http.disconnect`
once the app has finished its response (starlette/testclient.py), so a caller
who drops a stream halfway is a state it cannot reach — and a caller dropping the
fetch is the *only* cancel either app has for a chat (CLIENT-SURFACES.md's
closing section).

So these tests use a real server and a real socket, and close it for real. No
accelerator is involved: the engine underneath is still `tests/fake_engine.py`.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import httpx
import uvicorn

from crucible.engines import find_free_port

#: How long to wait for uvicorn to bind, and to unbind again afterwards.
STARTUP_TIMEOUT = 20.0
SHUTDOWN_TIMEOUT = 20.0


@contextmanager
def serve(app: Any) -> Iterator[str]:
    """Serve `app` on a free loopback port; yields its base URL."""
    port = find_free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            # The lifespan is where the job lane starts and where a resident
            # engine is shut down, so this is the whole app, not a fragment.
            lifespan="on",
        )
    )
    thread = threading.Thread(target=server.run, name="crucible-test-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError(f"the test server never bound 127.0.0.1:{port}")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=SHUTDOWN_TIMEOUT)
        if thread.is_alive():
            raise RuntimeError("the test server did not shut down")


def run_job(base: str, auth: dict[str, str], **body: Any) -> dict[str, Any]:
    """Submit a job over HTTP and poll it to a terminal state.

    The event stream is the richer door, but a test that only needs the model
    resident before it can ask the real question should not also be asserting on
    SSE framing — `tests/test_llm_api.py` does that against `TestClient`.
    """
    response = httpx.post(f"{base}/v1/jobs", headers=auth, json=body, timeout=30.0)
    if response.status_code != 202:
        raise AssertionError(f"the job was refused: {response.status_code} {response.text}")
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        state = httpx.get(f"{base}/v1/jobs/{job_id}", headers=auth, timeout=30.0).json()
        if state["status"] in ("done", "failed", "cancelled"):
            if state["status"] != "done":
                raise AssertionError(f"the job ended {state['status']}: {state['error']}")
            return state
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")
