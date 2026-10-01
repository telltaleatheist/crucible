from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import httpx

from crucible.api.serving import server_for
from crucible.engines import find_free_port

STARTUP_TIMEOUT = 20.0
SHUTDOWN_TIMEOUT = 20.0


@contextmanager
def serve(app: Any) -> Iterator[str]:
    port = find_free_port()
    server = server_for(app, host="127.0.0.1", port=port, log_level="warning")
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
