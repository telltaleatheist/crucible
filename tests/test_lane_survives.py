"""The queue's own bookkeeping must not be able to take the lane down.

These three tests exist because of one incident, found while merging phase 3a
into phase 4 on 2026-09-13. `asr` was written against a `JobType` protocol with
six members; `llm` added a seventh, `model_provenance`, in a branch that landed
the same night. Nothing complained: not at import, not at startup, not when a job
was accepted and queued.

The AttributeError surfaced inside `JobStore._finish`, on the event loop, while
writing a finished job's provenance sidecar. So the job never emitted a terminal
event, the SSE stream its client was reading never ended, and — because
`_run_lane` awaited `_execute` unguarded — the exception escaped the lane
coroutine and killed the worker outright. Every job submitted afterwards was
accepted with a 200 and sat at `queued` forever.

Three things were wrong and each gets a test: the missing member, the fact that a
missing member was possible at all, and the fact that the lane could die.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.jobs.base import Job, JobContext, JobTypeStatus, ModelDescriptor
from crucible.jobs.queue import JobStore


class _TypeMissingAMember:
    """A plugin written against an older `JobType`. It has no `model_provenance`."""

    name = "incomplete"

    def describe_models(self) -> list[ModelDescriptor]:
        return []

    def vram_estimate(self, model: str | None) -> int:
        return 0

    def check(self, backend: Any) -> JobTypeStatus:
        return JobTypeStatus(ready=True, detail="test double")

    def preflight(self, model: str | None, params: dict[str, Any]) -> None:
        return None

    def run(self, job: Job, ctx: JobContext) -> None:
        return None


def test_a_type_missing_a_protocol_member_is_refused_at_build() -> None:
    """By name, at startup — not three merges later inside a finished job."""
    registry = {_TypeMissingAMember.name: _TypeMissingAMember()}
    with pytest.raises(TypeError) as caught:
        # `build_registry` is what the server and `doctor` both call; the guard
        # lives at its end so neither can construct a registry it cannot serve.
        from crucible.jobs import _assert_every_type_implements_the_protocol

        _assert_every_type_implements_the_protocol(registry)  # type: ignore[arg-type]
    message = str(caught.value)
    assert "incomplete" in message
    assert "model_provenance" in message


def test_the_real_registry_conforms(make_client: Callable[..., TestClient]) -> None:
    """Every type this build ships, checked the way the server checks it.

    `build_registry` raises on a mismatch, so constructing a server with every
    job type enabled is the assertion.
    """
    client = make_client(enable_echo=True, enable_llm=True, enable_asr=True)
    with client:
        assert client.get("/v1/ping").status_code == 200


def test_the_lane_survives_a_job_whose_finish_raises(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """A bug in the queue's bookkeeping costs one job, not the server.

    The failure is injected where the real one happened — in `provenance()`,
    which `_finish` calls for every job — and the proof is not that the first
    job fails cleanly but that a *second* job submitted afterwards still runs.
    That is the part that was broken: the lane coroutine had died, so everything
    after it waited at `queued` with no worker left to take it.
    """
    client = make_client(enable_echo=True)
    with client:
        store: JobStore = client.app.state.store
        real_provenance = store.provenance
        exploded: list[str] = []

        def explode_once(job: Job, finished: str | None = None) -> dict[str, Any]:
            # Only on the call `_finish` makes, which passes a finish time.
            # `JobContext.artifact` calls this too, from the worker thread, where
            # `_execute` already catches everything and fails the job cleanly —
            # exploding there would test the path that was never broken. The
            # distinction is the whole reason this test is worth having: the
            # first version of it passed with the fix reverted.
            if finished is not None and job.id not in exploded:
                exploded.append(job.id)
                raise RuntimeError("the sidecar could not be written")
            return real_provenance(job, finished)

        store.provenance = explode_once  # type: ignore[assignment]

        first = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "inputs": {"a.txt": {"inline_base64": "aGk="}}},
        ).json()["job_id"]
        events = _drain(client, auth, first)
        assert events[-1]["event"] == "failed"
        assert events[-1]["data"]["error"]["code"] in ("job_failed", "queue_failed")

        store.provenance = real_provenance  # type: ignore[assignment]

        second = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "inputs": {"b.txt": {"inline_base64": "aGk="}}},
        ).json()["job_id"]
        events = _drain(client, auth, second)
        assert events[-1]["event"] == "done", (
            "the lane died with the first job: everything after it would sit at "
            "`queued` forever while the server went on answering 200"
        )


#: How long a job gets to emit a terminal event before this file calls it dead.
#: Generous for an echo job that takes milliseconds, and short enough that a
#: regression is a failed test in seconds rather than a CI job that hangs.
TERMINAL_DEADLINE_SECONDS = 10.0


def _drain(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict[str, Any]]:
    """Every event of one job, with a deadline, read from the store.

    Two deliberate choices, both made after watching this test hang.

    It waits for the terminal **event**, not for `status`. `_finish` sets the
    status first and appends the event last, so a job whose bookkeeping raised in
    between reports `done` while no client has been told anything — which is
    exactly the state the bug produced, and a test that polled the status would
    sail straight past it.

    And it reads the events out of the store rather than off the SSE stream, even
    though the stream is what a client reads. The regression here is a terminal
    event that never arrives; reading the stream reproduces that by BLOCKING,
    which is how the bug was found and is the worst possible shape for a test.
    The stream itself is covered in `tests/test_jobs.py`. Here the subject is the
    lane.
    """
    store: JobStore = client.app.state.store
    deadline = time.monotonic() + TERMINAL_DEADLINE_SECONDS
    while True:
        events = list(store.get(job_id).events)
        if events and events[-1]["event"] in ("done", "failed", "cancelled"):
            return events
        if time.monotonic() >= deadline:
            job = store.get(job_id)
            raise AssertionError(
                f"job {job_id} emitted no terminal event in "
                f"{TERMINAL_DEADLINE_SECONDS:.0f}s (status {job.status!r}, "
                f"{len(events)} events). A job that never says it finished is a "
                "stream that never ends: this is the lane having died."
            )
        time.sleep(0.02)
