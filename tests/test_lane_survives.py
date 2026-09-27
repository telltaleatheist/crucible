from __future__ import annotations

import time
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.jobs.base import Job, JobContext, JobTypeStatus, ModelDescriptor
from crucible.jobs.queue import JobStore


class _TypeMissingAMember:

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
    registry = {_TypeMissingAMember.name: _TypeMissingAMember()}
    with pytest.raises(TypeError) as caught:
        from crucible.jobs import _assert_every_type_implements_the_protocol

        _assert_every_type_implements_the_protocol(registry)
    message = str(caught.value)
    assert "incomplete" in message
    assert "model_provenance" in message


def test_the_real_registry_conforms(make_client: Callable[..., TestClient]) -> None:
    client = make_client(enable_echo=True, enable_llm=True, enable_asr=True)
    with client:
        assert client.get("/v1/ping").status_code == 200


def test_the_lane_survives_a_job_whose_finish_raises(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    client = make_client(enable_echo=True)
    with client:
        store: JobStore = client.app.state.store
        real_provenance = store.provenance
        exploded: list[str] = []

        def explode_once(job: Job, finished: str | None = None) -> dict[str, Any]:
            if finished is not None and job.id not in exploded:
                exploded.append(job.id)
                raise RuntimeError("the sidecar could not be written")
            return real_provenance(job, finished)

        store.provenance = explode_once

        first = client.post(
            "/v1/jobs",
            headers=auth,
            json={"type": "echo", "inputs": {"a.txt": {"inline_base64": "aGk="}}},
        ).json()["job_id"]
        events = _drain(client, auth, first)
        assert events[-1]["event"] == "failed"
        assert events[-1]["data"]["error"]["code"] in ("job_failed", "queue_failed")

        store.provenance = real_provenance

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


TERMINAL_DEADLINE_SECONDS = 10.0


def _drain(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict[str, Any]]:
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
