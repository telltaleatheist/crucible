"""A recovered job's event stream replays its ending and closes.

B-Side, 2026-10-04: a song killed by a deploy's restart had GET /v1/jobs/{id} saying
failed while GET /v1/jobs/{id}/events answered 200 and then sent nothing, forever.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from .conftest import configure_box, parse_sse

# Recent, and it has to be: a job finished longer ago than [jobs] retention_days is reaped at
# startup, and its stream answers 404. A fixed date here passed until 2026-10-11, then failed.
FINISHED = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()


def _ended_job(home: Path, job_id: str, **fields: object) -> None:
    directory = home / "jobs" / job_id
    directory.mkdir(parents=True)
    document = {
        "job_id": job_id, "type": "echo", "model": None, "status": "running", "progress": 0.4,
        "error": None, "artifacts": [], "chunks_done": [], "created": "2026-10-04T17:17:00+00:00",
        "started": "2026-10-04T17:17:01+00:00", "finished": None, "interrupted_at": None,
        "client": "b-side", "client_ref": None, "done_extra": {},
    }
    document.update(fields)
    (directory / "job.json").write_text(json.dumps(document), encoding="utf-8")


@pytest.mark.parametrize(
    ("fields", "events"),
    [
        ({"status": "failed", "finished": FINISHED,
          "error": {"code": "worker_failed", "message": "the worker exited -15"}}, ["failed"]),
        ({"status": "done", "finished": FINISHED, "artifacts": ["out.txt"]}, ["done"]),
        ({}, ["note"]),  # left running: it comes back interrupted
    ],
)
@pytest.mark.parametrize("last_event_id", [None, "0", "1"])
def test_the_stream_replays_the_ending_and_closes(
    home: Path, make_client: Callable[..., TestClient], auth: dict[str, str],
    fields: dict[str, object], events: list[str], last_event_id: str | None,
) -> None:
    configure_box(home)
    _ended_job(home, "1609a5808d44440a9840e568c691563a", **fields)
    headers = dict(auth)
    if last_event_id is not None:
        headers["Last-Event-ID"] = last_event_id
    with make_client() as client:
        # Returning at all is the point: before the fix this read never ended.
        with client.stream("GET", "/v1/jobs/1609a5808d44440a9840e568c691563a/events", headers=headers) as stream:
            got = [frame["event"] for frame in parse_sse(line for line in stream.iter_lines())]
    assert got == ([] if last_event_id == "1" else events)
