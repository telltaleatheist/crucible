"""A song keeps what it runs with on disk until it ends `done` (Owen, 2026-10-10).

Victoria's job 1f3da14c failed in synthesizing after composing ran to 8960 of 9000 tokens,
and its job.json had no params and no seed: nothing could run it again. A job that calls
`ctx.keep_request` (the audio type) writes `request.json` beside its record as it starts;
`done` drops it (done_extra.audio is the record then), failed / cancelled / interrupted keep
it until the job's directory is reaped. No other type keeps anything of its request:
a render's params are a chapter of somebody's book (56e6dee, 2026-09-20).
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from crucible import clock
from crucible.errors import JobCancelled, JobError
from crucible.jobs.base import (
    CANCELLED,
    DONE,
    FAILED,
    INTERRUPTED,
    RUNNING,
    Job,
    JobContext,
)
from crucible.jobs.queue import JobStore

KEPT = {"type": "song", "model": "m", "params": {"tags": "pop", "lyrics": "la", "seed": 41}}


class _Config:
    def __init__(self, root: Path) -> None:
        self.jobs_dir = root
        self.retention_days = 7


class _Song:
    """A job type that keeps its request, then ends the way it is told to."""

    name = "song"

    def __init__(self, ending: str) -> None:
        self.ending = ending
        self.on_disk_while_running: Any = None

    def run(self, job: Job, ctx: JobContext) -> None:
        ctx.keep_request(dict(KEPT))
        path = job.dir / JobStore.REQUEST_NAME
        self.on_disk_while_running = json.loads(path.read_text("utf-8"))
        if self.ending == FAILED:
            raise JobError("worker_failed", "CUDA out of memory in synthesizing")
        if self.ending in (CANCELLED, INTERRUPTED):
            raise JobCancelled(f"job {job.id} was cancelled")


class _Render:
    """A job type that keeps nothing: a render's params are a chapter of a book."""

    name = "tts"

    def __init__(self, ending: str) -> None:
        self.ending = ending

    def run(self, job: Job, ctx: JobContext) -> None:
        if self.ending == FAILED:
            raise JobError("worker_failed", "the narrator died")


def _run(tmp_path: Path, plugin: Any, ending: str) -> tuple[JobStore, Job]:
    store = JobStore(_Config(tmp_path), backend=None, registry={plugin.name: plugin})
    job = store.create(plugin.name, "m", {"tags": "pop", "lyrics": "la"})
    if ending == INTERRUPTED:
        store._interrupted_by_stop.add(job.id)
    asyncio.run(store._execute(job))
    assert job.status == ending
    return store, job


def _kept(job: Job) -> Path:
    return job.dir / JobStore.REQUEST_NAME


def test_a_song_keeps_its_request_on_disk_as_it_starts(tmp_path: Path) -> None:
    plugin = _Song(FAILED)
    _run(tmp_path, plugin, FAILED)
    assert plugin.on_disk_while_running == KEPT


def test_a_song_that_ends_done_drops_it(tmp_path: Path) -> None:
    _, job = _run(tmp_path, _Song(DONE), DONE)
    assert not _kept(job).exists()
    assert job.request is None
    record = json.loads((job.dir / JobStore.RECORD_NAME).read_text("utf-8"))
    assert "params" not in record and "request" not in record


@pytest.mark.parametrize("ending", [FAILED, CANCELLED, INTERRUPTED])
def test_a_song_that_does_not_end_done_keeps_it(tmp_path: Path, ending: str) -> None:
    _, job = _run(tmp_path, _Song(ending), ending)
    assert json.loads(_kept(job).read_text("utf-8")) == KEPT
    assert job.request == KEPT


@pytest.mark.parametrize("ending", [FAILED, CANCELLED, INTERRUPTED])
def test_a_kept_request_comes_back_after_a_restart(tmp_path: Path, ending: str) -> None:
    _, job = _run(tmp_path, _Song(ending), ending)
    again = JobStore(_Config(tmp_path), backend=None, registry={})
    assert again.restore() == [job.id]
    assert again.get(job.id).request == KEPT
    assert again.get(job.id).params == {}


def test_a_song_the_restart_interrupted_keeps_it(tmp_path: Path) -> None:
    store = JobStore(_Config(tmp_path), backend=None, registry={})
    job = store.create("song", "m", {})
    job.status = RUNNING
    store._persist(job)
    store.keep_request(job, dict(KEPT))

    again = JobStore(_Config(tmp_path), backend=None, registry={})
    again.restore()
    recovered = again.get(job.id)
    assert recovered.status == INTERRUPTED
    assert recovered.request == KEPT
    assert _kept(job).is_file()


def test_a_done_record_found_with_a_request_drops_it_on_restore(tmp_path: Path) -> None:
    """A stop between `_finish` writing `done` and removing the file."""
    store = JobStore(_Config(tmp_path), backend=None, registry={})
    job = store.create("song", "m", {})
    store.keep_request(job, dict(KEPT))
    job.status, job.finished = DONE, "2026-10-10T08:00:00+00:00"
    store._persist(job)

    again = JobStore(_Config(tmp_path), backend=None, registry={})
    again.restore()
    assert again.get(job.id).request is None
    assert not _kept(job).exists()


def test_a_reaped_song_takes_its_request_with_it(tmp_path: Path) -> None:
    store, job = _run(tmp_path, _Song(FAILED), FAILED)
    assert _kept(job).is_file()
    job.finished = (clock.now() - timedelta(days=8)).isoformat()
    reaped = store.reap()
    assert [r.job_id for r in reaped] == [job.id]
    assert not job.dir.exists()


@pytest.mark.parametrize("ending", [DONE, FAILED])
def test_a_render_never_has_its_request_on_disk(tmp_path: Path, ending: str) -> None:
    _, job = _run(tmp_path, _Render(ending), ending)
    assert not _kept(job).exists()
    assert job.request is None
    record = json.loads((job.dir / JobStore.RECORD_NAME).read_text("utf-8"))
    assert "params" not in record and "request" not in record


def test_only_the_audio_type_keeps_its_request() -> None:
    """The store keeps what a type hands it; which types hand it anything is this list.
    Narration, chat and every other type must never be on it."""
    package = Path(__file__).resolve().parents[1] / "crucible"
    callers = sorted(
        str(path.relative_to(package)).replace("\\", "/")
        for path in package.rglob("*.py")
        if re.search(r"ctx\.keep_request\(", path.read_text(encoding="utf-8"))
    )
    assert callers == ["jobs/audio/__init__.py"]
