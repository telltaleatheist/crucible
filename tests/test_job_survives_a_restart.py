"""A job survives the server that was running it.

THE INCIDENT, 2026-09-20. A deploy restarted this server six minutes into a
fine-tuning ladder's first real render. The render had been writing each chunk to
`artifacts/` as narrator answered it — that part always worked — but
`JobStore._jobs` was a plain in-memory dict, so after the restart the job simply
did not exist: `GET /v1/jobs/<id>` was a 404 while the finished FLACs sat in a
directory nothing could name. The audio was on disk and unreachable, and the
reaper eventually deleted it.

So the record is written beside the artifacts, and read back at startup.

`interrupted` IS NOT `failed`, and the distinction is the point. `failed` is this
server judging the work; BookForge sends such a row to a person. An interruption
is weather — the right answer is to collect what landed and re-ask for the rest,
which BookForge's `artifacts-owed.ts` already does for a dropped stream. Agreed
with that session, 2026-09-20.

NO RESUME ENDPOINT, deliberately. The render door's chunk `index` is the
CLIENT's and is never renumbered, and the seed is a pure function of
(index, take), so a chunk resubmitted an hour later renders identically. A resume
is therefore an ordinary new job carrying the chunks whose artifacts are missing
— which is why `chunks_done` is on the record and why these tests care that it
is exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.jobs.base import (
    CANCELLED,
    DONE,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    TERMINAL_STATES,
)


class _Config:
    def __init__(self, root: Path) -> None:
        self.jobs_dir = root
        self.retention_days = 7


def _store(root: Path) -> Any:
    from crucible.jobs.queue import JobStore

    return JobStore(_Config(root), backend=None, registry={})


def _record(directory: Path, **fields: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    document = {
        "job_id": directory.name,
        "type": "tts",
        "model": "deathstalker",
        "status": RUNNING,
        "progress": 0.22,
        "error": None,
        "artifacts": [],
        "chunks_done": [],
        "created": "2026-09-20T18:28:50+00:00",
        "started": "2026-09-20T18:28:51+00:00",
        "finished": None,
        "interrupted_at": None,
        "client": "bookforge",
        "client_ref": None,
        "done_extra": {},
    }
    document.update(fields)
    (directory / "job.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------- the recovery


def test_a_job_left_running_comes_back_interrupted(tmp_path: Path) -> None:
    """THE DEDUCTION, and it is not a guess: this process is the only thing that
    runs jobs, it has just started, and it is running none. Whatever wrote
    `running` on a disk is gone."""
    _record(tmp_path / "abc123", status=RUNNING)
    store = _store(tmp_path)

    assert store.restore() == ["abc123"]
    job = store.get("abc123")
    assert job.status == INTERRUPTED
    assert job.interrupted_at, "an interruption must say when it was noticed"


def test_a_queued_job_is_interrupted_too(tmp_path: Path) -> None:
    """It never started, so it produced nothing — but it is not `failed` and it
    is certainly not still `queued`: this server's lane is empty."""
    _record(tmp_path / "def456", status=QUEUED, started=None, progress=0.0)
    store = _store(tmp_path)
    store.restore()
    assert store.get("def456").status == INTERRUPTED


@pytest.mark.parametrize("status", [DONE, FAILED, CANCELLED])
def test_a_job_that_already_ended_comes_back_as_it_ended(
    tmp_path: Path, status: str
) -> None:
    """A restart must not rewrite history. `failed` stays failed — this server
    judged that work — and `done` stays done."""
    _record(tmp_path / "ghi789", status=status, finished="2026-09-20T18:30:00+00:00")
    store = _store(tmp_path)
    store.restore()
    job = store.get("ghi789")
    assert job.status == status
    assert job.interrupted_at is None


def test_interrupted_is_terminal_so_nothing_re_runs_it(tmp_path: Path) -> None:
    """THE SERVER NEVER RESUMES. Re-queuing a recovered job would re-render
    chunks whose audio is already on disk, and would do it against whatever
    voice happens to be resident now."""
    assert INTERRUPTED in TERMINAL_STATES
    _record(tmp_path / "jkl012", status=RUNNING)
    store = _store(tmp_path)
    store.restore()
    assert store.get("jkl012").id not in list(store.queued())


# ------------------------------------------------- what a resume differences on


def test_the_chunks_it_finished_survive_with_it(tmp_path: Path) -> None:
    """THE WHOLE POINT. Six minutes of rendering is worth recovering only if a
    client can tell which six minutes it was."""
    _record(
        tmp_path / "mno345",
        status=RUNNING,
        artifacts=["0.flac", "1.flac", "4.flac"],
        chunks_done=[0, 1, 4],
    )
    store = _store(tmp_path)
    store.restore()
    job = store.get("mno345")

    assert job.status == INTERRUPTED
    assert sorted(job.chunks_done) == [0, 1, 4]
    assert job.artifacts == ["0.flac", "1.flac", "4.flac"]
    # And the gap is a set difference, not filename parsing.
    assert sorted(set(range(6)) - set(job.chunks_done)) == [2, 3, 5]


def test_the_clients_own_reference_survives(tmp_path: Path) -> None:
    """After BOTH sides restart, the client has to match this job to whatever it
    was doing. A job id it may have lost with everything else is a poor key."""
    _record(tmp_path / "pqr678", status=RUNNING, client_ref="step-9f21")
    store = _store(tmp_path)
    store.restore()
    assert store.get("pqr678").client_ref == "step-9f21"


def test_the_params_are_not_on_disk(tmp_path: Path) -> None:
    """A render's params carry a chapter of somebody's book, and this record
    would be a second copy of it for a week. The client keeps its own text; what
    it needs from here is which chunks are done."""
    from crucible.jobs.queue import JobStore

    directory = tmp_path / "stu901"
    _record(directory, status=RUNNING)
    written = json.loads((directory / JobStore.RECORD_NAME).read_text("utf-8"))
    assert "params" not in written

    store = _store(tmp_path)
    store.restore()
    assert store.get("stu901").params == {}


# ------------------------------------------------------------- the failure ways


def test_a_directory_with_no_record_is_left_alone(tmp_path: Path) -> None:
    """A job directory from before this record existed. Inventing a record for
    it would be inventing facts; it is the reaper's business."""
    (tmp_path / "vwx234").mkdir()
    store = _store(tmp_path)
    assert store.restore() == []


def test_an_unreadable_record_is_skipped_loudly_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ONE BAD RECORD MUST NOT COST THE OTHERS. A restore that raised would take
    down a server whose other jobs were perfectly recoverable."""
    (tmp_path / "yza567").mkdir()
    (tmp_path / "yza567" / "job.json").write_text("{not json", encoding="utf-8")
    _record(tmp_path / "bcd890", status=RUNNING)

    store = _store(tmp_path)
    assert store.restore() == ["bcd890"]
    assert "could not read the record" in capsys.readouterr().err


def test_a_record_that_is_not_a_job_is_skipped(tmp_path: Path) -> None:
    """Valid JSON is not the same as a job."""
    (tmp_path / "efg123").mkdir()
    (tmp_path / "efg123" / "job.json").write_text("[1, 2, 3]", encoding="utf-8")
    store = _store(tmp_path)
    assert store.restore() == []


def test_restoring_twice_does_not_duplicate_anything(tmp_path: Path) -> None:
    """`restore()` runs once at startup, but a second call must be a no-op
    rather than a second copy of every job."""
    _record(tmp_path / "hij456", status=RUNNING)
    store = _store(tmp_path)
    assert store.restore() == ["hij456"]
    assert store.restore() == []


# ------------------------------------------- done/total and a pace, on the record


def test_the_denominator_and_the_last_chunk_stamp_survive_with_it(tmp_path: Path) -> None:
    """The ladder's ask (2026-09-21): done/total and a pace from the RECORD, not
    from an engine log in /tmp that died with the engine."""
    _record(
        tmp_path / "pqr678",
        status=RUNNING,
        artifacts=["0.flac", "1.flac"],
        chunks_done=[0, 1],
        chunks_total=128,
        chunk_at="2026-09-22T02:30:08+00:00",
    )
    store = _store(tmp_path)
    store.restore()
    job = store.get("pqr678")
    assert job.chunks_total == 128
    assert job.chunk_at == "2026-09-22T02:30:08+00:00"


def test_a_record_written_before_the_fields_existed_reads_them_as_null(tmp_path: Path) -> None:
    """A job.json from 1.0.21 has neither key. Null, not a KeyError and not 0 —
    0 would read as "asked for nothing"."""
    _record(tmp_path / "stu901", status=RUNNING)
    store = _store(tmp_path)
    store.restore()
    job = store.get("stu901")
    assert job.chunks_total is None
    assert job.chunk_at is None


def test_publishing_a_chunk_stamps_the_record_and_writes_it_through(tmp_path: Path) -> None:
    """`expect_chunks` lands the denominator; each indexed artifact moves the
    stamp; both are on disk before the next chunk, which is what a restart reads."""
    store = _store(tmp_path)
    job = store.create("tts", "deathstalker", {})
    assert job.chunks_total is None and job.chunk_at is None

    store.record_chunks_total(job, 3)
    assert job.chunks_total == 3
    store.record_artifact(job, "0.flac", index=0)
    first = job.chunk_at
    assert first is not None
    store.record_artifact(job, "1.flac", index=1)
    assert job.chunk_at is not None and job.chunk_at >= first
    # An artifact that is not a chunk moves nothing.
    store.record_artifact(job, "notes.json")
    assert job.chunk_at is not None and sorted(job.chunks_done) == [0, 1]

    on_disk = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert on_disk["chunks_total"] == 3
    assert on_disk["chunk_at"] == job.chunk_at

    again = _store(tmp_path)
    again.restore()
    recovered = again.get(job.id)
    assert recovered.chunks_total == 3
    assert recovered.chunk_at == job.chunk_at
