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


def test_a_job_left_running_comes_back_interrupted(tmp_path: Path) -> None:
    _record(tmp_path / "abc123", status=RUNNING)
    store = _store(tmp_path)

    assert store.restore() == ["abc123"]
    job = store.get("abc123")
    assert job.status == INTERRUPTED
    assert job.interrupted_at, "an interruption must say when it was noticed"


def test_a_queued_job_is_interrupted_too(tmp_path: Path) -> None:
    _record(tmp_path / "def456", status=QUEUED, started=None, progress=0.0)
    store = _store(tmp_path)
    store.restore()
    assert store.get("def456").status == INTERRUPTED


@pytest.mark.parametrize("status", [DONE, FAILED, CANCELLED])
def test_a_job_that_already_ended_comes_back_as_it_ended(
    tmp_path: Path, status: str
) -> None:
    _record(tmp_path / "ghi789", status=status, finished="2026-09-20T18:30:00+00:00")
    store = _store(tmp_path)
    store.restore()
    job = store.get("ghi789")
    assert job.status == status
    assert job.interrupted_at is None


def test_interrupted_is_terminal_so_nothing_re_runs_it(tmp_path: Path) -> None:
    assert INTERRUPTED in TERMINAL_STATES
    _record(tmp_path / "jkl012", status=RUNNING)
    store = _store(tmp_path)
    store.restore()
    assert store.get("jkl012").id not in list(store.queued())


def test_the_chunks_it_finished_survive_with_it(tmp_path: Path) -> None:
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
    assert sorted(set(range(6)) - set(job.chunks_done)) == [2, 3, 5]


def test_the_clients_own_reference_survives(tmp_path: Path) -> None:
    _record(tmp_path / "pqr678", status=RUNNING, client_ref="step-9f21")
    store = _store(tmp_path)
    store.restore()
    assert store.get("pqr678").client_ref == "step-9f21"


def test_the_params_are_not_on_disk(tmp_path: Path) -> None:
    from crucible.jobs.queue import JobStore

    directory = tmp_path / "stu901"
    _record(directory, status=RUNNING)
    written = json.loads((directory / JobStore.RECORD_NAME).read_text("utf-8"))
    assert "params" not in written

    store = _store(tmp_path)
    store.restore()
    assert store.get("stu901").params == {}


def test_a_directory_with_no_record_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "vwx234").mkdir()
    store = _store(tmp_path)
    assert store.restore() == []


def test_an_unreadable_record_is_skipped_loudly_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "yza567").mkdir()
    (tmp_path / "yza567" / "job.json").write_text("{not json", encoding="utf-8")
    _record(tmp_path / "bcd890", status=RUNNING)

    store = _store(tmp_path)
    assert store.restore() == ["bcd890"]
    assert "could not read the record" in capsys.readouterr().err


def test_a_record_that_is_not_a_job_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "efg123").mkdir()
    (tmp_path / "efg123" / "job.json").write_text("[1, 2, 3]", encoding="utf-8")
    store = _store(tmp_path)
    assert store.restore() == []


def test_restoring_twice_does_not_duplicate_anything(tmp_path: Path) -> None:
    _record(tmp_path / "hij456", status=RUNNING)
    store = _store(tmp_path)
    assert store.restore() == ["hij456"]
    assert store.restore() == []


def test_the_denominator_and_the_last_chunk_stamp_survive_with_it(tmp_path: Path) -> None:
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
    _record(tmp_path / "stu901", status=RUNNING)
    store = _store(tmp_path)
    store.restore()
    job = store.get("stu901")
    assert job.chunks_total is None
    assert job.chunk_at is None


def test_publishing_a_chunk_stamps_the_record_and_writes_it_through(tmp_path: Path) -> None:
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
