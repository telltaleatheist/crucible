from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from . import clock
from .errors import ApiError

JOURNALS_DIRNAME = "journals"

JOURNAL_FORMAT = 1

GONE = "_gone"

PROGRESS_WRITE_SECONDS = 2.0

REPLACE_ATTEMPTS = 10

_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_RESUME_ID = re.compile(r"^[0-9a-f]{32}$")

LIVE_STATES = frozenset({"queued", "running"})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def write_atomically(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    data = (json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    _fsync_directory(path.parent)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Identity:
    job_type: str
    model: str | None
    revision: str | None
    format_version: int
    params: dict[str, Any]


@dataclass(frozen=True)
class InputDigest:
    name: str
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "bytes": self.bytes}


class Journal:
    def __init__(self, directory: Path, manifest: dict[str, Any], retention_days: float) -> None:
        self._dir = directory
        self._manifest = manifest
        self._retention_days = retention_days
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._last_manifest_write = 0.0
        self._manifest_dirty = False
        self._keys: set[str] | None = None

    @property
    def id(self) -> str:
        return str(self._manifest["resume_id"])

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def manifest(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._manifest))

    @property
    def units_dir(self) -> Path:
        return self._dir / "units"


    def keys(self) -> set[str]:
        with self._lock:
            if self._keys is None:
                found: set[str] = set()
                if self.units_dir.is_dir():
                    for entry in self.units_dir.iterdir():
                        name = entry.name
                        if name.startswith(".") or not name.endswith(".json"):
                            continue
                        found.add(name[: -len(".json")])
                self._keys = found
            return set(self._keys)

    def get(self, key: str) -> Any | None:
        _check_key(key)
        path = self.units_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            envelope = _read(path)
        except (OSError, ValueError) as exc:
            print(
                f"crucible: journal {self.id} unit {key} is unreadable and will be "
                f"redone: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        return envelope.get("data")

    def put(self, key: str, data: Any) -> None:
        _check_key(key)
        now = clock.now()
        write_atomically(
            self.units_dir / f"{key}.json",
            {"key": key, "saved_at": now.isoformat(), "data": data},
        )
        with self._lock:
            if self._keys is not None:
                self._keys.add(key)
            self._touch(now)
        self._write_manifest(force=False)

    def put_file(self, key: str, source: Path, data: Any) -> None:
        _check_key(key)
        target = self.units_dir / f"{key}.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        with Path(source).open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
        _fsync_directory(target.parent)
        self.put(key, {"file": target.name, "bytes": target.stat().st_size, "data": data})

    def file(self, key: str) -> Path | None:
        unit = self.get(key)
        if not isinstance(unit, dict) or "file" not in unit:
            return None
        path = self.units_dir / str(unit["file"])
        return path if path.is_file() else None

    def progress(self, done: int, total: int, sentence: str, *, force: bool = False) -> None:
        with self._lock:
            self._manifest["units_done"] = int(done)
            self._manifest["units_total"] = int(total)
            self._manifest["progress"] = sentence
            self._manifest_dirty = True
        self._write_manifest(force=force)

    def flush(self) -> None:
        self._write_manifest(force=True)


    def _touch(self, now: datetime) -> None:
        self._manifest["last_saved"] = now.isoformat()
        self._manifest["expires_at"] = (
            now + timedelta(days=self._retention_days)
        ).isoformat()
        self._manifest_dirty = True

    def _write_manifest(self, *, force: bool) -> None:
        with self._write_lock:
            with self._lock:
                if not self._manifest_dirty:
                    return
                moment = time.monotonic()
                if not force and moment - self._last_manifest_write < PROGRESS_WRITE_SECONDS:
                    return
                document = json.loads(json.dumps(self._manifest))
                self._manifest_dirty = False
                self._last_manifest_write = moment
            write_atomically(self._dir / "manifest.json", document)

    def set_writer(self, job_id: str, state: str, *, resumed: bool | None = None) -> None:
        with self._lock:
            writer = {"job_id": job_id, "state": state, "at": clock.now().isoformat()}
            self._manifest["writer"] = writer
            history = list(self._manifest.get("jobs") or [])
            for row in history:
                if row.get("job_id") == job_id:
                    row["state"] = state
                    break
            else:
                history.append(
                    {"job_id": job_id, "state": state, "resumed": bool(resumed)}
                )
            self._manifest["jobs"] = history
            self._manifest_dirty = True
        self._write_manifest(force=True)


def _check_key(key: str) -> None:
    if not _KEY.match(key):
        raise ValueError(
            f"journal unit key {key!r} is not a plain file name: letters, digits, "
            "'.', '_' and '-', starting with a letter or digit"
        )


class Journals:
    def __init__(
        self,
        root: Path | None,
        retention_days: Callable[[], float],
        live_state: Callable[[str], str | None],
    ) -> None:
        self._root: Path = Path(root) if root is not None else Path()
        self._rooted = root is not None
        self._retention_days = retention_days
        self._live_state = live_state
        self._open: dict[str, Journal] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        return self._root if self._rooted else None


    def create(
        self, identity: Identity, inputs: list[InputDigest], job_id: str
    ) -> Journal:
        if not self._rooted:
            raise ApiError(
                500,
                "journal_unwritable",
                "this job store has no Crucible home, so it has nowhere to keep a "
                "resume journal",
            )
        resume_id = uuid.uuid4().hex
        now = clock.now()
        manifest = {
            "journal_format": JOURNAL_FORMAT,
            "resume_id": resume_id,
            "job_type": identity.job_type,
            "model": {"id": identity.model, "revision": identity.revision},
            "format_version": identity.format_version,
            "params": identity.params,
            "params_sha256": hashlib.sha256(
                canonical(identity.params).encode("utf-8")
            ).hexdigest(),
            "inputs": [digest.to_dict() for digest in inputs],
            "units_done": 0,
            "units_total": None,
            "progress": "nothing finished yet",
            "created": now.isoformat(),
            "last_saved": now.isoformat(),
            "expires_at": (now + timedelta(days=self._retention_days())).isoformat(),
            "job_id": job_id,
            "writer": {"job_id": job_id, "state": "queued", "at": now.isoformat()},
            "jobs": [{"job_id": job_id, "state": "queued", "resumed": False}],
        }
        directory = self._root / resume_id
        try:
            (directory / "units").mkdir(parents=True, exist_ok=False)
            write_atomically(directory / "manifest.json", manifest)
        except OSError as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise ApiError(
                500,
                "journal_unwritable",
                f"this server could not start a resume journal under {self._root}: "
                f"{type(exc).__name__}: {exc}",
                {"path": str(self._root)},
            ) from None
        journal = Journal(directory, manifest, self._retention_days())
        with self._lock:
            self._open[resume_id] = journal
        return journal

    def open(self, resume_id: str) -> Journal:
        with self._lock:
            cached = self._open.get(resume_id)
        if cached is not None:
            return cached
        manifest = self._manifest_or_refuse(resume_id)
        journal = Journal(self._root / resume_id, manifest, self._retention_days())
        with self._lock:
            return self._open.setdefault(resume_id, journal)

    def adopt(self, journal: Journal, job_id: str) -> None:
        journal.set_writer(job_id, "queued", resumed=True)

    def ended(self, resume_id: str, job_id: str, state: str) -> None:
        try:
            journal = self.open(resume_id)
            journal.flush()
            journal.set_writer(job_id, state)
        except Exception as exc:
            print(
                f"crucible: could not record on journal {resume_id} that job "
                f"{job_id} ended {state}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    def forget_new(self, journal: Journal) -> None:
        with self._lock:
            self._open.pop(journal.id, None)
        shutil.rmtree(journal.directory, ignore_errors=True)


    def verify(
        self, resume_id: str, identity: Identity, inputs: list[InputDigest]
    ) -> Journal:
        if not isinstance(resume_id, str) or not _RESUME_ID.match(resume_id):
            raise ApiError(
                400,
                "invalid_resume_id",
                f"resume is {resume_id!r}; a resume id is the 32 hex digits a job's "
                "202 answered with, or one GET /v1/resumable lists",
                {"resume": resume_id},
            )
        journal = self.open(resume_id)
        manifest = journal.manifest
        writer = self.writer_state(manifest)
        if writer["state"] in LIVE_STATES:
            raise ApiError(
                409,
                "resume_in_use",
                f"journal {resume_id} is being written by job {writer['job_id']}, "
                f"which is {writer['state']}; resume it once that job has ended",
                {"resume_id": resume_id, "job_id": writer["job_id"], "state": writer["state"]},
            )
        problem = _difference(manifest, identity, inputs)
        if problem is not None:
            sentence, what = problem
            raise ApiError(
                409,
                "resume_mismatch",
                f"journal {resume_id} is not this work: {sentence}. Resume only "
                "continues the same work; leave out resume to start fresh (the "
                "journal is kept either way)",
                {"resume_id": resume_id, "differs": what},
            )
        return journal

    def _manifest_or_refuse(self, resume_id: str) -> dict[str, Any]:
        path = self._root / resume_id / "manifest.json"
        if self._rooted and path.is_file():
            try:
                manifest = _read(path)
            except (OSError, ValueError) as exc:
                raise ApiError(
                    500,
                    "journal_unreadable",
                    f"journal {resume_id}'s manifest cannot be read: "
                    f"{type(exc).__name__}: {exc}",
                    {"resume_id": resume_id},
                ) from None
            expires = _parse_time(manifest.get("expires_at"))
            if expires is not None and clock.now() > expires and (
                self.writer_state(manifest)["state"] not in LIVE_STATES
            ):
                raise ApiError(
                    410,
                    "resume_expired",
                    f"journal {resume_id} expired at {manifest['expires_at']}: it "
                    f"was last saved {manifest.get('last_saved')} and this server "
                    "keeps a journal for [jobs] retention_days after that. Start "
                    "fresh",
                    {"resume_id": resume_id, "expires_at": manifest["expires_at"]},
                )
            return manifest
        tombstone = self._root / GONE / f"{resume_id}.json"
        if self._rooted and tombstone.is_file():
            try:
                gone = _read(tombstone)
            except (OSError, ValueError):
                gone = {}
            raise ApiError(
                410,
                "resume_expired",
                f"journal {resume_id} is gone: {gone.get('detail', 'it was removed')}. "
                "Start fresh",
                {"resume_id": resume_id, "why": gone.get("why"), "when": gone.get("when")},
            )
        raise ApiError(
            404,
            "unknown_resume_id",
            f"this server has no journal {resume_id}; GET /v1/resumable lists the "
            "ones it has",
            {"resume_id": resume_id},
        )


    def writer_state(self, manifest: dict[str, Any]) -> dict[str, Any]:
        writer = dict(manifest.get("writer") or {})
        job_id = writer.get("job_id")
        state = writer.get("state")
        if state in LIVE_STATES:
            live = None if job_id is None else self._live_state(str(job_id))
            writer["state"] = live if live is not None else "interrupted"
        return writer

    def entry(self, resume_id: str) -> dict[str, Any]:
        if not _RESUME_ID.match(resume_id):
            raise ApiError(
                404,
                "unknown_resume_id",
                f"{resume_id!r} is not a resume id; GET /v1/resumable lists them",
                {"resume_id": resume_id},
            )
        journal = self._open.get(resume_id)
        manifest = journal.manifest if journal is not None else None
        if manifest is None:
            path = self._root / resume_id / "manifest.json"
            if not self._rooted or not path.is_file():
                self._manifest_or_refuse(resume_id)
            manifest = _read(path)
        return self._describe(manifest)

    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self._rooted or not self._root.is_dir():
            return rows
        for directory in self._root.iterdir():
            if not directory.is_dir() or not _RESUME_ID.match(directory.name):
                continue
            journal = self._open.get(directory.name)
            try:
                manifest = (
                    journal.manifest
                    if journal is not None
                    else _read(directory / "manifest.json")
                )
            except (OSError, ValueError):
                continue
            rows.append(self._describe(manifest))
        rows.sort(key=lambda row: str(row.get("last_saved") or ""), reverse=True)
        return rows

    def _describe(self, manifest: dict[str, Any]) -> dict[str, Any]:
        writer = self.writer_state(manifest)
        done = manifest.get("units_done") or 0
        total = manifest.get("units_total")
        return {
            "resume_id": manifest["resume_id"],
            "job_type": manifest.get("job_type"),
            "model": manifest.get("model"),
            "format_version": manifest.get("format_version"),
            "inputs": list(manifest.get("inputs") or []),
            "params": manifest.get("params"),
            "units_done": done,
            "units_total": total,
            "progress": manifest.get("progress"),
            "created": manifest.get("created"),
            "last_saved": manifest.get("last_saved"),
            "expires_at": manifest.get("expires_at"),
            "job_id": manifest.get("job_id"),
            "last_job_id": writer.get("job_id"),
            "state": writer.get("state"),
            "jobs": list(manifest.get("jobs") or []),
        }


    def discard(self, resume_id: str) -> dict[str, Any]:
        entry = self.entry(resume_id)
        if entry["state"] in LIVE_STATES:
            raise ApiError(
                409,
                "resume_in_use",
                f"journal {resume_id} is being written by job {entry['last_job_id']}, "
                f"which is {entry['state']}; cancel that job first, or wait for it",
                {"resume_id": resume_id, "job_id": entry["last_job_id"], "state": entry["state"]},
            )
        self._remove(resume_id, "discarded", "it was discarded through DELETE /v1/resumable")
        return {
            "resume_id": resume_id,
            "discarded": True,
            "units_done": entry["units_done"],
            "units_total": entry["units_total"],
        }

    def _remove(self, resume_id: str, why: str, because: str) -> bool:
        when = clock.now().isoformat()
        directory = self._root / resume_id
        write_atomically(
            self._root / GONE / f"{resume_id}.json",
            {
                "resume_id": resume_id,
                "why": why,
                "when": when,
                "detail": f"it was removed at {when} because {because}",
            },
        )
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            print(
                f"crucible: could not remove journal {resume_id} ({why}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return False
        with self._lock:
            self._open.pop(resume_id, None)
        print(f"crucible: removed journal {resume_id} — {because}", file=sys.stderr)
        return True


    def reap(self, now: datetime | None = None) -> list[str]:
        moment = now if now is not None else clock.now()
        taken: list[str] = []
        if not self._rooted or not self._root.is_dir():
            return taken
        horizon = timedelta(days=self._retention_days())
        for directory in sorted(self._root.iterdir()):
            if not directory.is_dir() or not _RESUME_ID.match(directory.name):
                continue
            journal = self._open.get(directory.name)
            try:
                manifest = (
                    journal.manifest
                    if journal is not None
                    else _read(directory / "manifest.json")
                )
            except (OSError, ValueError):
                try:
                    age = moment.timestamp() - directory.stat().st_mtime
                except OSError:
                    continue
                if age > horizon.total_seconds() and self._remove(
                    directory.name, "aged", "its manifest was unreadable and it was older than retention_days"
                ):
                    taken.append(directory.name)
                continue
            if self.writer_state(manifest)["state"] in LIVE_STATES:
                continue
            saved = _parse_time(manifest.get("last_saved"))
            if saved is None or moment - saved <= horizon:
                continue
            if self._remove(
                directory.name,
                "aged",
                f"it was last saved {manifest.get('last_saved')}, more than "
                "[jobs] retention_days ago",
            ):
                taken.append(directory.name)
        return taken


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _difference(
    manifest: dict[str, Any], identity: Identity, inputs: list[InputDigest]
) -> tuple[str, dict[str, Any]] | None:
    if manifest.get("job_type") != identity.job_type:
        return (
            f"it is a {manifest.get('job_type')!r} journal and this is a "
            f"{identity.job_type!r} job",
            {"field": "job_type", "journal": manifest.get("job_type"), "job": identity.job_type},
        )
    model = manifest.get("model") or {}
    if model.get("id") != identity.model:
        return (
            f"it was written by model {model.get('id')!r} and this job names "
            f"{identity.model!r}",
            {"field": "model", "journal": model.get("id"), "job": identity.model},
        )
    if model.get("revision") != identity.revision:
        return (
            f"it was written by {identity.model!r} at revision "
            f"{model.get('revision')!r}, and this server has "
            f"{identity.revision!r} now",
            {"field": "revision", "journal": model.get("revision"), "job": identity.revision},
        )
    if manifest.get("format_version") != identity.format_version:
        return (
            f"its units are {identity.job_type} format version "
            f"{manifest.get('format_version')}, and this build writes version "
            f"{identity.format_version}; a unit's meaning changed between them",
            {
                "field": "format_version",
                "journal": manifest.get("format_version"),
                "job": identity.format_version,
            },
        )
    journal_inputs = {row["name"]: row for row in manifest.get("inputs") or []}
    job_inputs = {digest.name: digest for digest in inputs}
    for name in sorted(set(journal_inputs) | set(job_inputs)):
        before, now = journal_inputs.get(name), job_inputs.get(name)
        if before is None:
            return (
                f"this job has an input {name!r} the journal's job did not "
                f"(it had {sorted(journal_inputs)})",
                {"field": "inputs", "input": name, "journal": None, "job": now.sha256 if now else None},
            )
        if now is None:
            return (
                f"the journal's job had an input {name!r} this job does not "
                f"(this one has {sorted(job_inputs)})",
                {"field": "inputs", "input": name, "journal": before["sha256"], "job": None},
            )
        if before["sha256"] != now.sha256:
            return (
                f"the input {name!r} is not the same file: sha256 "
                f"{before['sha256']} ({before['bytes']:,} bytes) in the journal vs "
                f"{now.sha256} ({now.bytes:,} bytes) now",
                {"field": "inputs", "input": name, "journal": before["sha256"], "job": now.sha256},
            )
    journal_params = manifest.get("params") or {}
    job_params = json.loads(canonical(identity.params))
    for key in sorted(set(journal_params) | set(job_params)):
        before, now = journal_params.get(key, _ABSENT), job_params.get(key, _ABSENT)
        if before != now:
            return (
                f"the param {key!r} was {_show(before)} and is {_show(now)} now",
                {
                    "field": "params",
                    "param": key,
                    "journal": None if before is _ABSENT else before,
                    "job": None if now is _ABSENT else now,
                },
            )
    return None


class _Absent:
    def __repr__(self) -> str:
        return "absent"


_ABSENT = _Absent()


def _show(value: Any) -> str:
    if value is _ABSENT:
        return "absent"
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:77] + "..."
