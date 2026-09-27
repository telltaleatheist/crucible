"""The resume journal: a job's finished units of work, on disk as each lands.

docs/RESUMABLE-JOBS.md is the contract. The rulings, verbatim:

- Owen, 2026-09-27: *"we should definitely be writing work to disk, so if
  something fails, we dont lose everything. preferably writing to disk often.
  the nature of crucible necessitates very long jobs. sometimes stretched over
  days. one failure would lose a lot of work."*
- Owen, 2026-09-27: *"im thinking resuming can be a specific flag. if the user
  doesnt send the resume flag then it starts fresh. if they do send a resume
  flag, it continues from where they left off. maybe we could even have a call
  that shows what's available to resume?"*
- Owen, 2026-09-27: Crucible does not queue — *"It grants and releases leases.
  That's it"*. So nothing here retries anything. A journal is kept; whether and
  when to resume it is the app's decision, made with `resume: "<resume_id>"`.

WHERE IT LIVES AND WHY. `<CRUCIBLE_HOME>/journals/<resume_id>/`, beside `jobs/`
and not inside a job's directory: `JobStore.reap` deletes a job's directory
the moment its artifacts are fetched, and a failed job's the moment it ages
out, and the work a journal holds is exactly the work those jobs did NOT
finish handing over. The journal outlives every job that writes it.

    journals/<resume_id>/manifest.json     what the work IS, and how far it got
    journals/<resume_id>/units/<key>.json  one finished unit each
    journals/_gone/<resume_id>.json        the tombstone of a reaped or
                                           discarded journal, so its id is
                                           refused as expired, not unknown

A UNIT IS WRITTEN WHOLE OR NOT AT ALL. Every file here is written to a
temporary name in the same directory, flushed and fsynced, then `os.replace`d
into place, and the directory is fsynced where the platform allows it (not on
Windows, which has no directory handle to sync). A crash mid-write leaves a
dot-named temporary file that no reader ever looks at, never a half unit that
reads as finished.

EXPLICIT RESUME ONLY. A job submitted without `resume` gets a NEW journal and
never reads an old one; earlier journals are kept, so forgetting the flag
destroys nothing. A job WITH `resume` continues one only after `verify` has
found it the same work: job type, model and exact revision, every
output-affecting param, every input's sha256, and the job type's format
version. Anything else is refused before the job exists, naming what differs.
"""

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .errors import ApiError

#: The version of THIS container: the manifest's shape and the unit file's
#: envelope. A job type's own `format_version` is the meaning of its units and
#: moves independently.
JOURNAL_FORMAT = 1

#: The name of the directory the tombstones live in. Underscored so it can
#: never be a resume id (those are 32 hex digits).
GONE = "_gone"

#: How often a running job may rewrite the manifest for progress alone. The
#: units are the truth and are written the moment each lands; the manifest's
#: counts are what `GET /v1/resumable` shows, and a count a few seconds stale
#: costs nothing a restart would notice.
PROGRESS_WRITE_SECONDS = 2.0

#: How many times a replace is retried when Windows says the target is open.
#: A reader holds a manifest for microseconds; a writer that gave up on the
#: first `PermissionError` would lose a unit to a listing.
REPLACE_ATTEMPTS = 10

_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_RESUME_ID = re.compile(r"^[0-9a-f]{32}$")

#: The states a journal's last writer can be in, beside the job ones.
LIVE_STATES = frozenset({"queued", "running"})


def utcnow() -> datetime:
    """The journal's clock, in one place so a check can move it."""
    return datetime.now(timezone.utc)


def canonical(value: Any) -> str:
    """The one spelling of a params object that two runs can compare."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Make a rename durable where the platform lets us. Windows cannot open a
    directory for syncing, and NTFS journals its metadata anyway."""
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
    """Write `document` as JSON to `path`: temporary, fsync, replace, fsync dir.

    THE ONLY WAY ANYTHING IN A JOURNAL IS WRITTEN. The temporary name starts
    with a dot, so a crash between the write and the replace leaves a file no
    reader opens (`Journal.get` and `keys` read `<key>.json` names only).
    """
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
            # Windows: somebody has the target open for a read. Wait it out.
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    _fsync_directory(path.parent)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Identity:
    """What a journal is the work OF. Two runs with equal identities produce
    the same units; anything that could change a unit belongs here.

    `params` is only what changes the output, as the job type decides it:
    never `resume` itself, and never a knob that changes how fast the work goes
    and not what it says.
    """

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
    """One journal, open for reading and writing by the job that runs it.

    Written from the job's worker thread only; read from the event loop by
    the listing, which only ever reads whole files (`write_atomically`).
    """

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
        """A copy, safe to read on another thread while the job writes."""
        with self._lock:
            return json.loads(json.dumps(self._manifest))

    @property
    def units_dir(self) -> Path:
        return self._dir / "units"

    # ---------------------------------------------------------------- units

    def keys(self) -> set[str]:
        """Every unit on disk. Read once, then kept current by `put`."""
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
        """The unit's data, or None when it was never finished."""
        _check_key(key)
        path = self.units_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            envelope = _read(path)
        except (OSError, ValueError) as exc:
            # Unreachable by construction (a unit is replaced into place whole),
            # so said loudly and treated as not done: the work is redone rather
            # than stitched from a file nobody can read.
            print(
                f"crucible: journal {self.id} unit {key} is unreadable and will be "
                f"redone: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None
        return envelope.get("data")

    def put(self, key: str, data: Any) -> None:
        """Write one finished unit, durably, before returning."""
        _check_key(key)
        now = utcnow()
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
        """A unit whose result is BYTES (a converted file, a separated stem).

        For the job types whose units are audio (rvc, denoise; the contract in
        docs/RESUMABLE-JOBS.md). The bytes are copied beside the unit as
        `<key>.bin` first, fsynced and replaced into place, and only then is the
        `<key>.json` written that says the unit is finished — so a crash
        between the two leaves bytes no reader trusts, never a finished unit
        without its bytes. Read back with `file(key)`.
        """
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
        """The bytes of a finished `put_file` unit, or None."""
        unit = self.get(key)
        if not isinstance(unit, dict) or "file" not in unit:
            return None
        path = self.units_dir / str(unit["file"])
        return path if path.is_file() else None

    def progress(self, done: int, total: int, sentence: str, *, force: bool = False) -> None:
        """The job type's own count of its units, and a sentence a person reads."""
        with self._lock:
            self._manifest["units_done"] = int(done)
            self._manifest["units_total"] = int(total)
            self._manifest["progress"] = sentence
            self._manifest_dirty = True
        self._write_manifest(force=force)

    def flush(self) -> None:
        self._write_manifest(force=True)

    # ------------------------------------------------------------- manifest

    def _touch(self, now: datetime) -> None:
        self._manifest["last_saved"] = now.isoformat()
        self._manifest["expires_at"] = (
            now + timedelta(days=self._retention_days)
        ).isoformat()
        self._manifest_dirty = True

    def _write_manifest(self, *, force: bool) -> None:
        # ONE WRITER AT A TIME, snapshot and replace together: the job's
        # thread (units) and the lane (how the job ended) both write it, and a
        # stale snapshot replaced after a fresh one would undo the fresh one.
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
        """Record which job is writing this journal now, and how it ended."""
        with self._lock:
            writer = {"job_id": job_id, "state": state, "at": utcnow().isoformat()}
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
    """Every journal under one Crucible home. Owned by the `JobStore`.

    Open journals are cached by id so the job writing one and the lane
    recording how that job ended share one manifest in memory.
    """

    def __init__(
        self,
        root: Path | None,
        retention_days: Callable[[], float],
        live_state: Callable[[str], str | None],
    ) -> None:
        #: None for a store with no home (`crucible doctor`): it keeps none,
        #: lists none, and refuses to start one by name.
        self._root: Path = Path(root) if root is not None else Path()
        self._rooted = root is not None
        self._retention_days = retention_days
        #: `job_id -> "queued" | "running" | None`: what the lane says about
        #: a job right now, so a journal whose writer died with its server
        #: reads `interrupted` rather than `running` for ever.
        self._live_state = live_state
        self._open: dict[str, Journal] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        return self._root if self._rooted else None

    # ------------------------------------------------------------- lifecycle

    def create(
        self, identity: Identity, inputs: list[InputDigest], job_id: str
    ) -> Journal:
        """A NEW journal for a job submitted without `resume`. Never reuses one."""
        if not self._rooted:
            raise ApiError(
                500,
                "journal_unwritable",
                "this job store has no Crucible home, so it has nowhere to keep a "
                "resume journal",
            )
        resume_id = uuid.uuid4().hex
        now = utcnow()
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
        """The journal, or `unknown_resume_id` / `resume_expired` by name."""
        with self._lock:
            cached = self._open.get(resume_id)
        if cached is not None:
            return cached
        manifest = self._manifest_or_refuse(resume_id)
        journal = Journal(self._root / resume_id, manifest, self._retention_days())
        with self._lock:
            return self._open.setdefault(resume_id, journal)

    def adopt(self, journal: Journal, job_id: str) -> None:
        """A resumed job takes over writing this journal."""
        journal.set_writer(job_id, "queued", resumed=True)

    def ended(self, resume_id: str, job_id: str, state: str) -> None:
        """Record how the job writing this journal ended. Never raises."""
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
        """Delete a journal created for a submission that was then refused."""
        with self._lock:
            self._open.pop(journal.id, None)
        shutil.rmtree(journal.directory, ignore_errors=True)

    # ---------------------------------------------------------------- verify

    def verify(
        self, resume_id: str, identity: Identity, inputs: list[InputDigest]
    ) -> Journal:
        """The journal `resume` names, if it is the same work. Refuses by name.

        Every difference is its own sentence, and the refusal names the first
        one it finds in the order a person would check: the job type, the
        model, the format, each input, then each param.
        """
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
            if expires is not None and utcnow() > expires and (
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

    # --------------------------------------------------------------- listing

    def writer_state(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Which job last wrote this journal and how that ended, as of now.

        A writer recorded `queued` or `running` whose job the lane is not
        running is a writer whose server stopped under it: `interrupted`, for
        `jobs/base.py`'s INTERRUPTED reason.
        """
        writer = dict(manifest.get("writer") or {})
        job_id = writer.get("job_id")
        state = writer.get("state")
        if state in LIVE_STATES:
            live = None if job_id is None else self._live_state(str(job_id))
            writer["state"] = live if live is not None else "interrupted"
        return writer

    def entry(self, resume_id: str) -> dict[str, Any]:
        """One journal as `GET /v1/resumable/{id}` answers it."""
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
                self._manifest_or_refuse(resume_id)  # raises: expired or unknown
            manifest = _read(path)
        return self._describe(manifest)

    def list(self) -> list[dict[str, Any]]:
        """Every journal, newest first by when it was last saved."""
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

    # ---------------------------------------------------------------- delete

    def discard(self, resume_id: str) -> dict[str, Any]:
        """`DELETE /v1/resumable/{id}`. Refused while a job is writing it."""
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
        when = utcnow().isoformat()
        directory = self._root / resume_id
        # The tombstone FIRST: a crash between the two leaves a journal that
        # also has a tombstone, which is read as the journal (it is checked
        # first) and taken again on the next tick.
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

    # ------------------------------------------------------------------ reap

    def reap(self, now: datetime | None = None) -> list[str]:
        """Remove every journal past its `expires_at`. Called by `JobStore.reap`.

        THE SAME COLLECTOR AS THE JOB DIRECTORIES, not a second one (Owen,
        2026-09-25: *"a garbage collector clean up files older than 7 days"*):
        a journal expires `retention_days` after it was last saved, and a
        journal whose writer is queued or running is never taken, however old,
        for `reap`'s own reason — a job eight days into a book is this
        afternoon's work.
        """
        moment = now if now is not None else utcnow()
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
                # A directory with no readable manifest is a create that never
                # finished; its age is the directory's.
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
    """The first way this submission is not the journal's work, or None."""
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
