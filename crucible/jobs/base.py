"""The job-type contract and the per-job run context.

A job type is a plugin module under `crucible/jobs/<type>/` declaring what it can
serve and how to run it (DESIGN.md section 3). Phase 1 ships exactly one: `echo`.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..errors import JobCancelled

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({DONE, FAILED, CANCELLED})


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelDescriptor:
    """One model a job type can serve, as advertised by GET /info."""

    id: str
    revision: str
    source: str
    resident: bool
    vram_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "source": self.source,
            "resident": self.resident,
            "vram_bytes": self.vram_bytes,
        }


@dataclass(frozen=True)
class JobTypeStatus:
    """What `crucible doctor` prints for one job type."""

    ready: bool
    detail: str


@dataclass
class Job:
    id: str
    type: str
    model: str | None
    params: dict[str, Any]
    dir: Path
    created: str
    status: str = QUEUED
    progress: float = 0.0
    started: str | None = None
    finished: str | None = None
    error: dict[str, str] | None = None
    artifacts: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    @property
    def inputs_dir(self) -> Path:
        return self.dir / "inputs"

    @property
    def artifacts_dir(self) -> Path:
        return self.dir / "artifacts"


@runtime_checkable
class JobType(Protocol):
    """What every job-type plugin declares."""

    name: str

    def describe_models(self) -> list[ModelDescriptor]:
        """Models this type can serve on this host. Empty means the type takes no model."""

    def vram_estimate(self, model: str | None) -> int:
        """Bytes of accelerator memory a run of `model` needs."""

    def check(self, backend: Any) -> JobTypeStatus:
        """Whether this type can run here right now, and why not if it cannot."""

    def run(self, job: Job, ctx: "JobContext") -> None:
        """Do the work. Blocking; the queue runs it on a worker thread."""


class JobContext:
    """Handed to `run()`. Progress and artifacts go back through here.

    `run()` executes on a worker thread, so every mutation of the job is marshalled
    back onto the event loop with `call_soon_threadsafe`. That also keeps event
    ordering exactly as the job emitted it.
    """

    def __init__(self, store: Any, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        self._store = store
        self._job = job
        self._loop = loop

    @property
    def job(self) -> Job:
        return self._job

    @property
    def scratch(self) -> Path:
        return self._job.dir

    def inputs(self) -> dict[str, Path]:
        """The job's inputs, already materialised on disk, by name."""
        directory = self._job.inputs_dir
        if not directory.is_dir():
            return {}
        return {p.name: p for p in sorted(directory.iterdir()) if p.is_file()}

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested

    def raise_if_cancelled(self) -> None:
        if self._job.cancel_requested:
            raise JobCancelled(f"job {self._job.id} was cancelled")

    def progress(self, fraction: float, message: str) -> None:
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise TypeError(f"progress fraction must be a number, got {fraction!r}")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"progress fraction must be in [0, 1], got {fraction!r}")
        self._loop.call_soon_threadsafe(
            self._store.append_event,
            self._job,
            "progress",
            {"fraction": float(fraction), "message": message},
        )

    def artifact(self, name: str, path: Path) -> Path:
        """Publish `path` as artifact `name`, with its provenance sidecar.

        Returns the artifact's final path. The sidecar is written immediately, so
        `<name>.provenance.json` exists for as long as the artifact does.
        """
        validate_member_name(name)
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"artifact source {source} is not a file")
        destination = self._job.artifacts_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        sidecar = self._job.artifacts_dir / f"{name}.provenance.json"
        sidecar.write_text(
            json.dumps(self._store.provenance(self._job), indent=2) + "\n",
            encoding="utf-8",
        )
        self._loop.call_soon_threadsafe(self._store.record_artifact, self._job, name)
        return destination


_FORBIDDEN_NAME_PARTS = ("/", "\\", "\x00")


def validate_member_name(name: str) -> None:
    """Input and artifact names are single path members, never traversals."""
    if name == "" or name in (".", ".."):
        raise ValueError(f"invalid name {name!r}")
    for part in _FORBIDDEN_NAME_PARTS:
        if part in name:
            raise ValueError(f"invalid name {name!r}: contains {part!r}")
    if name.startswith("."):
        raise ValueError(f"invalid name {name!r}: must not start with a dot")
