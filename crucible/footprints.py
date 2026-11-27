"""What a model actually takes on this machine, measured while it serves, not declared.

Owen, 2026-11-27: "we have measurements to tell us how much memory is used for each process
right? how much each model ACTUALLY takes in practice rather than in theory? can we update
those numbers to be more accurate?" and, of the host's memory: "that should probably be part
of the calculation too". A manifest's `memory_bytes_estimate` is a forecast. Every load is
watched while it is resident, and what it held on the card and in host memory is kept here,
per model and per the estimate it was loaded under (one model at two widths or contexts is
two shapes). The next load of the same shape is weighed against what the last ones took.

What counts:

- Only a residency that served work. A model that loaded and sat idle never reached its
  working peak (a torch worker's activations, a KV cache filling), so it says nothing about
  what the next load will need; such a run is not kept.
- The card: what the card's used memory rose by from just before the load, at its highest
  while the model was resident (cuda-linux and llama-windows; under WSL2 the driver names no
  process, so the rise is the only figure there is). A program that opens on the card
  meanwhile reads as the model's (it errs large); one that closes meanwhile hides some of the
  model's (it errs small). So the figure used is the LARGEST of the last `KEPT_RUNS` served
  runs: one bad reading lasts until it is outvoted, and a low one is never trusted alone
  against higher.
- Host memory: what every process the model's engine leads holds, by the system's own count
  (crucible/procmemory.py: Pss on Linux, the physical footprint on a Mac, the working set on
  Windows), summed, at its highest; the largest of the kept runs is used the same way. On a
  Mac the card and host memory are one pool, so this is the model's whole figure there.

Every backend is watched (Owen, 2026-11-27: "also remember some systems wont have wsl. some
systems will be using a mac, or windows").
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import procmemory
from .accelerator import ProbeError

FOOTPRINTS_NAME = "footprints.json"

# How many served runs of one shape are kept; the largest of them is the figure used.
KEPT_RUNS = 3

# How often a resident model's memory is read while it is resident.
SAMPLE_S = 5.0

# A peak is written through to disk when it has grown by this much since the last write, so
# a residency that ends without an unload (the server killed) still leaves its figure.
WRITE_GROWTH_BYTES = 64 * 1024**2

@dataclass(frozen=True)
class Run:
    """One served residency: what it held at its highest."""

    run_id: str
    card_bytes: int | None
    ram_bytes: int | None
    measured_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "card_bytes": self.card_bytes,
            "ram_bytes": self.ram_bytes,
            "measured_at": self.measured_at,
        }


@dataclass(frozen=True)
class Footprint:
    """The kept runs of one model at one shape."""

    model: str
    estimate_bytes: int
    runs: tuple[Run, ...]

    @property
    def card_bytes(self) -> int | None:
        figures = [run.card_bytes for run in self.runs if run.card_bytes is not None]
        return max(figures) if figures else None

    @property
    def ram_bytes(self) -> int | None:
        figures = [run.ram_bytes for run in self.runs if run.ram_bytes is not None]
        return max(figures) if figures else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "estimate_bytes": self.estimate_bytes,
            "card_bytes": self.card_bytes,
            "ram_bytes": self.ram_bytes,
            "runs": [run.to_dict() for run in self.runs],
        }


def footprints_path(home: Path) -> Path:
    return Path(home) / FOOTPRINTS_NAME


def _key(model: str, estimate_bytes: int) -> str:
    return f"{model}|{estimate_bytes}"


def _read_all(home: Path) -> dict[str, Any]:
    path = footprints_path(home)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON object")
    return document


def _run_from(row: dict[str, Any]) -> Run:
    return Run(
        run_id=str(row["run_id"]),
        card_bytes=row["card_bytes"],
        ram_bytes=row["ram_bytes"],
        measured_at=str(row["measured_at"]),
    )


_LOCK = threading.Lock()


def read(home: Path, model: str, estimate_bytes: int) -> Footprint | None:
    """The kept runs of ``model`` loaded under ``estimate_bytes``; None before any served."""
    with _LOCK:
        row = _read_all(home).get(_key(model, estimate_bytes))
    if row is None:
        return None
    return Footprint(
        model=model,
        estimate_bytes=estimate_bytes,
        runs=tuple(_run_from(run) for run in row["runs"]),
    )


def read_every(home: Path) -> list[Footprint]:
    with _LOCK:
        document = _read_all(home)
    return [
        Footprint(
            model=str(row["model"]),
            estimate_bytes=int(row["estimate_bytes"]),
            runs=tuple(_run_from(run) for run in row["runs"]),
        )
        for row in document.values()
    ]


def record(home: Path, model: str, estimate_bytes: int, run: Run) -> Footprint:
    """Keep ``run`` (replacing an earlier write of the same run), newest last, at most
    `KEPT_RUNS`."""
    with _LOCK:
        document = _read_all(home)
        key = _key(model, estimate_bytes)
        row = document.get(key) or {"model": model, "estimate_bytes": estimate_bytes, "runs": []}
        runs = [r for r in row["runs"] if r["run_id"] != run.run_id]
        runs.append(run.to_dict())
        row["runs"] = runs[-KEPT_RUNS:]
        document[key] = row
        path = footprints_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        writing = path.with_name(path.name + ".writing")
        writing.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(writing, path)
    return Footprint(
        model=model,
        estimate_bytes=estimate_bytes,
        runs=tuple(_run_from(r) for r in row["runs"]),
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Watch:
    """Reads a resident model's card and host memory every `SAMPLE_S` while it is resident,
    keeps the highest of each, and writes the run once it has served work.

    ``card_used`` reads the card's used bytes, raising ProbeError when it cannot (unread
    where ``card_before`` is None: a Mac, whose one pool is the host memory read); ``pids``
    names every process the model's engine leads; ``working`` says whether Crucible has work
    in flight. A failed read is said once and skipped, never taken as zero."""

    def __init__(
        self,
        *,
        home: Path,
        model: str,
        estimate_bytes: int,
        card_before: int | None,
        card_used: Callable[[], int],
        pids: Callable[[], Iterable[int]],
        working: Callable[[], bool],
        say: Callable[[str], None],
        sample_s: float = SAMPLE_S,
        ram_of: Callable[[Iterable[int]], int] = procmemory.process_bytes,
    ) -> None:
        self._home = home
        self._model = model
        self._estimate = estimate_bytes
        self._card_before = card_before
        self._card_used = card_used
        self._pids = pids
        self._working = working
        self._say = say
        self._sample_s = sample_s
        self._ram_of = ram_of
        self._run_id = f"{model}@{utcnow()}"
        self._card_peak: int | None = None
        self._ram_peak: int | None = None
        self._ram_now: int | None = None
        self._served = False
        self._written: tuple[int, int] = (0, 0)
        self._complained: set[str] = set()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name=f"footprint-{model}", daemon=True
        )

    @property
    def ram_now(self) -> int | None:
        """What the model holds in host memory at the last read: what unloading it frees."""
        return self._ram_now

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def _complain_once(self, what: str, exc: BaseException) -> None:
        if what in self._complained:
            return
        self._complained.add(what)
        self._say(f"could not read {what} for {self._model}'s footprint ({exc}); "
                  "that read is skipped, never taken as zero")

    def sample(self) -> None:
        working = self._working()
        card: int | None = None
        if self._card_before is not None:
            try:
                used: int | None = self._card_used()
            except ProbeError as exc:
                self._complain_once("the card", exc)
                used = None
            if used is not None:
                card = max(0, used - self._card_before)
        try:
            ram: int | None = self._ram_of(self._pids())
        except ProbeError as exc:
            self._complain_once("host memory", exc)
            ram = None
        with self._lock:
            if ram is not None:
                self._ram_now = ram
            if card is not None:
                self._card_peak = card if self._card_peak is None else max(self._card_peak, card)
            if ram is not None:
                self._ram_peak = ram if self._ram_peak is None else max(self._ram_peak, ram)
            self._served = self._served or working
            grew = (
                (self._card_peak or 0) - self._written[0] >= WRITE_GROWTH_BYTES
                or (self._ram_peak or 0) - self._written[1] >= WRITE_GROWTH_BYTES
            )
        if self._served and grew:
            self._write()

    def _loop(self) -> None:
        while not self._stop.wait(self._sample_s):
            self.sample()

    def _write(self) -> Footprint | None:
        with self._lock:
            if not self._served:
                return None
            run = Run(
                run_id=self._run_id,
                card_bytes=self._card_peak,
                ram_bytes=self._ram_peak,
                measured_at=utcnow(),
            )
            self._written = (self._card_peak or 0, self._ram_peak or 0)
        return record(self._home, self._model, self._estimate, run)

    def finish(self) -> Footprint | None:
        """Stop reading; keep the run if it served work. The last read is taken first, so a
        short residency is not left with only its load-time figure."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._sample_s * 2)
        self.sample()
        return self._write()
