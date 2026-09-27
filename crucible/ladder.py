"""The measurement ladder: what THIS card can do, found out by running it.

docs/PROPOSAL-GPU-LADDER.md is the design; this is the build. Owen, 2026-09-26:

> *"we can quantize if we need to. no less than 4. ... our measurement tool
> should determine how much space is available, whether tensors are available,
> cuda graphs, vllm, etc. and install the best the user can use"*

The declared half (`backend.CardFacts` off the compute capability) answers
what a spec sheet can: bf16, FlashAttention 2, fp8, tensor cores. The ladder
answers what only a run can, and records it where capability reads it:

    rung          touches the GPU?   answers
    card          no (nvidia-smi)    total / free / desktop memory, disk space,
                                     driver, the declared features
    env           yes                each installed env's torch initialises on
                                     this card, and how fast each dtype's
                                     matmul is (bf16 is EMULATED below 8.0)
    cuda_graphs   yes                a CUDA graph captures and replays in the
                                     llm env -> `backend.CUDA_GRAPHS`
    vllm          yes                vLLM starts the smallest installed model
                                     and answers one request -> `VLLM_STARTS`

A rung that fails is a MEASUREMENT (MEASUREMENTS.md: "Failures are
measurements"), recorded with its first error line. A rung that could not run
cleanly — the card was in use, somebody else's work appeared mid-run — is
`waiting` or `interrupted`, and is never read as a failure.

ON A CARD SOMEBODY IS USING. Every GPU rung is preceded by the accelerator
guard's own test (`accelerator.guard`: a foreign compute app over the floor, or
unattributed VRAM past the desktop allowance under WSL2) and does not start if
it fails. While a rung runs, `Watch` samples nvidia-smi at 1 Hz, as
`scripts/calibrate-kv.sh` does, and a foreign compute app appearing makes the
rung `interrupted`. Crucible never evicts anything, and a measurement is no
exception. A job that reaches the server while a rung holds the card is
refused `accelerator_busy` by the server's own guard for those seconds — the
same answer it gives for any other tenant.

WHAT IS CONSUMED, AND WHAT IS ONLY RECORDED (the proposal's calls 3 and 4,
settled by the ruling above; docs/PROPOSAL-GPU-LADDER.md section 7):

* `cuda_graphs` false -> vLLM is started `--enforce-eager` (slower; it runs).
* `vllm` false -> every vLLM candidate is refused here, quoting the run.
* speed is REPORTED and never refuses: "install the best the user can use" is
  the best that runs, and a slow card still runs it.
* memory is RECORDED, not consumed for fit: PHASE9's ruling that declared
  estimates decide fit stands until Owen reverses it. The desktop the card
  rung sees is reported by `crucible doctor` beside the allowance.

THE RECORD is `<home>/ladder/card.json`, keyed on the card (name, compute
capability, total memory) and this Crucible's version. A different key reads
as no record — nothing measured, nothing refused — and `doctor` says it is
stale. Rewritten whole, atomically, after every rung.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import VERSION, accelerator
from .backend import (
    CUDA_GRAPHS,
    CUDA_LINUX,
    MLX_DARWIN,
    VLLM_STARTS,
    WSL_NVIDIA_SMI,
    Backend,
    CardFacts,
    Gpu,
    declared_card,
    nvidia_smi_path,
)
from .config import DEFAULT_DESKTOP_ALLOWANCE_BYTES
from .errors import ApiError, CrucibleError

LADDER_SCHEMA = 1
RECORD_NAME = "card.json"

#: The rungs, in the order they run. Each needs the one before it to have
#: passed where it builds on it (`cuda_graphs` and `vllm` need the llm env's
#: torch to have initialised in `env`).
CARD = "card"
ENV = "env"
GRAPHS = "cuda_graphs"
VLLM = "vllm"
RUNGS: tuple[str, ...] = (CARD, ENV, GRAPHS, VLLM)
#: The rungs that put work on the GPU. Everything else reads nvidia-smi.
GPU_RUNGS: frozenset[str] = frozenset({ENV, GRAPHS, VLLM})

PASSED = "passed"
FAILED = "failed"
INTERRUPTED = "interrupted"
WAITING = "waiting"
SKIPPED = "skipped"

#: How many one-second samples of `memory.used` the card rung takes to see the
#: desktop's size. A sampling window, not a threshold: long enough to see a
#: compositor breathe, short enough that install does not stall on it.
DESKTOP_SAMPLES = 5

#: What a torch smoke rung asks the guard for. The aligner's measurement of
#: 2026-09-26 put the CUDA context and cuBLAS workspaces at 0.95 GiB
#: (align/qwen3-aligner.toml); the smoke's own tensors are a few MiB.
SMOKE_NEED_BYTES = 1024 ** 3

#: The longest a torch smoke subprocess may take, cold import included. A
#: ceiling on a stuck process, not an estimate of how long it takes.
SMOKE_TIMEOUT_SECONDS = 300.0

#: The matmul the env rung times: n x n, ten times, per dtype.
SMOKE_MATMUL_N = 2048
SMOKE_MATMUL_REPEATS = 10

#: The prefix the smoke scripts print their one JSON line after, so a library
#: that prints to stdout cannot be mistaken for the result.
RESULT_PREFIX = "LADDER "

#: What the guard is told the ladder is called, in its refusals.
LADDER_SUBJECT = "the measurement ladder"


class LadderError(CrucibleError):
    """The ladder could not do what it was asked (not a rung that failed)."""


# ------------------------------------------------------------------ the record


@dataclass
class RungResult:
    rung: str
    outcome: str
    measured_at: str
    #: What the rung found, as data (bytes, seconds, booleans).
    facts: dict[str, Any] = field(default_factory=dict)
    #: One sentence: why it failed, waited or was skipped, or what it saw.
    detail: str = ""
    #: What else was on the card while it ran (`Watch.summary`), or None for a
    #: rung that put nothing on the GPU.
    contention: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "outcome": self.outcome,
            "basis": "measured",
            "measured_at": self.measured_at,
            "facts": self.facts,
            "detail": self.detail,
            "contention": self.contention,
        }


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record_path(home: Path) -> Path:
    return home / "ladder" / RECORD_NAME


def card_key(gpu: Gpu) -> dict[str, Any]:
    """What a record is TRUE OF. Another card, or another Crucible, is another
    record: the envs a release pins are part of what vLLM starting means."""
    return {
        "name": gpu.name,
        "compute_capability": gpu.compute_capability,
        "vram_bytes": gpu.vram_bytes,
        "crucible_version": VERSION,
    }


def load_record(home: Path) -> dict[str, Any] | None:
    """The record on disk, or None. A file that will not parse is None too:
    a measurement nobody can read is no measurement, and refusing work on it
    would be refusing on a guess."""
    path = record_path(home)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != LADDER_SCHEMA:
        return None
    return document


def stale_reason(home: Path, gpu: Gpu) -> str | None:
    """Why the record on disk does not describe this card, or None if it does
    (or there is none — absence is not staleness)."""
    document = load_record(home)
    if document is None:
        return None
    recorded = document.get("key", {})
    current = card_key(gpu)
    changed = [name for name in current if recorded.get(name) != current[name]]
    if not changed:
        return None
    return (
        "the measurement record was taken with a different "
        + ", ".join(changed)
        + " than this host has now; `crucible ladder` measures again"
    )


def _write_record(home: Path, gpu: Gpu, results: dict[str, RungResult]) -> Path:
    path = record_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": LADDER_SCHEMA,
        "key": card_key(gpu),
        "written_at": _now(),
        "rungs": {name: results[name].to_dict() for name in RUNGS if name in results},
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _results_from(document: dict[str, Any] | None) -> dict[str, RungResult]:
    found: dict[str, RungResult] = {}
    if document is None:
        return found
    for name, row in (document.get("rungs") or {}).items():
        if not isinstance(row, dict) or name not in RUNGS:
            continue
        found[name] = RungResult(
            rung=name,
            outcome=str(row.get("outcome", "")),
            measured_at=str(row.get("measured_at", "")),
            facts=dict(row.get("facts") or {}),
            detail=str(row.get("detail", "")),
            contention=row.get("contention"),
        )
    return found


#: Which rung answers which measured feature.
_FEATURE_OF_RUNG: dict[str, str] = {GRAPHS: CUDA_GRAPHS, VLLM: VLLM_STARTS}


def card_for(home: Path, gpu: Gpu) -> CardFacts:
    """THE card every decision reads: declared facts, plus the ladder's measured
    ones when its record is of THIS card and THIS Crucible.

    One function, so `crucible capability`, the install step, the API's
    capability rows, the settings recompute and the engine's argv all ask the
    same thing. A stale or missing record gives the declared card alone, whose
    measured facts are unknown — and unknown refuses nothing.
    """
    declared = declared_card(gpu)
    if stale_reason(home, gpu) is not None:
        return declared
    results = _results_from(load_record(home))
    measured: dict[str, bool] = {}
    detail: dict[str, str] = {}
    latest: str | None = None
    for rung, feature in _FEATURE_OF_RUNG.items():
        result = results.get(rung)
        if result is None or result.outcome not in (PASSED, FAILED):
            continue
        measured[feature] = result.outcome == PASSED
        if result.outcome == FAILED and result.detail:
            detail[feature] = result.detail
        if latest is None or result.measured_at > latest:
            latest = result.measured_at
    return CardFacts(
        name=declared.name,
        compute_capability=declared.compute_capability,
        measured=measured,
        measured_detail=detail,
        measured_at=latest,
    )


def summary(home: Path, gpu: Gpu) -> dict[str, Any]:
    """What `crucible doctor` and `crucible ladder --json` print of the record."""
    document = load_record(home)
    return {
        "path": str(record_path(home)),
        "present": document is not None,
        "stale": stale_reason(home, gpu),
        "rungs": {
            name: result.to_dict()
            for name, result in _results_from(document).items()
        },
    }


# ------------------------------------------------------------------ the watch


class Watch:
    """nvidia-smi at 1 Hz while a GPU rung runs: what else was on the card.

    `owned` is the set of pids the rung itself started, read live because an
    engine's pids are only known once it has spawned. A compute app outside it
    holding more than the guard's floor makes the rung `interrupted`.

    UNDER WSL2 THE COMPUTE-APP LIST IS EMPTY even for processes inside the VM
    (accelerator.py's measured limitation), so there `foreign` can never be
    set and the samples of `memory.used` and `utilization.gpu` are the record
    of contention: a number taken while they moved is marked `contended` by
    the reader, not trusted as clean.
    """

    def __init__(self, owned: Callable[[], frozenset[int]]) -> None:
        self._owned = owned
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.max_used_bytes = 0
        self.max_utilization = 0
        self.foreign: dict[int, str] = {}
        self.samples = 0

    def __enter__(self) -> "Watch":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                lines = accelerator._nvidia_smi(
                    "--query-gpu=memory.used,utilization.gpu", "the watch"
                )
                used_mib, utilization = (int(p.strip()) for p in lines[0].split(","))
                self.max_used_bytes = max(self.max_used_bytes, used_mib * 1024 * 1024)
                self.max_utilization = max(self.max_utilization, utilization)
                owned = self._owned()
                for app in accelerator.probe_compute_apps():
                    if app.pid in owned:
                        continue
                    if (
                        app.used_bytes is None
                        or app.used_bytes > accelerator.FOREIGN_PROCESS_FLOOR_BYTES
                    ):
                        self.foreign[app.pid] = app.describe()
                self.samples += 1
            except (accelerator.ProbeError, ValueError):
                pass  # a missed sample is a gap in the record, not a verdict
            self._stop.wait(1.0)

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "max_used_bytes": self.max_used_bytes,
            "max_utilization_percent": self.max_utilization,
            "foreign": sorted(self.foreign.values()),
            "compute_apps_visible": not os.path.exists(WSL_NVIDIA_SMI),
        }


# ------------------------------------------------------------------ rung 0


def _query(fields: str) -> list[str]:
    lines = accelerator._nvidia_smi(f"--query-gpu={fields}", "the card rung")
    return [part.strip() for part in lines[0].split(",")]


@dataclass(frozen=True)
class DesktopSample:
    """`memory.used` over `DESKTOP_SAMPLES` seconds: what the desktop held.

    Device-wide, on purpose. `memory.used` is the whole card's, so it counts
    the desktop's graphics memory whether or not the driver lists the processes
    holding it. That matters twice: on Windows the desktop's processes show as
    compute apps with no memory figure (`[N/A]`, accelerator.py's module
    docstring), and under WSL2 the guest's driver shim lists no compute apps at
    all. What the code already assumes of WSL2 — `unattributed_bytes` subtracts
    the allowance from the guest's `memory.used`, and accelerator.py says
    *"`memory.free` under WSL2 is accurate for the whole card"* — was measured
    on 2026-09-18 (MEASUREMENTS.md, Finding 1): *"Windows' own nvidia-smi agrees
    with WSL's nvidia-smi (3_286 vs 3_319 MiB used)"*. So a sample taken inside
    the guest sees the Windows desktop. What it cannot see is WHOSE the memory
    is: a Windows-side job running at that moment is counted as desktop, which
    `desktop_allowance_from`'s ceiling bounds.
    """

    least_bytes: int
    peak_bytes: int
    total_bytes: int
    samples: int
    #: The UTC date the samples were taken, for the config's note.
    on: str


def sample_desktop() -> DesktopSample:
    """Sample the card's used memory once a second for `DESKTOP_SAMPLES` seconds.

    THE ONE PLACE THE DESKTOP IS SAMPLED (ARCHITECTURE.md R1): the card rung
    reports it, and `measure_desktop_reserve` sizes the reserve from it, so the
    two can never disagree about how the desktop was seen. Raises
    `accelerator.ProbeError` (or `ValueError` on an unparseable answer) rather
    than returning a number nobody read.
    """
    total = int(_query("memory.total")[0]) * 1024 * 1024
    used: list[int] = []
    for sample in range(DESKTOP_SAMPLES):
        used.append(int(_query("memory.used")[0]) * 1024 * 1024)
        if sample + 1 < DESKTOP_SAMPLES:
            time.sleep(1.0)
    return DesktopSample(
        least_bytes=min(used),
        peak_bytes=max(used),
        total_bytes=total,
        samples=len(used),
        on=datetime.now(timezone.utc).date().isoformat(),
    )


def desktop_allowance_from(peak_bytes: int) -> int:
    """The reserve a measured desktop gets: its peak, doubled, and at least one
    desktop-scale process more — never above `DEFAULT_DESKTOP_ALLOWANCE_BYTES`.

        allowance = min(peak + max(peak, FOREIGN_PROCESS_FLOOR_BYTES), 3 GiB)

    WHY HEADROOM AT ALL. The sample is five seconds of a desktop as it was
    then. A desktop grows when somebody opens a browser or plays a video, and
    the reserve has to cover the desktop somebody is USING, not the one that
    sat idle while `crucible init` ran — a job sized to the idle figure would
    be refused, or would squeeze Kylie's browser, the first time she watched
    something.

    WHY `FOREIGN_PROCESS_FLOOR_BYTES` (1 GiB) AS THE FLOOR. That constant is
    already Crucible's ruled boundary between "the desktop" and "somebody's job"
    (PHASE2-LLM.md section 4: *"a compositor or a video decoder, not somebody's
    job"*). A browser playing a video is exactly one such process, so the
    reserve leaves room for one more of them than was open when it was sampled
    — the same line the guard draws, not a second number for the same idea.

    WHY DOUBLE THE PEAK when it is larger than that floor. What a new window
    costs scales with what the desktop is already drawing: its framebuffers
    are per monitor and per pixel, and a desktop sampled at 1.5 GiB is driving
    more or bigger screens than one at 0.3 GiB, so its next browser costs more
    too. Doubling lets the desktop grow by its own size again.

    WHY THE CEILING. 3 GiB is what owens-pc — a streaming, multi-monitor
    desktop on a 24 GB card — has lived with since phase 2; no desktop Crucible
    has met needs more. It also bounds what a sample can get wrong: a busy
    moment, or (under WSL2, where the guest cannot see whose memory it is) a
    Windows-side job, can at worst give a card the reserve it had before this
    rule existed.

    A GTX 1660 SUPER whose single low-resolution monitor holds 0.3 GiB gets
    0.3 + 1.0 = 1.3 GiB, and a job gets 4.7 GiB of its 6 GiB instead of 3.0.
    """
    headroom = max(peak_bytes, accelerator.FOREIGN_PROCESS_FLOOR_BYTES)
    return min(peak_bytes + headroom, DEFAULT_DESKTOP_ALLOWANCE_BYTES)


@dataclass(frozen=True)
class DesktopReserve:
    """A measured reserve: the allowance, and the note the config keeps with it."""

    allowance_bytes: int
    sample: DesktopSample

    @property
    def note(self) -> str:
        gib = 1024 ** 3
        return (
            f"desktop held {self.sample.least_bytes / gib:.2f}-"
            f"{self.sample.peak_bytes / gib:.2f} GiB of "
            f"{self.sample.total_bytes / gib:.1f} GiB over {self.sample.samples}s on "
            f"{self.sample.on}; kept peak + max(peak, "
            f"{accelerator.FOREIGN_PROCESS_FLOOR_BYTES / gib:.0f} GiB), at most "
            f"{DEFAULT_DESKTOP_ALLOWANCE_BYTES / gib:.0f} GiB"
        )


def _server_blocker(url: str, token: str | None) -> str | None:
    """What a Crucible server at `url` holds that rules a sample out, or None.

    With a token, `GET /v1/accelerator` is asked and its `resident` and
    Crucible-owned holders are the answer. Without one (a fresh init, which
    has no token for a server another config started), `GET /v1/ping`
    answering at all is the answer: something of Crucible's is running here
    and cannot be asked what it holds. A refused connection is the one "no
    server"; a slow or odd answer is not read as nothing.
    """
    from . import local

    path = "/v1/accelerator" if token is not None else "/v1/ping"
    try:
        state = local.request(f"{url}{path}", token=token, timeout=10)
    except urllib.error.HTTPError as exc:
        return (
            f"a Crucible server at {url} answered HTTP {exc.code} when asked "
            "what it holds, so it cannot be ruled out"
        )
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ConnectionRefusedError):
            return None  # nothing listening: nothing loaded by it
        return (
            f"a Crucible server at {url} did not answer what it holds "
            f"({reason}), so it cannot be ruled out"
        )
    except (local.LocalError, ValueError) as exc:
        return f"could not ask the Crucible server at {url} what it holds: {exc}"
    if token is None:
        return (
            f"a Crucible server is already answering at {url}, and this init has "
            "no token to ask it what it holds; stop it, or measure later with "
            "`crucible capability --measure-desktop`"
        )
    resident = state.get("resident")
    if resident:
        return (
            f"this machine's Crucible server has {resident.get('kind')} "
            f"{resident.get('id')} loaded; unload it (or stop the server) and "
            "measure again"
        )
    owned = [h for h in state.get("holders") or [] if h.get("owned_by_crucible")]
    if owned:
        return "this machine's Crucible server has processes on the card: " + ", ".join(
            str(h.get("name")) for h in owned
        )
    return None


def desktop_blocker(
    config: Any | None, backend: Backend, *, port: int | None = None
) -> str | None:
    """Why the desktop cannot be measured right now, or None when it can.

    A sample is only the desktop when nothing else is on the card, so this
    refuses — by name, never silently — when:

    * the Crucible server this config describes (or, with no config, whatever
      answers on `port` of this machine) is up and holds something, or cannot
      be asked (`_server_blocker`);
    * the driver names a `llama-server` (an engine of ours, maybe a crashed
      run's orphan: accelerator.py finds those by image), or any compute app
      holding `FOREIGN_PROCESS_FLOOR_BYTES` or more — somebody's job, which a
      sample would count as desktop.

    Under WSL2 the compute-app list is empty by the driver shim's design, so
    the second check is blind there; the server check and
    `desktop_allowance_from`'s ceiling are what remain.
    """
    if config is not None:
        # The server THIS config describes, on loopback, with its own token —
        # read off the config rather than `local.connection`, which on win32
        # answers for the host's fixed door and not for the config in hand.
        host = config.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        blocked = _server_blocker(f"http://{host}:{config.port}", config.token)
        if blocked is not None:
            return blocked
    elif port is not None:
        blocked = _server_blocker(f"http://127.0.0.1:{port}", None)
        if blocked is not None:
            return blocked
    if os.path.exists(WSL_NVIDIA_SMI):
        return None
    try:
        apps = accelerator.probe_compute_apps()
    except accelerator.ProbeError as exc:
        return f"could not list what is on the card: {exc}"
    for app in apps:
        if accelerator.is_llama_server(app.name):
            return f"a llama-server (pid {app.pid}) is on the card"
        if app.used_bytes is not None and app.used_bytes >= accelerator.FOREIGN_PROCESS_FLOOR_BYTES:
            return (
                f"{app.name} (pid {app.pid}) holds "
                f"{app.used_bytes / 1024 ** 3:.1f} GiB; that is somebody's job, "
                "not the desktop"
            )
    return None


def measure_desktop_reserve(
    config: Any | None, backend: Backend, *, port: int | None = None
) -> tuple[DesktopReserve | None, str]:
    """Measure this card's desktop and size its reserve: (reserve, "") or
    (None, why not).

    NVIDIA cards only — `cuda-linux`, and `llama-windows` on a card nvidia-smi
    answers for (its config is what the Windows-to-WSL move carries into the
    guest). A Mac's reserve is a share of unified memory, not a desktop on a
    card, and is untouched (`config.default_desktop_allowance_bytes`).
    """
    if backend.kind == MLX_DARWIN or backend.gpu.vendor != "nvidia":
        return None, f"{backend.kind} on {backend.gpu.name} has no card desktop to sample"
    if nvidia_smi_path() is None:
        return None, "there is no nvidia-smi to sample the card with"
    blocked = desktop_blocker(config, backend, port=port)
    if blocked is not None:
        return None, blocked
    try:
        sample = sample_desktop()
    except (accelerator.ProbeError, ValueError) as exc:
        return None, f"nvidia-smi: {exc}"
    return DesktopReserve(desktop_allowance_from(sample.peak_bytes), sample), ""


def rung_card(home: Path, backend: Backend, desktop_allowance_bytes: int) -> RungResult:
    """Rung 0: the card, the desktop on it, and the disk. nvidia-smi only.

    "how much space is available" is both halves, so both are here: the
    card's total and free memory, what the desktop holds of it (the least
    `memory.used` over `DESKTOP_SAMPLES` seconds with nothing of ours loaded),
    and the free disk under Crucible's home, where every weight goes.
    """
    started = _now()
    facts: dict[str, Any] = {}
    try:
        facts["disk_free_bytes"] = shutil.disk_usage(home if home.exists() else home.parent).free
    except OSError as exc:
        facts["disk_free_bytes"] = None
        facts["disk_error"] = str(exc)
    card = declared_card(backend.gpu)
    facts["declared"] = card.features()
    if backend.kind == MLX_DARWIN or nvidia_smi_path() is None:
        facts["pool_total_bytes"] = backend.gpu.vram_bytes
        return RungResult(
            CARD,
            PASSED,
            started,
            facts,
            detail=(
                f"{backend.gpu.name}: {backend.gpu.vram_bytes / 1024 ** 3:.1f} GiB "
                "pool; no NVIDIA card, so no compute capability or desktop sample"
            ),
        )
    try:
        name, driver, uuid, total, free, cap = _query(
            "name,driver_version,uuid,memory.total,memory.free,compute_cap"
        )
        # The same sampler `measure_desktop_reserve` sizes the reserve from.
        desktop = sample_desktop()
        used = [desktop.least_bytes, desktop.peak_bytes]
    except (accelerator.ProbeError, ValueError) as exc:
        return RungResult(CARD, FAILED, started, facts, detail=f"nvidia-smi: {exc}")
    facts.update(
        {
            "name": name,
            "driver": driver,
            "uuid": uuid,
            "compute_capability": cap,
            "total_bytes": int(total) * 1024 * 1024,
            "free_bytes": int(free) * 1024 * 1024,
            "desktop_bytes": min(used),
            "desktop_bytes_max": max(used),
            "desktop_allowance_bytes": desktop_allowance_bytes,
        }
    )
    over = min(used) - desktop_allowance_bytes
    detail = (
        f"{name}, driver {driver}, compute capability {cap}: "
        f"{int(total) / 1024:.1f} GiB, {int(free) / 1024:.1f} GiB free; the "
        f"desktop held {min(used) / 1024 ** 3:.1f}-{max(used) / 1024 ** 3:.1f} GiB"
    )
    if over > 0:
        detail += (
            f", {over / 1024 ** 3:.1f} GiB more than the "
            f"{desktop_allowance_bytes / 1024 ** 3:.1f} GiB desktop allowance"
        )
    return RungResult(CARD, PASSED, started, facts, detail=detail)


# ------------------------------------------------------------------ GPU rungs


#: Rung 1's script, run with each env's own python. Prints one JSON line.
ENV_SMOKE = f"""
import json, time
out = {{"ok": False}}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False in this env")
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    out["device"] = props.name
    cap = "sm_%d%d" % (props.major, props.minor)
    out["capability"] = "%d.%d" % (props.major, props.minor)
    out["arch_list"] = torch.cuda.get_arch_list()
    out["arch_supported"] = cap in out["arch_list"] or any(
        a.startswith("compute_") and int(a.split("_")[1]) <= props.major * 10 + props.minor
        for a in out["arch_list"]
    )
    out["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    n = {SMOKE_MATMUL_N}
    timings = {{}}
    for name in ("float32", "float16", "bfloat16"):
        dt = getattr(torch, name)
        try:
            a = torch.randn(n, n, device=dev, dtype=dt)
            b = torch.randn(n, n, device=dev, dtype=dt)
            a @ b
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range({SMOKE_MATMUL_REPEATS}):
                a @ b
            torch.cuda.synchronize()
            each = (time.perf_counter() - t) / {SMOKE_MATMUL_REPEATS}
            timings[name] = {{"seconds": each, "tflops": 2 * n ** 3 / each / 1e12}}
        except Exception as exc:
            timings[name] = {{"error": "%s: %s" % (type(exc).__name__, exc)}}
    out["matmul"] = timings
    out["ok"] = True
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print({RESULT_PREFIX!r} + json.dumps(out), flush=True)
"""

#: The CUDA-graph rung's script, run with the llm env's python: capture one
#: matmul, replay it on new input, compare with eager.
GRAPHS_SMOKE = f"""
import json
out = {{"ok": False}}
try:
    import torch
    dev = torch.device("cuda:0")
    x = torch.randn(512, 512, device=dev)
    w = torch.randn(512, 512, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            y = x @ w
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x @ w
    x.copy_(torch.randn(512, 512, device=dev))
    graph.replay()
    torch.cuda.synchronize()
    out["ok"] = bool(torch.allclose(y, x @ w, rtol=1e-3, atol=1e-3))
    if not out["ok"]:
        out["error"] = "the replayed graph did not match the eager result"
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
print({RESULT_PREFIX!r} + json.dumps(out), flush=True)
"""


def _preflight(
    backend: Backend, desktop_allowance_bytes: int, need_bytes: int
) -> tuple[str, str] | None:
    """The guard's own answer before a GPU rung: `(outcome, why)`, or None to go.

    Busy or short of room is `waiting` — the card is somebody else's right
    now, and that says nothing about what it can do.
    """
    if backend.kind != CUDA_LINUX:
        return (
            SKIPPED,
            f"the GPU rungs measure the Linux engines' envs, and this host is "
            f"{backend.kind}",
        )
    try:
        accelerator.guard(
            backend.kind,
            model_id=LADDER_SUBJECT,
            need_bytes=need_bytes,
            desktop_allowance_bytes=desktop_allowance_bytes,
        )
    except ApiError as exc:
        return WAITING, f"{exc.code}: {exc.message}"
    return None


def _run_script(python: Path, script: str) -> tuple[dict[str, Any] | None, str, Watch]:
    """One smoke script under a watch: `(result or None, stderr tail, watch)`."""
    process = subprocess.Popen(
        [str(python), "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with Watch(lambda: frozenset({process.pid})) as watch:
        try:
            stdout, stderr = process.communicate(timeout=SMOKE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return (
                None,
                f"did not finish within {SMOKE_TIMEOUT_SECONDS:.0f} s",
                watch,
            )
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX) :]), "", watch
            except ValueError:
                break
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    return None, f"exited {process.returncode}: {tail}", watch


def installed_env_pythons(home: Path, backend_kind: str) -> dict[str, Path]:
    """Every env this host has built that runs torch on the card, by name.

    Empty off `cuda-linux`: the GPU rungs measure the Linux engines' envs, and
    no other backend has an llm env of that kind to ask for."""
    from . import jobenv

    found: dict[str, Path] = {}
    if backend_kind != CUDA_LINUX:
        return found
    llm = jobenv.env_python(home, jobenv.llm_env(backend_kind))
    if llm.is_file():
        found["llm"] = llm
    for job_type in jobenv.WORKER_JOB_TYPES:
        python = jobenv.env_python(home, jobenv.worker_env(job_type, backend_kind))
        if python.is_file():
            found[job_type] = python
    return found


def rung_env(home: Path, backend: Backend, desktop_allowance_bytes: int) -> RungResult:
    """Rung 1: torch initialises on this card in each installed env, and how
    fast each dtype's matmul is. The bf16 figure below compute capability 8.0
    is the emulation penalty the aligner pays; it is reported, not refused."""
    started = _now()
    waiting = _preflight(backend, desktop_allowance_bytes, SMOKE_NEED_BYTES)
    if waiting is not None:
        return RungResult(ENV, waiting[0], started, detail=waiting[1])
    envs = installed_env_pythons(home, backend.kind)
    if not envs:
        return RungResult(ENV, SKIPPED, started, detail="no env is installed yet")
    facts: dict[str, Any] = {}
    failures: list[str] = []
    contention: dict[str, Any] = {}
    for name, python in envs.items():
        result, why, watch = _run_script(python, ENV_SMOKE)
        contention[name] = watch.summary()
        if watch.foreign:
            return RungResult(
                ENV,
                INTERRUPTED,
                started,
                facts,
                detail=f"another process came onto the card: {sorted(watch.foreign.values())}",
                contention=contention,
            )
        if result is None:
            facts[name] = {"ok": False, "error": why}
            failures.append(f"{name}: {why}")
            continue
        facts[name] = result
        if not result.get("ok"):
            failures.append(f"{name}: {result.get('error', 'no error given')}")
    outcome = FAILED if failures else PASSED
    detail = "; ".join(failures) if failures else f"torch runs on this card in {sorted(envs)}"
    return RungResult(ENV, outcome, started, facts, detail, contention)


def rung_graphs(home: Path, backend: Backend, desktop_allowance_bytes: int) -> RungResult:
    """Rung 2: a CUDA graph captures and replays in the llm env."""
    started = _now()
    waiting = _preflight(backend, desktop_allowance_bytes, SMOKE_NEED_BYTES)
    if waiting is not None:
        return RungResult(GRAPHS, waiting[0], started, detail=waiting[1])
    python = installed_env_pythons(home, backend.kind).get("llm")
    if python is None:
        return RungResult(
            GRAPHS, SKIPPED, started, detail="the llm env is not installed; vLLM is what captures graphs"
        )
    result, why, watch = _run_script(python, GRAPHS_SMOKE)
    if watch.foreign:
        return RungResult(
            GRAPHS,
            INTERRUPTED,
            started,
            detail=f"another process came onto the card: {sorted(watch.foreign.values())}",
            contention=watch.summary(),
        )
    if result is None:
        return RungResult(GRAPHS, FAILED, started, {"ok": False}, why, watch.summary())
    if result.get("ok"):
        return RungResult(
            GRAPHS, PASSED, started, result, "a CUDA graph captured and replayed", watch.summary()
        )
    return RungResult(
        GRAPHS, FAILED, started, result, str(result.get("error", "")), watch.summary()
    )


def _smallest_vllm_model(config: Any) -> tuple[Any, Any, Any] | None:
    """The smallest model with a vLLM block whose weights are on this server:
    `(manifest, spec, installed)`, or None. The rung asks whether vLLM starts
    at all, so the cheapest thing that answers it is the right thing to load."""
    from . import weights
    from .manifests import load_all_manifests
    from .precision import below_floor, weight_bits

    choices = []
    for manifest in load_all_manifests().values():
        spec = manifest.backends.get(CUDA_LINUX)
        if spec is None or spec.engine != "vllm" or manifest.weights_of is not None:
            continue
        if below_floor(weight_bits(spec)):
            continue
        found = weights.installed(config, manifest, spec)
        if found is None:
            continue
        choices.append((spec.memory_bytes_estimate, manifest.id, manifest, spec, found))
    if not choices:
        return None
    choices.sort(key=lambda row: (row[0], row[1]))
    _size, _id, manifest, spec, found = choices[0]
    return manifest, spec, found


def rung_vllm(
    config: Any, backend: Backend, card: CardFacts
) -> RungResult:
    """Rung 3: vLLM starts the smallest installed model here and answers once.

    Started exactly as a `load-model` would start it — the same argv composer
    (`Residency._engine_args`), the same KV plan sized against the card this
    second (`vram.plan_vllm_memory`, which states the pool in bytes and so
    removes vLLM's profiling window, MEASUREMENTS 2026-09-18 Finding 4), and
    this card's own args (`engines.vllm.card_args`: fp16 without bf16,
    `--enforce-eager` without CUDA graphs) — at the smallest context a load
    may ask for. A plan that does not fit is `waiting`, never a failure: room
    is the card's state, not what it can do.
    """
    from . import vram
    from .capability import MIN_LOAD_CONTEXT
    from .engines import EngineError, build_engine, engine_model_name, find_free_port, logs_dir
    from .engines.vllm import card_args
    from .residency import DEFAULT_READY_TIMEOUT_SECONDS, Residency

    started = _now()
    python = installed_env_pythons(config.home, backend.kind).get("llm")
    if backend.kind != CUDA_LINUX:
        return RungResult(VLLM, SKIPPED, started, detail=f"vLLM is not this backend's ({backend.kind})")
    if python is None:
        return RungResult(VLLM, SKIPPED, started, detail="the llm env is not installed")
    chosen = _smallest_vllm_model(config)
    if chosen is None:
        return RungResult(
            VLLM,
            SKIPPED,
            started,
            detail="no vLLM model's weights are on this server yet; this rung runs once one is pulled",
        )
    manifest, spec, installed = chosen
    waiting = _preflight(backend, config.desktop_allowance_bytes, spec.memory_bytes_estimate)
    if waiting is not None:
        return RungResult(VLLM, waiting[0], started, detail=waiting[1])
    try:
        state = accelerator.read_state(backend.kind, config.desktop_allowance_bytes)
    except accelerator.ProbeError as exc:
        return RungResult(VLLM, WAITING, started, detail=f"accelerator_unreadable: {exc}")
    plan = vram.plan_vllm_memory(
        model_id=manifest.id,
        spec=spec,
        context=MIN_LOAD_CONTEXT,
        card=state,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        reclaimable_bytes=0,
    )
    if plan is not None and not plan.fits:
        return RungResult(VLLM, WAITING, started, detail=plan.sentence())
    extra = card_args(spec, card)
    args = Residency._engine_args(
        manifest, spec, installed.path, plan, context=MIN_LOAD_CONTEXT, card_args=extra
    )
    engine = build_engine(spec.engine, python, logs_dir(config.home) / "ladder-vllm.log")
    served = engine_model_name(spec.engine, installed.path, manifest.id)
    facts: dict[str, Any] = {"model": manifest.id, "card_args": list(extra), "context": MIN_LOAD_CONTEXT}
    began = time.monotonic()
    with Watch(lambda: engine.pids) as watch:
        try:
            engine.start(installed.path, served, find_free_port(), args)
            engine.ready(DEFAULT_READY_TIMEOUT_SECONDS)
            facts["load_seconds"] = round(time.monotonic() - began, 1)
            body = json.dumps(
                {"model": served, "prompt": "Hello", "max_tokens": 1}
            ).encode()
            request = urllib.request.Request(
                engine.base_url + "/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            outcome, detail = PASSED, (
                f"vLLM started {manifest.id} in {facts['load_seconds']} s and answered"
            )
        except (EngineError, urllib.error.URLError, OSError) as exc:
            first = (str(exc).strip().splitlines() or ["no message"])[0]
            outcome, detail = FAILED, f"{manifest.id}: {first}"
        finally:
            try:
                engine.stop()
            except EngineError as exc:
                detail = f"{detail}; and it would not stop: {exc}"
    if watch.foreign:
        return RungResult(
            VLLM,
            INTERRUPTED,
            started,
            facts,
            f"another process came onto the card: {sorted(watch.foreign.values())}",
            watch.summary(),
        )
    return RungResult(VLLM, outcome, started, facts, detail, watch.summary())


# ------------------------------------------------------------------ the run


def run(
    config: Any,
    backend: Backend,
    rungs: tuple[str, ...] = RUNGS,
    *,
    gpu: bool = True,
    on_line: Callable[[str], None] | None = None,
) -> dict[str, RungResult]:
    """Run `rungs` in order, write the record after each, return every result.

    `gpu=False` runs the nvidia-smi rung alone and leaves the others as they
    were recorded: the door for "tell me what you know without touching the
    card". A rung that builds on one that did not pass is skipped, saying
    which.
    """
    unknown = [name for name in rungs if name not in RUNGS]
    if unknown:
        raise LadderError(f"{unknown} are not rungs; the rungs are {list(RUNGS)}")
    say = on_line or (lambda line: None)
    home = config.home
    previous = None if stale_reason(home, backend.gpu) else load_record(home)
    results = _results_from(previous)
    for name in RUNGS:
        if name not in rungs:
            continue
        if name in GPU_RUNGS and not gpu:
            continue
        if name in (GRAPHS, VLLM):
            env = results.get(ENV)
            if env is not None and env.outcome == FAILED:
                results[name] = RungResult(
                    name, SKIPPED, _now(), detail=f"the env rung failed: {env.detail}"
                )
                _write_record(home, backend.gpu, results)
                say(f"{name}: skipped — the env rung failed")
                continue
        say(f"{name}: measuring")
        if name == CARD:
            result = rung_card(home, backend, config.desktop_allowance_bytes)
        elif name == ENV:
            result = rung_env(home, backend, config.desktop_allowance_bytes)
        elif name == GRAPHS:
            result = rung_graphs(home, backend, config.desktop_allowance_bytes)
        else:
            # Read back through `card_for`, so the CUDA-graph answer this run
            # just wrote decides `--enforce-eager` exactly as a load would.
            result = rung_vllm(config, backend, card_for(home, backend.gpu))
        results[name] = result
        _write_record(home, backend.gpu, results)
        say(f"{name}: {result.outcome} — {result.detail}")
    return results


__all__ = [
    "CARD",
    "ENV",
    "GPU_RUNGS",
    "GRAPHS",
    "RUNGS",
    "VLLM",
    "DesktopReserve",
    "DesktopSample",
    "desktop_allowance_from",
    "desktop_blocker",
    "measure_desktop_reserve",
    "sample_desktop",
    "LadderError",
    "RungResult",
    "Watch",
    "card_for",
    "card_key",
    "load_record",
    "record_path",
    "run",
    "stale_reason",
    "summary",
]
