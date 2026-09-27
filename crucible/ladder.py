from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import accelerator
from .backend import (
    CUDA_LINUX,
    MLX_DARWIN,
    WSL_NVIDIA_SMI,
    Backend,
    CardFacts,
    Gpu,
    declared_card,
    nvidia_smi_path,
)
from .cardfacts import (
    CARD,
    ENV,
    FAILED,
    GPU_RUNGS,
    GRAPHS,
    INTERRUPTED,
    LADDER_SCHEMA,
    PASSED,
    RUNGS,
    SKIPPED,
    VLLM,
    WAITING,
    RungResult,
    card_for,
    card_key,
    load_record,
    record_path,
    results_from as _results_from,
    stale_reason,
)
from .clock import utcnow_to_the_second as _now
from .config import DEFAULT_DESKTOP_ALLOWANCE_BYTES
from .errors import ApiError, CrucibleError

DESKTOP_SAMPLES = 5

SMOKE_NEED_BYTES = 1024 ** 3

SMOKE_TIMEOUT_SECONDS = 300.0

SMOKE_MATMUL_N = 2048
SMOKE_MATMUL_REPEATS = 10

RESULT_PREFIX = "LADDER "

LADDER_SUBJECT = "the measurement ladder"


class LadderError(CrucibleError):
    ...


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


def summary(home: Path, gpu: Gpu) -> dict[str, Any]:
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


class Watch:
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
                pass
            self._stop.wait(1.0)

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "max_used_bytes": self.max_used_bytes,
            "max_utilization_percent": self.max_utilization,
            "foreign": sorted(self.foreign.values()),
            "compute_apps_visible": not os.path.exists(WSL_NVIDIA_SMI),
        }


def _query(fields: str) -> list[str]:
    lines = accelerator._nvidia_smi(f"--query-gpu={fields}", "the card rung")
    return [part.strip() for part in lines[0].split(",")]


@dataclass(frozen=True)
class DesktopSample:
    least_bytes: int
    peak_bytes: int
    total_bytes: int
    samples: int
    on: str


def sample_desktop() -> DesktopSample:
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
    headroom = max(peak_bytes, accelerator.FOREIGN_PROCESS_FLOOR_BYTES)
    return min(peak_bytes + headroom, DEFAULT_DESKTOP_ALLOWANCE_BYTES)


@dataclass(frozen=True)
class DesktopReserve:
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
            return None
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
    if config is not None:
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
    from . import procgroup

    process = subprocess.Popen(
        [str(python), "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **procgroup.own_group(),
    )
    with Watch(lambda: frozenset({process.pid})) as watch:
        try:
            stdout, stderr = process.communicate(timeout=SMOKE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            late = f"did not finish within {SMOKE_TIMEOUT_SECONDS:.0f} s"
            stop_it = (
                "Crucible does not SIGKILL a process holding CUDA: stop it with "
                f"`kill {process.pid}` (never -9), then run `crucible ladder` again"
            )
            try:
                procgroup.ask_to_stop(process)
                process.communicate(timeout=procgroup.STOP_TIMEOUT_SECONDS)
            except procgroup.ProcessGroupError as exc:
                return None, f"{late}, and {exc}. {stop_it}", watch
            except subprocess.TimeoutExpired:
                return (
                    None,
                    f"{late}, and pid {process.pid} did not exit within "
                    f"{procgroup.STOP_TIMEOUT_SECONDS:.0f} s of SIGTERM. {stop_it}",
                    watch,
                )
            return None, late, watch
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX) :]), "", watch
            except ValueError:
                break
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    return None, f"exited {process.returncode}: {tail}", watch


def installed_env_pythons(home: Path, backend_kind: str) -> dict[str, Path]:
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


def installed_torch_env_pythons(home: Path, backend_kind: str) -> dict[str, Path]:
    from . import jobenv

    found: dict[str, Path] = {}
    for name, python in installed_env_pythons(home, backend_kind).items():
        spec = (
            jobenv.llm_env(backend_kind)
            if name == "llm"
            else jobenv.worker_env(name, backend_kind)
        )
        if "torch" in jobenv.recipe_pins(jobenv.recipe_for(spec)):
            found[name] = python
    return found


def rung_env(home: Path, backend: Backend, desktop_allowance_bytes: int) -> RungResult:
    started = _now()
    waiting = _preflight(backend, desktop_allowance_bytes, SMOKE_NEED_BYTES)
    if waiting is not None:
        return RungResult(ENV, waiting[0], started, detail=waiting[1])
    envs = installed_torch_env_pythons(home, backend.kind)
    if not envs:
        return RungResult(ENV, SKIPPED, started, detail="no env that ships torch is installed yet")
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


def run(
    config: Any,
    backend: Backend,
    rungs: tuple[str, ...] = RUNGS,
    *,
    gpu: bool = True,
    on_line: Callable[[str], None] | None = None,
) -> dict[str, RungResult]:
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
