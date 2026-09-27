from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import procgroup
from ..accelerator import proc_entries
from ..errors import JobCancelled
from ..narratorengines import HIGGS_V3, VoicesDocumentView
from .base import LOG_TAIL_LINES, EngineError, SubprocessEngine, find_free_port

MODULE = "narrator.serve"

ENGINE_VARIABLE = "NARRATOR_ENGINE"

STACK_VARIABLE = "HIGGS_STACK"
MAX_NUM_SEQS_VARIABLE = "HIGGS_MAX_NUM_SEQS"

MEM_FRACTION_VARIABLE = "HIGGS_SGL_MEM_FRACTION"

CONTEXT_LENGTH_VARIABLE = "HIGGS_CONTEXT_LENGTH"

STACK_ENV_PREFIX_VARIABLE: dict[str, str] = {
    "sglang-omni": "HIGGS_SGL_ENV",
}
STACK_LAUNCH_BINARY: dict[str, str] = {
    "sglang-omni": "sgl-omni",
}
STACK_PORT_VARIABLE: dict[str, str] = {
    "sglang-omni": "HIGGS_SGL_PORT",
}


OWNER_MARKER_VARIABLE = "NARRATOR_HIGGS3_OWNER"
LAUNCHED_SERVER_GRACE_SECONDS = 180.0
LAUNCHED_SERVER_POLL_SECONDS = 1.0


def processes_launched_by(owner_pid: int) -> frozenset[int]:
    marker = f"{OWNER_MARKER_VARIABLE}={owner_pid}".encode()
    found = set()
    for pid, entry in proc_entries():
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker in environ.split(b"\0"):
            found.add(pid)
    return frozenset(found)


def _wait_until_gone(owner_pid: int, seconds: float) -> frozenset[int]:
    deadline = time.monotonic() + seconds
    while True:
        left = processes_launched_by(owner_pid)
        if not left or time.monotonic() >= deadline:
            return left
        time.sleep(LAUNCHED_SERVER_POLL_SECONDS)


def env_prefix_variable_for(serving_stack: str) -> str:
    variable = STACK_ENV_PREFIX_VARIABLE.get(serving_stack)
    if variable is None:
        raise EngineError(
            f"no env-prefix variable for serving stack {serving_stack!r}; this "
            f"build knows {sorted(STACK_ENV_PREFIX_VARIABLE)}. Each launcher "
            "reads only its own name for the prefix it runs out of, and the "
            "SGLang one DEFAULTS to a conda env rather than refusing — so a "
            "guess here is a server started out of a directory nobody named"
        )
    return variable


MLX_BATCH_VARIABLE = "NARRATOR_HIGGS3_MLX_BATCH"
MLX_MEM_BUDGET_VARIABLE = "NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB"
MLX_CACHE_LIMIT_VARIABLE = "HIGGS_MLX_CACHE_LIMIT_GB"

HIGGS_V3_MLX_WEIGHTS_GB = 8.5


@dataclass(frozen=True)
class MlxTier:
    name: str
    min_total_mib: int
    width: int
    mem_budget_gb: float
    cache_limit_gb: float


MLX_TIERS: dict[str, tuple[MlxTier, ...]] = {
    HIGGS_V3: (
        MlxTier("extreme", 60_000, 64, 42.0, 8.0),
        MlxTier("fast", 44_000, 72, 34.0, 8.0),
        MlxTier("moderate", 28_000, 48, 22.0, 6.0),
        MlxTier("light", 0, 24, 13.0, 3.0),
    ),
}


def mlx_render_profile(narrator_engine: str, total_bytes: int) -> MlxTier:
    rows = MLX_TIERS.get(narrator_engine)
    if rows is None:
        raise EngineError(
            f"no measured MLX render tiers for narrator engine "
            f"{narrator_engine!r}; this build knows {sorted(MLX_TIERS)}. "
            f"{MLX_BATCH_VARIABLE} is the width narrator's in-process backend "
            "batches at and it defaults to 1 — one chunk at a time — so "
            "leaving it unset is a measured 7x, not a safe fallback"
        )
    if not isinstance(total_bytes, int) or total_bytes <= 0:
        raise EngineError(
            f"cannot choose an MLX render tier for {narrator_engine!r} from "
            f"total_bytes={total_bytes!r}: the tier is a BAND of this "
            f"machine's own memory ({MLX_BATCH_VARIABLE} and "
            f"{MLX_MEM_BUDGET_VARIABLE} come out of the same row), and "
            "`accelerator.probe_unified_memory` is what reads it. A machine "
            "whose memory could not be read gets no row — the 64 GB Mac's is "
            "not a safe stand-in for a 16 GB one"
        )
    total_mib = round(total_bytes / (1024 * 1024))
    for row in rows:
        if total_mib >= row.min_total_mib:
            break
    else:
        raise EngineError(
            f"the {narrator_engine!r} MLX tier table has no row for "
            f"{total_mib} MiB; its last row must have a floor of 0"
        )
    headroom = row.mem_budget_gb - HIGGS_V3_MLX_WEIGHTS_GB - row.cache_limit_gb
    if headroom <= 0:
        raise EngineError(
            f"the {narrator_engine!r} MLX tier {row.name!r} budgets "
            f"{row.mem_budget_gb:g} GB, which cannot hold "
            f"{HIGGS_V3_MLX_WEIGHTS_GB:g} GB of weights plus a "
            f"{row.cache_limit_gb:g} GB pinned buffer cache "
            f"({MLX_CACHE_LIMIT_VARIABLE}) — narrator refuses that sum at load "
            "and there is no KV left to batch with"
        )
    return row

QUIT_GRACE_SECONDS = 210.0

POLL_SECONDS = 0.5

READER_JOIN_SECONDS = POLL_SECONDS * 4

CANCEL_GRACE_SECONDS = 120.0

LOAD_SILENCE_TIMEOUT_SECONDS = 900.0


def higgs_env_prefix(python: Path, serving_stack: str) -> Path:
    variable = env_prefix_variable_for(serving_stack)
    binary = STACK_LAUNCH_BINARY[serving_stack]
    root = Path(python).parent.parent
    if (root / "bin" / binary).exists():
        return root
    if (root / "pyvenv.cfg").is_file():
        return root
    if (root / "conda-meta").is_dir():
        return root
    raise EngineError(
        f"{variable} is the prefix narrator's server runs out of, "
        f"and {root} — the prefix of the tts env python {python} — is not "
        f"one: it carries no bin/{binary}, no pyvenv.cfg (a venv) "
        "and no conda-meta/ (a conda env). narrator's launch script builds "
        f"CUDA_HOME, PATH, LD_LIBRARY_PATH and the {binary} binary from that "
        "prefix and refuses when it is unset"
    )


class EngineWouldNotStop(EngineError):
    ...


@dataclass(frozen=True)
class _Garbled:
    line: str


class _Ended:
    ...


_ENDED = _Ended()


class NarratorEngine(SubprocessEngine):
    env_job_type = "tts"

    pull_command = "crucible voices pull"

    binds_a_port = False

    def __init__(
        self,
        narrator_engine: str,
        python: Path,
        log_path: Path,
        *,
        serving_stack: str | None,
        max_num_seqs: int | None,
        mem_fraction: float | None,
        context_length: int | None,
        voices: VoicesDocumentView | None,
        mlx_total_bytes: int | None,
    ) -> None:
        super().__init__(python=python, log_path=log_path)
        self._narrator_engine = narrator_engine
        self._serving_stack = serving_stack
        self._max_num_seqs = max_num_seqs
        self._mem_fraction = mem_fraction
        self._context_length = context_length
        if narrator_engine == HIGGS_V3:
            if voices is None:
                raise EngineError(
                    f"cannot start {self.name} without a voices document: "
                    "narrator resolves a Higgs v3 voice by name in the "
                    "NARRATOR_HIGGS_VOICES document and refuses a modelDir on the "
                    "load message, on the served arm and the MLX arm alike. "
                    "crucible/narratorvoices.py writes it from the voice "
                    "manifest and the pulled weights at every load"
                )
        elif voices is not None:
            raise EngineError(
                f"{self.name} was given a voices document ({voices.path}), but "
                f"only {HIGGS_V3!r} resolves a voice by name in one; any other "
                "engine takes its weights on the load message"
            )
        self._voices = voices
        if narrator_engine == HIGGS_V3 and serving_stack is not None:
            if max_num_seqs is None:
                raise EngineError(
                    f"cannot start {self.name} on the {serving_stack} stack "
                    f"without {MAX_NUM_SEQS_VARIABLE}: it is stage 0's "
                    "admission width and the width of narrator's own batch, "
                    "and narrator refuses it by name "
                    "(v3_served.serve_concurrency). It comes from the voice "
                    "manifest's [voice.serving].max_num_seqs"
                )
            if max_num_seqs < 1:
                raise EngineError(
                    f"{MAX_NUM_SEQS_VARIABLE}={max_num_seqs} for {self.name} "
                    "must be at least 1"
                )
            try:
                self._env_prefix: Path | None = higgs_env_prefix(
                    python, serving_stack)
            except EngineError as refusal:
                raise EngineError(
                    f"cannot start {self.name}: {refusal}"
                ) from refusal
        elif serving_stack is not None:
            raise EngineError(
                f"{self.name} was given serving_stack={serving_stack!r}, but "
                f"only {HIGGS_V3!r} starts a server underneath narrator and "
                "reads the HIGGS_* variables. Either the env recipe installed "
                "a stack this engine cannot use, or jobenv.tts_env named one "
                "it should not have"
            )
        else:
            self._env_prefix = None
        if narrator_engine == HIGGS_V3 and serving_stack is None:
            if mlx_total_bytes is None:
                raise EngineError(
                    f"cannot start {self.name} on the in-process arm without "
                    "the machine's total memory: the batch width "
                    f"({MLX_BATCH_VARIABLE}) and the budget it is narrowed "
                    f"against ({MLX_MEM_BUDGET_VARIABLE}) come out of ONE "
                    "measured row chosen by that figure, and narrator's own "
                    "defaults for them are 1 — one chunk at a time, a measured "
                    "7x — and 42 GB, which is a 64 GB machine's number. "
                    "`accelerator.probe_unified_memory` reads it and "
                    "`residency.load_voice` passes it"
                )
            self._mlx_tier: MlxTier | None = mlx_render_profile(
                narrator_engine, mlx_total_bytes
            )
        else:
            if mlx_total_bytes is not None:
                raise EngineError(
                    f"{self.name} was given mlx_total_bytes={mlx_total_bytes}, "
                    "but only the in-process arm (a higgs-v3 engine whose env "
                    "starts no serving stack) sizes a batch from this "
                    "machine's memory. A served narrator's memory is its "
                    "launcher's GPU fractions"
                )
            self._mlx_tier = None
        self._writer = threading.Lock()
        self._inbox: queue.Queue[dict[str, Any] | _Garbled | _Ended] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._ready_message: dict[str, Any] | None = None


    @property
    def name(self) -> str:
        return f"narrator ({self._narrator_engine})"

    @property
    def narrator_engine(self) -> str:
        return self._narrator_engine

    @property
    def base_url(self) -> str:
        raise EngineError(
            f"{self.name} has no base url: its wire is newline-delimited JSON "
            "over stdin and stdout, not HTTP. Talk to it through this engine "
            "object (PHASE3-TTS.md section 4)"
        )

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        return [str(self._python), "-m", MODULE]

    def environment(self) -> dict[str, str]:
        environment = {
            ENGINE_VARIABLE: self._narrator_engine,
            "PYTHONUNBUFFERED": "1",
        }
        if self._env_prefix is not None:
            assert self._serving_stack is not None
            assert self._max_num_seqs is not None
            environment[STACK_VARIABLE] = self._serving_stack
            environment[env_prefix_variable_for(self._serving_stack)] = str(
                self._env_prefix
            )
            environment[MAX_NUM_SEQS_VARIABLE] = str(self._max_num_seqs)
            port_variable = STACK_PORT_VARIABLE.get(self._serving_stack)
            if port_variable is not None:
                environment[port_variable] = str(find_free_port())
        if self._mem_fraction is not None:
            environment[MEM_FRACTION_VARIABLE] = f"{self._mem_fraction:g}"
        if self._context_length is not None:
            environment[CONTEXT_LENGTH_VARIABLE] = str(self._context_length)
        if self._mlx_tier is not None:
            environment[MLX_BATCH_VARIABLE] = str(self._mlx_tier.width)
            environment[MLX_MEM_BUDGET_VARIABLE] = f"{self._mlx_tier.mem_budget_gb:g}"
            environment[MLX_CACHE_LIMIT_VARIABLE] = (
                f"{self._mlx_tier.cache_limit_gb:g}"
            )
        if self._voices is not None:
            environment.update(self._voices.environment())
        return environment

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        return {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": log_handle,
            "text": True,
            "encoding": "utf-8",
            "errors": "strict",
            "bufsize": 1,
        }


    def attach(self, process: subprocess.Popen[Any]) -> None:
        self._ready_message = None
        self._reader = threading.Thread(
            target=self._pump,
            args=(process.stdout,),
            name=f"narrator-stdout-{self._narrator_engine}",
            daemon=True,
        )
        self._reader.start()

    def _pump(self, stream: Any) -> None:
        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    self._inbox.put(_Garbled(text))
                    continue
                if not isinstance(message, dict) or not isinstance(
                    message.get("type"), str
                ):
                    self._inbox.put(_Garbled(text))
                    continue
                if message["type"] == "ready" and self._ready_message is None:
                    self._ready_message = message
                    continue
                self._inbox.put(message)
        except (ValueError, OSError):
            pass
        finally:
            self._inbox.put(_ENDED)

    def detach(self) -> None:
        process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        reader = self._reader
        if reader is not None:
            reader.join(timeout=READER_JOIN_SECONDS)
        if (
            process is not None
            and process.stdout is not None
            and (reader is None or not reader.is_alive())
        ):
            try:
                process.stdout.close()
            except OSError:
                pass
        self._reader = None
        self._ready_message = None
        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break


    def announced_ready(self) -> str | None:
        message = self._ready_message
        if message is None:
            return None
        return (
            f"{self.name} is ready on {message.get('device')} "
            f"(backend {message.get('backend')})"
        )

    def announces_item_take(self) -> bool:
        message = self._ready_message
        if message is None:
            return False
        return message.get("itemTake") is True

    def readiness_description(self) -> str:
        return "print a ready line on stdout"


    def send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise EngineError(
                f"{self.name} is not running, so there is nothing to send to it"
            )
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._writer:
            try:
                process.stdin.write(line)
                process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} and could not be sent a "
                        f"{message.get('action')!r}. Last {LOG_TAIL_LINES} lines "
                        f"of {self.log_path}:\n" + self.log_tail()
                    ) from None
                raise EngineError(
                    f"could not write to {self.name}: {exc}. Last "
                    f"{LOG_TAIL_LINES} lines of {self.log_path}:\n" + self.log_tail()
                ) from None

    def converse(
        self,
        message: dict[str, Any],
        *,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        self.send(message)
        return self._until(terminal, silence_timeout, cancelled)

    def _until(
        self,
        terminal: frozenset[str],
        silence_timeout: float,
        cancelled: Callable[[], bool] | None,
    ) -> Iterator[dict[str, Any]]:
        process = self._process
        if process is None:
            raise EngineError(f"{self.name} is not running")
        deadline = time.monotonic() + silence_timeout
        cancel_sent = False
        cancel_deadline = 0.0
        while True:
            if cancelled is not None and cancelled() and not cancel_sent:
                self.send({"action": "cancel"})
                cancel_sent = True
                cancel_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
            if cancel_sent and time.monotonic() >= cancel_deadline:
                raise EngineWouldNotStop(
                    f"{self.name} was sent a cancel {CANCEL_GRACE_SECONDS:.0f}s "
                    "ago and has not finished what it was doing. Its stdin "
                    "reader sets a flag the moment a cancel lands, so an engine "
                    "still working after this long is one whose rendering arm "
                    "does not read that flag — this is the wire's contract "
                    "being broken, not a slow render. Last "
                    f"{LOG_TAIL_LINES} lines of {self.log_path}:\n"
                    + self.log_tail()
                )
            try:
                item = self._inbox.get(timeout=POLL_SECONDS)
            except queue.Empty:
                code = process.poll()
                if code is not None:
                    raise EngineError(
                        f"{self.name} exited {code} in the middle of a request. "
                        f"Last {LOG_TAIL_LINES} lines of {self.log_path}:\n"
                        + self.log_tail()
                    )
                if time.monotonic() >= deadline:
                    raise EngineError(
                        f"{self.name} said nothing at all for "
                        f"{silence_timeout:.0f}s. Last {LOG_TAIL_LINES} lines of "
                        f"{self.log_path}:\n" + self.log_tail()
                    )
                continue

            deadline = time.monotonic() + silence_timeout

            if isinstance(item, _Ended):
                raise EngineError(
                    f"{self.name} closed its stdout in the middle of a request "
                    f"(exit {process.poll()}). Last {LOG_TAIL_LINES} lines of "
                    f"{self.log_path}:\n" + self.log_tail()
                )
            if isinstance(item, _Garbled):
                raise EngineError(
                    f"{self.name} wrote a line to stdout that is not a protocol "
                    f"message: {item.line[:300]!r}. stdout carries the wire and "
                    "nothing else; narrator's own diagnostics belong on stderr, "
                    f"which is {self.log_path}"
                )
            if item["type"] == "error":
                raise EngineError(
                    f"{self.name} refused the request: "
                    f"{item.get('message', '(no message)')}"
                )
            yield item
            if item["type"] in terminal:
                if cancel_sent:
                    raise JobCancelled(f"{self.name} was cancelled mid-request")
                return


    def load(
        self,
        *,
        voice: str,
        weights_dir: Path,
        warm: bool,
        on_progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "action": "load",
            "voice": voice,
            "warm": warm,
        }
        if self._voices is None:
            request["modelDir"] = str(weights_dir)
        else:
            named = self._voices.weights_for(voice)
            if named != weights_dir:
                raise EngineError(
                    f"{self.name} was asked to load {voice!r} from {weights_dir}, "
                    f"but {self._voices.path} names {named} for it. The document "
                    "is what narrator reads; two directories for one voice is a "
                    "load nobody can vouch for"
                )
        loaded: dict[str, Any] | None = None
        for message in self.converse(
            request,
            terminal=frozenset({"loaded"}),
            silence_timeout=LOAD_SILENCE_TIMEOUT_SECONDS,
        ):
            if message["type"] == "loaded":
                loaded = message
            elif on_progress is not None:
                on_progress(f"{self.name}: {message['type']} {message}")
        if loaded is None:
            raise EngineError(f"{self.name} ended its load without a loaded line")
        return loaded


    @property
    def pids(self) -> frozenset[int]:
        process = self._process
        if process is None:
            return frozenset()
        return super().pids | processes_launched_by(process.pid)

    @property
    def stop_budget_seconds(self) -> float:
        return (
            QUIT_GRACE_SECONDS
            + READER_JOIN_SECONDS
            + super().stop_budget_seconds
            + 2 * (LAUNCHED_SERVER_GRACE_SECONDS + LAUNCHED_SERVER_POLL_SECONDS)
        )

    def stop(self) -> None:
        process = self._process
        owner = None if process is None else process.pid
        self._quit_narrator()
        if owner is not None:
            self._outlive_launched_servers(owner)

    def _outlive_launched_servers(self, owner: int) -> None:
        left = _wait_until_gone(owner, LAUNCHED_SERVER_GRACE_SECONDS)
        if not left:
            return
        unsignalled = procgroup.ask_groups_to_stop(left)
        left = _wait_until_gone(owner, LAUNCHED_SERVER_GRACE_SECONDS)
        if left:
            pids = " ".join(str(pid) for pid in sorted(left))
            refused = (
                f" Crucible could not signal pids {sorted(unsignalled)}."
                if unsignalled
                else ""
            )
            raise EngineError(
                f"{self.name}: the server narrator launched is still running "
                f"(pids {sorted(left)}) {2 * LAUNCHED_SERVER_GRACE_SECONDS:.0f}s "
                f"after narrator stopped, and did not exit on SIGTERM.{refused} "
                "Crucible does not SIGKILL a process holding CUDA: stop it with "
                f"`kill {pids}` (never -9) and load again. Its log is "
                f"{self.log_path}"
            )

    def _quit_narrator(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                self.send({"action": "quit"})
            except EngineError:
                pass
            else:
                try:
                    process.wait(timeout=QUIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        super().stop()
