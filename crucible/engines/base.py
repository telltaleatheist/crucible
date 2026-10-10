from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import hosttools, procgroup
from ..enginespec import flag_value
from ..errors import EngineError, JobCancelled
from ..logtail import led_by_first_error, tail_of_last_run

STOP_TIMEOUT_SECONDS = procgroup.STOP_TIMEOUT_SECONDS
READY_POLL_SECONDS = 2.0
LOG_TAIL_LINES = procgroup.LOG_TAIL_LINES

PORT_IN_USE = "port_in_use"

BIND_FAILURE_LINES: tuple[str, ...] = (
    "address already in use",
    "bind: address in use",
)

BIND_SCAN_LINES = 200


def int_flag(args: "list[str] | tuple[str, ...]", flag: str) -> int | None:
    found = flag_value(args, flag)
    if found is None:
        return None
    try:
        return int(found)
    except ValueError as exc:
        raise EngineError(f"{flag} {found!r} is not an integer") from exc


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def port_in_use_error(
    engine_name: str, port: int | None, model_id: str | None, line: str
) -> "EngineError":
    return EngineError(
        f"{PORT_IN_USE}: another server is answering on port {port}: "
        f"127.0.0.1:{port} was free when Crucible chose it and is taken now, so "
        f"{model_id} was not started. Crucible never adopts a server it did not "
        "start, and picks a fresh port on retry: run the load again. "
        f"{engine_name} said: {line}"
    )


def weights_subject_id(model_dir: Path) -> str:
    return Path(model_dir).parent.name


LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"


def library_path(dirs: "tuple[Path, ...]", inherited: str) -> str:
    """`dirs` ahead of whatever the server was started with: an engine binary whose shared
    libraries live in an env (llama-server on cuda-linux, crucible/llamacpp.py) finds
    those first, and the host's own entries (WSL's /usr/lib/wsl/lib) still resolve."""
    ours = [str(entry) for entry in dirs]
    theirs = [entry for entry in inherited.split(os.pathsep) if entry]
    return os.pathsep.join([*ours, *theirs])


def plan_flags(plan: Any) -> list[str]:
    return [] if plan is None else list(plan.flags())


def logs_dir(home: Path) -> Path:
    directory = home / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class SubprocessEngine:
    name = "subprocess"

    chat_concurrency: int | None = None

    chat_concurrency_basis: str | None = None

    chat_concurrency_flag: str | None = None

    decide_logprobs: bool = False

    max_logprobs: int | None = None

    decide_basis: str | None = None

    decide_items_batched: bool = False

    decide_items_basis: str | None = None

    decide_questions_batched: bool = False

    decide_questions_basis: str | None = None

    chat_prefill: bool = False

    chat_prefill_basis: str | None = None

    structured_output_formats: frozenset[str] = frozenset()
    """The `response_format` types beyond `text` the engine enforces."""

    structured_output_fields: frozenset[str] = frozenset()
    """The other body fields (structured.GRAMMAR_FIELDS) the engine enforces."""

    structured_output_basis: str | None = None
    """Where both were read. The chat door refuses `structured_output_not_served` for a
    constraint the engine does not state, so it is never sent to be ignored."""

    json_whitespace_compact: bool = False
    """Whether the engine keeps `"json_whitespace": "compact"` (no whitespace between
    JSON tokens) when the door writes it into the schema (structured.with_compact_json)."""

    json_whitespace_basis: str | None = None
    """Where that was read. Required of an engine that enforces any structured output;
    the chat door refuses `json_whitespace_not_served` with it."""

    decide_likelihood_route: str | None = None
    """How the engine scores a likelihood question's candidates: `items` (the
    batched items route, every candidate a row) or `prompt-logprobs` (one
    request per candidate, the log-probability of every prompt token read back)
    or `forced-tokens` (one request per candidate, the candidate's tokens forced
    as the generated continuation and each one's raw log-probability read back).
    None: it cannot, and `decide_likelihood_basis` says why."""

    decide_likelihood_images: bool = False

    decide_likelihood_basis: str | None = None

    sigterm_wait_seconds: float = STOP_TIMEOUT_SECONDS

    env_job_type = "llm"

    pull_command = "crucible models pull"

    binds_a_port = True

    def __init__(
        self, python: Path, log_path: Path, library_dirs: tuple[Path, ...] = ()
    ) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._library_dirs = tuple(Path(entry) for entry in library_dirs)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._port: int | None = None
        self._served_name: str | None = None
        self._phase_seen: tuple[str, float] | None = None

    @classmethod
    def load_args(
        cls,
        spec: Any,
        weights_dir: Path,
        context: int,
        plan: Any,
        *,
        card_flags: tuple[str, ...] = (),
        source: str = "",
    ) -> list[str]:
        return [*spec.engine_args, *plan_flags(plan)]

    @classmethod
    def served_name(cls, weights_dir: Path, model_id: str) -> str:
        return model_id

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        return {}

    @property
    def stop_budget_seconds(self) -> float:
        return procgroup.stop_budget_seconds(self.sigterm_wait_seconds)

    def missing_executable_hint(self) -> str:
        return (
            f"the {self.env_job_type} env is not installed: run "
            f"`crucible install {self.env_job_type}` and load again"
        )

    def subject_id(self, model_dir: Path, served_name: str) -> str:
        return served_name

    def missing_model_hint(self, model_dir: Path, served_name: str) -> str:
        return (
            f"the weights are not pulled: run "
            f"`{self.pull_command} {self.subject_id(model_dir, served_name)}` "
            "and load again"
        )

    def bind_failure_in_log(self) -> str | None:
        for line in self.log_tail(BIND_SCAN_LINES).splitlines():
            lowered = line.lower()
            if any(needle in lowered for needle in BIND_FAILURE_LINES):
                return line.strip()
        return None

    def refuse_a_taken_port(self) -> None:
        if not self.binds_a_port:
            return
        line = self.bind_failure_in_log()
        if line is not None:
            raise port_in_use_error(self.name, self._port, self._served_name, line)

    def announced_ready(self) -> str | None:
        served = self._probe_models(f"{self.base_url}/v1/models")
        if served is None:
            return None
        if self._served_name is not None and self._served_name not in served:
            raise EngineError(
                f"{self.name} on port {self._port} is serving {served}, not "
                f"{self._served_name!r}: another server is answering on port "
                f"{self._port}. Crucible will not proxy a model it did not ask "
                "for, and picks a fresh port on retry: run the load again"
            )
        return f"{self.name} is serving {self._served_name!r}"

    def readiness_description(self) -> str:
        return f"answer {self.base_url}/v1/models"

    def confirm(
        self,
        deadline: float,
        on_progress: Callable[[str], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        return None

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        return {
            "stdin": subprocess.DEVNULL,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }

    def attach(self, process: subprocess.Popen[Any]) -> None:
        return None

    def detach(self) -> None:
        return None


    @property
    def base_url(self) -> str:
        if self._port is None:
            raise EngineError(f"{self.name} has not been started, so it has no url")
        return f"http://127.0.0.1:{self._port}"

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def pids(self) -> frozenset[int]:
        if self._process is None or self._process.poll() is not None:
            return frozenset()
        return frozenset({self._process.pid})

    @property
    def exit_code(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        if self._process is not None:
            raise EngineError(
                f"{self.name} is already running as pid {self._process.pid}"
            )
        if not self._python.is_file():
            raise EngineError(
                f"{self.name} cannot start: {self._python} does not exist; "
                + self.missing_executable_hint()
            )
        if not model_dir.is_dir():
            raise EngineError(
                f"{self.name} cannot start: no model directory at {model_dir}; "
                + self.missing_model_hint(model_dir, served_name)
            )

        try:
            compiler = hosttools.compiler_environment(self._python.parent.parent)
        except hosttools.HostToolError as exc:
            raise EngineError(f"{self.name} cannot start: {exc}") from None
        command = self.command(model_dir, served_name, port, args)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        existed = self._log_path.is_file() and self._log_path.stat().st_size > 0
        self._log_handle = self._log_path.open("ab")
        header = (
            ("\n" if existed else "")
            + f"=== crucible {self.name} engine, {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=== {' '.join(command)}\n"
        ).encode("utf-8")
        self._log_handle.write(header)
        self._log_handle.flush()

        environment = dict(os.environ)
        environment.update(self.environment())
        environment.update(compiler)
        if self._library_dirs:
            environment[LIBRARY_PATH_ENV] = library_path(
                self._library_dirs, environment.get(LIBRARY_PATH_ENV, "")
            )
        try:
            group = procgroup.own_group()
        except procgroup.ProcessGroupError as exc:
            self._close_log()
            raise EngineError(str(exc)) from exc
        try:
            self._process = subprocess.Popen(
                command,
                env=environment,
                **group,
                **self.stdio(self._log_handle),
            )
        except OSError as exc:
            self._close_log()
            raise EngineError(f"could not spawn {command[0]}: {exc}") from exc
        self._port = port
        self._served_name = served_name
        try:
            self.attach(self._process)
        except Exception as exc:
            try:
                self.stop()
            except EngineError:
                pass
            raise EngineError(
                f"{self.name} started as pid {self._process.pid if self._process else '?'} "
                f"but Crucible could not attach to it: {exc}"
            ) from exc

    def ready(
        self,
        timeout: float,
        on_progress: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Wait for the engine to answer. ``cancelled`` is the owning job's cancel
        flag: a load nobody wants any more stops waiting at the next poll rather
        than at the end of the load (a fleet's losing server held its card ~80 s)."""
        if self._process is None or self._port is None:
            raise EngineError(f"{self.name} has not been started")
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            self.raise_if_cancelled(cancelled)
            code = self._process.poll()
            self.refuse_a_taken_port()
            if code is not None:
                raise EngineError(
                    f"{self.name} exited {code} before it was ready. " + self.log_report()
                )
            announcement = self.announced_ready()
            if announcement is not None:
                if on_progress is not None:
                    on_progress(announcement)
                self.confirm(deadline, on_progress, cancelled)
                return
            if time.monotonic() >= deadline:
                raise EngineError(
                    f"{self.name} did not {self.readiness_description()} within "
                    f"{timeout:.0f}s. " + self.log_report()
                )
            attempt += 1
            if on_progress is not None:
                on_progress(self.warming_message(attempt, deadline))
            time.sleep(READY_POLL_SECONDS)

    def raise_if_cancelled(self, cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise JobCancelled(f"{self.name} was cancelled while it was starting")

    def warming_message(self, attempt: int, deadline: float) -> str:
        remaining = max(0.0, deadline - time.monotonic())
        base = (
            f"{self.name} loading; {attempt * READY_POLL_SECONDS:.0f}s elapsed, "
            f"{remaining:.0f}s before give-up"
        )
        phase = self.starting_phase()
        if phase is not None:
            return f"{base} — {phase}, {self._seconds_in(phase):.0f}s so far"
        last = self.log_tail(1).strip()
        return f"{base} — {last}" if last else base

    def starting_phase(self) -> str | None:
        """What the engine is doing while it starts, in words a client can show, when
        the engine's own log says it; None when it does not (the warming message then
        quotes the log's last line). Said to the client while it waits, never decided
        on (docs/ARCHITECTURE.md R4)."""
        return None

    def _seconds_in(self, phase: str) -> float:
        """How long Crucible has seen ``phase`` as the latest: from the first poll that
        read it, not from the engine's own clock."""
        now = time.monotonic()
        seen = self._phase_seen
        if seen is None or seen[0] != phase:
            self._phase_seen = (phase, now)
            return 0.0
        return now - seen[1]

    def _probe_models(self, url: str) -> list[str] | None:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None
        return [entry.get("id") for entry in data if isinstance(entry, dict)]

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            procgroup.stop_gracefully(
                process, self.name, self.sigterm_wait_seconds, self._log_path
            )
        except procgroup.ProcessGroupError as exc:
            self.detach()
            self._close_log()
            raise EngineError(str(exc)) from exc
        self.detach()
        self._close_log()
        self._process = None
        self._port = None
        self._served_name = None

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        return tail_of_last_run(self._log_path, lines)

    def log_report(self) -> str:
        """What a refusal quotes of this engine's log: the first error its last run
        printed, which the tail may have cut off, then the tail."""
        return led_by_first_error(
            self._log_path,
            f"Last {LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail(),
        )

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
