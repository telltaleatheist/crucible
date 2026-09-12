"""The `Engine` interface and the managed-subprocess machinery behind it.

PHASE2-LLM.md section 3: `start(model_dir, served_name, port, args)`, `ready()`
polling the engine's own `/v1/models`, `stop()` by SIGTERM with a wait and
**never SIGKILL** (a killed CUDA process wedges WSL until Windows reboots), and
`base_url`. The engine binds 127.0.0.1 on a free port; only Crucible talks to it.
Its stdout and stderr go to `~/.crucible/logs/engine-<id>.log`.
"""

from __future__ import annotations

import errno
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from ..errors import CrucibleError

#: How long `stop()` waits for SIGTERM to be honoured before it gives up and says
#: so. It never escalates to SIGKILL.
STOP_TIMEOUT_SECONDS = 180.0
#: How often `ready()` asks, and how much of the log a failure reports.
READY_POLL_SECONDS = 2.0
LOG_TAIL_LINES = 40


class EngineError(CrucibleError):
    """An engine would not start, would not answer, or would not stop."""


@runtime_checkable
class Engine(Protocol):
    """What the load-model job needs from vLLM, mlx-lm, or a test double."""

    name: str

    def start(
        self,
        model_dir: Path,
        served_name: str,
        port: int,
        args: list[str],
    ) -> None:
        """Spawn the server. Returns as soon as the process exists, not when ready."""

    def ready(
        self,
        timeout: float,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Block until the engine answers /v1/models, or raise EngineError."""

    def stop(self) -> None:
        """SIGTERM and wait. Never SIGKILL."""

    @property
    def base_url(self) -> str:
        """`http://127.0.0.1:<port>` — the root the OpenAI routes hang off."""

    @property
    def pids(self) -> frozenset[int]:
        """Every pid this engine owns, for the accelerator guard."""


def find_free_port() -> int:
    """A port nothing is listening on right now.

    Bind-and-release: the engine claims it milliseconds later. Nothing else on
    this host is handing out ports at that rate, and a collision surfaces as the
    engine failing to bind, which `ready()` reports by name rather than hiding.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def logs_dir(home: Path) -> Path:
    directory = home / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class SubprocessEngine:
    """Common body for every engine: spawn, poll, SIGTERM.

    Subclasses supply `name` and `command()`. The process is started in its own
    session so that SIGTERM reaches the workers an engine forks (vLLM forks one
    per GPU) and not just the parent.
    """

    name = "subprocess"

    def __init__(self, python: Path, log_path: Path) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._port: int | None = None
        self._served_name: str | None = None

    # ------------------------------------------------------------- subclass

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        """Extra environment for the engine process. Merged over os.environ."""
        return {}

    # ------------------------------------------------------------- lifecycle

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

    def start(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> None:
        if self._process is not None:
            raise EngineError(
                f"{self.name} is already running as pid {self._process.pid}"
            )
        if not self._python.is_file():
            raise EngineError(
                f"no interpreter at {self._python}; the llm env is not installed"
            )
        if not model_dir.is_dir():
            raise EngineError(f"no model directory at {model_dir}")

        command = self.command(model_dir, served_name, port, args)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self._log_path.open("wb")
        header = (
            f"=== crucible {self.name} engine, {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"=== {' '.join(command)}\n"
        ).encode("utf-8")
        self._log_handle.write(header)
        self._log_handle.flush()

        environment = dict(os.environ)
        environment.update(self.environment())
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=environment,
            )
        except OSError as exc:
            self._close_log()
            raise EngineError(f"could not spawn {command[0]}: {exc}") from exc
        self._port = port
        self._served_name = served_name

    def ready(
        self, timeout: float, on_progress: Callable[[str], None] | None = None
    ) -> None:
        if self._process is None or self._port is None:
            raise EngineError(f"{self.name} has not been started")
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}/v1/models"
        attempt = 0
        while True:
            code = self._process.poll()
            if code is not None:
                raise EngineError(
                    f"{self.name} exited {code} before it was ready. Last "
                    f"{LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail()
                )
            served = self._probe_models(url)
            if served is not None:
                if self._served_name is not None and self._served_name not in served:
                    raise EngineError(
                        f"{self.name} is serving {served}, not "
                        f"{self._served_name!r}; Crucible will not proxy a model it "
                        "did not ask for"
                    )
                if on_progress is not None:
                    on_progress(f"{self.name} is serving {self._served_name!r}")
                return
            if time.monotonic() >= deadline:
                raise EngineError(
                    f"{self.name} did not answer {url} within {timeout:.0f}s. Last "
                    f"{LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail()
                )
            attempt += 1
            if on_progress is not None:
                on_progress(self.warming_message(attempt, deadline))
            time.sleep(READY_POLL_SECONDS)

    def warming_message(self, attempt: int, deadline: float) -> str:
        remaining = max(0.0, deadline - time.monotonic())
        last = self.log_tail(1).strip()
        base = (
            f"{self.name} loading; {attempt * READY_POLL_SECONDS:.0f}s elapsed, "
            f"{remaining:.0f}s before give-up"
        )
        return f"{base} — {last}" if last else base

    def _probe_models(self, url: str) -> list[str] | None:
        """The served model ids, or None while the engine is not answering yet."""
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
        """SIGTERM the engine's process group, then wait. Never SIGKILL."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                if exc.errno != errno.ESRCH:
                    raise EngineError(
                        f"could not signal {self.name} (pid {process.pid}): {exc}"
                    ) from exc
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                self._close_log()
                raise EngineError(
                    f"{self.name} (pid {process.pid}) did not exit within "
                    f"{STOP_TIMEOUT_SECONDS:.0f}s of SIGTERM. Crucible does not "
                    "SIGKILL a process holding CUDA — that wedges WSL2 until "
                    f"Windows reboots. Kill it by hand if you must: {self._log_path}"
                ) from None
        self._close_log()
        self._process = None
        self._port = None
        self._served_name = None

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        if not self._log_path.is_file():
            return ""
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
