from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .. import procgroup
from ..errors import CrucibleError
from ..logtail import tail_of_last_run

STOP_TIMEOUT_SECONDS = 180.0
READY_POLL_SECONDS = 2.0
LOG_TAIL_LINES = 40


class EngineError(CrucibleError):
    ...


@runtime_checkable
class Engine(Protocol):
    name: str

    def start(
        self,
        model_dir: Path,
        served_name: str,
        port: int,
        args: list[str],
    ) -> None:
        pass

    def ready(
        self,
        timeout: float,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def base_url(self) -> str:
        pass

    @property
    def pids(self) -> frozenset[int]:
        pass


def int_flag(args: "list[str] | tuple[str, ...]", flag: str) -> int | None:
    found: str | None = None
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            found = args[index + 1]
        elif arg.startswith(flag + "="):
            found = arg.split("=", 1)[1]
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

    def __init__(self, python: Path, log_path: Path) -> None:
        self._python = Path(python)
        self._log_path = Path(log_path)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._port: int | None = None
        self._served_name: str | None = None


    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        return {}

    def announced_ready(self) -> str | None:
        served = self._probe_models(f"{self.base_url}/v1/models")
        if served is None:
            return None
        if self._served_name is not None and self._served_name not in served:
            raise EngineError(
                f"{self.name} is serving {served}, not "
                f"{self._served_name!r}; Crucible will not proxy a model it "
                "did not ask for"
            )
        return f"{self.name} is serving {self._served_name!r}"

    def readiness_description(self) -> str:
        return f"answer {self.base_url}/v1/models"

    def confirm(
        self, deadline: float, on_progress: Callable[[str], None] | None
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
                f"no interpreter at {self._python}; the llm env is not installed"
            )
        if not model_dir.is_dir():
            raise EngineError(f"no model directory at {model_dir}")

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
        self, timeout: float, on_progress: Callable[[str], None] | None = None
    ) -> None:
        if self._process is None or self._port is None:
            raise EngineError(f"{self.name} has not been started")
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            code = self._process.poll()
            if code is not None:
                raise EngineError(
                    f"{self.name} exited {code} before it was ready. Last "
                    f"{LOG_TAIL_LINES} lines of {self._log_path}:\n" + self.log_tail()
                )
            announcement = self.announced_ready()
            if announcement is not None:
                if on_progress is not None:
                    on_progress(announcement)
                self.confirm(deadline, on_progress)
                return
            if time.monotonic() >= deadline:
                raise EngineError(
                    f"{self.name} did not {self.readiness_description()} within "
                    f"{timeout:.0f}s. Last "
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
        if process.poll() is None:
            try:
                delivered = procgroup.ask_to_stop(process)
                if not delivered and procgroup.platform_kind() == procgroup.WIN32:
                    procgroup.terminate_tree(process, self.name)
            except procgroup.ProcessGroupError as exc:
                raise EngineError(
                    f"could not stop {self.name} (pid {process.pid}): {exc}"
                ) from exc
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                if procgroup.platform_kind() == procgroup.WIN32:
                    try:
                        procgroup.terminate_tree(process, self.name)
                    except procgroup.ProcessGroupError as exc:
                        self.detach()
                        self._close_log()
                        raise EngineError(str(exc)) from exc
                else:
                    self.detach()
                    self._close_log()
                    raise EngineError(
                        f"{self.name} (pid {process.pid}) did not exit within "
                        f"{STOP_TIMEOUT_SECONDS:.0f}s of SIGTERM. Crucible does not "
                        "SIGKILL a process holding CUDA — that wedges WSL2 until "
                        f"Windows reboots. Kill it by hand if you must: {self._log_path}"
                    ) from None
        self.detach()
        self._close_log()
        self._process = None
        self._port = None
        self._served_name = None

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        return tail_of_last_run(self._log_path, lines)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None
