from __future__ import annotations

import codecs
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RunResult:
    code: int | None
    stdout: str
    stderr: str
    failure: str | None

    @property
    def ok(self) -> bool:
        return self.failure is None and self.code == 0

    def output_tail(self) -> str:
        for candidate in (self.stderr.strip(), self.stdout.strip(), self.failure):
            if candidate:
                return candidate if len(candidate) <= 400 else "..." + candidate[-400:]
        return f"exit {self.code}"

    said = output_tail


class Runner(Protocol):
    @property
    def platform(self) -> str: ...

    @property
    def env(self) -> Mapping[str, str]: ...

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        ...

    def stream(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        on_line: "Callable[[str, str], None]",
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        ...

    def download(
        self,
        url: str,
        destination: "Path",
        *,
        timeout_s: float,
        on_progress: "Callable[[int, int | None, str], None] | None" = None,
        attempts: int = 1,
    ) -> RunResult:
        ...

    def get(self, url: str, *, timeout_s: float) -> int | None:
        ...

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> "Child":
        ...


class Child(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None:
        ...

    def terminate(self) -> None: ...

    def wait(self, timeout_s: float) -> int | None: ...


class ControlledChild:
    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()

    def wait(self, timeout_s: float) -> int:
        return self._process.wait(timeout=timeout_s)


class ProcessRunner:
    def __init__(
        self, platform: str, env: Mapping[str, str], cwd: str | None = None
    ) -> None:
        self._platform = platform
        self._env = dict(env)
        self._cwd = cwd

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def env(self) -> Mapping[str, str]:
        return self._env

    def _child_env(self, env: Mapping[str, str] | None) -> dict[str, str] | None:
        if env is None:
            return None
        merged = dict(self._env)
        merged.update(env)
        return merged

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                timeout=timeout_s,
                env=self._child_env(env),
                cwd=self._cwd,
                creationflags=_no_window_flag(self._platform),
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                code=None,
                stdout="",
                stderr="",
                failure=f"timed out after {timeout_s:.0f}s",
            )
        except OSError as exc:
            return RunResult(code=None, stdout="", stderr="", failure=str(exc))
        return RunResult(
            code=completed.returncode,
            stdout=_decode_pipe(completed.stdout),
            stderr=_decode_pipe(completed.stderr),
            failure=None,
        )

    def stream(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        on_line: Callable[[str, str], None],
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        collected: dict[str, list[str]] = {"stdout": [], "stderr": []}
        try:
            child = subprocess.Popen(
                list(argv),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._child_env(env),
                cwd=self._cwd,
                creationflags=_no_window_flag(self._platform),
            )
        except OSError as exc:
            return RunResult(code=None, stdout="", stderr="", failure=str(exc))

        def pump(pipe: object, name: str) -> None:
            assert pipe is not None
            for raw in pipe:
                for line in _decode_pipe(raw).splitlines():
                    collected[name].append(line)
                    on_line(line, name)

        threads = [
            threading.Thread(target=pump, args=(child.stdout, "stdout"), daemon=True),
            threading.Thread(target=pump, args=(child.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            code: int | None = child.wait(timeout=timeout_s)
            failure = None
        except subprocess.TimeoutExpired:
            code, failure = None, _ask_to_stop(child, timeout_s)
        for thread in threads:
            thread.join(timeout=30.0)
        return RunResult(
            code=code,
            stdout="\n".join(collected["stdout"]),
            stderr="\n".join(collected["stderr"]),
            failure=failure,
        )

    def download(
        self,
        url: str,
        destination: Path,
        *,
        timeout_s: float,
        on_progress: Callable[[int, int | None, str], None] | None = None,
        attempts: int = 1,
    ) -> RunResult:
        from ..interpreter import InterpreterError, fetch

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch(
                url,
                destination,
                on_progress=on_progress,
                timeout=int(timeout_s),
                attempts=attempts,
            )
        except InterpreterError as exc:
            return RunResult(code=None, stdout="", stderr="", failure=exc.message)
        return RunResult(code=0, stdout=str(destination), stderr="", failure=None)

    def get(self, url: str, *, timeout_s: float) -> int | None:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> Child:
        controlled = list(argv)[1:] == ["-m", "crucible.cli", "serve", "--controller-stdin"]
        child = subprocess.Popen(
            list(argv),
            env=self._child_env(env),
            cwd=self._cwd,
            creationflags=_no_window_flag(self._platform),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE if controlled else subprocess.DEVNULL,
        )
        return ControlledChild(child) if controlled else child


STOP_WAIT_SECONDS = 30.0

_UTF16_SNIFF_BYTES = 64


def _ask_to_stop(child: subprocess.Popen, timeout_s: float) -> str:
    child.terminate()
    try:
        child.wait(timeout=STOP_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        return (
            f"timed out after {timeout_s:.0f}s; pid {child.pid} was asked to stop "
            f"and had not exited {STOP_WAIT_SECONDS:.0f}s later. Nothing was "
            "force-killed: end it in Task Manager once it has let go of the GPU"
        )
    return f"timed out after {timeout_s:.0f}s and was asked to stop"


def _decode_pipe(raw: bytes | None) -> str:
    if not raw:
        return ""
    if raw.startswith(codecs.BOM_UTF16_LE):
        return _newlines(raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le", errors="replace"))
    window = raw[:_UTF16_SNIFF_BYTES]
    if len(window) >= 2 and all(window[i] == 0 for i in range(1, len(window), 2)):
        return _newlines(raw.decode("utf-16-le", errors="replace"))
    return _newlines(raw.decode("utf-8", errors="replace"))


def _newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _no_window_flag(platform: str) -> int:
    if platform != "win32":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
