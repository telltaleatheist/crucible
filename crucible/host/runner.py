"""The one door to a subprocess and to the engine's `/v1/ping`.

Everything the host does to the machine goes through a `Runner`, for the reason
`sdk/bootstrap/src/runner.ts` gives on the other side of this phase: a test
supplies a scripted one and asserts on the ARGV that would have run, so the boot
recipe, the recovery recipes and the whole install sequence are exercised on a
machine that is not Windows and has no `crucible` distro.

Three rules carried over from that file, each of which was a defect somewhere:

1. **Argument arrays, never shell strings.** Nothing here is joined into a
   command line. (BookForge's memory `wsl-exe-implicit-shell-trap.md`: wsl.exe
   PRE-EXPANDS `$var` unless `--exec`.)
2. **Every call has a timeout.** A booting distro or a blocking profile makes
   `wsl.exe` never return, and a tray that hangs is a tray with no menu.
3. **A failure is a RESULT, not an exception.** The caller decides what a
   non-zero exit means; several of them mean "not yet" rather than "wrong".
"""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RunResult:
    """What a command said. `code` is None when it never produced one."""

    code: int | None
    stdout: str
    stderr: str
    #: Set when there is no exit code to report: a spawn error, or the timeout.
    failure: str | None

    @property
    def ok(self) -> bool:
        return self.failure is None and self.code == 0

    def said(self) -> str:
        """The most useful line to put in a log or a refusal."""
        for candidate in (self.stderr.strip(), self.stdout.strip(), self.failure):
            if candidate:
                return candidate[:400]
        return f"exit {self.code}"


class Runner(Protocol):
    """What the host is allowed to do to the machine."""

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
        """Run and collect. Never raises for a non-zero exit or a timeout."""
        ...

    def get(self, url: str, *, timeout_s: float) -> int | None:
        """The HTTP status of a GET, or None when nothing answered."""
        ...

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> "Child":
        """Start a long-lived child (the host-mode server) and return a handle."""
        ...


class Child(Protocol):
    """A process the host started and is responsible for."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None:
        """The exit code, or None while it is still running."""
        ...

    def terminate(self) -> None: ...

    def wait(self, timeout_s: float) -> int | None: ...


class ControlledChild:
    """An owned engine exits through its lifespan, never TerminateProcess."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def poll(self) -> int | None:
        return self._process.poll()

    def terminate(self) -> None:
        # Idempotent; closing the writer also handles an engine already exiting.
        if self._process.stdin is not None:
            self._process.stdin.close()

    def wait(self, timeout_s: float) -> int:
        return self._process.wait(timeout=timeout_s)


class ProcessRunner:
    """The real one. `subprocess` plus `urllib`, and nothing else."""

    def __init__(self, platform: str, env: Mapping[str, str]) -> None:
        self._platform = platform
        self._env = dict(env)

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
                # A tray program has no console; a child that opens one is a
                # window flashing on somebody's desktop every fifteen seconds.
                creationflags=_no_window_flag(self._platform),
                text=True,
                errors="replace",
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
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            failure=None,
        )

    def get(self, url: str, *, timeout_s: float) -> int | None:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            # It ANSWERED. 401 from `/v1/ping` would still mean a server is
            # there, and "there is a server and it refused me" is not the same
            # fact as "nothing is listening" — see `presence.py`.
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
            creationflags=_no_window_flag(self._platform),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE if controlled else subprocess.DEVNULL,
        )
        return ControlledChild(child) if controlled else child  # type: ignore[return-value]



def _no_window_flag(platform: str) -> int:
    """`CREATE_NO_WINDOW` on Windows, 0 elsewhere.

    Read off `subprocess` rather than written as `0x08000000`, but only when
    the attribute is there: the constant does not exist on Linux, and this
    module is imported by the test suite, which runs in WSL.
    """
    if platform != "win32":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
