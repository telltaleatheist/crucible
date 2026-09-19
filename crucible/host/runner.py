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

And a fourth this file learned on its own, deploy 1.0.4:

4. **The pipes are BYTES, and this module decides what they say.** `wsl.exe`
   writes its own diagnostics as UTF-16LE; see `_decode_pipe`.
"""

from __future__ import annotations

import codecs
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

    def __init__(
        self, platform: str, env: Mapping[str, str], cwd: str | None = None
    ) -> None:
        self._platform = platform
        self._env = dict(env)
        #: WHERE CHILDREN START, and it is not cosmetic. A child inherits this
        #: process's working directory, and the orchestrator's is inside its own
        #: installation (installation.json records
        #: `...\Crucible\host\Lib\site-packages`, deliberately, so that
        #: `-m crucible.cli` imports). A `wsl.exe` child therefore holds a handle
        #: on `Crucible\host` — and keeps holding it after the orchestrator
        #: exits, which is what stopped an upgrade on 2026-09-16 with
        #: "Move-Item: the process cannot access the file because it is being
        #: used by another process", naming nothing. Sysinternals `handle64`
        #: found two orphaned wsl.exe and a wslhost.exe on that directory.
        #:
        #: The caller passes CRUCIBLE_HOME: the server's own state directory,
        #: which is what the systemd unit uses as WorkingDirectory for the same
        #: reason, and which the installer never moves. NOT the user's home —
        #: `console_script` records the ImportError that follows from a working
        #: directory landing on sys.path.
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
                # A tray program has no console; a child that opens one is a
                # window flashing on somebody's desktop every fifteen seconds.
                creationflags=_no_window_flag(self._platform),
                # NOT `text=True`. Asking subprocess to decode means asking it
                # to decode with the LOCALE codec, and one of the two programs
                # on the other end of this pipe does not use it — `_decode_pipe`.
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
            cwd=self._cwd,
            creationflags=_no_window_flag(self._platform),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.PIPE if controlled else subprocess.DEVNULL,
        )
        return ControlledChild(child) if controlled else child  # type: ignore[return-value]



#: How many bytes of a stream are enough to recognise UTF-16LE by its shape.
#: A whole `wsl -l -v` table would do as well; the point of a window is that
#: the guest's own output, which can be megabytes, is not walked twice.
_UTF16_SNIFF_BYTES = 64


def _decode_pipe(raw: bytes | None) -> str:
    """Turn one captured pipe into text, and MEASURE which codec wrote it.

    `wsl.exe` IS THE ONLY TOOL THIS HOST RUNS THAT NEEDS THIS, and the reason
    is that it is really two programs. Its OWN messages — "There is no
    distribution with the supplied name.", "Error code: Wsl/Service/
    WSL_E_DISTRO_NOT_FOUND", the `wsl -l -v` table — are written by the Windows
    side as UTF-16LE, which is what the Windows console API takes. Everything
    it `--exec`s is a program inside the guest, and its bytes are RELAYED, so
    they arrive exactly as Linux wrote them: UTF-8. One command can therefore
    produce a UTF-16 stderr and a UTF-8 stdout, which is why this decides per
    stream and not once per call.

    MEASURED 2026-09-19, deploy 1.0.4. The runner asked `subprocess.run` for
    text, so both streams were decoded with the locale codec — under which
    every NUL of a UTF-16LE string is a perfectly good character — and the one
    sentence naming why the deploy had failed reached `host.log` as
    `T\\x00h\\x00e\\x00r\\x00e\\x00 \\x00i\\x00s\\x00 …`. Nothing was lost; it
    was simply unreadable, which for a diagnostic is the same thing.

    Two facts identify it, both from the bytes themselves rather than from the
    argv, because a runner that decided by command name would be a second
    owner of the question "what is wsl.exe": a UTF-16LE byte-order mark, and —
    for the streams that carry none, which is what 1.0.4 measured — a high
    byte of zero under every ASCII character in the opening window.

    `errors="replace"` and never `"ignore"`: a byte nothing can decode becomes
    U+FFFD, a character a person reading the log can SEE, rather than a hole in
    a sentence that reads as if it were complete.
    """
    if not raw:
        return ""
    if raw.startswith(codecs.BOM_UTF16_LE):
        return _newlines(raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le", errors="replace"))
    window = raw[:_UTF16_SNIFF_BYTES]
    if len(window) >= 2 and all(window[i] == 0 for i in range(1, len(window), 2)):
        return _newlines(raw.decode("utf-16-le", errors="replace"))
    return _newlines(raw.decode("utf-8", errors="replace"))


def _newlines(text: str) -> str:
    """CRLF and CR to LF — what `text=True` used to do on the way past.

    Not cosmetic and not new behaviour: `parse_wsl_list` and every other reader
    in this package was written against universal-newline output, and wsl.exe
    is a Windows program that ends its lines the Windows way.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _no_window_flag(platform: str) -> int:
    """`CREATE_NO_WINDOW` on Windows, 0 elsewhere.

    Read off `subprocess` rather than written as `0x08000000`, but only when
    the attribute is there: the constant does not exist on Linux, and this
    module is imported by the test suite, which runs in WSL.
    """
    if platform != "win32":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
