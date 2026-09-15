"""llama.cpp's `llama-server`, the `llama-windows` engine.

PHASE15-HOST.md 3.10 and 7.4's item 3. Structurally this is what
`crucible/engines/vllm.py` and `mlx_lm.py` are: a `SubprocessEngine` the
residency spawns per resident model, leases, settles and kills. What is
different is only that `command()` is a BINARY rather than a Python module —
llama.cpp ships an executable, not a package — and that the whole thing runs
on win32, where the POSIX process group the base class signals does not
exist.

THE FOUR THINGS THIS FILE DECIDES, AND WHY
-------------------------------------------
1. **`--alias <crucible id>`** (7.4's recorded decision). llama-server names
   the model it serves after the GGUF file unless told otherwise, which would
   make `/v1/models` answer `dots.ocr-Q8_0.gguf`. With the alias the name
   is the Crucible id, so `engine_model_name()` returns the id, the OpenAI
   proxy forwards `model` VERBATIM (no rewrite, unlike mlx-lm), and readiness
   becomes *"the name equals the id this server started"* rather than 3.10
   fact 4's *"a name ending in dots.ocr"* — strictly stricter, and it
   generalises to the two text models the same mechanism serves.
   `pages_engine_wrong_model` keeps its name: it is what the base class's
   `announced_ready` raises when the served name is not the asked-for one.
2. **`-m` and `--mmproj` are composed by the RESIDENCY**, not carried in the
   manifest's `engine_args`. Only the server knows where it put the weights.
   The manifest carries `--parallel 1` and nothing else, and `-c` comes from
   the block's `context_default`.
3. **Nothing is ever adopted** (3.10, fact 5). Foundry used a server already
   answering on port 8000 and never stopped it; a process with two owners is
   the shape ARCHITECTURE.md R1 forbids. Crucible starts its own child on an
   ephemeral loopback port, and a port that is somehow taken is
   `port_in_use` by name.
4. **Stop is 30 s graceful, then kill** (3.10, fact 6), and that is a
   DEVIATION from the base class's never-SIGKILL rule, stated here so it is
   not read as an oversight. That rule exists because SIGKILLing a process
   holding a CUDA device inside WSL2 wedges the distro until Windows reboots
   (`crucible/engines/base.py`). This engine does not run inside WSL2 — it
   runs natively on Windows, against the Windows driver, where terminating a
   CUDA process is what Task Manager does every day. The graceful half is
   `CTRL_BREAK_EVENT` to the child's own process group, which llama-server
   handles; `terminate()` follows if it does not.

THE FATAL LINES
---------------
`/v1/models` alone turns every failure into the full startup timeout — five
minutes of a person watching a spinner for a missing DLL. So the log is read
while readiness is polled and the lines that mean *never coming* end the wait
at once, as `pages_engine_failed` with the line. They are matched case-
insensitively on substrings taken from the three failures Foundry's launcher
actually met (`fatalReason()`): a CUDA out-of-memory, a missing CUDA runtime
DLL, and a GGUF the loader will not read.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .base import SubprocessEngine, EngineError

#: The engine's name, as a manifest's `[backends.llama-windows] engine` spells
#: it and as `crucible/manifests.py`'s `BACKEND_ENGINES` pins it.
ENGINE_NAME = "llama-server"

#: How long the child is given to exit on its own before it is terminated.
#: 3.10, fact 6. Thirty seconds, not the three minutes an engine inside WSL2
#: gets: there is no distro to wedge here, and a page reader that will not
#: close its own log in half a minute is not going to.
GRACEFUL_STOP_SECONDS = 30.0

#: And how long the terminate itself is given before this gives up and says
#: so. A process that survives `TerminateProcess` is in a state no further
#: signal reaches, and pretending otherwise would be a stop that reports
#: success over a port still held.
KILL_WAIT_SECONDS = 10.0

#: The two refusals a child that never comes up is reported as. `port_in_use`
#: is its own because it is the one that is not about this model at all
#: (3.10, fact 5): something else on the machine took the port between
#: `find_free_port` choosing it and the child binding it, and the answer is
#: to retry, not to look at the weights.
PAGES_ENGINE_FAILED = "pages_engine_failed"
PORT_IN_USE = "port_in_use"

#: How much of the log the fatal-line scan reads. The interesting lines are
#: llama.cpp's last words, and a full log of a long load is megabytes.
FATAL_SCAN_LINES = 200

#: Substrings that mean the child will NEVER become ready, each with the code
#: it is reported under and the sentence a person reads. Lower-cased before
#: comparison. Taken from the failures Foundry's launcher actually met
#: (`fatalReason()`), each named rather than summarised so a new one is added
#: deliberately and not by widening a regex.
FATAL_LINES: tuple[tuple[str, str, str], ...] = (
    (
        "cuda error: out of memory",
        PAGES_ENGINE_FAILED,
        "the card has no room for this model",
    ),
    (
        "out of memory",
        PAGES_ENGINE_FAILED,
        "the card, or this machine's RAM, has no room",
    ),
    (
        "cudart64_",
        PAGES_ENGINE_FAILED,
        "a CUDA runtime DLL is missing: the cudart asset did not unpack",
    ),
    (
        "the code execution cannot proceed",
        PAGES_ENGINE_FAILED,
        "a DLL beside llama-server is missing",
    ),
    (
        "failed to load model",
        PAGES_ENGINE_FAILED,
        "llama.cpp will not read this GGUF",
    ),
    (
        "error loading model",
        PAGES_ENGINE_FAILED,
        "llama.cpp will not read this GGUF",
    ),
    (
        "unknown model architecture",
        PAGES_ENGINE_FAILED,
        "this llama.cpp build does not know this model",
    ),
    (
        "address already in use",
        PORT_IN_USE,
        "something else on this machine took the port",
    ),
    (
        "bind: address in use",
        PORT_IN_USE,
        "something else on this machine took the port",
    ),
)


def fatal_reason(line: str) -> tuple[str, str] | None:
    """`(code, sentence)` when this line means the child is never coming.

    A pure function over ONE line, so it is testable without a process and so
    the table above is the only place a failure becomes fatal.
    """
    lowered = line.lower()
    for needle, code, reason in FATAL_LINES:
        if needle in lowered:
            return (code, reason)
    return None


class LlamaServerEngine(SubprocessEngine):
    """One `llama-server.exe` per resident model.

    `python` on the constructor is the path to `llama-server.exe` and not an
    interpreter. The base class calls it `_python` because every engine before
    this one was a Python module; the residency passes whatever
    `jobenv`-equivalent resolves for the backend, and for `llama-windows` that
    is `crucible/llamacpp.py`'s `server_path()`. The name is the base class's
    and renaming it across four engines to suit this one would be a rename
    for a word.
    """

    name = ENGINE_NAME

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        """`llama-server -m … [--mmproj …] -c … --alias <id> --host --port`.

        `args` arrives from `Residency._engine_args` already carrying `-m`,
        `--mmproj` (for a vision model) and `-c`, because those are composed
        from the SPEC and the weights directory, which only the server knows.
        What is added here is what is true of every llama-server this engine
        starts: where it listens, and what it calls itself.
        """
        return [
            str(self._python),
            *args,
            "--alias",
            served_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    def stdio(self, log_handle: Any) -> dict[str, Any]:
        """Both streams to the log, and — on win32 — its OWN process group.

        `CREATE_NEW_PROCESS_GROUP` is what makes `CTRL_BREAK_EVENT` reach the
        child and only the child. Without it a break would go to this server's
        whole group, which includes this server.
        """
        wiring = dict(super().stdio(log_handle))
        if sys.platform == "win32":
            wiring["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        return wiring

    def readiness_description(self) -> str:
        return f"answer {self.base_url}/v1/models with {self._served_name!r}"

    def announced_ready(self) -> str | None:
        """The base class's `/v1/models` check, plus the fatal-line early exit.

        The order matters: a fatal line is read BEFORE the probe, because a
        child that has already printed "out of memory" will keep not answering
        for the whole five-minute timeout and the useful sentence is already
        in the log.
        """
        fatal = self._fatal_in_log()
        if fatal is not None:
            code, reason, line = fatal
            if code == PORT_IN_USE:
                raise port_in_use_error(self._port, self._served_name, line)
            raise EngineError(
                f"{code}: {self.name} will not come up: {reason}. It said: {line}"
            )
        return super().announced_ready()

    def _fatal_in_log(self) -> tuple[str, str, str] | None:
        """`(code, sentence, the line)` for the first fatal line, or None."""
        for line in self.log_tail(FATAL_SCAN_LINES).splitlines():
            found = fatal_reason(line)
            if found is not None:
                return (found[0], found[1], line.strip())
        return None

    def stop(self) -> None:
        """30 s graceful, then terminate. See the module docstring's point 4.

        On POSIX (a test, or a future llama.cpp on Linux) this is SIGTERM to
        the process group — the base class's mechanism, with this engine's
        shorter clock — and then `kill()`. On win32 it is `CTRL_BREAK_EVENT`
        to the child's own group and then `terminate()`.
        """
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            self._ask_it_to_stop(process)
            try:
                process.wait(timeout=GRACEFUL_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=KILL_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    self.detach()
                    self._close_log()
                    raise EngineError(
                        f"{self.name} (pid {process.pid}) survived both a "
                        f"graceful stop and a kill. Its port is still held and "
                        f"nothing here can take it back: {self._log_path}"
                    ) from None
        self.detach()
        self._close_log()
        self._process = None
        self._port = None
        self._served_name = None

    def _ask_it_to_stop(self, process: "subprocess.Popen[bytes]") -> None:
        """The polite half, in the platform's own vocabulary."""
        import os
        import signal

        try:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # It is already gone, or the group is. `wait` below is what
            # decides; a signal that did not land is not itself news.
            pass

    def confirm(
        self, deadline: float, on_progress: Callable[[str], None] | None
    ) -> None:
        """Nothing. `llama-server` binds its OpenAI routes after the load.

        Unlike mlx-lm, whose list route is served by a thread that does not
        wait for the weights (`crucible/engines/mlx_lm.py`, point 3), llama.cpp
        reads the GGUF before it listens — so a 200 from `/v1/models` means
        the model is in memory. Stated rather than inherited silently, because
        the day that stops being true this is the docstring that is wrong.
        """
        return None


def port_in_use_error(
    port: int | None, model_id: str | None, line: str
) -> EngineError:
    """`port_in_use`, by name (3.10, fact 5).

    Crucible chose this port milliseconds ago from `find_free_port`, so this
    is a genuine race with something else on the machine and not a
    misconfiguration. It is NOT an invitation to adopt whatever is there: a
    process Crucible did not start is somebody else's, and serving through it
    would make the running engine a fact with two owners.
    """
    return EngineError(
        f"{PORT_IN_USE}: 127.0.0.1:{port} was free when Crucible chose it and "
        f"is taken now, so {model_id} was not started. Crucible never adopts a "
        f"server it did not start: retry, and it will choose another port. "
        f"llama-server said: {line}"
    )


__all__ = [
    "ENGINE_NAME",
    "FATAL_LINES",
    "GRACEFUL_STOP_SECONDS",
    "KILL_WAIT_SECONDS",
    "LlamaServerEngine",
    "PAGES_ENGINE_FAILED",
    "PORT_IN_USE",
    "fatal_reason",
    "port_in_use_error",
]
