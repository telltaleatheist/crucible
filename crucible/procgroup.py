"""A child in its own process group, and how to end it, on each platform.

THE ONE OWNER of three facts every process Crucible spawns depends on — an
engine (`engines/base.py`), llama-server (`engines/llama_server.py`) and a job
worker (`workers.py`):

1. **How a child is put in its own group.** POSIX: `start_new_session=True`
   (setsid), so a SIGTERM to the group reaches whatever the child forks — vLLM's
   per-GPU workers, ffmpeg under `asr`, a urvc batch under `rvc`. win32:
   `CREATE_NEW_PROCESS_GROUP`, which is what lets `CTRL_BREAK_EVENT` reach the
   child and ONLY the child (without it a break goes to this server's own group
   too). `start_new_session` is silently IGNORED on win32, so passing it there
   was a child in no group of its own and a stop with nothing to aim at.
2. **How it is asked to stop.** POSIX: SIGTERM to the group. win32:
   `CTRL_BREAK_EVENT` to the child's group — exactly what
   `LlamaServerEngine` already sent, and the reason it is the win32 answer here.
3. **What happens when it will not.** POSIX: NOTHING — the caller reports the
   timeout by name and never SIGKILLs, because SIGKILLing a process in a WSL2
   dxg GPU wait wedges the distro until Windows reboots. win32: the tree is
   TERMINATED (`taskkill /T /F`), because that reason does not exist there: a
   native Windows process is terminated against the Windows driver, which is
   what Task Manager does every day. `llama_server.py`'s module docstring point
   4 made the same deviation for the same reason; this is it made once.

WHY THIS MODULE EXISTS (measured 2026-09-23). Every stop path called
`os.killpg`, which does not exist on win32. Each one raised `AttributeError`
out of a stop, so the child was never signalled and nothing waited for it: a
Windows test run left 114 `mlx_vlm_serve.py` servers and a trail of
`fake_align_worker.py` / `fake_asr_worker.py` / `fake_narrator.py` processes
alive (some for three days), and a cancelled `align` job on a Windows host
reported `AttributeError: module 'os' has no attribute 'killpg'` and left its
worker running. The tests were only where it showed; the defect is the
production stop path, so it is fixed here and not in the tests.

AN UNKNOWN PLATFORM IS REFUSED BY NAME rather than guessed at: a stop that
picks a mechanism for a platform nobody has read is how the win32 case above
went unnoticed.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
from typing import Any

from .errors import CrucibleError

#: The POSIX platforms whose `os.killpg` and `setsid` Crucible relies on.
POSIX_PLATFORMS = frozenset({"linux", "darwin"})
WIN32 = "win32"

#: How long a win32 tree-kill is waited on before it is reported as having
#: failed. Well inside the 30 s margin `residency.CLEARANCE_TIMEOUT_SECONDS`
#: adds to an engine's own stop deadline, so a settlement waiting on a stop
#: still sees the stop's own answer rather than racing it.
KILL_WAIT_SECONDS = 10.0


class ProcessGroupError(CrucibleError):
    """This platform has no mechanism here, or a kill did not take."""


def platform_kind(platform: str | None = None) -> str:
    """`"posix"` or `"win32"`, or a refusal naming the platform."""
    name = sys.platform if platform is None else platform
    if name in POSIX_PLATFORMS:
        return "posix"
    if name == WIN32:
        return WIN32
    raise ProcessGroupError(
        f"platform {name!r} has no process-group mechanism in Crucible: "
        f"{sorted(POSIX_PLATFORMS)} use setsid + SIGTERM and win32 uses "
        "CREATE_NEW_PROCESS_GROUP + CTRL_BREAK_EVENT. A stop that guessed would "
        "leave a child running that nothing can see"
    )


def own_group(platform: str | None = None) -> dict[str, Any]:
    """The `Popen` keyword arguments that put a child in its own group."""
    if platform_kind(platform) == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def ask_to_stop(process: "subprocess.Popen[Any]") -> bool:
    """The polite half: SIGTERM to the group, or CTRL_BREAK on win32.

    Returns True if the signal was delivered, False if there was nothing to
    deliver it to (the process or its group is already gone) or — on win32 —
    no console to deliver it through. A Crucible host started detached has no
    console, and `GenerateConsoleCtrlEvent` then fails; the caller treats that
    as "the polite door does not exist" and goes to its second step at once
    rather than waiting out a clock nothing will stop.

    Any other failure raises: a signal that could not be sent for a reason
    nobody anticipated is not something to wait politely on.
    """
    if process.poll() is not None:
        return False
    if platform_kind() == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return False
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            raise ProcessGroupError(
                f"could not signal pid {process.pid}'s group: {exc}"
            ) from exc
        return True
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    except OSError:
        # No console to route the event through, or the group is gone. Either
        # way nothing was delivered; `terminate_tree` is what remains.
        return False
    return True


def terminate_tree(process: "subprocess.Popen[Any]", what: str) -> None:
    """win32 only: terminate the child AND everything it started, and wait.

    `taskkill /T` because Windows has no group kill: `Popen.kill()` is
    `TerminateProcess` on the one pid and would orphan an ffmpeg a worker
    started. Refused by name on POSIX, where the answer to a process that
    ignored SIGTERM is a sentence, never a SIGKILL (module docstring, 3).
    """
    if platform_kind() != WIN32:
        raise ProcessGroupError(
            f"{what} (pid {process.pid}) would be force-killed, and off win32 "
            "Crucible never does that: a killed CUDA process in a WSL2 GPU wait "
            "wedges the distro until Windows reboots"
        )
    if process.poll() is not None:
        return
    result = subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=KILL_WAIT_SECONDS,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        process.wait(timeout=KILL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        raise ProcessGroupError(
            f"{what} (pid {process.pid}) survived `taskkill /T /F` "
            f"(exit {result.returncode}: "
            f"{(result.stdout + result.stderr).strip() or 'no output'})"
        ) from None


__all__ = [
    "KILL_WAIT_SECONDS",
    "POSIX_PLATFORMS",
    "ProcessGroupError",
    "WIN32",
    "ask_to_stop",
    "own_group",
    "platform_kind",
    "terminate_tree",
]
