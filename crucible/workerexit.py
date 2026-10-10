"""Says how a worker process ended, in words a person can act on.

A worker that exits with a code says so ("exited 1"). One that a signal ended has a
negative return code, and "exited -9" reads like a crash of the worker's own making when
on Linux it almost always means the kernel's out-of-memory killer: Crucible never sends
SIGKILL (crucible/procgroup.py). On Victoria's laptop (2026-10-10, YuE2 on an 8 GB card)
the guest ran out of memory, the kernel killed the worker, and the job said only that
the worker "exited -9".

So when a worker is started its owner takes an `OomCount` - the kernel's count of
out-of-memory kills, read from the server's own cgroup (`memory.events`, which counts
the kills of every process in it, the workers included, global OOM or not) or, with no
cgroup v2 to read, the host-wide `oom_kill` in /proc/vmstat. When the worker ends by
SIGKILL, a count that moved while it ran is the evidence the out-of-memory killer did
it; a count that did not move says someone else did; a count that cannot be read says
the question is open, and the words say all three apart.
"""
from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path

PROC_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_VMSTAT = Path("/proc/vmstat")


@dataclass(frozen=True)
class OomCount:
    """The out-of-memory kill count at one moment, and where it was read."""

    kills: int
    source: str


def _field(text: str, name: str) -> int | None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _own_cgroup_events(proc_cgroup: Path, cgroup_root: Path) -> Path | None:
    try:
        text = proc_cgroup.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        # cgroup v2 is the one line "0::<path>".
        if line.startswith("0::"):
            relative = line[3:].strip().lstrip("/")
            return cgroup_root / relative / "memory.events"
    return None


def read_oom_count(
    *,
    proc_cgroup: Path = PROC_CGROUP,
    cgroup_root: Path = CGROUP_ROOT,
    proc_vmstat: Path = PROC_VMSTAT,
) -> OomCount | None:
    """The kill count the server's own cgroup keeps, else the host's; None off Linux or
    when neither can be read."""
    events = _own_cgroup_events(proc_cgroup, cgroup_root)
    if events is not None:
        try:
            kills = _field(events.read_text(encoding="utf-8"), "oom_kill")
        except OSError:
            kills = None
        if kills is not None:
            return OomCount(kills, str(events))
    try:
        kills = _field(proc_vmstat.read_text(encoding="utf-8"), "oom_kill")
    except OSError:
        return None
    return None if kills is None else OomCount(kills, str(proc_vmstat))


def signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"signal {number}"


@dataclass(frozen=True)
class Ending:
    """How a worker ended: `phrase` follows its name ("exited 1", "was killed by the
    out-of-memory killer (SIGKILL, signal 9)"), `why` is the sentence that says what that
    means, empty when the phrase says it all."""

    phrase: str
    why: str = ""

    def sentence(self) -> str:
        return f" {self.why}" if self.why else ""


def how_it_ended(
    code: int | None,
    at_start: OomCount | None,
    now: OomCount | None,
) -> Ending:
    if code is None:
        return Ending("has not exited")
    if code >= 0:
        return Ending(f"exited {code}")
    number = -code
    named = f"{signal_name(number)}, signal {number}"
    if number != signal.SIGKILL:
        return Ending(f"was killed by a signal ({named})")
    if at_start is not None and now is not None and at_start.source == now.source:
        if now.kills > at_start.kills:
            return Ending(
                f"was killed by the out-of-memory killer ({named})",
                "The machine ran out of memory while it ran: the kernel's count of "
                f"out-of-memory kills in {now.source} went from {at_start.kills} to "
                f"{now.kills}. Free memory on the machine, or run something smaller, "
                "before running it again.",
            )
        return Ending(
            f"was killed ({named})",
            "It was not the out-of-memory killer: the kernel's count of out-of-memory "
            f"kills in {now.source} stayed at {now.kills}. Crucible never sends "
            "SIGKILL, so a person or another program did.",
        )
    return Ending(
        f"was killed ({named}; out of memory?)",
        "Crucible never sends SIGKILL, and the kernel's out-of-memory killer is what "
        "usually does; this machine keeps no out-of-memory count Crucible can read to "
        "say for certain.",
    )


__all__ = ["Ending", "OomCount", "how_it_ended", "read_oom_count", "signal_name"]
