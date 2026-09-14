"""The PREDICATES for 4c's table, and the walk that uses them.

`wsl_states.py` beside this file is GENERATED from
`sdk/bootstrap/src/wsl-states.ts` and holds the table's DATA — the codes, their
order, the probe argv, the sentences, the actions. What cannot be generated is
`means`: whether a probe's output MEANS a state is code, not data, and a
generator that emitted predicates would be emitting a second implementation.

So the predicates are written once here, keyed by the generated codes, and
`tests/test_host_wslstate.py` asserts that the two sets are exactly equal. That
is the same seam `crucible/envpack.py`'s `SMOKE_IMPORT` has with
`cli.INSTALLABLE_JOB_TYPES` — "tied by a check instead of by an import" — and it
is why a row renamed in TypeScript fails a Python test rather than silently
never matching.

The ORDER is the generated order and this file does not choose it: deepest
cause first, so a machine with VT-x off in the firmware is never told to press
"Enable WSL", which it would then press forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping

from .errors import HostError
from .runner import RunResult, Runner
from .wsl_states import WSL_STATE_CODES, WSL_STATES, WslStateDef

#: What `wsl --install` / `--status` print when the hypervisor is not there.
#: The same expression `wsl-states.ts` carries, and the same reason: `--status`
#: fails for "the feature is off" AND for "virtualization is off", and only the
#: text tells them apart.
NO_HYPERVISOR = re.compile(
    r"HCS_E_HYPERV_NOT_INSTALLED|0x80370102|hypervisor|virtual machine platform",
    re.IGNORECASE,
)

PROBE_TIMEOUT_SECONDS = 60.0


@dataclass
class Evidence:
    """Everything the probes have answered so far, for rows that need two facts."""

    results: dict[str, RunResult]
    distros: list[str]
    #: What the caller measured, for the rows that are about numbers.
    required_bytes: int = 0
    app_distro: str | None = None
    guest_user: str | None = None
    release: str = ""


def _said(result: RunResult) -> str:
    return result.said()


def _free_bytes(result: RunResult) -> int | None:
    first = result.stdout.strip().split()
    if not first or not first[0].isdigit():
        return None
    return int(first[0]) * 1024


def gib(value: int) -> str:
    """`gib()` in `wsl-states.ts`, to one decimal, so the two sentences match."""
    return f"{value / 1024 ** 3:.1f} GiB"


def _systemd_on(result: RunResult) -> bool:
    return re.search(r"systemd\s*=\s*true", result.stdout, re.IGNORECASE) is not None


# --------------------------------------------------------------- predicates
#
# One per generated code. Each takes (result, evidence) and answers "does this
# evidence mean this state". Nothing here reaches the machine: the walk does
# the running, and these are pure so the whole table can be exercised with
# fabricated output.

Predicate = Callable[[RunResult, Evidence], bool]

MEANS: dict[str, Predicate] = {
    "virtualization_disabled": lambda result, _: (
        not result.ok
        and NO_HYPERVISOR.search(f"{result.stdout}{result.stderr}{result.failure or ''}")
        is not None
    ),
    "wsl_missing": lambda result, _: not result.ok,
    "wsl1_only": lambda result, _: re.search(
        r"Default Version:\s*1\b", result.stdout, re.IGNORECASE
    )
    is not None,
    "no_crucible_distro": lambda _result, seen: "crucible" not in seen.distros,
    "distro_not_systemd": lambda result, _: not _systemd_on(result),
    "foreign_distro_not_systemd": lambda result, seen: (
        seen.app_distro is not None
        and seen.app_distro in seen.distros
        and not _systemd_on(result)
    ),
    "guest_no_network": lambda result, _: not result.ok,
    "pack_disk": lambda result, seen: (
        _free_bytes(result) is not None and _free_bytes(result) < seen.required_bytes  # type: ignore[operator]
    ),
    "linger_unreadable": lambda result, _: not result.ok or result.stdout.strip() != "0",
    # THE LAST ROW IS TOTAL. `detect()` returning None would make "nothing is
    # wrong" a null every caller has to interpret; it is a state with a name.
    "wsl_ready": lambda _result, _seen: True,
}


@dataclass(frozen=True)
class WslState:
    """A state, as detected: the row, the words, and what the probe said."""

    code: str
    sentence: str
    action_kind: str
    action_argv: tuple[str, ...]
    action_text: str
    action_url: str
    evidence: str


def render(text: str, result: RunResult, seen: Evidence) -> str:
    """Fill the generated template's placeholders from the evidence.

    `str.format` is NOT used: a sentence is prose and may legitimately contain
    a brace, and a `KeyError` from somebody's error message is a worse failure
    than the one being reported.
    """
    free = _free_bytes(result)
    replacements: Mapping[str, str] = {
        "{said}": _said(result),
        "{app_distro}": seen.app_distro or "",
        "{guest_user}": seen.guest_user or "",
        "{release}": seen.release,
        "{required}": gib(seen.required_bytes),
        "{free}": "an unreadable amount" if free is None else gib(free),
    }
    out = text
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def parse_distro_names(text: str) -> list[str]:
    """The names out of `wsl -l -v`. Shared with `presence.py`'s reader."""
    from .presence import parse_wsl_list

    return parse_wsl_list(text)


def detect(
    runner: Runner,
    *,
    release: str,
    required_bytes: int = 0,
    app_distro: str | None = None,
    guest_user: str | None = None,
    check_network: bool = False,
    timeout_s: float = PROBE_TIMEOUT_SECONDS,
) -> WslState:
    """The FIRST row that matches, with its probes run lazily and cached.

    A healthy machine costs `--status`, `-l -v`, one `cat`, and — only if the
    caller asked — one `curl`, one `df` and one `id`. A machine with no WSL
    costs exactly one command.

    The two rows that cost something are OFF unless asked for, which is
    `wsl-states.ts`'s rule and its reason: reading the facts about a machine
    must not reach the internet.
    """
    seen = Evidence(
        results={},
        distros=[],
        required_bytes=required_bytes,
        app_distro=app_distro,
        guest_user=guest_user,
        release=release,
    )

    def substitute(word: str) -> str:
        return (
            word.replace("{app_distro}", app_distro or "")
            .replace("{guest_user}", guest_user or "")
            .replace("{release}", release)
        )

    def ask(state: WslStateDef) -> RunResult:
        cached = seen.results.get(state.probe)
        if cached is not None:
            return cached
        argv = [substitute(word) for word in state.probe_argv]
        result = runner.run(argv, timeout_s=timeout_s)
        seen.results[state.probe] = result
        if state.probe == "wsl-list" and result.ok:
            seen.distros = parse_distro_names(result.stdout)
        return result

    for state in WSL_STATES:
        if state.optional:
            wanted = (state.code == "guest_no_network" and check_network) or (
                state.code == "pack_disk" and required_bytes > 0
            )
            if not wanted:
                continue
        if state.code == "foreign_distro_not_systemd" and (
            app_distro is None or app_distro == "crucible"
        ):
            continue
        means = MEANS.get(state.code)
        if means is None:
            raise HostError(
                "wsl_state_unknown",
                f"the generated table has a row {state.code!r} and this build has no "
                "predicate for it. Regenerate with `npm run gen:install` in "
                "sdk/bootstrap and add the predicate to crucible/host/wslstate.py; "
                "a row that can never match is a machine state nobody answers.",
            )
        result = ask(state)
        if not means(result, seen):
            continue
        return WslState(
            code=state.code,
            sentence=render(state.sentence, result, seen),
            action_kind=state.action_kind,
            action_argv=tuple(substitute(word) for word in state.action_argv),
            action_text=render(state.action_text, result, seen),
            action_url=render(state.action_url, result, seen),
            evidence=_said(result),
        )
    # Unreachable while `wsl_ready` matches everything. A table whose last row
    # stops being total is a bug, not a state.
    raise HostError(
        "wsl_state_unknown",
        f"no row of {list(WSL_STATE_CODES)} matched, and the table must be total",
    )


def elevated_argv(state: WslState) -> list[str]:
    """The PowerShell that runs a `run-elevated` action under a UAC prompt.

    The host runs this; bootstrap only ever spells it (`wsl-states.ts`:
    "a library that raises one from a probe is a library that pops a dialog
    nobody asked for"). The host is the process a person can SEE, started from
    their own tray menu, so the dialog has an author they recognise.
    """
    if state.action_kind != "run-elevated":
        raise HostError(
            "wsl_state_unknown",
            f"{state.code}'s action is {state.action_kind}, which is not elevated",
        )
    program, *rest = state.action_argv
    quoted = ",".join("'" + word.replace("'", "''") + "'" for word in rest)
    arguments = "" if quoted == "" else f" -ArgumentList {quoted}"
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        f"Start-Process -Verb RunAs -Wait -FilePath '{program}'{arguments}",
    ]
