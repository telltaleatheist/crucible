"""The PREDICATES for 4c's table, and the walk that uses them.

`wsl_states.py` beside this file is GENERATED from
`sdk/bootstrap/src/wsl-states.ts` and holds the table's DATA — the codes, their
order, the probe argv, the sentences, the actions. What cannot be generated is
`means`: whether a probe's output MEANS a state is code, not data, and a
generator that emitted predicates would be emitting a second implementation.

So the predicates are written once here, keyed by the generated codes, and
`tests/test_host_wslstate.py` asserts that the two sets are exactly equal. That
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


# ------------------------------------------------ what wsl.exe's answer MEANS
#
# FRESH-INSTALL #10, #15, #17 (kylies-pc, 2026-09-26). Three readings of
# wsl.exe went wrong on one machine in one hour:
#
#   * before WSL was live, `wsl -l -v` printed wsl.exe's whole usage screen, and
#     the tray logged all of it as the "presence";
#   * after the features were enabled but before a restart had committed them,
#     wsl.exe answered `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED`, the table read
#     that as `wsl_missing`, and the resume ran `wsl --install` under UAC AGAIN,
#     which re-pended the very servicing transaction the restart was for;
#   * `Win32_OptionalFeature InstallState = 1` said "enabled" the whole time,
#     while CBS had the package at "Install Pending" and there was no lxss.sys.
#
# So "is WSL live" is keyed on what wsl.exe ANSWERS, by its machine-readable
# code, and "is a restart still owed" on what servicing says (CBS
# RebootPending, PendingFileRenameOperations). InstallState is read too, but
# only as a fact for the log (#11) and as a gate on the noisy pending signals:
# never on its own.

#: The machine-readable part of a wsl.exe refusal: `Error code:
#: Wsl/Service/WSL_E_DISTRO_NOT_FOUND` (measured on owens-pc, 2026-09-26), or
#: an HCS code. Codes are not localised; the sentences around them are.
WSL_ERROR_CODE = re.compile(r"\b((?:WSL|HCS)_E_[A-Z0-9_]+)\b")

#: The answers that mean "WSL is not live yet, and enabling it is what's owed".
COMPONENT_REQUIRED_CODES: frozenset[str] = frozenset(
    {"WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED", "WSL_E_OPTIONAL_COMPONENT_NOT_ENABLED"}
)

#: The answers that mean WSL IS live and simply has nothing registered.
NO_DISTRO_CODES: frozenset[str] = frozenset(
    {"WSL_E_DEFAULT_DISTRO_NOT_FOUND", "WSL_E_DISTRO_NOT_FOUND"}
)

#: Option names from wsl.exe's usage screen. They are not localised, and the
#: inbox stub on a machine without the feature prints the usage screen instead
#: of an answer. Three of them in one reply is the usage screen and not a
#: sentence that happens to name an option.
_USAGE_OPTIONS = ("--install", "--list", "--exec", "--distribution", "--shutdown", "--help")

#: The two features `wsl --install --no-distribution` enables.
WSL_FEATURES: tuple[str, ...] = ("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")

#: `Win32_OptionalFeature.InstallState`: 1 enabled, 2 disabled, 3 absent.
FEATURE_STATES = {1: "on", 2: "off", 3: "absent", 4: "unknown"}

FEATURE_QUERY_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class WslAnswer:
    """What ONE wsl.exe reply means, as a word, and the code it carried.

    `kind` is one of:
      live                the command worked
      no_distros          WSL is live and has nothing registered
      component_required  the features are not live (off, or awaiting a restart)
      stub                the inbox wsl.exe answered with its usage screen
      no_hypervisor       Windows cannot start a virtual machine
      error               another WSL_E_/HCS_E_ code
      unreadable          no code, no usage screen: nothing to key on
    """

    kind: str
    code: str = ""
    #: The first line of the reply, kept ONLY when there is no code to name
    #: (`unreadable`): one line of evidence, never the whole screen.
    first: str = ""

    @property
    def live(self) -> bool:
        return self.kind in ("live", "no_distros")

    def line(self) -> str:
        """One line for host.log. Never wsl.exe's own prose (#10, #17)."""
        words = {
            "live": "WSL is live",
            "no_distros": "WSL is live with no distributions",
            "component_required": "WSL is not live yet: its Windows features are off or waiting for a restart",
            "stub": "WSL is not installed yet (wsl.exe answered with its usage text)",
            "no_hypervisor": "Windows cannot start a virtual machine",
            "error": "wsl.exe refused",
            "unreadable": "wsl.exe gave no answer this build can read",
        }[self.kind]
        if self.code == "WSL_E_DISTRO_NOT_FOUND":
            words = "WSL is live and that distribution is not registered"
        if self.code:
            return f"{words} ({self.code})"
        return f"{words}: {self.first}" if self.first else words


def read_wsl_answer(result: RunResult) -> WslAnswer:
    """Classify a wsl.exe reply by its code, never by its localised sentence."""
    if result.ok:
        return WslAnswer("live")
    text = f"{result.stdout}\n{result.stderr}\n{result.failure or ''}".replace("\x00", "")
    found = WSL_ERROR_CODE.search(text)
    code = found.group(1) if found else ""
    if code in COMPONENT_REQUIRED_CODES:
        return WslAnswer("component_required", code)
    if code in NO_DISTRO_CODES:
        return WslAnswer("no_distros", code)
    if (
        code == "HCS_E_HYPERV_NOT_INSTALLED"
        or "0x80370102" in text.lower()
        or (not code and NO_HYPERVISOR.search(text))
    ):
        return WslAnswer("no_hypervisor", code or "HCS_E_HYPERV_NOT_INSTALLED")
    if code:
        return WslAnswer("error", code)
    if sum(1 for option in _USAGE_OPTIONS if option in text) >= 3:
        return WslAnswer("stub")
    # The pre-store inbox wsl.exe with the feature off says so by HRESULT.
    if "0x8007019e" in text.lower():
        return WslAnswer("component_required", "0x8007019e")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return WslAnswer("unreadable", first=first[:160])


def wsl_answer_line(result: RunResult) -> str:
    """`read_wsl_answer(result).line()`, for callers that only log."""
    return read_wsl_answer(result).line()


def feature_query_argv() -> list[str]:
    """InstallState of both WSL features, one `Name=State` per line. No admin.

    `Get-CimInstance Win32_OptionalFeature` answers a standard user (measured on
    owens-pc, 2026-09-26, 1.1 s); `Get-WindowsOptionalFeature` needs elevation.
    """
    names = " or ".join(f"Name='{name}'" for name in WSL_FEATURES)
    return [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f'Get-CimInstance Win32_OptionalFeature -Filter "{names}" | '
        "ForEach-Object { $_.Name + '=' + $_.InstallState }",
    ]


def parse_features(text: str) -> dict[str, int | None]:
    """`{feature: InstallState}`, None for a feature the query did not name."""
    states: dict[str, int | None] = {name: None for name in WSL_FEATURES}
    for raw in text.splitlines():
        name, _, value = raw.strip().partition("=")
        if name in states and value.strip().isdigit():
            states[name] = int(value.strip())
    return states


#: Where servicing says a restart is still owed. Read, never written.
CBS_REBOOT_PENDING_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
)
WU_REBOOT_REQUIRED_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
)
SESSION_MANAGER_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager"


def servicing_signals() -> tuple[str, ...] | None:
    """Which restart-owed signals Windows is raising, or None off Windows.

    `cbs` (Component Based Servicing has a RebootPending key) is the one that
    speaks for the WSL features: DISM sets it when an enable needs a restart to
    commit. `pending-renames` (PendingFileRenameOperations) and `windows-update`
    (RebootRequired) are generic, and on a live machine one of them is often
    set for somebody else's reasons (owens-pc had 70 characters of renames with
    WSL perfectly live, 2026-09-26), so `LiveWsl.restart_owed` counts them only
    while the features say they are on and wsl.exe says they are not live.
    """
    try:
        import winreg
    except ImportError:
        return None
    signals: list[str] = []
    for name, key in (("cbs", CBS_REBOOT_PENDING_KEY), ("windows-update", WU_REBOOT_REQUIRED_KEY)):
        try:
            winreg.CloseKey(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key))
            signals.append(name)
        except OSError:
            pass
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, SESSION_MANAGER_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "PendingFileRenameOperations")
            if value:
                signals.append("pending-renames")
    except OSError:
        pass
    return tuple(signals)


def booted_at() -> float | None:
    """When this machine last booted, as a Unix time, or None off Windows.

    `GetTickCount64` runs through sleep and through a Fast Startup "shut down",
    which is the point: a Fast Startup boot never commits servicing (#14), so it
    must not count as the restart Crucible asked for either.
    """
    try:
        import ctypes

        tick = ctypes.windll.kernel32.GetTickCount64  # type: ignore[attr-defined]
    except (ImportError, AttributeError, OSError):
        return None
    import time

    tick.restype = ctypes.c_ulonglong
    return time.time() - tick() / 1000.0


@dataclass(frozen=True)
class LiveWsl:
    """Is WSL live, and if not, is a restart what it is waiting for (#15)."""

    answer: WslAnswer
    #: `{feature: InstallState}`; None where the query could not say.
    features: dict[str, int | None]
    #: `servicing_signals()`, or None when there is no registry to ask.
    signals: tuple[str, ...] | None

    @property
    def live(self) -> bool:
        return self.answer.live

    @property
    def features_on(self) -> bool:
        return all(self.features.get(name) == 1 for name in WSL_FEATURES)

    @property
    def restart_owed(self) -> bool:
        """Enabled, not live, and servicing says it is waiting on a restart.

        CBS RebootPending on its own is enough once wsl.exe says the component
        is required: that is exactly the enable-then-restart gap. The generic
        signals count only when both features also report on, so a stray
        rename queued by some other program never stands in for "enable WSL".
        """
        if self.live or self.answer.kind not in ("component_required", "stub", "unreadable"):
            return False
        signals = self.signals or ()
        if "cbs" in signals:
            return True
        return self.features_on and bool(signals)

    def features_line(self) -> str:
        return ", ".join(
            f"{name} {FEATURE_STATES.get(state, 'unreadable') if state is not None else 'unreadable'}"
            for name, state in self.features.items()
        )

    def line(self) -> str:
        """One line for host.log and the event stream."""
        signals = "unreadable" if self.signals is None else (", ".join(self.signals) or "none")
        owed = "; a restart is still owed" if self.restart_owed else ""
        return (
            f"{self.answer.line()}; features: {self.features_line()}; "
            f"restart signals: {signals}{owed}"
        )


def probe_live(
    runner: Runner,
    *,
    status: RunResult | None = None,
    signals: Callable[[], tuple[str, ...] | None] = servicing_signals,
) -> LiveWsl:
    """`wsl --status` by its code, the features, and servicing's signals.

    `status` is the `wsl --status` reply when the caller already has one (the
    walk does), so the probe is not asked twice.
    """
    if status is None:
        status = runner.run(["wsl.exe", "--status"], timeout_s=PROBE_TIMEOUT_SECONDS)
    queried = runner.run(feature_query_argv(), timeout_s=FEATURE_QUERY_TIMEOUT_SECONDS)
    features = parse_features(queried.stdout) if queried.ok else {name: None for name in WSL_FEATURES}
    return LiveWsl(answer=read_wsl_answer(status), features=features, signals=signals())


@dataclass
class Evidence:
    """Everything the probes have answered so far, for rows that need two facts."""

    results: dict[str, RunResult]
    distros: list[str]
    #: What the caller measured, for the rows that are about numbers.
    required_bytes: int = 0
    app_distro: str | None = None
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
    # By the CODE first (#15): a component-required answer whose prose happens
    # to mention the virtual machine platform is a feature to enable, not a
    # firmware setting, and only the code tells them apart. The expression is
    # still the reading for a reply that carries no code at all.
    "virtualization_disabled": lambda result, _: read_wsl_answer(result).kind == "no_hypervisor",
    # By what wsl.exe ANSWERS (#15): a live WSL with no default distribution
    # can fail `--status` with WSL_E_DEFAULT_DISTRO_NOT_FOUND, and reading
    # that as "missing" would put a live machine through `wsl --install`.
    "wsl_missing": lambda result, _: not read_wsl_answer(result).live,
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
    "guest_no_disk": lambda result, seen: (
        _free_bytes(result) is not None and _free_bytes(result) < seen.required_bytes  # type: ignore[operator]
    ),
    "guest_root_unreachable": lambda result, _: not result.ok or result.stdout.strip() != "0",
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
    #: The generated row's own answer to "can the tray carry a machine past
    #: this without a person" (PHASE19 2.1). Carried through rather than
    #: re-derived from `action_kind`: `wsl_ready` instructs and is automatic,
    #: and a second derivation here would be the second opinion the field
    #: exists to remove.
    automatic: bool
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
        "{release}": seen.release,
        "{required}": gib(seen.required_bytes),
        "{free}": "an unreadable amount" if free is None else gib(free),
    }
    out = text
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def install_index_urls(release: str) -> list[str]:
    """Every place a first install downloads from, in the order it needs them.

    PHASE19-AUTOMATIC-WSL.md 2.12. THREE OWNERS, ASKED, and no fourth list:

      * `crucible/jobenv.py` reads the RECIPES under `crucible/envs/` — pip's
        default index, every `--index-url` / `--extra-index-url` /
        `--find-links` any recipe names, and the Hugging Face endpoint the
        weights come from.
      * `crucible/interpreter.py` holds the python-build-standalone pin, and
        its `url` is the exact file `install.sh` fetches.
      * `crucible/host/wsl_states.py` holds the release wheel's URL, generated
        from `release.ts`.

    A list written down here instead would drift the first time a recipe gained
    an index, and the drift would be INVISIBLE: the probe would go on passing
    and pip would go on failing minutes later with pip's own message.
    """
    from ..interpreter import SERVER_PYTHON, pin_for
    from ..jobenv import recipe_index_urls

    urls: list[str] = []
    for url in recipe_index_urls():
        if url not in urls:
            urls.append(url)
    # The guest is Linux; `GUEST_BACKEND` in installer.py is the one place that
    # word is decided, and this is the same backend's interpreter.
    interpreter = pin_for("cuda-linux", SERVER_PYTHON).url
    if interpreter not in urls:
        urls.append(interpreter)
    wheel = _WHEEL_URL_TEMPLATE.replace("{release}", release)
    if wheel not in urls:
        urls.append(wheel)
    return urls


#: The release wheel's URL, as the generated table already spells it in the
#: `guest_no_network` row's `action_url`. Taken from there rather than composed
#: again: `release.ts` owns that URL and the generator carries it across.
def _wheel_url_template() -> str:
    for state in WSL_STATES:
        if state.code == "guest_no_network":
            return state.action_url
    raise HostError(
        "wsl_state_unknown",
        "the generated table has no `guest_no_network` row, so nothing here "
        "knows which wheel a first install fetches.",
    )


_WHEEL_URL_TEMPLATE = _wheel_url_template()


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
        release=release,
    )

    #: Built ONCE, and only when the network row is actually going to be
    #: probed: reading the recipes touches the disk, and `wsl-states.ts`'s rule
    #: is that reading a machine's facts costs nothing it was not asked for.
    indexes = " ".join(install_index_urls(release)) if check_network else ""

    def substitute(word: str) -> str:
        return (
            word.replace("{app_distro}", app_distro or "")
            .replace("{release}", release)
            .replace("{indexes}", indexes)
        )

    def ask(state: WslStateDef) -> RunResult:
        cached = seen.results.get(state.probe)
        if cached is not None:
            return cached
        argv = [substitute(word) for word in state.probe_argv]
        result = runner.run(argv, timeout_s=timeout_s)
        seen.results[state.probe] = result
        if state.probe == "wsl-list":
            # The one reader (`presence.read_wsl_distros`): no distros is [],
            # and an unreadable list is not silently taken as empty.
            from .presence import read_wsl_distros

            seen.distros = read_wsl_distros(result) or []
        return result

    for state in WSL_STATES:
        if state.optional:
            wanted = (state.code == "guest_no_network" and check_network) or (
                state.code == "guest_no_disk" and required_bytes > 0
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
            automatic=state.automatic,
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
