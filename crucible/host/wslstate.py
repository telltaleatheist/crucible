from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Mapping

from .. import wsl
from ..memorybudget import gib_text
from ..platform.errors import HostError
from ..platform.powershell import query_argv, runas_argv
from ..platform.runner import Runner, RunResult
from ..platform.wsl_table import WSL_STATE_CODES, WSL_STATES, WslStateDef

NO_HYPERVISOR = re.compile(
    r"HCS_E_HYPERV_NOT_INSTALLED|0x80370102|hypervisor|virtual machine platform",
    re.IGNORECASE,
)

PROBE_TIMEOUT_SECONDS = 60.0


WSL_ERROR_CODE = re.compile(r"\b((?:WSL|HCS)_E_[A-Z0-9_]+)\b")

COMPONENT_REQUIRED_CODES: frozenset[str] = frozenset(
    {"WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED", "WSL_E_OPTIONAL_COMPONENT_NOT_ENABLED"}
)

NO_DISTRO_CODES: frozenset[str] = frozenset(
    {"WSL_E_DEFAULT_DISTRO_NOT_FOUND", "WSL_E_DISTRO_NOT_FOUND"}
)

_USAGE_OPTIONS = ("--install", "--list", "--exec", "--distribution", "--shutdown", "--help")

WSL_FEATURES: tuple[str, ...] = ("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")

FEATURE_STATES = {1: "on", 2: "off", 3: "absent", 4: "unknown"}

FEATURE_QUERY_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class WslAnswer:
    kind: str
    code: str = ""
    first: str = ""

    @property
    def live(self) -> bool:
        return self.kind in ("live", "no_distros")

    def line(self) -> str:
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
    if "0x8007019e" in text.lower():
        return WslAnswer("component_required", "0x8007019e")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return WslAnswer("unreadable", first=first[:160])


def sets_default_user(result: RunResult) -> bool | None:
    """Whether this wsl.exe has `--manage <distro> --set-default-user`, from its own usage text.

    None when the answer is not usage text at all: that is not a "no", and the caller
    must not treat it as one. The option names are not translated, so the text is read
    the same on every Windows display language."""
    text = f"{result.stdout}\n{result.stderr}".replace("\x00", "")
    if sum(1 for option in _USAGE_OPTIONS if option in text) < 3:
        return None
    return wsl.SET_DEFAULT_USER_FLAG in text


LXSS_KEY = r"Software\Microsoft\Windows\CurrentVersion\Lxss"


def registered_wsl_distros() -> list[str] | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, LXSS_KEY)
    except FileNotFoundError:
        return []
    with key:
        return _distribution_names(winreg, key)


def _distribution_names(winreg, key) -> list[str]:
    names: list[str] = []
    index = 0
    while True:
        try:
            sub = winreg.EnumKey(key, index)
        except OSError:
            return names
        index += 1
        try:
            with winreg.OpenKey(key, sub) as distro:
                names.append(str(winreg.QueryValueEx(distro, "DistributionName")[0]))
        except OSError:
            continue


def read_wsl_distros(result: object) -> list[str] | None:
    if getattr(result, "ok"):
        return wsl.parse_distro_list(getattr(result, "stdout"))
    if registered_wsl_distros() == []:
        return []
    return None


def wsl_answer_line(result: RunResult) -> str:
    return read_wsl_answer(result).line()


def feature_query_argv() -> list[str]:
    names = " or ".join(f"Name='{name}'" for name in WSL_FEATURES)
    return query_argv(
        f'Get-CimInstance Win32_OptionalFeature -Filter "{names}" | '
        "ForEach-Object { $_.Name + '=' + $_.InstallState }"
    )


def parse_features(text: str) -> dict[str, int | None]:
    states: dict[str, int | None] = {name: None for name in WSL_FEATURES}
    for raw in text.splitlines():
        name, _, value = raw.strip().partition("=")
        if name in states and value.strip().isdigit():
            states[name] = int(value.strip())
    return states


CBS_REBOOT_PENDING_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending"
)
WU_REBOOT_REQUIRED_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
)
SESSION_MANAGER_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager"


def servicing_signals() -> tuple[str, ...] | None:
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
    try:
        import ctypes

        tick = ctypes.windll.kernel32.GetTickCount64
    except (ImportError, AttributeError, OSError):
        return None
    import time

    tick.restype = ctypes.c_ulonglong
    return time.time() - tick() / 1000.0


@dataclass(frozen=True)
class LiveWsl:
    answer: WslAnswer
    features: dict[str, int | None]
    signals: tuple[str, ...] | None

    @property
    def live(self) -> bool:
        return self.answer.live

    @property
    def features_on(self) -> bool:
        return all(self.features.get(name) == 1 for name in WSL_FEATURES)

    @property
    def restart_owed(self) -> bool:
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
    if status is None:
        status = runner.run(wsl.status_argv(), timeout_s=PROBE_TIMEOUT_SECONDS)
    queried = runner.run(feature_query_argv(), timeout_s=FEATURE_QUERY_TIMEOUT_SECONDS)
    features = parse_features(queried.stdout) if queried.ok else {name: None for name in WSL_FEATURES}
    return LiveWsl(answer=read_wsl_answer(status), features=features, signals=signals())


@dataclass
class Evidence:
    results: dict[str, RunResult]
    distros: list[str]
    required_bytes: int = 0
    app_distro: str | None = None
    release: str = ""


def _free_bytes(result: RunResult) -> int | None:
    first = result.stdout.strip().split()
    if not first or not first[0].isdigit():
        return None
    return int(first[0]) * 1024


def _systemd_on(result: RunResult) -> bool:
    return re.search(r"systemd\s*=\s*true", result.stdout, re.IGNORECASE) is not None


Predicate = Callable[[RunResult, Evidence], bool]

MEANS: dict[str, Predicate] = {
    "virtualization_disabled": lambda result, _: read_wsl_answer(result).kind == "no_hypervisor",
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
        _free_bytes(result) is not None and _free_bytes(result) < seen.required_bytes
    ),
    "guest_root_unreachable": lambda result, _: not result.ok or result.stdout.strip() != "0",
    "wsl_ready": lambda _result, _seen: True,
}


@dataclass(frozen=True)
class WslState:
    code: str
    sentence: str
    action_kind: str
    action_argv: tuple[str, ...]
    action_text: str
    action_url: str
    automatic: bool
    evidence: str


def render(text: str, result: RunResult, seen: Evidence) -> str:
    free = _free_bytes(result)
    replacements: Mapping[str, str] = {
        "{said}": result.output_tail(),
        "{app_distro}": seen.app_distro or "",
        "{release}": seen.release,
        "{required}": gib_text(seen.required_bytes),
        "{free}": "an unreadable amount" if free is None else gib_text(free),
    }
    out = text
    for key, value in replacements.items():
        out = out.replace(key, value)
    return out


def install_index_urls(release: str) -> list[str]:
    from ..interpreter import SERVER_PYTHON, pin_for
    from ..jobenv import recipe_index_urls

    urls: list[str] = []
    for url in recipe_index_urls():
        if url not in urls:
            urls.append(url)
    interpreter = pin_for("cuda-linux", SERVER_PYTHON).url
    if interpreter not in urls:
        urls.append(interpreter)
    wheel = GUEST_WHEEL_URL_TEMPLATE.replace("{release}", release)
    if wheel not in urls:
        urls.append(wheel)
    return urls


NETWORK_ROW = "guest_no_network"


def state_row(code: str) -> WslStateDef:
    for state in WSL_STATES:
        if state.code == code:
            return state
    raise HostError(
        "wsl_state_unknown",
        f"the generated table has no `{code}` row, so nothing here "
        "knows which wheel a first install fetches.",
    )


GUEST_WHEEL_URL_TEMPLATE = state_row(NETWORK_ROW).action_url


class _Probes:
    def __init__(
        self, runner: Runner, seen: Evidence, *, indexes: str, timeout_s: float
    ) -> None:
        self._runner = runner
        self._seen = seen
        self._indexes = indexes
        self._timeout_s = timeout_s

    def substitute(self, word: str) -> str:
        return (
            word.replace("{app_distro}", self._seen.app_distro or "")
            .replace("{release}", self._seen.release)
            .replace("{indexes}", self._indexes)
        )

    def ask(self, state: WslStateDef) -> RunResult:
        cached = self._seen.results.get(state.probe)
        if cached is not None:
            return cached
        argv = [self.substitute(word) for word in state.probe_argv]
        result = self._runner.run(argv, timeout_s=self._timeout_s)
        self._seen.results[state.probe] = result
        if state.probe == "wsl-list":
            self._seen.distros = read_wsl_distros(result) or []
        return result

    def matched(self, state: WslStateDef, result: RunResult) -> WslState:
        return WslState(
            code=state.code,
            sentence=render(state.sentence, result, self._seen),
            action_kind=state.action_kind,
            action_argv=tuple(self.substitute(word) for word in state.action_argv),
            action_text=render(state.action_text, result, self._seen),
            action_url=render(state.action_url, result, self._seen),
            automatic=state.automatic,
            evidence=result.output_tail(),
        )


def _row_applies(
    state: WslStateDef, *, check_network: bool, required_bytes: int, app_distro: str | None
) -> bool:
    if state.optional:
        wanted = (state.code == NETWORK_ROW and check_network) or (
            state.code == "guest_no_disk" and required_bytes > 0
        )
        if not wanted:
            return False
    return not (
        state.code == "foreign_distro_not_systemd"
        and (app_distro is None or app_distro == "crucible")
    )


def _predicate(state: WslStateDef) -> Predicate:
    means = MEANS.get(state.code)
    if means is None:
        raise HostError(
            "wsl_state_unknown",
            f"the generated table has a row {state.code!r} and this build has no "
            "predicate for it. Regenerate with `npm run gen:install` in "
            "sdk/bootstrap and add the predicate to crucible/host/wslstate.py; "
            "a row that can never match is a machine state nobody answers.",
        )
    return means


def detect(
    runner: Runner,
    *,
    release: str,
    required_bytes: int = 0,
    app_distro: str | None = None,
    check_network: bool = False,
    timeout_s: float = PROBE_TIMEOUT_SECONDS,
) -> WslState:
    seen = Evidence(
        results={},
        distros=[],
        required_bytes=required_bytes,
        app_distro=app_distro,
        release=release,
    )
    indexes = " ".join(install_index_urls(release)) if check_network else ""
    probes = _Probes(runner, seen, indexes=indexes, timeout_s=timeout_s)
    for state in WSL_STATES:
        if not _row_applies(
            state, check_network=check_network, required_bytes=required_bytes, app_distro=app_distro
        ):
            continue
        means = _predicate(state)
        result = probes.ask(state)
        if means(result, seen):
            return probes.matched(state, result)
    raise HostError(
        "wsl_state_unknown",
        f"no row of {list(WSL_STATE_CODES)} matched, and the table must be total",
    )


def elevated_argv(state: WslState) -> list[str]:
    if state.action_kind != "run-elevated":
        raise HostError(
            "wsl_state_unknown",
            f"{state.code}'s action is {state.action_kind}, which is not elevated",
        )
    return runas_argv(state.action_argv)
