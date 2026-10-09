from __future__ import annotations

import errno
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .pairing import parse_pairing_line
from .platform import hostconfig, portholder
from .platform.errors import HostError, LocalError
from .platform.paths import ENGINE_PORT, INSTALL_ONE_LINER, LOG_NAME, door_url, engine_url
from .protocol import API_VERSION, DOOR_PORT, HANDOVER_HEADER, api_headers
from .wsl import CRUCIBLE_DISTRO, pairing_argv

CONTROLLER_URL = door_url()
ENGINE_URL = engine_url()

START_SECONDS = 90.0
POLL_SECONDS = 0.25
CALL_TIMEOUT_SECONDS = 3.0
GUEST_READ_SECONDS = 60.0
# The shutdown's own reads. The controller answers /v1/info only after asking the engine
# behind it, so a 3 s client timeout on it gave up first: on the PC (2026-10-02, twice)
# the read took 3-4 s, `local shutdown` said "timed out" without ever sending /quit, and
# the installer started a second orchestrator beside the old one, which hung.
SHUTDOWN_READ_SECONDS = 30.0

ORCHESTRATOR_ROLE = "orchestrator"

UPGRADE_NEXT_STEP = (
    "Restart Windows so the controller starts again from this release, then run "
    "`crucible local shutdown` again"
)

Send = Callable[..., dict]


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def open_url(url: str, *, token: str | None = None, method: str = "GET",
             timeout: float = CALL_TIMEOUT_SECONDS, body: Any = None,
             headers: dict[str, str] | None = None):
    sent = {} if token is None else api_headers(token)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        sent["Content-Type"] = "application/json"
    elif method == "POST":
        data = b"{}"
    sent.update(headers or {})
    return opener().open(urllib.request.Request(url, headers=sent, method=method, data=data),
                         timeout=timeout)


def request(url: str, *, token: str | None = None, method: str = "GET",
            timeout: float = CALL_TIMEOUT_SECONDS, headers: dict[str, str] | None = None) -> dict:
    with open_url(url, token=token, method=method, timeout=timeout, headers=headers) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise LocalError(
            f"local_protocol_invalid: {url} answered with something other than a JSON "
            "object, so it is not a Crucible endpoint this build understands. Run "
            "`crucible local status` to see what is answering there"
        )
    return value


def is_orchestrator(answer: dict) -> bool:
    return answer.get("crucible") is True and answer.get("role") == ORCHESTRATOR_ROLE


def wrong_controller(detail: str = "") -> LocalError:
    said = f" ({detail})" if detail else ""
    return LocalError(
        f"wrong_controller: what answers on port {DOOR_PORT} is not Crucible's "
        f"controller{said}: {portholder.held_sentence(DOOR_PORT)}"
    )


def controller_start_failed(home: Path, timeout: float) -> LocalError:
    return LocalError(
        f"controller_start_failed: Crucible's controller was started but did not "
        f"answer on port {DOOR_PORT} within {timeout:.0f} s. Its log is "
        f"{Path(home) / LOG_NAME}. If it never starts, reinstall from PowerShell "
        f"with: {INSTALL_ONE_LINER}"
    )


def token_mismatch(home: Path) -> LocalError:
    home = Path(home)
    return LocalError(
        "engine_token_mismatch: Crucible's controller on this PC accepts none of "
        f"the engine tokens this PC holds: not the one in {home / 'pairing'} (the "
        f"line apps pair with), not [auth].token in {home / 'config.toml'} (the "
        "Windows engine's own), and not the one the Linux engine publishes in "
        f'~/.crucible/pairing inside the "{CRUCIBLE_DISTRO}" distro. Nothing was '
        f"stopped or changed. The controller's log, {home / LOG_NAME}, names the "
        "token it checks against. Restart the controller so it re-reads them: "
        "`crucible local shutdown`, then start Crucible from the Start menu (or "
        "sign out and back in). If that does not settle it, run the install "
        f"again from PowerShell: {INSTALL_ONE_LINER} — it carries the token into "
        "the guest with `crucible init --force --config-from` so both sides hold "
        "one token"
    )


class _NotTheController(Exception):
    pass


def _identify(send: Send) -> dict:
    try:
        observed = send(CONTROLLER_URL + "/v1/ping")
    except urllib.error.HTTPError as exc:
        raise _NotTheController(f"HTTP {exc.code}") from exc
    except OSError:
        raise
    except (ValueError, http.client.HTTPException) as exc:
        raise _NotTheController(str(exc)) from exc
    if not is_orchestrator(observed):
        raise _NotTheController("")
    return observed


def ping(*, send: Send = request) -> dict:
    try:
        return _identify(send)
    except _NotTheController as exc:
        raise wrong_controller(str(exc)) from exc


def is_up(*, send: Send = request) -> bool:
    try:
        _identify(send)
    except (OSError, LocalError, _NotTheController):
        return False
    return True


def spawn(home: Path) -> None:
    from .platform.packaged import refuse_packaged

    refuse_packaged("starting Crucible's controller")
    executable = Path(sys.executable)
    if executable.name.lower() == "python.exe":
        executable = executable.with_name("pythonw.exe")
    env = dict(os.environ, CRUCIBLE_HOME=str(home))
    subprocess.Popen([str(executable), "-m", "crucible.cli", "orchestrator", "--headless"],
                     env=env, cwd=str(Path(__file__).resolve().parent.parent), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def answering(*, send: Send = request) -> bool:
    try:
        ping(send=send)
    except OSError:
        return False
    return True


def ensure_running(home: Path, *, timeout: float = START_SECONDS,
                   up: Callable[[], bool] = answering,
                   spawn: Callable[[Path], None] = spawn,
                   on_start: Callable[[], None] | None = None) -> None:
    if up():
        return
    if on_start is not None:
        on_start()
    spawn(home)
    deadline = time.monotonic() + timeout
    while not up():
        if time.monotonic() >= deadline:
            raise controller_start_failed(home, timeout)
        time.sleep(POLL_SECONDS)


def pairing_token(home: Path) -> str | None:
    try:
        return parse_pairing_line((Path(home) / "pairing").read_text(encoding="utf-8").strip()).token
    except (OSError, ValueError):
        return None


def bearer(home: Path) -> str | None:
    return pairing_token(home) or hostconfig.read_token(home)


def guest_tokens(home: Path) -> list[str]:
    distros = [CRUCIBLE_DISTRO]
    try:
        named = hostconfig.consented_distro(home)
    except HostError:
        named = None
    if named and named not in distros:
        distros.append(named)
    tokens: list[str] = []
    for distro in distros:
        try:
            done = subprocess.run(
                pairing_argv(distro), capture_output=True, timeout=GUEST_READ_SECONDS,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode != 0:
            continue
        try:
            tokens.append(parse_pairing_line(done.stdout.decode("utf-8", "replace").strip()).token)
        except ValueError:
            continue
    return tokens


def token_candidates(home: Path, *,
                     guest_tokens: Callable[[Path], list[str]] = guest_tokens) -> Iterator[str]:
    for token in (pairing_token(home), hostconfig.read_token(home)):
        if token is not None:
            yield token
    yield from guest_tokens(home)


def call(path: str, token: str, *, home: Path, method: str = "GET",
         timeout: float = CALL_TIMEOUT_SECONDS, send: Send = request,
         candidates: Iterable[str] | None = None) -> tuple[dict, str]:
    try:
        return send(CONTROLLER_URL + path, token=token, method=method, timeout=timeout), token
    except urllib.error.HTTPError as exc:
        if exc.code != 401 or sys.platform != "win32":
            raise
    tried = {token}
    for candidate in token_candidates(home) if candidates is None else candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            return send(CONTROLLER_URL + path, token=candidate, method=method, timeout=timeout), candidate
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
    raise token_mismatch(home)


QUIT_SECONDS = 15.0
QUIT_POLL_SECONDS = 0.1
REFUSED_WINERRORS = (10061, 10054)
NATIVE_BACKEND = "llama-windows"
GUEST_OWNERS = ("wsl-unit", "found")
KNOWN_OWNERS = (None, "child") + GUEST_OWNERS


def connection_refused(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)):
        return True
    return isinstance(reason, OSError) and (
        reason.errno in (errno.ECONNREFUSED, errno.ECONNRESET)
        or getattr(reason, "winerror", None) in REFUSED_WINERRORS)


def recorded_pid(home: Path) -> int | None:
    pid_file = Path(home) / "host.pid"
    raw = pid_file.read_text().strip() if pid_file.is_file() else ""
    return int(raw) if raw.isdigit() else None


def _engine_closed(send: Send) -> bool:
    try:
        send(ENGINE_URL + "/v1/ping")
    except (urllib.error.URLError, ConnectionError) as exc:
        if connection_refused(exc):
            return True
        raise LocalError(f"engine_shutdown_unknown: {exc}") from exc
    return False


def _engine_backend(send: Send, engine_token: Callable[[], str]) -> str | None:
    try:
        info = send(ENGINE_URL + "/v1/info", token=engine_token())
        return info.get("host", {}).get("backend")
    except (LocalError, urllib.error.URLError, ConnectionError, ValueError, OSError):
        return None


def _refuse_an_orphaned_native_engine(send: Send, engine_token: Callable[[], str]) -> None:
    if _engine_closed(send):
        return
    backend = _engine_backend(send, engine_token)
    if backend is not None and backend != NATIVE_BACKEND:
        return
    raise LocalError(
        "engine_unmanaged: a native engine is answering without its "
        f"controller; it was not stopped. {portholder.held_sentence(ENGINE_PORT)}"
        if backend == NATIVE_BACKEND else
        "engine_unmanaged: an engine is answering without its controller "
        f"and could not be asked what it is; it was not stopped. {portholder.held_sentence(ENGINE_PORT)}"
    )


def _controller_absent(home: Path, exc: BaseException, *, send: Send,
                       engine_token: Callable[[], str], alive: Callable[[int], bool]) -> None:
    if not connection_refused(exc):
        raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
    pid = recorded_pid(home)
    if pid is not None and alive(pid):
        raise LocalError(
            f"controller_shutdown_unknown: the controller (pid {pid}) is alive but "
            f"not answering on port {DOOR_PORT}; its log is {Path(home) / LOG_NAME}. "
            "Wait a minute and run `crucible local shutdown` again; if it never "
            f"answers, end pid {pid} alone (not its tree) in Task Manager"
        )
    _refuse_an_orphaned_native_engine(send, engine_token)


def _shutdown_contract(call: Callable[[str, str], tuple[dict, str]], token: str) -> tuple[str | None, str]:
    info, token = call("/v1/info", token)
    server = info.get("server")
    if not isinstance(server, dict) or info.get("role") != ORCHESTRATOR_ROLE or server.get("api_version") != API_VERSION:
        raise LocalError("controller_upgrade_unsupported: authenticated controller identity "
                         f"is incompatible. {UPGRADE_NEXT_STEP}")
    release = server.get("version")
    lifecycle = info.get("local_lifecycle_version")
    if type(lifecycle) is not int or lifecycle != 1:
        raise LocalError(f"controller_upgrade_unsupported: no supported shutdown contract "
                         f"for {release!r} (lifecycle {lifecycle!r}). {UPGRADE_NEXT_STEP}")
    engine = info.get("engine")
    owner = engine.get("owner") if isinstance(engine, dict) else None
    if owner not in KNOWN_OWNERS:
        raise LocalError("controller_upgrade_unsupported: the controller does not own the "
                         f"answering engine. {UPGRADE_NEXT_STEP}")
    return owner, token


def _controller_closed(send: Send) -> bool:
    try:
        send(CONTROLLER_URL + "/v1/ping")
    except (urllib.error.URLError, ConnectionError) as exc:
        if connection_refused(exc):
            return True
        raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
    return False


def _wait_for_exit(home: Path, pid: int, *, send: Send, alive: Callable[[int], bool], native: bool) -> None:
    deadline = time.monotonic() + QUIT_SECONDS
    while True:
        if _controller_closed(send) and not alive(pid):
            if native and not _engine_closed(send):
                raise LocalError(
                    "engine_shutdown_failed: the native engine still answers after its "
                    f"controller exited. {portholder.held_sentence(ENGINE_PORT)}"
                )
            return
        if time.monotonic() >= deadline:
            raise LocalError(
                f"controller_shutdown_failed: the controller (pid {pid}) "
                f"was asked to quit and did not exit within {QUIT_SECONDS:.0f} s; its log is "
                f"{Path(home) / LOG_NAME}. Nothing was force-killed: its process tree "
                "holds the session that keeps the Linux engine's distro up. Run "
                "`crucible local shutdown` again; if it never exits, end pid "
                f"{pid} alone (not its tree) in Task Manager"
            )
        time.sleep(QUIT_POLL_SECONDS)


def shutdown_controller(home: Path, *, send: Send, engine_token: Callable[[], str],
                        call: Callable[[str, str], tuple[dict, str]],
                        stop_engine: Callable[[], object], alive: Callable[[int], bool]) -> None:
    try:
        answer = send(CONTROLLER_URL + "/v1/ping")
    except (urllib.error.URLError, ConnectionError) as exc:
        _controller_absent(home, exc, send=send, engine_token=engine_token, alive=alive)
        return
    if not is_orchestrator(answer):
        raise wrong_controller()
    owner, token = _shutdown_contract(call, engine_token())
    pid = recorded_pid(home)
    if pid is None:
        raise LocalError(
            f"controller_shutdown_unknown: the controller has no valid process record in "
            f"{Path(home) / 'host.pid'}; its log is {Path(home) / LOG_NAME}. Run "
            "`crucible local shutdown` again in a minute"
        )
    native = owner not in GUEST_OWNERS
    if native:
        stop_engine()
    send(CONTROLLER_URL + "/quit", token=token, method="POST", headers={HANDOVER_HEADER: "1"},
         timeout=SHUTDOWN_READ_SECONDS)
    _wait_for_exit(home, pid, send=send, alive=alive, native=native)
