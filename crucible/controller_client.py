from __future__ import annotations

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
from .platform.errors import HostError
from .platform.paths import INSTALL_ONE_LINER, LOG_NAME, door_url, engine_url
from .protocol import DOOR_PORT, api_headers
from .wsl import CRUCIBLE_DISTRO, pairing_argv

CONTROLLER_URL = door_url()
ENGINE_URL = engine_url()

START_SECONDS = 90.0
POLL_SECONDS = 0.25
CALL_TIMEOUT_SECONDS = 3.0
GUEST_READ_SECONDS = 60.0

ORCHESTRATOR_ROLE = "orchestrator"

Send = Callable[..., dict]


class LocalError(RuntimeError):
    pass


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
        raise LocalError(f"local_protocol_invalid: {url} did not return an object")
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
