from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import webbrowser
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from . import VERSION, controller_client, processlock, traylife
from .config import crucible_home, load_config, own_engine_backend
from .controller_client import (
    CONTROLLER_URL,
    ENGINE_URL,
    QUIT_SECONDS,
    LocalError,
    controller_start_failed,
    request,
    token_mismatch,
    wrong_controller,
)
from .errors import ConfigError, CrucibleError
from .pairing import parse_pairing_line
from .platform.installation import RECORD, installed_control, publish_installation, release_order
from .platform.installation import RELEASE_PATTERN as _RELEASE
from .platform.paths import LOG_NAME
from .protocol import API_VERSION, DOOR_PORT

TRAY_VERBS = ("tray", "install-desktop", "remove-desktop")

BROWSER_VERBS = ("open-console", "connect")

INFO_TIMEOUT = 15.0

SETTLE_POLL_SECONDS = 0.25


def connection(home: Path) -> tuple[str, str, str]:
    if sys.platform == "win32":
        path = home / "pairing"
        try:
            pair = parse_pairing_line(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise LocalError(f"local_pairing_invalid: {path}: {exc}") from exc
        return ENGINE_URL, pair.name, pair.token
    config = load_config(home)
    host = config.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.port}", config.name, config.token


def _stopped_on_purpose(token: str) -> dict:
    try:
        observed = request(CONTROLLER_URL + "/local/status", token=token)
    except (OSError, ValueError, LocalError):
        return {}
    if observed.get("state") == "stopped" and observed.get("intentional") is True:
        return {"state": "stopped", "detail": "Stopped by the operator"}
    return {}


def _service_state(home: Path) -> dict:
    from . import service

    config = load_config(home)
    observed = service.status(service.mechanism_for(config.backend_kind),
                              service.user_home(), runner=service.subprocess_runner)
    if not observed.installed:
        return {"state": "broken",
                "detail": "The service definition is missing; run `crucible service install` to write it"}
    if observed.running is False:
        return {"state": "stopped", "detail": observed.detail}
    return {}


def _not_answering(home: Path, result: dict, token: str, exc: BaseException) -> dict:
    result["detail"] = f"Engine did not answer: {exc}"
    result.update(_stopped_on_purpose(token) if sys.platform == "win32" else _service_state(home))
    return result


def _identified(result: dict, url: str, name: str, token: str) -> dict:
    try:
        info = request(url + "/v1/info", token=token, timeout=INFO_TIMEOUT)
    except urllib.error.HTTPError as exc:
        return dict(result, state="unauthorized" if exc.code in (401, 403) else "unhealthy",
                    detail=f"Engine info returned HTTP {exc.code}")
    except (OSError, ValueError, LocalError) as exc:
        return dict(result, state="unhealthy",
                    detail=f"The engine answered ping but not /v1/info: {exc}")
    server = info.get("server")
    if not isinstance(server, dict) or server.get("name") != name or server.get("api_version") != API_VERSION:
        return dict(result, state="wrong_service", detail="The engine returned incompatible or unexpected identity information")
    machine = info.get("host")
    return dict(result, state="running", version=server.get("version"),
                backend=(machine.get("backend") if isinstance(machine, dict) else None),
                detail="The paired engine is answering")


def status(home: Path | None = None) -> dict:
    home = home if home is not None else crucible_home()
    url, name, token = connection(home)
    result = {"schema_version": 1, "state": "unreachable", "name": name,
              "url": url, "detail": ""}
    try:
        ping = request(url + "/v1/ping")
    except urllib.error.HTTPError as exc:
        return dict(result, state="wrong_service" if exc.code == 404 else "unhealthy",
                    detail=f"The endpoint answered ping with HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return _not_answering(home, result, token, exc)
    except (ValueError, LocalError) as exc:
        return dict(result, state="wrong_service", detail=str(exc))
    if ping.get("crucible") is not True or ping.get("name") != name:
        return dict(result, state="wrong_service", detail="The endpoint is not the paired Crucible engine")
    return _identified(result, url, name, token)


def _spawn_controller(home: Path) -> None:
    controller_client.spawn(home)


def controller_ping() -> dict:
    return controller_client.ping(send=request)


def controller_answering() -> bool:
    return controller_client.answering(send=request)


def ensure_controller(home: Path, timeout: float = controller_client.START_SECONDS) -> None:
    controller_client.ensure_running(home, timeout=timeout, up=controller_answering,
                                     spawn=_spawn_controller)


def _guest_tokens(home: Path) -> list[str]:
    return controller_client.guest_tokens(home)


def _token_candidates(home: Path):
    return controller_client.token_candidates(home, guest_tokens=_guest_tokens)


def door_call(path: str, home: Path, token: str, *, method: str = "GET",
              timeout: float = controller_client.CALL_TIMEOUT_SECONDS) -> tuple[dict, str]:
    return controller_client.call(path, token, home=home, method=method, timeout=timeout,
                                  send=request, candidates=_token_candidates(home))


def _wait_for_pairing(home: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not (home / "pairing").exists():
        if time.monotonic() >= deadline:
            raise controller_start_failed(home, timeout)
        time.sleep(controller_client.POLL_SECONDS)


def _open_console(action: str, url: str, name: str, token: str) -> dict:
    section = "?section=connect" if action == "connect" else ""
    webbrowser.open(url + "/" + section + "#token=" + quote(token, safe=""))
    return {"schema_version": 1, "state": "opened", "name": name, "url": url,
            "detail": "Opened Crucible's console"}


def _ask_controller(action: str, home: Path, token: str, timeout: float) -> None:
    if action == "start":
        ensure_controller(home)
    else:
        try:
            controller_ping()
        except OSError as exc:
            raise LocalError(
                f"controller_unreachable: Crucible's controller is not answering on "
                f"port {DOOR_PORT} ({exc}), so nothing here can {action} its engine. "
                f"Its log is {home / LOG_NAME}. Run `crucible local start` to bring "
                "the controller back, then run this again"
            ) from exc
    door_call("/local/" + action, home, token, method="POST", timeout=timeout)


def _ask_service(action: str, home: Path) -> None:
    from . import service

    config = load_config(home)
    mechanism = service.mechanism_for(config.backend_kind)
    operation = service.start if action == "start" else service.stop
    operation(mechanism, home=service.user_home(), runner=service.subprocess_runner)


def _refuse_a_stale_engine(home: Path, observed: dict) -> None:
    running_version = observed.get("version")
    if not isinstance(running_version, str) or running_version == VERSION:
        return
    try:
        own_backend = own_engine_backend(home)
    except (ConfigError, OSError) as exc:
        raise LocalError(
            "local_config_unreadable: this installation's config "
            f"could not be read ({exc}), so whether the engine "
            f"answering ({running_version}) is its own cannot be "
            "known. Run `crucible doctor` to see what is wrong with it"
        ) from exc
    if own_backend is not None and observed.get("backend") in (None, own_backend):
        raise LocalError(
            f"engine_version_stale: the engine answering is "
            f"{running_version}, but this installation is {VERSION}. "
            "Its service was not restarted onto the new definition. Run "
            "`crucible local stop`, then `crucible local start`"
        )


def _reconciled_sharing(home: Path) -> dict:
    from .sharing import reconcile

    try:
        return reconcile(home)
    except (OSError, ValueError, RuntimeError, CrucibleError) as exc:
        return {"state": "degraded", "detail": str(exc), "remote_reachability": "not_tested"}


def _still_settling(action: str, home: Path, observed: dict, deadline: float) -> None:
    if observed["state"] == "unauthorized" and sys.platform == "win32":
        if time.monotonic() >= deadline:
            raise token_mismatch(home)
        return
    if observed["state"] in ("wrong_service", "unauthorized", "broken"):
        raise LocalError(f"{observed['state']}: {observed['detail']}")
    if time.monotonic() >= deadline:
        where = f"{home / 'logs'}" + (f" and {home / LOG_NAME}" if sys.platform == "win32" else "")
        raise LocalError(f"local_{action}_failed: {observed['detail']}. Why is in {where}")


def _settle(action: str, home: Path, timeout: float) -> dict:
    wanted = "running" if action == "start" else "stopped"
    deadline = time.monotonic() + timeout
    while True:
        observed = status(home)
        if observed["state"] == wanted:
            if action == "start":
                _refuse_a_stale_engine(home, observed)
                observed["sharing"] = _reconciled_sharing(home)
            return observed
        _still_settling(action, home, observed, deadline)
        time.sleep(SETTLE_POLL_SECONDS)


def run_engine_verb(action: str, home: Path | None = None, *, timeout: float = 60) -> dict:
    home = home if home is not None else crucible_home()
    if sys.platform == "win32" and action == "start" and not (home / "pairing").exists():
        ensure_controller(home)
        _wait_for_pairing(home, timeout)
    url, name, token = connection(home)
    if action in BROWSER_VERBS:
        return _open_console(action, url, name, token)
    if sys.platform == "win32":
        _ask_controller(action, home, token, timeout)
    else:
        _ask_service(action, home)
    return _settle(action, home, timeout)


def act(action: str, home: Path | None = None, *, timeout: float = 60) -> dict:
    return run_engine_verb(action, home, timeout=timeout)


def _shutdown_service(home: Path) -> None:
    from . import service

    try:
        config = load_config(home)
    except ConfigError:
        return
    observed = service.status(service.mechanism_for(config.backend_kind),
                              service.user_home(), runner=service.subprocess_runner)
    if observed.installed:
        act("stop")


def shutdown() -> None:
    home = crucible_home()
    traylife.close_tray(home)
    if sys.platform != "win32":
        _shutdown_service(home)
        return
    controller_client.shutdown_controller(
        home,
        send=lambda url, **options: request(url, **options),
        engine_token=lambda: connection(home)[2],
        call=lambda path, token: door_call(path, home, token),
        stop_engine=lambda: act("stop"),
        alive=lambda pid: processlock.alive(pid),
    )


def refusal_text(exc: BaseException) -> str:
    if not isinstance(exc, urllib.error.HTTPError):
        return str(exc)
    try:
        body = json.loads(exc.read().decode("utf-8", "replace"))
    except (ValueError, OSError):
        return str(exc)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return str(exc)
    code, message = error.get("code"), error.get("message")
    if not isinstance(code, str) or not isinstance(message, str):
        return str(exc)
    return f"{exc} - {code}: {message}"


said = refusal_text


def _no_tray_verbs(verb: str) -> None:
    raise LocalError(
        f"local_tray_unavailable: `crucible local {verb}` needs the tray, which "
        f"only the `crucible` command line wires in; run `crucible local {verb}`"
    )


def _answer(action: str, tray_verbs: Callable[[str], None]) -> dict | None:
    if action == "shutdown":
        shutdown()
        return {"closed": True}
    if action == "close-tray":
        traylife.close_tray(crucible_home())
        return {"closed": True}
    if action == "register":
        return {"installation": str(publish_installation(crucible_home()))}
    if action == "install-cli":
        from .launcher import install

        home = crucible_home()
        control = installed_control(home)
        return install(home, control["command"], control["cwd"])
    if action in TRAY_VERBS:
        tray_verbs(action)
        return None
    return status() if action == "status" else act(action)


def command(args: argparse.Namespace, tray_verbs: Callable[[str], None] = _no_tray_verbs) -> int:
    try:
        answer = _answer(args.local_action, tray_verbs)
    except (OSError, ValueError, RuntimeError, CrucibleError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": {"code": "local_failed", "message": refusal_text(exc)}}), file=sys.stderr)
        return 1
    if answer is not None:
        print(json.dumps(answer))
    return 0


def add_parser(subparsers, tray_verbs: Callable[[str], None] | None = None) -> None:
    parser = subparsers.add_parser("local", help="Local installation and service lifecycle")
    parser.add_argument("local_action", choices=["register", "status", "start", "stop",
                                                "open-console", "connect", "tray", "install-cli", "install-desktop", "remove-desktop", "close-tray", "shutdown"])
    parser.add_argument("--json", action="store_true", help="Structured output (always enabled)")
    if tray_verbs is None:
        parser.set_defaults(func=command)
    else:
        parser.set_defaults(func=lambda args: command(args, tray_verbs))


__all__ = [
    "CONTROLLER_URL", "ENGINE_URL", "LocalError", "QUIT_SECONDS", "RECORD", "_RELEASE",
    "act", "command", "connection", "door_call", "publish_installation", "release_order",
    "refusal_text", "run_engine_verb", "said", "shutdown", "status", "token_mismatch",
    "wrong_controller",
]
