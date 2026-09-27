from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import webbrowser
from pathlib import Path
from urllib.parse import quote

from . import VERSION, controller_client, processlock, traylife
from .atomicjson import write_json
from .config import crucible_home, load_config, own_engine_backend
from .controller_client import (
    CONTROLLER_URL,
    ENGINE_URL,
    LocalError,
    controller_start_failed,
    request,
    token_mismatch,
    wrong_controller,
)
from .errors import ConfigError, CrucibleError
from .pairing import parse_pairing_line
from .platform.paths import LOG_NAME
from .protocol import API_VERSION, DOOR_PORT, HANDOVER_HEADER

RECORD = "installation.json"

QUIT_SECONDS = 15.0

_RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def release_order(left: str, right: str) -> int:
    numbers = []
    for value in (left, right):
        match = _RELEASE.match(value.strip())
        if match is None:
            raise LocalError(
                f"release_unreadable: {value!r} is not a Crucible release, so it "
                "cannot be compared with one"
            )
        numbers.append(tuple(int(part) for part in match.groups()))
    first, second = numbers
    if first == second:
        return 0
    return -1 if first < second else 1


def publish_installation(home: Path | None = None) -> Path:
    home = (home if home is not None else crucible_home()).resolve()
    executable = Path(sys.executable).absolute()
    if executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    if not executable.is_file():
        raise LocalError(f"local_runtime_missing: {executable}")
    record = {
        "schema_version": 1, "platform": sys.platform, "release": VERSION,
        "home": str(home),
        "control": {"command": str(executable), "args": ["-m", "crucible.cli", "local"],
                    "cwd": str(Path(__file__).resolve().parent.parent)},
    }
    return write_json(home / RECORD, record, private=True)


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


INFO_TIMEOUT = 15.0


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
        result["detail"] = f"Engine did not answer: {exc}"
        if sys.platform == "win32":
            try:
                observed = request(CONTROLLER_URL + "/local/status", token=token)
                if observed.get("state") == "stopped" and observed.get("intentional") is True:
                    result["state"] = "stopped"
                    result["detail"] = "Stopped by the operator"
            except (OSError, ValueError, LocalError):
                pass
        else:
            from . import service
            config = load_config(home)
            observed = service.status(service.mechanism_for(config.backend_kind),
                                      service.user_home(), runner=service.subprocess_runner)
            if not observed.installed:
                result.update(state="broken", detail="The service definition is missing")
            elif observed.running is False:
                result.update(state="stopped", detail=observed.detail)
        return result
    except (ValueError, LocalError) as exc:
        return dict(result, state="wrong_service", detail=str(exc))
    if ping.get("crucible") is not True or ping.get("name") != name:
        return dict(result, state="wrong_service", detail="The endpoint is not the paired Crucible engine")
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


def act(action: str, home: Path | None = None, *, timeout: float = 60) -> dict:
    home = home if home is not None else crucible_home()
    if sys.platform == "win32" and action == "start" and not (home / "pairing").exists():
        ensure_controller(home)
        _wait_for_pairing(home, timeout)
    url, name, token = connection(home)
    if action in ("open-console", "connect"):
        section = "?section=connect" if action == "connect" else ""
        webbrowser.open(url + "/" + section + "#token=" + quote(token, safe=""))
        return {"schema_version": 1, "state": "opened", "name": name, "url": url,
                "detail": "Opened Crucible's console"}
    if sys.platform == "win32":
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
    else:
        from . import service
        config = load_config(home)
        mechanism = service.mechanism_for(config.backend_kind)
        operation = service.start if action == "start" else service.stop
        operation(mechanism, home=service.user_home(), runner=service.subprocess_runner)
    deadline = time.monotonic() + timeout
    while True:
        observed = status(home)
        if observed["state"] == ("running" if action == "start" else "stopped"):
            if action == "start":
                running_version = observed.get("version")
                if isinstance(running_version, str) and running_version != VERSION:
                    try:
                        own_backend = own_engine_backend(home)
                    except (ConfigError, OSError) as exc:
                        raise LocalError(
                            "local_config_unreadable: this installation's config "
                            f"could not be read ({exc}), so whether the engine "
                            f"answering ({running_version}) is its own cannot be "
                            "known"
                        ) from exc
                    ours = own_backend is not None and observed.get("backend") in (
                        None,
                        own_backend,
                    )
                    if ours:
                        raise LocalError(
                            f"engine_version_stale: the engine answering is "
                            f"{running_version}, but this installation is {VERSION}. "
                            "Its service was not restarted onto the new definition"
                        )
                from .sharing import reconcile
                try:
                    observed["sharing"] = reconcile(home)
                except (OSError, ValueError, RuntimeError, CrucibleError) as exc:
                    observed["sharing"] = {"state": "degraded", "detail": str(exc),
                                           "remote_reachability": "not_tested"}
            return observed
        if observed["state"] == "unauthorized" and sys.platform == "win32":
            if time.monotonic() >= deadline:
                raise token_mismatch(home)
            time.sleep(0.25)
            continue
        if observed["state"] in ("wrong_service", "unauthorized", "broken"):
            raise LocalError(f"{observed['state']}: {observed['detail']}")
        if time.monotonic() >= deadline:
            raise LocalError(f"local_{action}_failed: {observed['detail']}")
        time.sleep(0.25)


def shutdown() -> None:
    home = crucible_home()
    traylife.close_tray(home)
    if sys.platform == "win32":
        def refused(exc: BaseException) -> bool:
            import errno
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, (ConnectionRefusedError, ConnectionResetError)):
                return True
            return isinstance(reason, OSError) and (
                reason.errno in (errno.ECONNREFUSED, errno.ECONNRESET)
                or getattr(reason, "winerror", None) in (10061, 10054))
        try:
            ping = request(CONTROLLER_URL + "/v1/ping")
        except (urllib.error.URLError, ConnectionError) as exc:
            if not refused(exc):
                raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
            pid = home / "host.pid"
            if pid.exists() and (raw := pid.read_text().strip()).isdigit() and processlock.alive(int(raw)):
                raise LocalError(
                    f"controller_shutdown_unknown: the controller (pid {raw}) is alive but "
                    f"not answering on port {DOOR_PORT}; its log is {home / LOG_NAME}. "
                    "Wait a minute and run `crucible local shutdown` again; if it never "
                    f"answers, end pid {raw} alone (not its tree) in Task Manager"
                )
            try:
                request(ENGINE_URL + "/v1/ping")
            except (urllib.error.URLError, ConnectionError) as engine_exc:
                if refused(engine_exc):
                    return
                raise LocalError(f"engine_shutdown_unknown: {engine_exc}") from engine_exc
            try:
                _, _, engine_token = connection(home)
                backend = (request(ENGINE_URL + "/v1/info", token=engine_token)
                           .get("host", {}).get("backend"))
            except (LocalError, urllib.error.URLError, ConnectionError, ValueError, OSError):
                backend = None
            if backend is not None and backend != "llama-windows":
                return
            raise LocalError(
                "engine_unmanaged: a native engine is answering without its "
                "controller; it was not stopped"
                if backend == "llama-windows" else
                "engine_unmanaged: an engine is answering without its controller "
                "and could not be asked what it is; it was not stopped"
            )
        if not controller_client.is_orchestrator(ping):
            raise wrong_controller()
        _, _, token = connection(home)
        info, token = door_call("/v1/info", home, token)
        server = info.get("server")
        if not isinstance(server, dict) or info.get("role") != "orchestrator" or server.get("api_version") != API_VERSION:
            raise LocalError("controller_upgrade_unsupported: authenticated controller identity is incompatible")
        release = server.get("version")
        lifecycle = info.get("local_lifecycle_version")
        if type(lifecycle) is not int or lifecycle != 1:
            raise LocalError(f"controller_upgrade_unsupported: no supported shutdown contract "
                             f"for {release!r} (lifecycle {lifecycle!r})")
        engine = info.get("engine")
        owner = engine.get("owner") if isinstance(engine, dict) else None
        if owner not in (None, "child", "wsl-unit", "found"):
            raise LocalError("controller_upgrade_unsupported: the controller does not own the answering engine")
        pid_file = home / "host.pid"
        raw = pid_file.read_text().strip() if pid_file.is_file() else ""
        if not raw.isdigit():
            raise LocalError("controller_shutdown_unknown: the controller has no valid process record")
        controller_pid = int(raw)
        if owner not in ("wsl-unit", "found"):
            act("stop")
        request(CONTROLLER_URL + "/quit", token=token, method="POST",
                headers={HANDOVER_HEADER: "1"})
        deadline = time.monotonic() + QUIT_SECONDS
        while True:
            closed = False
            try:
                request(CONTROLLER_URL + "/v1/ping")
            except (urllib.error.URLError, ConnectionError) as exc:
                if refused(exc):
                    closed = True
                else:
                    raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
            if closed and not processlock.alive(controller_pid):
                if owner not in ("wsl-unit", "found"):
                    try:
                        request(ENGINE_URL + "/v1/ping")
                    except (urllib.error.URLError, ConnectionError) as exc:
                        if refused(exc):
                            return
                        raise LocalError(f"engine_shutdown_unknown: {exc}") from exc
                    raise LocalError("engine_shutdown_failed: native engine still answers after its controller exited")
                return
            if time.monotonic() >= deadline:
                raise LocalError(
                    f"controller_shutdown_failed: the controller (pid {controller_pid}) "
                    f"was asked to quit and did not exit within {QUIT_SECONDS:.0f} s; its log is "
                    f"{home / LOG_NAME}. Nothing was force-killed: its process tree "
                    "holds the session that keeps the Linux engine's distro up. Run "
                    "`crucible local shutdown` again; if it never exits, end pid "
                    f"{controller_pid} alone (not its tree) in Task Manager"
                )
            time.sleep(0.1)
    else:
        from . import service
        try:
            config = load_config(home)
        except ConfigError:
            return
        observed = service.status(service.mechanism_for(config.backend_kind),
                                  service.user_home(), runner=service.subprocess_runner)
        if not observed.installed:
            return
        act("stop")


def command(args: argparse.Namespace) -> int:
    try:
        if args.local_action in ("shutdown", "close-tray"):
            if args.local_action == "shutdown":
                shutdown()
            else:
                traylife.close_tray(crucible_home())
            print(json.dumps({"closed": True}))
            return 0
        if args.local_action == "register":
            path = publish_installation()
            print(json.dumps({"installation": str(path)}))
            return 0
        if args.local_action == "install-cli":
            from .launcher import install
            home = crucible_home()
            record = json.loads((home / RECORD).read_text(encoding="utf-8"))
            print(json.dumps(install(home, record["control"]["command"], record["control"]["cwd"])))
            return 0
        if args.local_action in ("tray", "install-desktop", "remove-desktop"):
            from . import desktop
            getattr(desktop, args.local_action.replace("-", "_"))()
            return 0
        result = status() if args.local_action == "status" else act(args.local_action)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, RuntimeError, CrucibleError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": {"code": "local_failed", "message": said(exc)}}), file=sys.stderr)
        return 1


def said(exc: BaseException) -> str:
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


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("local", help="Local installation and service lifecycle")
    parser.add_argument("local_action", choices=["register", "status", "start", "stop",
                                                "open-console", "connect", "tray", "install-cli", "install-desktop", "remove-desktop", "close-tray", "shutdown"])
    parser.add_argument("--json", action="store_true", help="Structured output (always enabled)")
    parser.set_defaults(func=command)
