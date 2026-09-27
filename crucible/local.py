from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from urllib.parse import quote

from . import VERSION
from .config import crucible_home, load_config, own_engine_backend
from .pairing import parse_pairing_line
from .errors import ConfigError, CrucibleError
from .host.paths import DOOR_PORT, INSTALL_ONE_LINER, LOG_NAME, door_url, engine_url
from .host.wsl_states import CRUCIBLE_DISTRO

RECORD = "installation.json"


class LocalError(RuntimeError):
    pass


_RELEASE = __import__("re").compile(r"^v?(\d+)\.(\d+)\.(\d+)")


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
    home.mkdir(parents=True, exist_ok=True)
    path = home / RECORD
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=home,
                                     prefix="installation-", suffix=".tmp", delete=False) as f:
        json.dump(record, f, indent=2)
        f.write("\n")
        staged = Path(f.name)
    try:
        staged.chmod(0o600)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
    return path


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


def request(url: str, *, token: str | None = None, method: str = "GET",
            timeout: float = 3, headers: dict[str, str] | None = None) -> dict:
    sent = {} if token is None else {"Authorization": f"Bearer {token}", "X-Crucible-Api": "1"}
    sent.update(headers or {})
    req = urllib.request.Request(url, headers=sent, method=method,
                                 data=b"{}" if method == "POST" else None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise LocalError(f"local_protocol_invalid: {url} did not return an object")
    return value


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
    if not isinstance(server, dict) or server.get("name") != name or server.get("api_version") != 1:
        return dict(result, state="wrong_service", detail="The engine returned incompatible or unexpected identity information")
    machine = info.get("host")
    return dict(result, state="running", version=server.get("version"),
                backend=(machine.get("backend") if isinstance(machine, dict) else None),
                detail="The paired engine is answering")


def _spawn_controller(home: Path) -> None:
    executable = Path(sys.executable)
    if executable.name.lower() == "python.exe":
        executable = executable.with_name("pythonw.exe")
    env = dict(os.environ, CRUCIBLE_HOME=str(home))
    subprocess.Popen([str(executable), "-m", "crucible.cli", "orchestrator", "--headless"],
                     env=env, cwd=str(Path(__file__).resolve().parent.parent), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def wrong_controller(detail: str = "") -> LocalError:
    from .host.portholder import held_sentence

    said = f" ({detail})" if detail else ""
    return LocalError(
        f"wrong_controller: what answers on port {DOOR_PORT} is not Crucible's "
        f"controller{said}: {held_sentence(DOOR_PORT)}"
    )


def controller_start_failed(home: Path, timeout: float) -> LocalError:
    return LocalError(
        f"controller_start_failed: Crucible's controller was started but did not "
        f"answer on port {DOOR_PORT} within {timeout:.0f} s. Its log is "
        f"{Path(home) / LOG_NAME}. If it never starts, reinstall from PowerShell "
        f"with: {INSTALL_ONE_LINER}"
    )


def controller_ping() -> dict:
    try:
        observed = request(CONTROLLER_URL + "/v1/ping")
    except (urllib.error.HTTPError, ValueError) as exc:
        raise wrong_controller(f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else str(exc)) from exc
    if observed.get("crucible") is not True or observed.get("role") != "orchestrator":
        raise wrong_controller()
    return observed


HANDOVER_HEADER = "X-Crucible-Handover"

CONTROLLER_URL = door_url()

ENGINE_URL = engine_url()


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


def _guest_tokens(home: Path) -> list[str]:
    from .host.app import consented_distro
    from .host.errors import HostError
    from .host.presence import guest_pairing_argv
    from .host.wsl_states import CRUCIBLE_DISTRO

    distros = [CRUCIBLE_DISTRO]
    try:
        named = consented_distro(home)
    except HostError:
        named = None
    if named and named not in distros:
        distros.append(named)
    tokens: list[str] = []
    for distro in distros:
        try:
            done = subprocess.run(
                guest_pairing_argv(distro), capture_output=True, timeout=60,
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


def _token_candidates(home: Path):
    try:
        yield connection(home)[2]
    except LocalError:
        pass
    from .host.app import read_token

    configured = read_token(home)
    if configured is not None:
        yield configured
    yield from _guest_tokens(home)


def door_call(path: str, home: Path, token: str, *, method: str = "GET",
              timeout: float = 3) -> tuple[dict, str]:
    try:
        return request(CONTROLLER_URL + path, token=token, method=method, timeout=timeout), token
    except urllib.error.HTTPError as exc:
        if exc.code != 401 or sys.platform != "win32":
            raise
    tried = {token}
    for candidate in _token_candidates(home):
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            return (request(CONTROLLER_URL + path, token=candidate, method=method,
                            timeout=timeout), candidate)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
    raise token_mismatch(home)


def act(action: str, home: Path | None = None, *, timeout: float = 60) -> dict:
    home = home if home is not None else crucible_home()
    if sys.platform == "win32" and action == "start" and not (home / "pairing").exists():
        try:
            controller_ping()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            _spawn_controller(home)
        deadline = time.monotonic() + timeout
        while not (home / "pairing").exists():
            if time.monotonic() >= deadline:
                raise controller_start_failed(home, timeout)
            time.sleep(0.25)
    url, name, token = connection(home)
    if action in ("open-console", "connect"):
        section = "?section=connect" if action == "connect" else ""
        webbrowser.open(url + "/" + section + "#token=" + quote(token, safe=""))
        return {"schema_version": 1, "state": "opened", "name": name, "url": url,
                "detail": "Opened Crucible's console"}
    if sys.platform == "win32":
        try:
            ping = controller_ping()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if action != "start":
                raise LocalError("controller_unreachable: Start Crucible's controller before stopping its engine")
            _spawn_controller(home)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    ping = controller_ping()
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if time.monotonic() >= deadline:
                        raise controller_start_failed(home, timeout)
                    time.sleep(0.25)
        if ping.get("crucible") is not True or ping.get("role") != "orchestrator":
            raise wrong_controller()
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
    from .desktop import close_tray
    close_tray()
    home = crucible_home()
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
            from .host.app import _alive
            pid = home / "host.pid"
            if pid.exists() and (raw := pid.read_text().strip()).isdigit() and _alive(int(raw)):
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
        if ping.get("crucible") is not True or ping.get("role") != "orchestrator":
            raise wrong_controller()
        _, _, token = connection(home)
        info, token = door_call("/v1/info", home, token)
        server = info.get("server")
        if not isinstance(server, dict) or info.get("role") != "orchestrator" or server.get("api_version") != 1:
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
        from .host.app import _alive
        pid_file = home / "host.pid"
        raw = pid_file.read_text().strip() if pid_file.is_file() else ""
        if not raw.isdigit():
            raise LocalError("controller_shutdown_unknown: the controller has no valid process record")
        controller_pid = int(raw)
        if owner not in ("wsl-unit", "found"):
            act("stop")
        request(CONTROLLER_URL + "/quit", token=token, method="POST",
                headers={HANDOVER_HEADER: "1"})
        deadline = time.monotonic() + 15
        while True:
            closed = False
            try:
                request(CONTROLLER_URL + "/v1/ping")
            except (urllib.error.URLError, ConnectionError) as exc:
                if refused(exc):
                    closed = True
                else:
                    raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
            if closed and not _alive(controller_pid):
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
                    f"was asked to quit and did not exit within 15 s; its log is "
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
            from .desktop import close_tray
            if args.local_action == "shutdown":
                shutdown()
            else:
                close_tray()
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
