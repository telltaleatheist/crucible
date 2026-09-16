"""Installed local lifecycle contract. Apps never know the runtime's layout.

installation.json records installation facts, not health or credentials. Its
control command owns platform decisions; clients validate its structured result.
"""
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
from . import VERSION
from .config import crucible_home, load_config
from .pairing import parse_pairing_line
from .errors import CrucibleError

RECORD = "installation.json"


class LocalError(RuntimeError):
    pass


def publish_installation(home: Path | None = None) -> Path:
    home = (home if home is not None else crucible_home()).resolve()
    # Preserve venv/bin/python itself: resolving its symlink selects the base
    # interpreter and loses the installed environment on source-built POSIX.
    executable = Path(sys.executable).absolute()
    # pythonw cannot return JSON to a pipe. It is only used to launch the UI.
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
    # Unique staging file: two clients may adopt the same installation at once.
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
    """The locally installed engine, never an arbitrary saved remote server."""
    if sys.platform == "win32":
        path = home / "pairing"
        try:
            pair = parse_pairing_line(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise LocalError(f"local_pairing_invalid: {path}: {exc}") from exc
        # The host publishes the engine's pairing; controller is separate.
        return "http://127.0.0.1:7100", pair.name, pair.token
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
            timeout: float = 3) -> dict:
    headers = {} if token is None else {"Authorization": f"Bearer {token}", "X-Crucible-Api": "1"}
    req = urllib.request.Request(url, headers=headers, method=method,
                                 data=b"{}" if method == "POST" else None)
    # Local service access must not depend on the invoking shell's proxy env.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise LocalError(f"local_protocol_invalid: {url} did not return an object")
    return value


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
        # A timeout is not proof that a process is stopped.
        result["detail"] = f"Engine did not answer: {exc}"
        if sys.platform == "win32":
            try:
                observed = request("http://127.0.0.1:7101/local/status", token=token)
                if observed.get("state") == "stopped" and observed.get("intentional") is True:
                    result["state"] = "stopped"
                    result["detail"] = "Stopped by the operator"
            except (OSError, ValueError, LocalError):
                pass  # Keep the original, explicit unreachable state.
        else:
            from . import service
            config = load_config(home)
            observed = service.status(service.mechanism_for(config.backend_kind),
                                      service.user_home(), runner=service.subprocess_runner)
            if not observed.installed:
                result.update(state="broken", detail="The service definition is missing")
            elif not observed.running and (observed.detail.startswith("inactive/") or
                                           "not loaded" in observed.detail):
                result.update(state="stopped", detail=observed.detail)
        return result
    except (ValueError, LocalError) as exc:
        return dict(result, state="wrong_service", detail=str(exc))
    if ping.get("crucible") is not True or ping.get("name") != name:
        return dict(result, state="wrong_service", detail="The endpoint is not the paired Crucible engine")
    try:
        info = request(url + "/v1/info", token=token)
    except urllib.error.HTTPError as exc:
        return dict(result, state="unauthorized" if exc.code in (401, 403) else "unhealthy",
                    detail=f"Engine info returned HTTP {exc.code}")
    except (OSError, ValueError, LocalError) as exc:
        return dict(result, state="unhealthy", detail=str(exc))
    server = info.get("server")
    if not isinstance(server, dict) or server.get("name") != name or server.get("api_version") != 1:
        return dict(result, state="wrong_service", detail="The engine returned incompatible or unexpected identity information")
    # The engine's own version travels with the observation. It is reported
    # rather than judged here, because `status` answers for a running server
    # generally and two versions coexisting is not by itself a fault. The
    # START path below is where a mismatch IS one.
    return dict(result, state="running", version=server.get("version"),
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


def controller_ping() -> dict:
    """An answering HTTP error is an occupied port, never a missing process."""
    try:
        observed = request("http://127.0.0.1:7101/v1/ping")
    except (urllib.error.HTTPError, ValueError) as exc:
        raise LocalError(f"wrong_controller: port 7101 returned an incompatible response: {exc}") from exc
    if observed.get("crucible") is not True or observed.get("role") != "orchestrator":
        raise LocalError("wrong_controller: another service occupies Crucible's control port")
    return observed


def act(action: str, home: Path | None = None, *, timeout: float = 60) -> dict:
    home = home if home is not None else crucible_home()
    if sys.platform == "win32" and action == "start" and not (home / "pairing").exists():
        # Fresh Windows install: controller initializes the native engine and
        # pairing before there is any credential with which to control it.
        try:
            controller_ping()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            _spawn_controller(home)
        deadline = time.monotonic() + timeout
        while not (home / "pairing").exists():
            if time.monotonic() >= deadline:
                raise LocalError("controller_start_failed: native engine did not initialize; inspect host.log")
            time.sleep(0.25)
    url, name, token = connection(home)
    if action in ("open-console", "connect"):
        section = "?section=connect" if action == "connect" else ""
        webbrowser.open(url + "/" + section + "#token=" + quote(token, safe=""))
        return {"schema_version": 1, "state": "opened", "name": name, "url": url,
                "detail": "Opened Crucible's console"}
    if sys.platform == "win32":
        endpoint = "http://127.0.0.1:7101"
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
                        raise LocalError("controller_start_failed: inspect host.log")
                    time.sleep(0.25)
        if ping.get("crucible") is not True or ping.get("role") != "orchestrator":
            raise LocalError("wrong_controller: another service occupies Crucible's control port")
        request(endpoint + "/local/" + action, token=token, method="POST", timeout=timeout)
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
                # THE ENGINE ANSWERING MUST BE THE ONE THIS RELEASE INSTALLED.
                # `systemctl enable --now` does nothing to an already-running
                # unit, so an upgrade that rewrote ExecStart can leave the OLD
                # executable serving while every line of the install says it
                # succeeded (measured 2026-09-16: a guest reported 0.6.0 from
                # the previous release's path after installing 0.6.3).
                # `service.install` now restarts a definition that moved; this
                # is the check that would have caught it either way, and it
                # catches any other route to the same stale process.
                #
                # ABSENT is not MISMATCHED: an engine too old to report its
                # version is not evidence of staleness, and refusing it would
                # be inventing a fault out of a missing key.
                running_version = observed.get("version")
                if isinstance(running_version, str) and running_version != VERSION:
                    raise LocalError(
                        f"engine_version_stale: the engine answering is "
                        f"{running_version}, but this installation is {VERSION}. "
                        "Its service was not restarted onto the new definition"
                    )
                from .sharing import reconcile
                try:
                    observed["sharing"] = reconcile(home)
                except (OSError, ValueError, RuntimeError, CrucibleError) as exc:
                    # Optional networking must not turn a healthy local engine
                    # into a failed start, but its failure must remain visible.
                    observed["sharing"] = {"state": "degraded", "detail": str(exc),
                                           "remote_reachability": "not_tested"}
            return observed
        if observed["state"] in ("wrong_service", "unauthorized", "broken"):
            raise LocalError(f"{observed['state']}: {observed['detail']}")
        if time.monotonic() >= deadline:
            raise LocalError(f"local_{action}_failed: {observed['detail']}")
        time.sleep(0.25)


def shutdown() -> None:
    """Quiesce before an upgrade; absence is accepted only after refused connects."""
    from .desktop import close_tray
    close_tray()
    home = crucible_home()
    if sys.platform == "win32":
        def refused(exc: BaseException) -> bool:
            import errno
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            return isinstance(reason, ConnectionRefusedError) or (
                isinstance(reason, OSError) and (reason.errno == errno.ECONNREFUSED or getattr(reason, "winerror", None) == 10061))
        try:
            ping = request("http://127.0.0.1:7101/v1/ping")
        except (urllib.error.URLError, ConnectionError) as exc:
            if not refused(exc):
                raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
            from .host.app import _alive
            pid = home / "host.pid"
            if pid.exists() and (raw := pid.read_text().strip()).isdigit() and _alive(int(raw)):
                raise LocalError("controller_shutdown_unknown: its recorded process is still alive")
            try:
                request("http://127.0.0.1:7100/v1/ping")
            except (urllib.error.URLError, ConnectionError) as engine_exc:
                if refused(engine_exc):
                    return
                raise LocalError(f"engine_shutdown_unknown: {engine_exc}") from engine_exc
            raise LocalError("engine_unmanaged: an engine is answering without its controller; it was not stopped")
        if ping.get("crucible") is not True or ping.get("role") != "orchestrator":
            raise LocalError("wrong_controller: port 7101 is occupied by another service")
        _, _, token = connection(home)
        info = request("http://127.0.0.1:7101/v1/info", token=token)
        server = info.get("server")
        if not isinstance(server, dict) or info.get("role") != "orchestrator" or server.get("api_version") != 1:
            raise LocalError("controller_upgrade_unsupported: authenticated controller identity is incompatible")
        release = server.get("version")
        lifecycle = info.get("local_lifecycle_version")
        supported = type(lifecycle) is int and lifecycle == 1
        legacy = lifecycle is None and release == "0.6.0"
        if not supported and not legacy:
            raise LocalError(f"controller_upgrade_unsupported: no supported shutdown contract "
                             f"for {release!r} (lifecycle {lifecycle!r})")
        engine = info.get("engine")
        owner = engine.get("owner") if isinstance(engine, dict) else None
        if owner not in (None, "child", "wsl-unit"):
            raise LocalError("controller_upgrade_unsupported: the controller does not own the answering engine")
        from .host.app import _alive
        pid_file = home / "host.pid"
        raw = pid_file.read_text().strip() if pid_file.is_file() else ""
        if not raw.isdigit():
            raise LocalError("controller_shutdown_unknown: the controller has no valid process record")
        controller_pid = int(raw)
        # 0.6.0 has authenticated /info and /quit, but no /local/stop. Its
        # documented quit stops its native child and releases its guest hold.
        # A guest unit uses a different runtime and survives the Windows swap.
        if supported:
            act("stop")
        request("http://127.0.0.1:7101/quit", token=token, method="POST")
        deadline = time.monotonic() + 15
        while True:
            closed = False
            try:
                request("http://127.0.0.1:7101/v1/ping")
            except (urllib.error.URLError, ConnectionError) as exc:
                if refused(exc):
                    closed = True
                else:
                    raise LocalError(f"controller_shutdown_unknown: {exc}") from exc
            if closed and not _alive(controller_pid):
                if owner != "wsl-unit":
                    try:
                        request("http://127.0.0.1:7100/v1/ping")
                    except (urllib.error.URLError, ConnectionError) as exc:
                        if refused(exc):
                            return
                        raise LocalError(f"engine_shutdown_unknown: {exc}") from exc
                    raise LocalError("engine_shutdown_failed: native engine still answers after its controller exited")
                return
            if time.monotonic() >= deadline:
                raise LocalError("controller_shutdown_failed: controller did not exit")
            time.sleep(0.1)
    else:
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
        print(json.dumps({"error": {"code": "local_failed", "message": str(exc)}}), file=sys.stderr)
        return 1


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("local", help="Local installation and service lifecycle")
    parser.add_argument("local_action", choices=["register", "status", "start", "stop",
                                                "open-console", "connect", "tray", "install-cli", "install-desktop", "remove-desktop", "close-tray", "shutdown"])
    parser.add_argument("--json", action="store_true", help="Structured output (always enabled)")
    parser.set_defaults(func=command)
