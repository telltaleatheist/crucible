"""`POST /v1/server/restart`: this server starting itself again, through what runs it.

A server cannot start itself after it has stopped, so the restart belongs to whatever
runs it. Three cases, each answered by name:

- A Windows host started it (`$CRUCIBLE_HOST_DOOR` is set): the host restarts its
  engine, the same `engine-restart` task the page already hands it.
- systemd or launchd runs it, as `crucible service install` defines it: the server
  stops cleanly and exits SELF_RESTART_EXIT, and the manager starts it again (the unit's
  `Restart=always`, the agent's KeepAlive). No root and no `systemctl` call: the
  restart policy those definitions carry is the door, and it is checked before the
  server leaves, not assumed.
- Anything else (`crucible serve` typed in a shell): refused, because nothing would start it
  again and the page would be left talking to nothing.

The address a definition starts `serve` with is written into it, so a host or port the
config changed since is refused by name with the command that rewrites it
(`crucible service install`, which restarts the server onto the new address itself).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from . import service
from .backend import Backend
from .config import Config
from .errors import ApiError

NOT_SUPERVISED = "restart_not_supervised"
NEEDS_SERVICE_INSTALL = "restart_needs_service_install"

# Long enough for the response to leave before the server starts to stop.
EXIT_AFTER_S = 0.5


def install_command() -> str:
    """The command that rewrites the definition, as typed where the person is: a WSL
    guest's from Windows, through `crucible guest`."""
    return "crucible guest service install" if service.in_wsl() else "crucible service install"


def supervisor(
    config: Config,
    backend: Backend,
    *,
    runner: service.Runner,
    home: Path,
    pid: int,
) -> str:
    """The service manager that will start this server again, or a refusal by name."""
    try:
        mechanism = service.mechanism_for(backend.kind)
    except service.ServiceError as exc:
        raise ApiError(409, NOT_SUPERVISED, f"nothing here can start this server again: {exc}") from None
    state = service.status(mechanism, home, runner=runner)
    if state.pid != pid:
        runs = "no server" if state.pid is None else f"pid {state.pid}"
        raise ApiError(
            409,
            NOT_SUPERVISED,
            f"this server (pid {pid}) is not the one {mechanism} runs ({runs}; "
            f"{state.detail}). It was started from a shell, and nothing would start it "
            "again once it stopped. Stop it and start it where it was started, or run "
            f"`{install_command()}` so {mechanism} runs it and the page can restart it",
            {"mechanism": mechanism, "pid": pid, "service_pid": state.pid},
        )
    _require_current_definition(config, mechanism, home)
    return mechanism


def _require_current_definition(config: Config, mechanism: str, home: Path) -> None:
    defined = service.read_defined(mechanism, home)
    path = service.definition_path(mechanism, home)
    command = install_command()
    if defined is None or not defined.restarts_on_exit:
        raise ApiError(
            409,
            NEEDS_SERVICE_INSTALL,
            f"{path} does not start this server again when it exits; it was written by "
            f"another build. Run `{command}` on the server, which rewrites it and "
            "restarts the server",
            {"command": command, "definition": str(path)},
        )
    if (defined.host, defined.port) != (config.host, config.port):
        raise ApiError(
            409,
            NEEDS_SERVICE_INSTALL,
            f"{path} starts this server on {defined.host}:{defined.port}, and the "
            f"config now says {config.host}:{config.port}. The address is written into "
            f"the service's definition, which this server cannot rewrite from inside "
            f"it. Run `{command}` on the server: it writes the new address in and "
            "restarts the server onto it",
            {"command": command, "definition": str(path),
             "defined": f"{defined.host}:{defined.port}",
             "configured": f"{config.host}:{config.port}"},
        )


def exit_for_restart(app: Any) -> None:
    """Stop the server cleanly after the answer has left, exiting SELF_RESTART_EXIT
    (cli/serve.py reads `restart_asked`)."""
    server = app.state.uvicorn_server
    if server is None:
        raise ApiError(
            409,
            NOT_SUPERVISED,
            "this app is not being served by `crucible serve`, so there is no server "
            "process to stop and start again",
        )
    app.state.restart_asked = True

    def stop() -> None:
        server.should_exit = True

    asyncio.get_running_loop().call_later(EXIT_AFTER_S, stop)


def this_pid() -> int:
    return os.getpid()


__all__ = [
    "EXIT_AFTER_S",
    "NEEDS_SERVICE_INSTALL",
    "NOT_SUPERVISED",
    "exit_for_restart",
    "install_command",
    "supervisor",
    "this_pid",
]
