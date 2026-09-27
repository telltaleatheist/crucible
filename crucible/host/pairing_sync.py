from __future__ import annotations

from ..pairing import PairingFileError, pairing_line, parse_pairing_line, write_pairing_file
from ..platform.hostconfig import ConfigUnreadable, read_token, server_name_and_token
from ..platform.paths import engine_url
from . import operator_stop
from .context import HostContext
from .state import Owner

GUEST_OWNERS = (Owner.WSL_UNIT, Owner.FOUND)


def guest_pairing_line(context: HostContext) -> str | None:
    owner = context.presence.owner
    if owner is Owner.WSL_UNIT:
        return context.watcher.read_guest_pairing(context.watcher.distro)
    if owner is Owner.FOUND:
        return None if context.watcher.found is None else context.watcher.found.line
    return None


def _guest_token(context: HostContext) -> str | None:
    line = guest_pairing_line(context)
    if line is None:
        return None
    try:
        return parse_pairing_line(line).token
    except ValueError as exc:
        context.log.write(f"claim: the guest's pairing line will not parse ({exc})")
        return None


def _stopped_engine_token(context: HostContext) -> str | None:
    try:
        return parse_pairing_line((context.home / "pairing").read_text(encoding="utf-8").strip()).token
    except (OSError, ValueError) as exc:
        context.log.write(f"stopped engine pairing is invalid: {exc}")
        return None


def engine_token(context: HostContext) -> str | None:
    owner = context.presence.owner
    if owner in GUEST_OWNERS:
        return _guest_token(context)
    if owner is Owner.HOST_CHILD:
        return read_token(context.home)
    if owner is Owner.NONE and operator_stop.is_stopped(context.home):
        return _stopped_engine_token(context)
    return None


def engine_token_detail(context: HostContext) -> str:
    owner = context.presence.owner
    if owner in GUEST_OWNERS:
        return (
            f"the engine here is owner={owner.value} and its bearer comes from the "
            "guest's pairing line, which could not be read or would not parse. "
            "The host log says which."
        )
    if owner is Owner.HOST_CHILD:
        return (
            "this host runs its own engine and there is no token in its config "
            f"yet ({context.home}). It gets one the first time the Windows server "
            "is initialised, which is seconds after the host first starts."
        )
    return (
        "this orchestrator owns no engine (owner=none), so there is no bearer "
        "for it to check against. Its config is not the problem. An engine that "
        "is answering is adopted on the next watch tick; one that is not needs "
        "Restart engine, or `crucible local start`."
    )


def host_mode_pairing_line(context: HostContext) -> str | None:
    if not (context.home / "config.toml").is_file():
        context.log.write("pairing: no config yet, so no pairing file")
        return None
    try:
        name, token = server_name_and_token(context.home)
    except ConfigUnreadable as exc:
        context.log.write(f"pairing: {exc}")
        return None
    return pairing_line(name, engine_url(), token)


def _line_to_publish(context: HostContext) -> str | None:
    owner = context.presence.owner
    if owner in GUEST_OWNERS:
        line = guest_pairing_line(context)
        if line is None:
            context.log.write(
                "pairing: the engine on this machine is a guest's and its own "
                "pairing line could not be read, so nothing was written — a "
                "file with the wrong token is worse than no file (3.6)"
            )
        return line
    if owner is Owner.HOST_CHILD:
        return host_mode_pairing_line(context)
    context.log.write("pairing: there is no engine on this machine, so no file")
    return None


def write_pairing(context: HostContext) -> None:
    line = _line_to_publish(context)
    if line is None:
        return
    try:
        written = write_pairing_file(context.home, line, env=context.runner.env)
    except PairingFileError as exc:
        context.log.write(f"pairing: {exc.code}: {exc.message}")
        return
    context.log.write(f"pairing: {written}")
