from __future__ import annotations

import re
from typing import Sequence

from .platform.wsl_table import CRUCIBLE_DISTRO

WSL_EXE = "wsl.exe"

# Under the Windows home: where the installer imports the distro, so its disk
# (ext4.vhdx) lives here. Uninstall keeps it: the distro is never unregistered.
DISTRO_DIRNAME = "wsl"

GUEST_USER = "crucible"
ROOT_USER = "root"

GUEST_HOME = "${CRUCIBLE_HOME:-$HOME/.crucible}"
GUEST_CRUCIBLE = f"{GUEST_HOME}/server/bin/crucible"

LIST_HEADERS = frozenset({"NAME", "NOM", "NAAM"})

_DISTRO_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def default_user_argv(distro: str, argv: Sequence[str]) -> list[str]:
    return [WSL_EXE, "-d", distro, "--exec", *argv]


def as_user_argv(distro: str, user: str, argv: Sequence[str]) -> list[str]:
    return [WSL_EXE, "-d", distro, "-u", user, "--exec", *argv]


def guest_argv(distro: str, argv: Sequence[str]) -> list[str]:
    if distro == CRUCIBLE_DISTRO:
        return as_user_argv(distro, GUEST_USER, argv)
    return default_user_argv(distro, argv)


def root_argv(distro: str, argv: Sequence[str] = ()) -> list[str]:
    return as_user_argv(distro, ROOT_USER, argv)


def guest_shell_argv(distro: str, script: str) -> list[str]:
    return guest_argv(distro, ["bash", "-lc", script])


def guest_file_argv(distro: str, name: str) -> list[str]:
    return guest_shell_argv(distro, f'cat "{GUEST_HOME}/{name}"')


def guest_home_argv(distro: str) -> list[str]:
    return guest_shell_argv(distro, f'printf %s "{GUEST_HOME}"')


def pairing_argv(distro: str) -> list[str]:
    return guest_file_argv(distro, "pairing")


def list_argv(*, running: bool = False) -> list[str]:
    return [WSL_EXE, "-l", "-v", *(["--running"] if running else [])]


def status_argv() -> list[str]:
    return [WSL_EXE, "--status"]


def terminate_argv(distro: str) -> list[str]:
    return [WSL_EXE, "--terminate", distro]


def whoami_argv(distro: str) -> list[str]:
    return default_user_argv(distro, ["id", "-un"])


SET_DEFAULT_USER_FLAG = "--set-default-user"


def help_argv() -> list[str]:
    return [WSL_EXE, "--help"]


def set_default_user_argv(distro: str, user: str) -> list[str]:
    return [WSL_EXE, "--manage", distro, SET_DEFAULT_USER_FLAG, user]


def import_argv(distro: str, destination: str, archive: str) -> list[str]:
    return [WSL_EXE, "--import", distro, destination, archive, "--version", "2"]


def parse_distro_list(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.replace("\x00", "").splitlines():
        line = raw.strip().lstrip("*").strip()
        parts = line.split()
        if not parts or parts[0].upper() in LIST_HEADERS:
            continue
        if len(parts) >= 3 and parts[-1].isdigit():
            names.append(" ".join(parts[:-2]))
        elif len(parts) == 1 and _DISTRO_NAME.match(parts[0]):
            names.append(parts[0])
    return names
