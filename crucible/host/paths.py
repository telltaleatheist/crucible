"""Where the host's files are. PHASE15-HOST.md 3.6 and 4.1.

ONE PLACE, and every function takes the environment rather than reading it, for
the reason the whole package is shaped this way: pytest runs inside WSL, where
`LOCALAPPDATA` does not exist, and a path rule that can only be exercised on the
machine it is for is a path rule nothing pins.

`%LOCALAPPDATA%\\Crucible\\` is the host's root and it already has three members
that other files own — `wsl\\` and `downloads\\` from
`sdk/bootstrap/src/distro.ts`, and `host\\` from PHASE15 4.4. The host adds
`config.toml`, `pairing` and `host.log` beside them, and runs with
`CRUCIBLE_HOME` set to that directory (3.6), so the host-mode server's config
lands there too rather than in `~/.crucible`, which on Windows is a directory no
app looks in.

`LOCALAPPDATA` is READ and never assembled from a username — `distro.ts` states
the same rule for the same reason: a username with a space, a roaming profile or
a redirected AppData all make `C:\\Users\\<name>\\AppData\\Local` a guess.
"""

from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Mapping

from .errors import HostError

#: The directory under `%LOCALAPPDATA%` that everything Crucible puts on a
#: Windows machine lives in. Shared with `sdk/bootstrap/src/distro.ts`'s
#: `crucibleAppData`, which spells the same two segments — tied by
#: `tests/test_host_paths.py`, which reads that file.
APPDATA_DIRNAME = "Crucible"

#: The host pack unpacks here (4.4), so this is where `crucible.cmd` is.
PACK_SUBDIR = "host"

#: The tray's log (4.1). One file, appended, rolled at `LOG_ROLL_BYTES`.
LOG_NAME = "host.log"
LOG_PREVIOUS_NAME = "host.log.1"
LOG_ROLL_BYTES = 2 * 1024 * 1024

#: The relocatable entry point the Windows pack carries (4.4). NOT an `.exe`:
#: pip's `Scripts\\*.exe` launchers bake the build tree's interpreter path into
#: the binary and do not survive the move to `%LOCALAPPDATA%`.
CONSOLE_CMD = "crucible.cmd"

#: The interpreter that starts the tray. `pythonw` and not `python`, because a
#: console window is not a thing a login item may open (4.1).
PYTHONW = "pythonw.exe"

#: The loopback door bootstrap talks to (4.3). Loopback and a bearer, not a
#: pipe: one transport, and it is the one every other Crucible door already is.
DOOR_HOST = "127.0.0.1"
DOOR_PORT = 7101

#: Where the engine answers, on every machine (3.5: apps connect to the same
#: address whether the server is the guest's or the host-mode child).
ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 7100


def local_app_data(env: Mapping[str, str]) -> PureWindowsPath:
    """`%LOCALAPPDATA%`, from the environment. Refused by name when unset."""
    value = env.get("LOCALAPPDATA")
    if value is None or value.strip() == "":
        raise HostError(
            "host_no_localappdata",
            "LOCALAPPDATA is not set, so there is no per-user directory for the "
            "host's config, log and pairing file. It is read from the "
            "environment and never assembled from a username.",
        )
    return PureWindowsPath(value)


def crucible_root(env: Mapping[str, str]) -> PureWindowsPath:
    """`%LOCALAPPDATA%\\Crucible` — and the host's `CRUCIBLE_HOME`."""
    override = env.get("CRUCIBLE_HOME")
    if override:
        root = PureWindowsPath(override)
        if not root.is_absolute():
            raise HostError("host_invalid_home", "CRUCIBLE_HOME must be an absolute Windows path")
        return root
    return local_app_data(env) / APPDATA_DIRNAME


def host_pack_dir(env: Mapping[str, str]) -> PureWindowsPath:
    """`%LOCALAPPDATA%\\Crucible\\host` — where the host pack unpacked."""
    return crucible_root(env) / PACK_SUBDIR


def console_cmd_path(env: Mapping[str, str]) -> PureWindowsPath:
    """`…\\host\\crucible.cmd`. Its existence is what "the host is installed" means."""
    return host_pack_dir(env) / CONSOLE_CMD


def pythonw_path(env: Mapping[str, str]) -> PureWindowsPath:
    """`…\\host\\pythonw.exe`, which is what the Startup shortcut points at."""
    return host_pack_dir(env) / PYTHONW


def log_path(env: Mapping[str, str]) -> PureWindowsPath:
    return crucible_root(env) / LOG_NAME


def previous_log_path(env: Mapping[str, str]) -> PureWindowsPath:
    return crucible_root(env) / LOG_PREVIOUS_NAME


def engine_url(path: str = "") -> str:
    """`http://127.0.0.1:7100<path>`. The engine's address, spelled once."""
    return f"http://{ENGINE_HOST}:{ENGINE_PORT}{path}"


def door_url(path: str = "") -> str:
    """`http://127.0.0.1:7101<path>`. The install door's address, spelled once."""
    return f"http://{DOOR_HOST}:{DOOR_PORT}{path}"


def as_text(path: PurePath) -> str:
    """A path as the OS spells it, for an argv or a log line."""
    return str(path)
