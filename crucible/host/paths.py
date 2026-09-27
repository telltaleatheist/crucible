from __future__ import annotations

from pathlib import PurePath, PureWindowsPath
from typing import Mapping

from .errors import HostError

APPDATA_DIRNAME = "Crucible"

PACK_SUBDIR = "host"

LOG_NAME = "host.log"
LOG_PREVIOUS_NAME = "host.log.1"
LOG_ROLL_BYTES = 2 * 1024 * 1024

CONSOLE_CMD = "crucible.cmd"

PYTHONW = "pythonw.exe"

DOOR_HOST = "127.0.0.1"
DOOR_PORT = 7101

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 7100


def local_app_data(env: Mapping[str, str]) -> PureWindowsPath:
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
    override = env.get("CRUCIBLE_HOME")
    if override:
        root = PureWindowsPath(override)
        if not root.is_absolute():
            raise HostError("host_invalid_home", "CRUCIBLE_HOME must be an absolute Windows path")
        return root
    return local_app_data(env) / APPDATA_DIRNAME


def host_pack_dir(env: Mapping[str, str]) -> PureWindowsPath:
    return crucible_root(env) / PACK_SUBDIR


def console_cmd_path(env: Mapping[str, str]) -> PureWindowsPath:
    return host_pack_dir(env) / CONSOLE_CMD


def pythonw_path(env: Mapping[str, str]) -> PureWindowsPath:
    return host_pack_dir(env) / PYTHONW


def log_path(env: Mapping[str, str]) -> PureWindowsPath:
    return crucible_root(env) / LOG_NAME


def previous_log_path(env: Mapping[str, str]) -> PureWindowsPath:
    return crucible_root(env) / LOG_PREVIOUS_NAME


def engine_url(path: str = "") -> str:
    return f"http://{ENGINE_HOST}:{ENGINE_PORT}{path}"


def door_url(path: str = "") -> str:
    return f"http://{DOOR_HOST}:{DOOR_PORT}{path}"


def as_text(path: PurePath) -> str:
    return str(path)
