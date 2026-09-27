from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .errors import HostError

CONFIG_NAME = "config.toml"

CONSENT_TABLE = "orchestrator"
CONSENT_KEY = "distro"

WSL_KEY = "wsl"
WSL_NEVER = "never"


class ConfigUnreadable(ValueError):
    def __init__(self, path: Path, reason: BaseException) -> None:
        super().__init__(f"{path} could not be read as TOML ({reason})")
        self.path = path
        self.reason = reason


def config_path(home: Path) -> Path:
    return Path(home) / CONFIG_NAME


def document(home: Path) -> dict[str, Any] | None:
    path = config_path(home)
    if not path.is_file():
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigUnreadable(path, exc) from exc


def _string(home: Path, table: str, key: str) -> str | None:
    try:
        read = document(home)
    except ConfigUnreadable:
        return None
    section = None if read is None else read.get(table)
    value = section.get(key) if isinstance(section, dict) else None
    return value if isinstance(value, str) and value else None


def read_token(home: Path) -> str | None:
    return _string(home, "auth", "token")


def read_backend_kind(home: Path) -> str | None:
    return _string(home, "backend", "kind")


def server_name_and_token(home: Path) -> tuple[str, str]:
    read = document(home)
    if read is None:
        raise ConfigUnreadable(config_path(home), FileNotFoundError("there is no such file"))
    try:
        return read["server"]["name"], read["auth"]["token"]
    except (KeyError, TypeError) as exc:
        raise ConfigUnreadable(config_path(home), KeyError(f"no {exc}")) from exc


def declined_wsl(home: Path) -> bool:
    try:
        read = document(home)
    except ConfigUnreadable as exc:
        raise HostError(
            "orchestrator_wsl_invalid",
            f"{exc}, so whether this machine declined the Linux engine cannot be "
            "known. Nothing is moved until the file parses.",
        ) from exc
    table = None if read is None else read.get(CONSENT_TABLE)
    if not isinstance(table, dict) or WSL_KEY not in table:
        return False
    value = table[WSL_KEY]
    if value != WSL_NEVER:
        raise HostError(
            "orchestrator_wsl_invalid",
            f"{CONSENT_TABLE}.{WSL_KEY} in {config_path(home)} is {value!r}. The one value "
            f'this key takes is "{WSL_NEVER}", which keeps this machine on its '
            "native Windows engine; remove the key to let Crucible install the "
            "Linux one.",
        )
    return True


def consented_distro(home: Path) -> str | None:
    try:
        read = document(home)
    except ConfigUnreadable as exc:
        raise HostError(
            "orchestrator_distro_invalid",
            f"{exc}, so whether this orchestrator was given a distro to manage "
            "cannot be known. It manages none until the file parses.",
        ) from exc
    table = None if read is None else read.get(CONSENT_TABLE)
    if table is None:
        return None
    path = config_path(home)
    if not isinstance(table, dict):
        raise HostError(
            "orchestrator_distro_invalid",
            f"[{CONSENT_TABLE}] in {path} is a {type(table).__name__} and not "
            "a table.",
        )
    if CONSENT_KEY not in table:
        return None
    name = table[CONSENT_KEY]
    if not isinstance(name, str) or name.strip() == "":
        raise HostError(
            "orchestrator_distro_invalid",
            f"{CONSENT_TABLE}.{CONSENT_KEY} in {path} is "
            f"{name!r}; it names a WSL distribution, as `wsl -l -v` spells it "
            '(e.g. distro = "Ubuntu").',
        )
    return name.strip()
