"""Config and the bearer token.

Everything Crucible keeps on disk lives under one root:

    <CRUCIBLE_HOME>/config.toml      mode 0600, holds the token
    <CRUCIBLE_HOME>/jobs/<id>/       job scratch (inputs/, artifacts/)
    <CRUCIBLE_HOME>/uploads/         blobs from POST /uploads

`CRUCIBLE_HOME` defaults to `~/.crucible` and is read from the environment on every
call, so a test (or a second server on one host) can point it somewhere else.
"""

from __future__ import annotations

import os
import secrets
import socket
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from .errors import ConfigError

CRUCIBLE_HOME_ENV = "CRUCIBLE_HOME"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7100
TOKEN_BYTES = 32

#: VRAM the host's own desktop holds that is not anybody's job. A headless Linux
#: box wants 0; a Windows machine running WSL2 wants about 3 GiB, because the
#: desktop compositor, the browser and the editor are all on the same card and
#: the WSL2 driver shim does not list them as compute apps. The accelerator guard
#: subtracts this before calling unaccounted VRAM "somebody else's job"
#: (crucible/accelerator.py). It is a declared fact about the host, written by
#: `crucible init`, not a fudge factor the code picks.
DEFAULT_DESKTOP_ALLOWANCE_BYTES = 3 * 1024 ** 3


def crucible_home() -> Path:
    """The root of this server's state. Honours $CRUCIBLE_HOME."""
    override = os.environ.get(CRUCIBLE_HOME_ENV)
    if override is not None and override != "":
        return Path(override).expanduser()
    return Path.home() / ".crucible"


def config_path(home: Path | None = None) -> Path:
    return (home if home is not None else crucible_home()) / "config.toml"


def mint_token() -> str:
    """A 32-byte urlsafe bearer token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def default_server_name() -> str:
    return f"crucible@{socket.gethostname()}"


@dataclass(frozen=True)
class Config:
    path: Path
    home: Path
    name: str
    host: str
    port: int
    token: str
    backend_kind: str
    enable_echo: bool
    enable_llm: bool
    desktop_allowance_bytes: int

    @property
    def jobs_dir(self) -> Path:
        return self.home / "jobs"

    @property
    def uploads_dir(self) -> Path:
        return self.home / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"


def _require(table: dict[str, Any], section: str, key: str, kind: type) -> Any:
    if section not in table:
        raise ConfigError(f"config is missing the [{section}] section")
    if key not in table[section]:
        raise ConfigError(f"config is missing {section}.{key}")
    value = table[section][key]
    wrong_type = not isinstance(value, kind)
    # bool is a subclass of int; a bool where an int is wanted is still wrong.
    if kind is int and isinstance(value, bool):
        wrong_type = True
    if wrong_type:
        raise ConfigError(
            f"config key {section}.{key} must be {kind.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


def load_config(home: Path | None = None) -> Config:
    """Read config.toml. Raises ConfigError naming the missing piece."""
    root = home if home is not None else crucible_home()
    path = config_path(root)
    if not path.exists():
        raise ConfigError(
            f"no config at {path} — run `crucible init` (or set {CRUCIBLE_HOME_ENV})"
        )
    try:
        with path.open("rb") as handle:
            table = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    return Config(
        path=path,
        home=root,
        name=_require(table, "server", "name", str),
        host=_require(table, "server", "host", str),
        port=_require(table, "server", "port", int),
        token=_require(table, "auth", "token", str),
        backend_kind=_require(table, "backend", "kind", str),
        enable_echo=_require(table, "jobs", "enable_echo", bool),
        enable_llm=_require(table, "jobs", "enable_llm", bool),
        desktop_allowance_bytes=_require(
            table, "accelerator", "desktop_allowance_bytes", int
        ),
    )


def write_config(
    home: Path,
    *,
    name: str,
    host: str,
    port: int,
    token: str,
    backend_kind: str,
    enable_echo: bool,
    enable_llm: bool,
    desktop_allowance_bytes: int,
) -> Path:
    """Write config.toml at mode 0600 under a 0700 home. Returns the path."""
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    path = config_path(home)
    document = {
        "server": {"name": name, "host": host, "port": port},
        "auth": {"token": token},
        "backend": {"kind": backend_kind},
        "jobs": {"enable_echo": enable_echo, "enable_llm": enable_llm},
        "accelerator": {"desktop_allowance_bytes": desktop_allowance_bytes},
    }
    # Create with 0600 from the outset so the token is never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(tomli_w.dumps(document).encode("utf-8"))
    os.chmod(path, 0o600)
    return path


def config_mode(path: Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))
