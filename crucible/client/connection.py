from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..config import crucible_home
from ..errors import ConfigError
from ..pairing import parse_pairing_line
from .errors import ClientRefusal

PAIRING_ENV = "CRUCIBLE_PAIRING"

SERVERS_DIR = "servers"

SOURCE_LOCAL = "local"
SOURCE_URL_TOKEN = "--url/--token"


@dataclass(frozen=True)
class Connection:

    url: str
    token: str
    name: str | None
    source: str


def server_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in name.strip())
    slug = slug.strip(".-")
    if not slug:
        raise ClientRefusal(f"server_name_invalid: {name!r} names no server")
    return slug


def saved_pairing_path(name: str) -> Path:
    return crucible_home() / SERVERS_DIR / f"{server_slug(name)}.pairing"


def _refuse_two_sources(named: list[tuple[str, str | None]]) -> None:
    sources = [name for name, value in named if value is not None]
    if len(sources) > 1:
        raise ClientRefusal(
            f"connection_overspecified: {', '.join(sources)} each name a server; "
            "pass exactly one. (An environment variable counts: unset "
            f"{PAIRING_ENV} to use a flag.)"
        )


def _paired_file(server: str) -> str:
    pairing_file = str(saved_pairing_path(server))
    if not Path(pairing_file).is_file():
        raise ClientRefusal(
            f"server_not_paired: this computer has not paired with "
            f"{server!r}. Run `crucible pair <its address>` once first"
        )
    return pairing_file


def _line_in(pairing_file: str) -> str:
    try:
        lines = [
            line.strip()
            for line in Path(pairing_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise ClientRefusal(f"pairing_file_unreadable: {pairing_file}: {exc}") from None
    if len(lines) != 1:
        raise ClientRefusal(
            f"pairing_file_invalid: {pairing_file} holds {len(lines)} non-empty "
            "lines; it must hold exactly one pairing line"
        )
    return lines[0]


def _from_pairing(line: str, source: str) -> Connection:
    try:
        pair = parse_pairing_line(line)
    except ValueError as exc:
        raise ClientRefusal(f"pairing_line_invalid: {exc}") from None
    return Connection(url=pair.url.rstrip("/"), token=pair.token, name=pair.name, source=source)


def _from_url_and_token(url: str | None, token: str | None) -> Connection | None:
    if url is not None and token is None:
        raise ClientRefusal(
            "token_required: --url names a server this machine may not be, so it "
            "must come with --token. This command will not send the local "
            "engine's bearer to an address that was typed on the command line"
        )
    if token is not None and url is None:
        raise ClientRefusal(
            "url_required: --token is for a server elsewhere, so it must come "
            "with --url. To use the local engine's own token, pass neither"
        )
    if url is None or token is None:
        return None
    return Connection(url=url.rstrip("/"), token=token, name=None, source=SOURCE_URL_TOKEN)


def local_connection() -> Connection:
    from ..local import LocalError, connection

    try:
        url, name, token = connection(crucible_home())
    except (LocalError, ConfigError, OSError) as exc:
        raise ClientRefusal(
            f"no_local_engine: {exc}. Pass --url and --token, or --pairing, to "
            "reach a server that is not this machine's"
        ) from None
    return Connection(url=url.rstrip("/"), token=token, name=name, source=SOURCE_LOCAL)


def resolve(
    *,
    url: str | None = None,
    token: str | None = None,
    pairing: str | None = None,
    pairing_file: str | None = None,
    server: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Connection:
    environment = os.environ if environment is None else environment
    pairing_env = environment.get(PAIRING_ENV) or None
    _refuse_two_sources([
        ("--pairing", pairing),
        ("--pairing-file", pairing_file),
        ("--server", server),
        (f"${PAIRING_ENV}", pairing_env),
        (SOURCE_URL_TOKEN, url or token),
    ])
    if server is not None:
        pairing_file = _paired_file(server)
    if pairing_file is not None:
        pairing, source = _line_in(pairing_file), f"--pairing-file {pairing_file}"
    elif pairing_env is not None:
        pairing, source = pairing_env.strip(), f"${PAIRING_ENV}"
    else:
        source = "--pairing"
    if pairing is not None and (url is not None or token is not None):
        raise ClientRefusal(
            "connection_overspecified: --pairing already carries the address and "
            "the token, so it cannot be combined with --url or --token. Pass one "
            "of the two forms"
        )
    if pairing is not None:
        return _from_pairing(pairing, source)
    typed = _from_url_and_token(url, token)
    return typed if typed is not None else local_connection()
