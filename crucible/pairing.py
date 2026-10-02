from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from .errors import CrucibleError
from .interfaces import ipv4_addresses

SCHEME = "crucible"


def _authority(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def reachable_urls(
    host: str, port: int, advertise: Sequence[str] = ()
) -> list[str]:
    if host in ("0.0.0.0", "::", ""):
        derived = [f"http://{_authority(address, port)}" for address in ipv4_addresses()]
    else:
        derived = [f"http://{_authority(host, port)}"]
    for authority in advertise:
        url = (
            f"http://{authority}"
            if _has_port(authority)
            else f"http://{_authority(authority, port)}"
        )
        if url not in derived:
            derived.append(url)
    return derived


def _has_port(authority: str) -> bool:
    tail = authority.rsplit("]", 1)[-1] if authority.startswith("[") else authority
    head, separator, port = tail.rpartition(":")
    if separator != ":" or not port.isdigit():
        return False
    return head != "" or authority.startswith("[")


def pairing_line(name: str, url: str, token: str) -> str:
    authority = urlsplit(url).netloc
    if authority == "":
        raise ValueError(
            f"{url!r} has no authority to build a pairing line from; a URL here "
            "is one of `reachable_urls`' entries, e.g. http://192.168.68.20:7100"
        )
    return (
        f"{SCHEME}://{quote(name, safe='')}@{authority}/#{quote(token, safe='')}"
    )


def pairing_lines(name: str, urls: list[str], token: str) -> list[str]:
    return [pairing_line(name, url, token) for url in urls]


@dataclass(frozen=True)
class Pairing:
    name: str
    url: str
    token: str


def parse_pairing_line(line: str) -> Pairing:
    parts = urlsplit(line.strip())
    if parts.scheme != SCHEME:
        raise ValueError(
            f"{line.strip()[:60]!r} is not a {SCHEME}:// pairing line"
        )
    name, _, authority = parts.netloc.rpartition("@")
    if name == "" or authority == "":
        raise ValueError(
            "a pairing line is `crucible://<name>@<host>:<port>/#<token>`; "
            f"{parts.netloc!r} has no name or no authority"
        )
    token = unquote(parts.fragment)
    if token == "":
        raise ValueError("a pairing line's fragment is its token, and this one is empty")
    return Pairing(name=unquote(name), url=f"http://{authority}", token=token)


PAIRING_FILENAME = "pairing"

ICACLS_TIMEOUT_SECONDS = 30.0


class PairingFileError(CrucibleError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def pairing_file_path(home: Path) -> Path:
    return Path(home) / PAIRING_FILENAME


def icacls_argv(path: Path, user: str) -> Sequence[str]:
    return [
        "icacls",
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"{user}:(R,W)",
    ]


def _windows_user(env: Mapping[str, str]) -> str:
    user = env.get("USERNAME")
    if user is None or user.strip() == "":
        raise PairingFileError(
            "pairing_acl_failed",
            "USERNAME is not set, so there is no account to give the pairing "
            "file to. It is read from the environment and never guessed.",
        )
    return user.strip()


def write_pairing_file(
    home: Path,
    line: str,
    *,
    platform: str = sys.platform,
    env: Mapping[str, str] | None = None,
    run: "object | None" = None,
) -> Path:
    return write_pairing_line(
        pairing_file_path(Path(home)), line, platform=platform, env=env, run=run,
        private_directory=True,
    )


def write_pairing_line(
    path: Path,
    line: str,
    *,
    platform: str = sys.platform,
    env: Mapping[str, str] | None = None,
    run: "object | None" = None,
    private_directory: bool = False,
) -> Path:
    environment = os.environ if env is None else env
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    body = line if line.endswith("\n") else line + "\n"

    if platform == "win32":
        path.write_text(body, encoding="utf-8", newline="\n")
        user = _windows_user(environment)
        runner = subprocess.run if run is None else run
        completed = runner(
            list(icacls_argv(path, user)),
            capture_output=True,
            text=True,
            timeout=ICACLS_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            path.unlink(missing_ok=True)
            raise PairingFileError(
                "pairing_acl_failed",
                f"icacls would not restrict {path} to {user} "
                f"({(completed.stderr or completed.stdout or '').strip() or f'exit {completed.returncode}'}). "
                "The file has been deleted rather than left with a bearer token "
                "in it that anybody on this machine can read.",
            )
        return path

    if private_directory:
        os.chmod(directory, 0o700)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(body.encode("utf-8"))
    os.chmod(path, 0o600)
    return path


def read_pairing_file(home: Path) -> str | None:
    path = pairing_file_path(Path(home))
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


__all__ = [
    "PAIRING_FILENAME",
    "Pairing",
    "SCHEME",
    "PairingFileError",
    "icacls_argv",
    "pairing_file_path",
    "pairing_line",
    "pairing_lines",
    "parse_pairing_line",
    "reachable_urls",
    "read_pairing_file",
    "write_pairing_file",
    "write_pairing_line",
]
