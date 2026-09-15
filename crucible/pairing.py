"""The pairing line: a whole connect door in one string.

    crucible://crucible%40mac-studio@192.168.68.20:7100/#bXktdG9rZW4

PHASE13-OPERATOR.md section 2.1 is the contract and this module is its producer.
The consumer is `parsePairing` in `@crucible/client`, in another language, which
is the one place in Crucible where one rule has two implementations — so the
rule is written in the doc rather than in either of them, and the two are tested
against the same literal line from both ends.

WHY THE NAME IS PERCENT-ENCODED
-------------------------------
A server's name contains an `@`: `config.default_server_name()` builds
`crucible@<hostname>`, and that is what an operator sees everywhere else. Put
it raw into a URI's userinfo and the authority has two `@` — which some parsers
split at the first and some at the last. A format whose meaning depends on the
parser is not a format, so the name (and the token, for the day a token is made
of something other than urlsafe base64) is written as RFC 3986 userinfo:
unreserved characters pass, everything else is `%XX`.

`urllib.parse.quote(value, safe="")` is exactly that set — Python's
`always_safe` is letters, digits, `_ . - ~` — and it emits uppercase hex, which
is what RFC 3986 says to produce.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit

from .errors import CrucibleError
from .interfaces import ipv4_addresses

#: The scheme an app's connect door recognises. Not `http`, deliberately: the
#: line carries a secret in its fragment and must never be something a browser
#: will navigate to by accident out of a chat window.
SCHEME = "crucible"


def _authority(host: str, port: int) -> str:
    """`host:port`, with IPv6 bracketed as a URI requires."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def reachable_urls(host: str, port: int) -> list[str]:
    """The bind address, made into addresses something else can dial.

    A **wildcard** bind (`0.0.0.0`, `::`, or an empty host) is not an address,
    so it becomes one URL per non-loopback IPv4 interface, in the order the OS
    lists them (`crucible/interfaces.py`).

    A **concrete** bind becomes exactly one URL, and that includes
    `127.0.0.1`: the operator stated it, it is where the server really is, and
    an app on the same machine reaches it there. Substituting a LAN address for
    a loopback bind would hand out a URL nothing answers on.
    """
    if host in ("0.0.0.0", "::", ""):
        return [f"http://{_authority(address, port)}" for address in ipv4_addresses()]
    return [f"http://{_authority(host, port)}"]


def pairing_line(name: str, url: str, token: str) -> str:
    """One `crucible://` line for one URL of one server.

    `url` is a `reachable_urls` entry; only its authority is used, because the
    scheme is this line's own and a path would have nowhere to go. The trailing
    `/` before the fragment is part of the format: without it the fragment
    would be read as part of the authority by a lenient parser and as a syntax
    error by a strict one.
    """
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
    """One line per URL, in the same order. The `pairing` field of `/v1/setup`."""
    return [pairing_line(name, url, token) for url in urls]


@dataclass(frozen=True)
class Pairing:
    """The four facts a pairing line carries, read back out of one."""

    name: str
    url: str
    token: str


def parse_pairing_line(line: str) -> Pairing:
    """The inverse of {@link pairing_line}. Raises `ValueError` on anything else.

    Added by PHASE17, which needs the TOKEN out of a line for the first time:
    the orchestrator claims its engine with the engine's own bearer, and on a
    machine whose engine is a guest's, the only place that token exists on the
    Windows side is the line the orchestrator copied (PHASE15 3.6, 4.1a).

    `rsplit` on the LAST `@` is the half of the contract the reader owes —
    {@link pairing_line} percent-encodes the name precisely so a name
    containing `@` cannot make the authority ambiguous, and
    `crucible/host/presence.py`'s `pairing_line_authority` already reads it
    the same way. The URL comes back as `http://<authority>`, which is where
    the server is; the line itself carries no scheme for it, because a
    Crucible is HTTP and the `crucible://` scheme belongs to the line.
    """
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


# ------------------------------------------------------------ the pairing FILE
#
# PHASE15-HOST.md 3.6. The line above is a string; this is where it is written
# down so an app on the same machine can read it and never ask a person to type
# a token. One line, trailing newline, user-only.
#
# WHERE, per platform, is 3.6's table and not this file's invention:
#
#   linux / darwin / inside the WSL guest   <CRUCIBLE_HOME>/pairing
#   win32                                   %LOCALAPPDATA%\Crucible\pairing
#
# and `CRUCIBLE_HOME` in the environment overrides the directory on every
# platform. On Windows the host runs WITH `CRUCIBLE_HOME` set to
# `%LOCALAPPDATA%\Crucible`, so the two rules agree rather than compete.


#: The file's name, everywhere. One word, one owner.
PAIRING_FILENAME = "pairing"

#: `icacls` is how a Windows file is given an ACL of one user. It ships with
#: Windows, which is the whole reason it is used instead of pywin32: this pack
#: (PHASE15 4.4) carries an interpreter, a wheel and a tray, and a COM/ACL
#: dependency for one file that is written once is a dependency to build,
#: pin and ship forever.
ICACLS_TIMEOUT_SECONDS = 30.0


class PairingFileError(CrucibleError):
    """The pairing file could not be written with the permissions it needs."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def pairing_file_path(home: Path) -> Path:
    """`<home>/pairing`. The caller resolves `home`; 3.6 says which it is."""
    return Path(home) / PAIRING_FILENAME


def icacls_argv(path: Path, user: str) -> Sequence[str]:
    """`icacls <file> /inheritance:r /grant:r <user>:(R,W)` — 4.4's exact line.

    `/inheritance:r` REMOVES the inherited entries rather than adding one:
    a file under `%LOCALAPPDATA%` inherits Administrators and SYSTEM, and a
    grant without the removal would be a file with a token in it that three
    principals can read. `/grant:r` replaces rather than accumulates, so
    running this twice leaves one entry and not two.
    """
    return [
        "icacls",
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"{user}:(R,W)",
    ]


def _windows_user(env: Mapping[str, str]) -> str:
    """`%USERNAME%`, read and never assembled. Refused by name when unset."""
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
    """Write the pairing line to `<home>/pairing`, readable by this user only.

    On linux and darwin that is mode 0600, created 0600 from the outset so the
    token is never briefly world-readable — the same rule and the same reason
    as `config.write_config`.

    On Windows a mode is not a thing, so the ACL is set with `icacls` (4.4).
    **If that fails the file is DELETED**, because a pairing file is a bearer
    token on disk and a token that everybody on the machine can read is worse
    than no pairing file at all: the absent case is a fact an app knows how to
    handle (3.6: "an absent file means no local server"), and the readable case
    is a silent credential leak.

    `run` is the subprocess runner, injectable, defaulting to
    `subprocess.run` — the one legitimate default here, because it is the
    platform's own and not a guess at a value.
    """
    environment = os.environ if env is None else env
    directory = Path(home)
    directory.mkdir(parents=True, exist_ok=True)
    path = pairing_file_path(directory)
    body = line if line.endswith("\n") else line + "\n"

    if platform == "win32":
        path.write_text(body, encoding="utf-8", newline="\n")
        user = _windows_user(environment)
        runner = subprocess.run if run is None else run
        completed = runner(  # type: ignore[operator]
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

    # POSIX: 0600 from the outset, under a 0700 home.
    os.chmod(directory, 0o700)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(body.encode("utf-8"))
    os.chmod(path, 0o600)
    return path


def read_pairing_file(home: Path) -> str | None:
    """The line, or None when there is none.

    None is a FACT and not a fallback (3.6): "no local server" is what the
    caller does something about, and it is never an error and never a retry.
    """
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
]
