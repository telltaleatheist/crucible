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

from urllib.parse import quote, urlsplit

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


__all__ = ["SCHEME", "pairing_line", "pairing_lines", "reachable_urls"]
