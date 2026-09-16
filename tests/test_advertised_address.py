"""An address something else forwards to us on, declared because it cannot be seen.

WHY THIS EXISTS. `reachable_urls` answers "where am I" by looking at the bind
and, for a wildcard, at this machine's interfaces. That is right, and it is
complete exactly while a server's reachability is its own.

It is not, on Owen's PC. The engine runs in WSL and binds 127.0.0.1; `tailscale
serve` on the Windows side forwards into the guest, so the Mac reaches it at
owens-pc.owenmorgan.com:7100. From inside the guest there is no interface, no
socket and no route that says so, so no amount of looking finds it. Measured
2026-09-15: the PC's console offered `crucible://...@127.0.0.1:7100/#...` and
the Mac could do nothing with it, while the Mac's console (binding 0.0.0.0)
offered two usable lines. The asymmetry is not a bug in the looking; it is a
fact that has to be told.
"""

from __future__ import annotations

import pytest

from crucible.errors import ConfigError
from crucible.pairing import reachable_urls


def test_a_loopback_bind_still_reports_loopback_when_nothing_is_declared() -> None:
    """The unchanged case, and the one almost every server is in."""
    assert reachable_urls("127.0.0.1", 7100) == ["http://127.0.0.1:7100"]


def test_the_declared_address_is_ADDED_to_the_derived_one() -> None:
    """Both readers are served, and the local one is served FIRST.

    An app on this machine should take the loopback line: it needs no network
    and cannot be intercepted. The declared line is for the app that is
    somewhere else. A console offering only one of them is wrong for exactly
    one of its two readers.
    """
    assert reachable_urls("127.0.0.1", 7100, ["owens-pc.owenmorgan.com"]) == [
        "http://127.0.0.1:7100",
        "http://owens-pc.owenmorgan.com:7100",
    ]


def test_a_bare_host_takes_the_servers_own_port() -> None:
    """A forward that keeps the number is the overwhelmingly common one."""
    assert reachable_urls("127.0.0.1", 7100, ["box"])[-1] == "http://box:7100"


def test_a_declared_port_is_kept_when_the_forward_changes_it() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["box:9000"])[-1] == "http://box:9000"


def test_declaring_something_already_derived_yields_ONE_line() -> None:
    """Stating a true thing twice is not an error, and not two rows either."""
    assert reachable_urls("127.0.0.1", 7100, ["127.0.0.1:7100"]) == [
        "http://127.0.0.1:7100"
    ]


def test_an_ipv6_literal_keeps_its_brackets_and_is_not_read_as_a_port() -> None:
    """The colons in `[::1]` are not a port separator.

    The only question is what follows the closing bracket, which is what
    `_has_port` looks at — a naive rpartition(':') would read `1]` as a port
    and produce an address nothing dials.
    """
    assert reachable_urls("127.0.0.1", 7100, ["[fd7a:115c:a1e0::1]"])[-1] == (
        "http://[fd7a:115c:a1e0::1]:7100"
    )
    assert reachable_urls("127.0.0.1", 7100, ["[fd7a:115c:a1e0::1]:7100"])[-1] == (
        "http://[fd7a:115c:a1e0::1]:7100"
    )


def test_a_wildcard_bind_still_enumerates_and_still_takes_a_declaration() -> None:
    """The Mac's case plus a forward: derived interfaces, then the declared one.

    Asserted about the LAST entry only, because the derived ones are this
    machine's real interfaces and a test that named them would be a test about
    the machine it runs on.
    """
    urls = reachable_urls("0.0.0.0", 7100, ["elsewhere.example"])
    assert urls[-1] == "http://elsewhere.example:7100"
    assert len(urls) >= 1


# ----------------------------------------------------------------- the config


def _parse(advertise_line: str):
    from crucible import config as config_module

    text = (
        '[server]\nname = "s"\nhost = "127.0.0.1"\nport = 7100\n'
        f"{advertise_line}"
        '\n[auth]\ntoken = "t"\n'
    )
    return config_module, text


def test_a_scheme_in_an_advertised_entry_is_refused_not_trimmed() -> None:
    """Because a scheme means somebody believes this field takes URLs.

    Quietly dropping it leaves them believing it — including the day they write
    `https://`, which a silent trim would then serve over http.
    """
    from crucible.config import _advertised

    with pytest.raises(ConfigError, match="carries a scheme"):
        _advertised({"server": {"advertise": ["http://box:7100"]}})


def test_a_path_in_an_advertised_entry_is_refused() -> None:
    from crucible.config import _advertised

    with pytest.raises(ConfigError, match="carries a path"):
        _advertised({"server": {"advertise": ["box:7100/v1"]}})


def test_an_empty_entry_is_refused_rather_than_skipped() -> None:
    from crucible.config import _advertised

    with pytest.raises(ConfigError, match="names no address"):
        _advertised({"server": {"advertise": [""]}})


def test_a_non_list_is_refused_by_name() -> None:
    from crucible.config import _advertised

    with pytest.raises(ConfigError, match="must be a list of strings"):
        _advertised({"server": {"advertise": "box:7100"}})


def test_absent_is_the_normal_case_and_means_nothing_forwards() -> None:
    """Not `_require`: almost every server's reachability is its own."""
    from crucible.config import _advertised

    assert _advertised({"server": {"name": "s"}}) == ()
    assert _advertised({}) == ()
