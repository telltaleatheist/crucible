from __future__ import annotations

import pytest

from crucible.errors import ConfigError
from crucible.pairing import reachable_urls


def test_a_loopback_bind_still_reports_loopback_when_nothing_is_declared() -> None:
    assert reachable_urls("127.0.0.1", 7100) == ["http://127.0.0.1:7100"]


def test_the_declared_address_is_ADDED_to_the_derived_one() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["owens-pc.owenmorgan.com"]) == [
        "http://127.0.0.1:7100",
        "http://owens-pc.owenmorgan.com:7100",
    ]


def test_a_bare_host_takes_the_servers_own_port() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["box"])[-1] == "http://box:7100"


def test_a_declared_port_is_kept_when_the_forward_changes_it() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["box:9000"])[-1] == "http://box:9000"


def test_declaring_something_already_derived_yields_ONE_line() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["127.0.0.1:7100"]) == [
        "http://127.0.0.1:7100"
    ]


def test_an_ipv6_literal_keeps_its_brackets_and_is_not_read_as_a_port() -> None:
    assert reachable_urls("127.0.0.1", 7100, ["[fd7a:115c:a1e0::1]"])[-1] == (
        "http://[fd7a:115c:a1e0::1]:7100"
    )
    assert reachable_urls("127.0.0.1", 7100, ["[fd7a:115c:a1e0::1]:7100"])[-1] == (
        "http://[fd7a:115c:a1e0::1]:7100"
    )


def test_a_wildcard_bind_still_enumerates_and_still_takes_a_declaration() -> None:
    urls = reachable_urls("0.0.0.0", 7100, ["elsewhere.example"])
    assert urls[-1] == "http://elsewhere.example:7100"
    assert len(urls) >= 1


def _parse(advertise_line: str):
    from crucible import config as config_module

    text = (
        '[server]\nname = "s"\nhost = "127.0.0.1"\nport = 7100\n'
        f"{advertise_line}"
        '\n[auth]\ntoken = "t"\n'
    )
    return config_module, text


def test_a_scheme_in_an_advertised_entry_is_refused_not_trimmed() -> None:
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
    from crucible.config import _advertised

    assert _advertised({"server": {"name": "s"}}) == ()
    assert _advertised({}) == ()
