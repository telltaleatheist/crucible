"""Who can reach this server, and the one thing that opens it (2026-10-09).

A friend's laptop ran Crucible in its WSL2 guest, bound to loopback, and nothing she
was shown said that only her PC could reach it, or what opened it. `crucible serve`
told her to pass `--host 0.0.0.0`, which inside a NAT'd guest opens nothing. These
hold the one answer every surface reads: `reach`.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import pairing, reach, service
from crucible.cli import serve
from crucible.platform import lan_door


def config(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**{
        "advertise": (), "tailscale_advertise": (), "lan_advertise": (),
        "path": "/home/me/.crucible/config.toml", **fields,
    })


def test_a_loopback_wsl_guest_points_at_windows_and_says_what_opening_changes() -> None:
    found = reach.for_server(config(), place=reach.WSL_GUEST, host="127.0.0.1", port=7100)
    assert found.reachable is False and found.urls == ()
    assert found.command == "crucible lan enable"
    assert "PowerShell" in found.how and "not inside WSL" in found.how
    assert "Share on your network" in found.how, "the window's action, beside the command"
    assert found.changes == lan_door.ELEVATION_SENTENCE
    for said in ("port forward", lan_door.RULE_NAME, "Private", "Public", "crucible lan disable"):
        assert said in found.changes


def test_a_wsl_guest_is_reachable_only_where_windows_advertises_it() -> None:
    shared = reach.for_server(config(lan_advertise=("192.168.1.5:7100",)),
                              place=reach.WSL_GUEST, host="127.0.0.1", port=7100)
    assert shared.reachable and shared.urls == ("http://192.168.1.5:7100",)
    assert shared.command is None


def test_a_wildcard_bind_inside_the_nat_guest_is_not_called_reachable(monkeypatch) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["172.24.10.2"])
    found = reach.for_server(config(), place=reach.WSL_GUEST, host="0.0.0.0", port=7100)
    assert found.reachable is False, "the guest's own vEthernet address is not the LAN"


def test_a_loopback_mac_names_the_config_line_and_the_restart() -> None:
    found = reach.for_server(config(), place=reach.POSIX, host="127.0.0.1", port=7100)
    assert not found.reachable and found.command is None
    assert 'host = "0.0.0.0"' in found.how and "/home/me/.crucible/config.toml" in found.how
    assert "crucible service restart" in found.how
    assert "crucible lan" not in found.how, "the lan door is the Windows crossing only"


def test_a_native_windows_engine_is_not_sent_to_the_lan_door() -> None:
    found = reach.for_server(config(path=r"C:\c\config.toml"), place=reach.WINDOWS,
                             host="127.0.0.1", port=7100)
    assert not found.reachable and found.command is None
    assert "refuses" in found.how and "administrator" in found.changes


def test_a_wildcard_bind_with_no_network_address_says_that_and_not_loopback(monkeypatch) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: [])
    found = reach.for_server(config(), place=reach.POSIX, host="0.0.0.0", port=7100)
    assert not found.reachable and "no network address" in found.sentence


def test_a_wildcard_mac_is_reachable_at_its_addresses(monkeypatch) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["192.168.68.20"])
    found = reach.for_server(config(), place=reach.POSIX, host="0.0.0.0", port=7100)
    assert found.reachable and found.urls == ("http://192.168.68.20:7100",)


@pytest.mark.parametrize("record, reachable, command", [
    (None, False, "crucible lan enable"),
    ({"authorities": ["192.168.1.5:7100"], "state": "configured"}, True, None),
    ({"authorities": [], "state": "degraded"}, False, "crucible lan status"),
    ({"authorities": ["192.168.1.5:7100"], "state": "open"}, False, "crucible lan status"),
])
def test_the_windows_door_record_is_shared_only_once_verified(record, reachable, command) -> None:
    found = reach.wsl_door_reach(record)
    assert found.reachable is reachable and found.command == command


def test_setup_says_whether_other_devices_can_reach_it(
    make_client: Callable[..., TestClient], auth: dict[str, str], monkeypatch,
) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    with make_client(enable_echo=True) as client:
        network = client.get("/v1/setup", headers=auth).json()["network"]
        assert network["reachable"] is False and network["urls"] == []
        assert network["command"] == "crucible lan enable"
        assert set(network) == {"reachable", "urls", "sentence", "how", "command", "changes"}
        put = client.put("/v1/settings", headers=auth, json={"lan_advertise": ["192.168.1.5:7100"]})
        assert put.status_code == 200, put.text
        network = client.get("/v1/setup", headers=auth).json()["network"]
    assert network["reachable"] is True and network["urls"] == ["http://192.168.1.5:7100"]


def test_setup_on_a_plain_linux_box_names_its_own_config(
    make_client: Callable[..., TestClient], auth: dict[str, str], monkeypatch,
) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: False)
    with make_client(enable_echo=True) as client:
        body = client.get("/v1/setup", headers=auth).json()
    assert body["network"]["command"] is None
    assert body["config_path"] in body["network"]["how"]


def test_serve_inside_the_guest_no_longer_says_to_bind_wide(monkeypatch) -> None:
    monkeypatch.setattr(service, "in_wsl", lambda: True)
    banner = serve.reach_banner(config(), "cuda-linux", "127.0.0.1", 7100)
    assert banner.startswith("bound to loopback: Only this PC can reach Crucible.")
    assert "crucible lan enable" in banner and "--host 0.0.0.0" not in banner


def test_serve_bound_wide_keeps_its_one_line() -> None:
    assert serve.reach_banner(config(), "cuda-linux", "0.0.0.0", 7100) == (
        "bound beyond loopback: the bearer token is the only lock."
    )


def test_lan_offer_on_a_mac_prints_the_same_answer_and_asks_nothing(tmp_path, monkeypatch, capsys) -> None:
    import argparse

    from crucible import lan

    monkeypatch.setattr(service, "in_wsl", lambda: False)
    monkeypatch.setattr(lan, "load_config", lambda home: config(
        host="127.0.0.1", port=7100, backend_kind="mlx-darwin"))
    monkeypatch.setattr(lan, "crucible_home", lambda: tmp_path)
    monkeypatch.setattr(lan.sys, "platform", "darwin")
    assert lan.command(argparse.Namespace(lan_action="offer", ask=True)) == 0
    out = capsys.readouterr().out
    assert out.startswith("Reaching Crucible from another device:\nOnly this machine can reach")
    assert "crucible service restart" in " ".join(out.split()) and "?" not in out
