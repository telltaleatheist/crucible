from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import interfaces, pairing
from crucible.interfaces import InterfaceError

PAIRING_LITERAL = "crucible://crucible%40mac-studio@192.168.68.20:7100/#s3cret-t0ken_x"


def test_the_name_is_percent_encoded_because_it_contains_an_at() -> None:
    line = pairing.pairing_line(
        "crucible@mac-studio", "http://192.168.68.20:7100", "s3cret-t0ken_x"
    )
    assert line == PAIRING_LITERAL
    assert line.split("://", 1)[1].count("@") == 1


def test_a_token_of_unreserved_characters_is_unchanged() -> None:
    token = "abcXYZ019-_"
    assert pairing.pairing_line("n", "http://h:1", token).endswith(f"#{token}")


def test_a_name_with_a_space_or_a_slash_survives() -> None:
    line = pairing.pairing_line("owen's box/2", "http://10.0.0.5:7100", "t")
    assert line == "crucible://owen%27s%20box%2F2@10.0.0.5:7100/#t"


def test_an_ipv6_url_keeps_its_brackets() -> None:
    urls = pairing.reachable_urls("fd00::1", 7100)
    assert urls == ["http://[fd00::1]:7100"]
    assert pairing.pairing_line("n", urls[0], "t") == "crucible://n@[fd00::1]:7100/#t"


def test_a_url_with_no_authority_is_refused_rather_than_written() -> None:
    with pytest.raises(ValueError, match="no authority"):
        pairing.pairing_line("n", "not-a-url", "t")


def test_reachable_urls_of_a_concrete_bind_is_exactly_that_one() -> None:
    assert pairing.reachable_urls("127.0.0.1", 7100) == ["http://127.0.0.1:7100"]


def test_a_wildcard_bind_lists_the_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pairing, "ipv4_addresses", lambda: ["192.168.68.20", "100.64.0.3"]
    )
    assert pairing.reachable_urls("0.0.0.0", 7100) == [
        "http://192.168.68.20:7100",
        "http://100.64.0.3:7100",
    ]


def test_this_host_answers_getifaddrs_at_all() -> None:
    found = interfaces.ipv4_addresses()
    assert isinstance(found, list)
    assert all(not address.startswith("127.") for address in found)
    assert all(not address.startswith("169.254.") for address in found)
    assert len(found) == len(set(found))


def test_setup_carries_the_token_and_a_line_per_url(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["10.1.2.3"])
    with make_client(enable_echo=True) as client:
        client.app.state.bind_host = "0.0.0.0"
        body: dict[str, Any] = client.get("/v1/setup", headers=auth).json()
    assert body["name"] == "crucible@test"
    assert body["backend"] == "cuda-linux"
    assert body["bind"] == "http://0.0.0.0:7100"
    assert body["urls"] == ["http://10.1.2.3:7100"]
    assert body["token"] == "test-token-not-minted"
    assert body["pairing"] == [
        "crucible://crucible%40test@10.1.2.3:7100/#test-token-not-minted"
    ]
    assert body["job_types"] == ["echo"]
    assert body["config_path"].endswith("config.toml")


def test_setup_repeats_info_s_job_types_from_the_same_producer(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_echo=True, enable_llm=True) as client:
        setup = client.get("/v1/setup", headers=auth).json()
        info = client.get("/v1/info", headers=auth).json()
    assert setup["job_types"] == info["job_types"]


def test_setup_needs_the_token(client: TestClient) -> None:
    assert client.get("/v1/setup").status_code == 401


def test_an_unreadable_interface_list_is_a_named_503_not_an_empty_urls(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse() -> list[str]:
        raise InterfaceError("getifaddrs(3) failed")

    monkeypatch.setattr(pairing, "ipv4_addresses", refuse)
    with make_client() as client:
        client.app.state.bind_host = "0.0.0.0"
        response = client.get("/v1/setup", headers=auth)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "interfaces_unreadable"
