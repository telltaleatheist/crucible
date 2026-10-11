"""A token is optional, like Ollama (docs/DESIGN.md section 9).

Owen, 2026-10-11: "lets make it so b sides and any other claude agent is capable of making
calls to crucible without requiring the access token. claude doesnt like handling access
tokens and it isnt necessary anyway since we're behind a firewall. ollama doesnt require
access tokens for calling its api".

`[auth] mode = "none"` answers every route without a header and ignores a bearer that is
sent anyway, right or wrong; `"token"` is the lock it always was. A config without the key
is `"token"`, so an upgrade opens nothing; a new `crucible init` writes `"none"`. The server
says which in GET /v1/ping, never hands its stored token out while nothing asks for it, and
every client works holding no token.
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from crucible import API_VERSION, cli, pairing
from crucible.cli import api_cmd
from crucible.client import transport
from crucible.client.pair import pair
from crucible.config import AUTH_NONE, AUTH_TOKEN, config_path, load_config, rewrite_config
from crucible.errors import ConfigError

from .conftest import TOKEN
from .live_server import serve

VERSION_ONLY = {"X-Crucible-Api": str(API_VERSION)}
WRONG_BEARER = {**VERSION_ONLY, "Authorization": "Bearer not-the-token"}
NO_TOKEN_LINE = "crucible://crucible%40mac-studio@192.168.68.20:7100/"


@pytest.fixture
def open_client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client(auth=AUTH_NONE) as instance:
        yield instance


# ------------------------------------------------------------------ the config


def test_a_config_without_the_key_keeps_the_lock(home: Path, client: TestClient) -> None:
    assert "mode" not in tomllib.loads(config_path(home).read_text())["auth"]
    assert load_config(home).auth == AUTH_TOKEN


def test_a_rewrite_keeps_the_mode_and_a_missing_key_stays_missing(
    home: Path, client: TestClient
) -> None:
    rewrite_config(load_config(home), retention_days=3)
    assert "mode" not in tomllib.loads(config_path(home).read_text())["auth"]
    rewrite_config(load_config(home), auth=AUTH_NONE)
    rewrite_config(load_config(home), retention_days=4)
    assert tomllib.loads(config_path(home).read_text())["auth"]["mode"] == AUTH_NONE
    assert load_config(home).token == TOKEN


def test_a_moved_server_keeps_its_mode(tmp_path: Path) -> None:
    # `crucible init --config-from`: a file without the key is a server that asked for its
    # token, so it keeps asking; a file that says "none" keeps saying it.
    from crucible.cli.init import carried_auth

    old = tmp_path / "old.toml"
    old.write_text('[auth]\ntoken = "t"\n', encoding="utf-8")
    assert carried_auth(old) == AUTH_TOKEN
    opened = tmp_path / "open.toml"
    opened.write_text('[auth]\ntoken = "t"\nmode = "none"\n', encoding="utf-8")
    assert carried_auth(opened) == AUTH_NONE


def test_a_mode_that_is_not_one_is_refused_by_name(home: Path, client: TestClient) -> None:
    text = config_path(home).read_text().replace("[auth]\n", '[auth]\nmode = "open"\n')
    config_path(home).write_text(text)
    with pytest.raises(ConfigError, match=r"\[auth\] mode: 'open' is not one of"):
        load_config(home)


# ------------------------------------------------------------------ the server


def test_a_no_auth_server_answers_without_a_header(open_client: TestClient) -> None:
    response = open_client.get("/v1/info", headers=VERSION_ONLY)
    assert response.status_code == 200, response.text
    assert response.json()["server"]["auth"] == AUTH_NONE


def test_a_no_auth_server_ignores_a_stray_token_right_or_wrong(open_client: TestClient) -> None:
    assert open_client.get("/v1/health", headers=WRONG_BEARER).status_code == 200
    right = {**VERSION_ONLY, "Authorization": f"Bearer {TOKEN}"}
    assert open_client.get("/v1/health", headers=right).status_code == 200
    garbled = {**VERSION_ONLY, "Authorization": "Basic x"}
    assert open_client.get("/v1/health", headers=garbled).status_code == 200


def test_the_openai_door_is_open_too(open_client: TestClient) -> None:
    # Not 401: whatever the chat door answers next, the token is not what stopped it.
    response = open_client.post("/openai/v1/chat/completions", headers=VERSION_ONLY, json={})
    assert response.status_code != 401


def test_a_token_server_still_refuses_without_one(client: TestClient) -> None:
    for headers in (VERSION_ONLY, WRONG_BEARER):
        response = client.get("/v1/info", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"


def test_ping_says_token_without_a_token(client: TestClient) -> None:
    assert client.get("/v1/ping").json()["auth"] == AUTH_TOKEN


def test_ping_says_none_without_a_token(open_client: TestClient) -> None:
    # (One server per test: both fixtures write the same CRUCIBLE_HOME.)
    assert open_client.get("/v1/ping").json()["auth"] == AUTH_NONE


def test_the_version_header_is_still_required_and_named_plainly(open_client: TestClient) -> None:
    response = open_client.get("/v1/health")
    assert response.status_code == 426
    error = response.json()["error"]
    assert error["code"] == "api_version_required"
    assert f"missing the X-Crucible-Api header" in error["message"]
    assert f"-H 'X-Crucible-Api: {API_VERSION}'" in error["message"]


def test_setup_on_a_no_auth_server_hands_out_no_token(
    open_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pairing, "ipv4_addresses", lambda: ["10.1.2.3"])
    body: dict[str, Any] = open_client.get("/v1/setup", headers=VERSION_ONLY).json()
    assert body["auth"] == AUTH_NONE
    assert body["token"] is None
    assert body["pairing"] == ["crucible://crucible%40test@127.0.0.1:7100/"]
    assert TOKEN not in json.dumps(body)


def test_setup_on_a_token_server_still_carries_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    body = client.get("/v1/setup", headers=auth).json()
    assert body["auth"] == AUTH_TOKEN
    assert body["token"] == TOKEN
    assert all(line.endswith(f"/#{TOKEN}") for line in body["pairing"])


def test_pairing_with_a_no_auth_server_needs_no_approval_and_carries_no_token(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(auth=AUTH_NONE, open_pairing=False) as instance:
        start = instance.post(
            "/v1/pairing/start", headers=VERSION_ONLY, json={"client_name": "b-sides"}
        ).json()
        assert start["approval_required"] is False
        poll = instance.post(
            "/v1/pairing/poll",
            headers=VERSION_ONLY,
            json={"id": start["id"], "device_code": start["device_code"]},
        ).json()
    assert poll == {"status": "approved", "name": "crucible@test", "token": None}


def test_rotating_a_token_nothing_asks_for_is_refused(open_client: TestClient) -> None:
    response = open_client.post("/v1/settings/token/rotate", headers=VERSION_ONLY)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "auth_off"


def test_settings_switch_the_mode_live_and_keep_the_token(
    home: Path, client: TestClient, auth: dict[str, str]
) -> None:
    assert client.get("/v1/settings", headers=auth).json()["auth"] == AUTH_TOKEN
    answer = client.put("/v1/settings", headers=auth, json={"auth": "none"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["auth"] == AUTH_NONE
    assert client.get("/v1/health", headers=VERSION_ONLY).status_code == 200
    back = client.put("/v1/settings", headers=VERSION_ONLY, json={"auth": "token"})
    assert back.status_code == 200, back.text
    assert client.get("/v1/health", headers=VERSION_ONLY).status_code == 401
    assert client.get("/v1/health", headers=auth).status_code == 200
    assert load_config(home).token == TOKEN


def test_settings_refuse_a_mode_that_is_not_one(client: TestClient, auth: dict[str, str]) -> None:
    response = client.put("/v1/settings", headers=auth, json={"auth": "open"})
    assert response.status_code == 400
    assert response.json()["error"]["details"]["field"] == "auth"


def test_a_change_to_the_file_applies_from_the_next_request(
    home: Path, client: TestClient
) -> None:
    assert client.get("/v1/health", headers=VERSION_ONLY).status_code == 401
    rewrite_config(load_config(home), auth=AUTH_NONE)
    assert client.get("/v1/health", headers=VERSION_ONLY).status_code == 200


# ------------------------------------------------------------------ the CLI


def test_crucible_auth_prints_and_sets_the_mode(
    home: Path, client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["auth"]) == 0
    assert '[auth] mode = "token"' in capsys.readouterr().out
    assert cli.main(["auth", "none"]) == 0
    said = capsys.readouterr().out
    assert '[auth] mode = "none"' in said
    assert "nothing needs restarting" in said
    assert load_config(home).auth == AUTH_NONE
    assert load_config(home).token == TOKEN
    assert cli.main(["auth", "none"]) == 0
    assert "already" in capsys.readouterr().out
    assert cli.main(["auth", "token"]) == 0
    assert load_config(home).auth == AUTH_TOKEN


def test_token_url_on_a_no_auth_server_prints_lines_without_the_token(
    make_client: Callable[..., TestClient], capsys: pytest.CaptureFixture[str]
) -> None:
    with make_client(auth=AUTH_NONE):
        assert cli.main(["token", "--url"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines()[0] == "crucible://crucible%40test@127.0.0.1:7100/"
    assert TOKEN not in captured.out
    assert "asks for no token" in captured.err


# ------------------------------------------------------------------ the clients


def test_the_pairing_line_without_a_token_round_trips() -> None:
    # The seam with sdk/ts/test/unit-pairing.test.ts, from this side.
    assert pairing.pairing_line(
        "crucible@mac-studio", "http://192.168.68.20:7100", None
    ) == NO_TOKEN_LINE
    parsed = pairing.parse_pairing_line(NO_TOKEN_LINE)
    assert (parsed.name, parsed.url, parsed.token) == (
        "crucible@mac-studio", "http://192.168.68.20:7100", None
    )
    assert pairing.parse_pairing_line(NO_TOKEN_LINE + "#").token is None


def test_the_cli_sends_no_authorization_header_without_a_token() -> None:
    headers = transport.request_headers(None)
    assert "Authorization" not in headers
    assert headers["X-Crucible-Api"] == str(API_VERSION)
    resolved = api_cmd.resolve(argparse.Namespace(url="http://h:7100", token=None, pairing=None))
    assert resolved.token is None


@pytest.fixture
def open_base(make_app: Callable[..., FastAPI]) -> Iterator[str]:
    with serve(make_app(auth=AUTH_NONE)) as url:
        yield url


def test_crucible_api_reaches_a_no_auth_server_by_url_alone(
    open_base: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["api", "--url", open_base, "info"]) == 0
    assert json.loads(capsys.readouterr().out)["server"]["auth"] == AUTH_NONE
    assert cli.main(["api", "--url", open_base, "--token", "stale", "health"]) == 0


def test_crucible_api_by_url_alone_against_a_token_server_says_what_to_do(
    make_app: Callable[..., FastAPI], capsys: pytest.CaptureFixture[str]
) -> None:
    with serve(make_app()) as base:
        assert cli.main(["api", "--url", base, "info"]) == 1
    err = capsys.readouterr().err
    assert "unauthorized" in err
    assert "none was sent" in err
    assert "crucible auth none" in err


def test_pair_with_a_no_auth_server_saves_a_line_at_once(open_base: str) -> None:
    name, url, token = pair(open_base, client_name="b-sides", sleep=lambda _: None)
    assert (name, url, token) == ("crucible@test", open_base, None)
