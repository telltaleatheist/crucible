from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from crucible import protocol
from crucible.cli import api_cmd, common
from crucible.client import connection, errors, pair, transport

CORE = (connection, errors, pair, transport)


@pytest.mark.parametrize("module", CORE, ids=lambda module: module.__name__)
def test_the_client_core_neither_parses_arguments_nor_prints(module: Any) -> None:
    tree = ast.parse(inspect.getsource(module))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "argparse" not in imported
    assert "print" not in called


def test_the_user_agent_is_built_once_from_the_protocol() -> None:
    assert transport.USER_AGENT == protocol.user_agent("cli")
    assert transport.request_headers("t")["User-Agent"] == transport.USER_AGENT
    assert "Authorization" not in transport.request_headers(None)


class _Answer(io.BytesIO):
    def __enter__(self) -> "_Answer":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_pairing_goes_through_the_one_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[dict[str, Any]] = []

    def open_url(origin: str, method: str, path: str, **kwargs: Any) -> _Answer:
        opened.append({"origin": origin, "method": method, "path": path, **kwargs})
        return _Answer(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(transport, "open_url", open_url)
    assert pair.pair_call("http://h:7100", "/v1/pairing/start", {"client_name": "c"}) == {
        "ok": True
    }
    assert opened == [{
        "origin": "http://h:7100", "method": "POST", "path": "/v1/pairing/start",
        "token": None, "body": b'{"client_name": "c"}', "content_type": "application/json",
        "timeout": pair.PAIR_TIMEOUT_SECONDS,
    }]


def test_the_servers_probe_sends_the_protocol_user_agent(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    sent: list[Any] = []

    class Opener:
        def open(self, request: Any, timeout: float) -> Any:
            sent.append(request)
            raise urllib.error.URLError("nobody home")

    monkeypatch.setattr(common.urllib.request, "build_opener", lambda *handlers: Opener())
    config = argparse.Namespace(host="127.0.0.1", port=1, token="t", name="n")
    assert common.server_here(config, None) is None
    assert sent[0].get_header("User-agent") == protocol.user_agent("cli")


def test_resolve_takes_keywords_and_no_namespace(tmp_path: Path) -> None:
    resolved = connection.resolve(url="http://h:1/", token="abc", environment={})
    assert resolved == connection.Connection(
        url="http://h:1", token="abc", name=None, source="--url/--token"
    )
    with pytest.raises(errors.ClientRefusal, match="connection_overspecified"):
        connection.resolve(url="http://h:1", token="abc",
                           environment={connection.PAIRING_ENV: "crucible://a@b:1/#t"})


def _leaves(verbs: tuple[api_cmd.Verb, ...]) -> list[api_cmd.Verb]:
    found: list[api_cmd.Verb] = []
    for verb in verbs:
        found.extend(_leaves(verb.verbs) if verb.verbs else [verb])
    return found


def test_every_api_verb_is_a_row_with_help_and_a_handler_or_children() -> None:
    for verb in api_cmd.API_VERBS:
        assert verb.help
        assert (verb.run is None) == bool(verb.verbs), verb.name
    assert all(callable(leaf.run) for leaf in _leaves(api_cmd.API_VERBS))


def test_the_verbs_live_in_api_cmd_and_nothing_forwards_to_them() -> None:
    assert importlib.util.find_spec("crucible.apiclient") is None
