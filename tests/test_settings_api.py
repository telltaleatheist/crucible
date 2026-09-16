"""The settings door: routes, upstreams, and a key that is never readable back.

PHASE15-HOST.md sections 3.1, 3.2, 3.3 and 3.9.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import capability, upstreams
from crucible.config import load_config

from .fake_upstream import ANTHROPIC_MODELS, OLLAMA_MODELS, FakeUpstream

#: A key long enough to pass the door's own length check and distinctive enough
#: that a grep for it cannot match anything else in a response.
ANTHROPIC_KEY = "sk-ant-zzzTESTKEYzzz-9f2k3A9"
OPENAI_KEY = "sk-proj-zzzOPENAIzzz-77bQ4z1"


def decided(total: int = 26 * 1024 ** 3, allowance: int = 3 * 1024 ** 3) -> Any:
    return capability.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=capability.decide_all(
            "cuda-linux",
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia",
            chosen={},
        ),
        routes={},
    )


@pytest.fixture
def settings_client(make_client: Callable[..., TestClient]):
    with make_client(enable_llm=True, capability=decided()) as instance:
        yield instance


def put(client: TestClient, auth: dict[str, str], patch: dict[str, Any]):
    return client.put("/v1/settings", headers=auth, json=patch)


def test_managed_sharing_survives_settings_writes_and_does_not_change_bind(settings_client, auth):
    client = settings_client
    before = client.get("/v1/setup", headers=auth).json()
    response = put(client, auth, {"tailscale_advertise": ["pc.tail.ts.net:7100"]})
    assert response.status_code == 200
    assert response.json()["tailscale_advertise"] == ["pc.tail.ts.net:7100"]
    assert put(client, auth, {"desktop_allowance_bytes": 1}).status_code == 200
    after = client.get("/v1/setup", headers=auth).json()
    assert after["urls"] == before["urls"] + ["http://pc.tail.ts.net:7100"]
    assert load_config(client.app.state.config.home).tailscale_advertise == ("pc.tail.ts.net:7100",)
    assert put(client, auth, {"tailscale_advertise": []}).status_code == 200
    assert client.get("/v1/setup", headers=auth).json()["urls"] == before["urls"]


def test_operator_addresses_survive_settings_write(settings_client, auth):
    client = settings_client
    path = client.app.state.config.path
    text = path.read_text(encoding="utf-8").replace("[server]", '[server]\nadvertise = ["manual.example:7100"]')
    path.write_text(text, encoding="utf-8")
    client.app.state.config.adopt(load_config(client.app.state.config.home))
    assert put(client, auth, {"tailscale_advertise": ["pc.tail.ts.net:7100"]}).status_code == 200
    assert put(client, auth, {"tailscale_advertise": []}).status_code == 200
    assert load_config(client.app.state.config.home).advertise == ("manual.example:7100",)


@pytest.mark.parametrize("address", ["https://pc.test", "0.0.0.0", "pc.test:99999", "x@y", "pc#token", "bad name"])
def test_managed_sharing_rejects_non_dialable_authorities(settings_client, auth, address):
    response = put(settings_client, auth, {"tailscale_advertise": [address]})
    assert response.status_code == 400
    assert response.json()["error"]["details"]["field"] == "tailscale_advertise"


# --------------------------------------------------------------------- GET


def test_the_document_names_every_class_and_every_upstream_configured_or_not(
    settings_client, auth
) -> None:
    """3.1's shape, whole, on a server nobody has configured.

    Three upstream cards are drawn whether or not anybody has a key, so all
    three names are always present: a key that came and went would make "not
    configured" and "this build does not know that upstream" one reading.
    """
    body = settings_client.get("/v1/settings", headers=auth).json()
    assert set(body["routes"]) == set(capability.ROUTABLE_CLASSES)
    assert set(body["routes"]) == {"clean", "translate", "simplify", "analysis"}
    for row in body["routes"].values():
        assert row["route"] == "local"
    # The local selection is carried so a window can draw "translate: local,
    # qwen3.8-27b-4bit" without a second call.
    assert body["routes"]["clean"]["model"] == "qwen3.5-9b"
    assert set(body["upstreams"]) == set(upstreams.UPSTREAM_NAMES)
    assert body["upstreams"] == {
        "anthropic": {"configured": False, "key_hint": None},
        "openai": {"configured": False, "key_hint": None},
        "ollama": {"configured": False, "url": None},
    }
    assert body["backend_kind"] == "cuda-linux"
    assert body["desktop_allowance_bytes"] == 3 * 1024 ** 3


def test_pages_is_not_routable_because_it_is_not_one_of_the_four(
    settings_client, auth
) -> None:
    """`pages` is an `llm` job type and is NOT a routable class.

    Declared on the class table rather than derived from `job_type == "llm"`:
    the VLM sends page images and "forward it to Anthropic" is a different
    feature nobody asked for.
    """
    body = settings_client.get("/v1/settings", headers=auth).json()
    assert "pages" not in body["routes"]
    assert "pages" not in capability.ROUTABLE_CLASSES


# --------------------------------------------------------------------- PUT


def test_an_upstream_and_a_route_land_in_one_request(settings_client, auth) -> None:
    """Section 5.2's move: paste a key and route a class, in one PUT.

    Upstreams are applied before routes, which is the whole reason this works
    in one call rather than two.
    """
    body = put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"translate": "anthropic/claude-sonnet-5"},
        },
    )
    assert body.status_code == 200, body.text
    document = body.json()
    assert document["upstreams"]["anthropic"]["configured"] is True
    assert document["routes"]["translate"] == {
        "route": "upstream",
        "model": "anthropic/claude-sonnet-5",
    }
    # The other three are untouched — a patch is partial.
    assert document["routes"]["clean"]["route"] == "local"


def test_the_key_hint_is_an_ellipsis_and_four_characters(
    settings_client, auth
) -> None:
    """Pinned by the contract: clients render the hint verbatim."""
    document = put(
        settings_client, auth, {"upstreams": {"anthropic": {"key": ANTHROPIC_KEY}}}
    ).json()
    assert document["upstreams"]["anthropic"]["key_hint"] == "…k3A9"
    assert document["upstreams"]["anthropic"]["key_hint"].startswith("…")


def test_the_whole_key_is_in_no_response_no_log_and_no_activity_row(
    settings_client, auth, caplog
) -> None:
    """3.9's grep, run four ways. A key is write-only and this is what that means.

    Every response body, every response HEADER, every log record emitted while
    the write happened, and `/v1/activity`'s settings history.
    """
    with caplog.at_level(logging.DEBUG):
        written = put(
            settings_client,
            auth,
            {
                "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
                "routes": {"translate": "anthropic/claude-sonnet-5"},
            },
        )
    assert written.status_code == 200
    reads = [
        written,
        settings_client.get("/v1/settings", headers=auth),
        settings_client.get("/v1/capability", headers=auth),
        settings_client.get("/v1/activity", headers=auth),
        settings_client.get("/v1/setup", headers=auth),
        settings_client.get("/v1/openai/models", headers=auth),
    ]
    for response in reads:
        assert ANTHROPIC_KEY not in response.text, response.url
        assert ANTHROPIC_KEY not in json.dumps(dict(response.headers))
    assert ANTHROPIC_KEY not in caplog.text
    # And the history says WHAT changed without saying what it was set to.
    writes = reads[3].json()["settings"]["writes"]
    assert writes[0]["changed"] == [
        "upstreams.anthropic set",
        "routes.translate = anthropic/claude-sonnet-5",
    ]


def test_the_act_and_the_client_are_recorded_on_a_settings_write(
    settings_client, auth
) -> None:
    put_headers = {**auth, "X-Crucible-Act": "translate", "User-Agent": "bookforge/1"}
    settings_client.put(
        "/v1/settings",
        headers=put_headers,
        json={"upstreams": {"ollama": {"url": "http://192.168.68.20:11434"}}},
    )
    row = settings_client.get("/v1/activity", headers=auth).json()["settings"][
        "writes"
    ][0]
    assert row["act"] == "translate"
    assert row["client"] == "bookforge/1"
    assert row["changed"] == ["upstreams.ollama set"]


def test_an_unknown_act_is_refused_before_anything_is_written(
    settings_client, auth
) -> None:
    body = settings_client.put(
        "/v1/settings",
        headers={**auth, "X-Crucible-Act": "transalte"},
        json={"upstreams": {"anthropic": {"key": ANTHROPIC_KEY}}},
    )
    assert body.status_code == 400
    assert body.json()["error"]["code"] == "unknown_act"
    after = settings_client.get("/v1/settings", headers=auth).json()
    assert after["upstreams"]["anthropic"]["configured"] is False


# ---------------------------------------------------------------- refusals


def test_route_not_routable_names_the_class_and_the_field(
    settings_client, auth
) -> None:
    body = put(settings_client, auth, {"routes": {"tts": "anthropic/whatever"}})
    assert body.status_code == 400
    error = body.json()["error"]
    assert error["code"] == "route_not_routable"
    assert error["details"]["field"] == "routes.tts"
    assert error["details"]["capability"] == "tts"


def test_route_bad_model_covers_a_missing_slash_and_an_unknown_upstream(
    settings_client, auth
) -> None:
    for value in ("claude-sonnet-5", "claud/claude-sonnet-5", "anthropic/"):
        body = put(settings_client, auth, {"routes": {"translate": value}})
        assert body.status_code == 400, value
        error = body.json()["error"]
        assert error["code"] == "route_bad_model", value
        assert error["details"]["field"] == "routes.translate"


def test_route_upstream_unconfigured_refuses_a_route_the_server_cannot_serve(
    settings_client, auth
) -> None:
    """*"the server never stores a route it cannot serve"* — 3.2."""
    body = put(settings_client, auth, {"routes": {"simplify": "openai/gpt-5"}})
    assert body.status_code == 409
    error = body.json()["error"]
    assert error["code"] == "route_upstream_unconfigured"
    assert error["details"]["field"] == "routes.simplify"
    assert error["details"]["upstream"] == "openai"


def test_upstream_in_use_names_the_classes_and_is_avoidable_in_one_request(
    settings_client, auth
) -> None:
    put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {
                "translate": "anthropic/claude-sonnet-5",
                "simplify": "anthropic/claude-sonnet-5",
            },
        },
    )
    refused = put(settings_client, auth, {"upstreams": {"anthropic": None}})
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "upstream_in_use"
    assert error["details"]["field"] == "upstreams.anthropic"
    assert error["details"]["classes"] == ["simplify", "translate"]
    # …and the caller re-routes first, in the same request if it likes.
    accepted = put(
        settings_client,
        auth,
        {
            "routes": {"translate": "local", "simplify": "local"},
            "upstreams": {"anthropic": None},
        },
    )
    assert accepted.status_code == 200, accepted.text
    document = accepted.json()
    assert document["upstreams"]["anthropic"]["configured"] is False
    assert document["routes"]["translate"]["route"] == "local"


def test_a_refusal_applies_nothing(settings_client, auth) -> None:
    """The whole point of resolving in memory before writing a byte.

    A request that configures an upstream and then names an impossible route
    must not leave the key behind: its sender believes the call failed.
    """
    body = put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"translate": "openai/gpt-5"},
        },
    )
    assert body.status_code == 409
    assert body.json()["error"]["code"] == "route_upstream_unconfigured"
    after = settings_client.get("/v1/settings", headers=auth).json()
    assert after["upstreams"]["anthropic"]["configured"] is False


def test_unknown_upstream_and_upstream_bad_field(settings_client, auth) -> None:
    unknown = put(settings_client, auth, {"upstreams": {"claude": {"key": "x" * 20}}})
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "unknown_upstream"
    assert unknown.json()["error"]["details"]["field"] == "upstreams.claude"

    wrong = put(
        settings_client, auth, {"upstreams": {"anthropic": {"url": "http://x"}}}
    )
    assert wrong.status_code == 400
    error = wrong.json()["error"]
    assert error["code"] == "upstream_bad_field"
    assert error["details"]["field"] == "upstreams.anthropic"
    assert error["details"]["takes"] == "key"


def test_every_refusal_carries_a_dotted_field(settings_client, auth) -> None:
    """Pinned by the contract: Foundry highlights one control from `details.field`."""
    patches = [
        {"nonsense": 1},
        {"routes": {"tts": "anthropic/x"}},
        {"routes": {"translate": "nope"}},
        {"routes": {"translate": 4}},
        {"routes": {"translate": "openai/gpt-5"}},
        {"upstreams": {"claude": {"key": "x" * 20}}},
        {"upstreams": {"anthropic": {"url": "http://x"}}},
        {"upstreams": {"anthropic": {"key": "short"}}},
        {"desktop_allowance_bytes": -1},
        ["not an object"],
    ]
    for patch in patches:
        body = put(settings_client, auth, patch)
        assert body.status_code >= 400, patch
        details = body.json()["error"].get("details")
        assert details is not None and "field" in details, patch


def test_the_allowance_is_written_and_recomputes_the_capability_rows(
    settings_client, auth, home: Path
) -> None:
    """A reserve that eats the card turns classes off, and the record says so."""
    before = settings_client.get("/v1/capability", headers=auth).json()
    assert {row["capability"]: row["enabled"] for row in before["classes"]}["clean"]
    body = put(settings_client, auth, {"desktop_allowance_bytes": 25 * 1024 ** 3})
    assert body.status_code == 200, body.text
    after = settings_client.get("/v1/capability", headers=auth).json()
    rows = {row["capability"]: row for row in after["classes"]}
    assert rows["clean"]["enabled"] is False
    assert "short by" in rows["clean"]["reason"]
    assert after["desktop_allowance_bytes"] == 25 * 1024 ** 3
    # …and it is on disk, not only in memory.
    assert load_config(home).desktop_allowance_bytes == 25 * 1024 ** 3


# ------------------------------------------------------------- capability


def test_capability_says_the_route_and_keeps_the_local_answer(
    settings_client, auth
) -> None:
    """3.3: `route` on every row, and the local sentence kept after routing."""
    before = {
        row["capability"]: row
        for row in settings_client.get("/v1/capability", headers=auth).json()[
            "classes"
        ]
    }
    local_reason = before["translate"]["reason"]
    assert before["translate"]["route"] == "local"

    put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"translate": "anthropic/claude-sonnet-5"},
        },
    )
    after = {
        row["capability"]: row
        for row in settings_client.get("/v1/capability", headers=auth).json()[
            "classes"
        ]
    }
    row = after["translate"]
    assert row["route"] == "upstream"
    assert row["enabled"] is True
    assert row["selected"] == "anthropic/claude-sonnet-5"
    assert row["reason"].startswith("routed to anthropic; ")
    assert capability.LOCAL_ANSWER_PREFIX + local_reason in row["reason"]
    # Every other row is untouched and still says `local`.
    assert after["clean"]["route"] == "local"
    assert after["clean"]["reason"] == before["clean"]["reason"]
    assert after["tts"]["route"] == "local"


def test_routing_back_to_local_restores_the_row_exactly(
    settings_client, auth
) -> None:
    """Nothing is lost by routing away: the record is rebuilt from `decide()`."""
    before = settings_client.get("/v1/capability", headers=auth).json()["classes"]
    put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"simplify": "anthropic/claude-sonnet-5"},
        },
    )
    put(settings_client, auth, {"routes": {"simplify": "local"}})
    after = settings_client.get("/v1/capability", headers=auth).json()["classes"]
    strip = lambda rows: [  # noqa: E731 - a local comparison, not an export
        {k: v for k, v in row.items() if k != "route"} for row in rows
    ]
    assert strip(after) == strip(before)


def test_the_operators_routes_survive_a_capability_rewrite(
    settings_client, auth, home: Path
) -> None:
    """`crucible install`'s `_write_capability` must not unroute a server.

    `write_config` writes the WHOLE document, so a rewrite that omitted the
    two new tables would quietly delete somebody's key — and the door that
    rewrites it is `crucible install`, which an operator runs months after
    pasting one.
    """
    from crucible.cli import _write_capability

    from .conftest import FAKE_BACKEND

    put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"translate": "anthropic/claude-sonnet-5"},
        },
    )
    config = load_config(home)
    decisions = capability.decide_all(
        "cuda-linux",
        total_bytes=FAKE_BACKEND.gpu.vram_bytes,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        gpu_vendor=FAKE_BACKEND.gpu.vendor,
        chosen={},
    )
    _write_capability(config, FAKE_BACKEND, decisions, {"enable_tts": True})
    after = load_config(home)
    assert after.route_model("translate") == "anthropic/claude-sonnet-5"
    assert after.upstream("anthropic").key == ANTHROPIC_KEY
    # …and the rewritten record still says the class is routed, rather than
    # putting the local row back over it.
    row = after.capability.row("translate")
    assert row.selected == "anthropic/claude-sonnet-5"
    assert row.reason.startswith("routed to anthropic; ")


# ------------------------------------------------------------------- test


def test_test_asks_the_upstream_what_it_serves(settings_client, auth, monkeypatch) -> None:
    with FakeUpstream() as upstream:
        monkeypatch.setattr(upstreams, "ANTHROPIC_BASE", upstream.url)
        body = settings_client.post(
            "/v1/settings/upstreams/anthropic/test",
            headers=auth,
            json={"key": ANTHROPIC_KEY},
        )
        assert body.status_code == 200, body.text
        assert body.json() == {"models": ANTHROPIC_MODELS}
        # The provider's own headers, not a bearer.
        assert upstream.headers_seen[-1]["x-api-key"] == ANTHROPIC_KEY
        assert upstream.headers_seen[-1]["anthropic-version"] == (
            upstreams.ANTHROPIC_VERSION
        )


def test_test_reads_ollamas_tags(settings_client, auth) -> None:
    with FakeUpstream() as upstream:
        body = settings_client.post(
            "/v1/settings/upstreams/ollama/test",
            headers=auth,
            json={"url": upstream.url},
        )
        assert body.status_code == 200, body.text
        assert body.json() == {"models": OLLAMA_MODELS}


def test_test_refuses_unreachable_rejected_and_unconfigured(
    settings_client, auth, monkeypatch
) -> None:
    unconfigured = settings_client.post(
        "/v1/settings/upstreams/openai/test", headers=auth
    )
    assert unconfigured.status_code == 400
    assert unconfigured.json()["error"]["code"] == "upstream_unconfigured"

    with FakeUpstream() as upstream:
        upstream.status = 401
        monkeypatch.setattr(upstreams, "OPENAI_BASE", upstream.url)
        rejected = settings_client.post(
            "/v1/settings/upstreams/openai/test",
            headers=auth,
            json={"key": OPENAI_KEY},
        )
        # **502, NOT 401.** A 401 from a Crucible route means THIS server
        # refused THIS client's bearer, and a client that saw one here would
        # tell a person their Crucible token was wrong about a key the
        # upstream rejected — measured by BookForge, 2026-09-14. The chat
        # door already answers 502 for every non-2xx but 429 (7.2); this is
        # the same decision at the other door.
        assert rejected.status_code == 502
        said = rejected.json()["error"]
        assert said["code"] == "upstream_rejected"
        # And the sentence names WHO did what.
        assert "openai rejected the credential" in said["message"]
        assert "not Crucible's about your token" in said["message"]
        assert said["details"]["upstream_status"] == 401
        dead = upstream.url

    unreachable = settings_client.post(
        "/v1/settings/upstreams/ollama/test", headers=auth, json={"url": dead}
    )
    assert unreachable.status_code == 502
    said = unreachable.json()["error"]
    assert said["code"] == "upstream_unreachable"
    # It NAMES THE URL that did not answer, in the message and in details:
    # "ollama did not answer" is unactionable when the operator has just
    # typed an address.
    assert dead in said["message"]
    assert said["details"]["url"].startswith(dead)


def test_test_of_an_upstream_this_server_does_not_know(settings_client, auth) -> None:
    body = settings_client.post("/v1/settings/upstreams/claude/test", headers=auth)
    assert body.status_code == 400
    assert body.json()["error"]["code"] == "unknown_upstream"


# ---------------------------------------------------------------- the file


def test_a_config_naming_an_unconfigured_upstream_does_not_load(
    settings_client, home: Path
) -> None:
    """A hand-edited config is refused at LOAD, by the same names.

    A server that started holding a route it cannot serve would refuse one
    capability for the rest of its life with a sentence about the wrong thing.
    """
    from crucible.errors import ConfigError

    path = home / "config.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text + '\n[routes]\ntranslate = "anthropic/claude-sonnet-5"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="route_upstream_unconfigured"):
        load_config(home)


def test_a_config_with_an_upstream_table_missing_its_field_does_not_load(
    settings_client, home: Path
) -> None:
    from crucible.errors import ConfigError

    path = home / "config.toml"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n[upstreams.anthropic]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="upstreams.anthropic.key"):
        load_config(home)


def test_local_is_never_written_to_the_config(home: Path, settings_client, auth) -> None:
    put(
        settings_client,
        auth,
        {
            "upstreams": {"anthropic": {"key": ANTHROPIC_KEY}},
            "routes": {"clean": "anthropic/claude-haiku-5", "translate": "local"},
        },
    )
    text = (home / "config.toml").read_text(encoding="utf-8")
    assert '"local"' not in text
    assert "translate" not in text.split("[routes]")[1].split("[")[0]
