"""`clean`, `translate`, `simplify` and `analysis` were retired to the apps on 2026-10-25
(crucible/classnames.py RETIRED_CLASSES): the apps pick their own text models and send
that work as a chat (`generate`) or to `decide`.

A server upgrading from before then has a config.toml and a capability record that may
name them. It must start and work: they are dropped on load, said once on the log, and
gone from the file once the server starts. A name that was never a class is still
refused. A door asked for one of them says it was retired, by name. And an ACT is still
the app's own label: BookForge keeps sending `X-Crucible-Act: clean`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Iterator

import pytest
import tomli_w
from fastapi.testclient import TestClient

from crucible import classnames, config, upstreams
from crucible.api import create_app
from crucible.config import load_config
from crucible.errors import ConfigError

from .conftest import FAKE_BACKEND, configure_box
from .fake_upstream import FakeUpstream
from .test_settings_api import ANTHROPIC_KEY, decided

RETIRED = ("clean", "translate", "simplify", "analysis")


def test_the_four_are_named_once_and_are_no_class() -> None:
    assert classnames.RETIRED_CLASSES == RETIRED
    for name in RETIRED:
        assert name not in classnames.CLASS_NAMES
        assert name not in classnames.ROUTABLE_CLASSES
        assert name not in classnames.SELECTABLE_CLASSES
        assert name not in classnames.CHAT_VERBS
    assert classnames.ROUTABLE_CLASSES == classnames.CHAT_VERBS == ("generate",)


def _old_rows(capability: Any) -> list[dict[str, Any]]:
    """The rows a record written before 2026-10-25 carried for the retired classes."""
    return [
        {
            "capability": name,
            "enabled": True,
            "selected": "qwen3.5-9b" if name == "clean" else "qwen3.8-27b-4bit",
            "reason": f"{name}: what this card decided before the retirement",
            "summary": f"can {name}",
            "shortfall_bytes": 0,
        }
        for name in RETIRED
    ]


def write_an_old_config(home: Path) -> None:
    """A config.toml as a server from before the retirement left it: every retired class
    in [local_models], [routes] (one of them to an upstream this file does not even
    configure) and the capability record, beside settings that must survive."""
    configure_box(home, enable_llm=True, capability=decided())
    path = home / "config.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["capability"]["classes"] = [
        *document["capability"]["classes"],
        *_old_rows(document["capability"]),
    ]
    document["upstreams"] = {"anthropic": {"key": ANTHROPIC_KEY}}
    document["routes"] = {
        "generate": "anthropic/claude-sonnet-5",
        "translate": "anthropic/claude-sonnet-5",
        "simplify": "openai/gpt-5",
        "clean": "anthropic/claude-haiku-5",
        "analysis": "anthropic/claude-sonnet-5",
    }
    document["local_models"] = {
        "decide": "qwen3.5-4b",
        "clean": "qwen3.5-9b",
        "translate": "qwen3.8-27b-4bit",
        "simplify": "qwen3.5-9b",
        "analysis": "qwen3.5-9b",
    }
    path.write_text(tomli_w.dumps(document), encoding="utf-8")


EXPECTED_DROPS = (
    "[local_models] analysis",
    "[local_models] clean",
    "[local_models] simplify",
    "[local_models] translate",
    "[routes] analysis",
    "[routes] clean",
    "[routes] simplify",
    "[routes] translate",
    "[capability] row clean",
    "[capability] row translate",
    "[capability] row simplify",
    "[capability] row analysis",
)


def test_an_old_config_loads_without_the_retired_classes_and_keeps_the_rest(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_an_old_config(home)
    loaded = load_config(home)

    assert loaded.retired_dropped == EXPECTED_DROPS
    assert [(r.capability, r.model) for r in loaded.routes] == [
        ("generate", "anthropic/claude-sonnet-5")
    ]
    assert [(m.capability, m.model) for m in loaded.local_models] == [
        ("decide", "qwen3.5-4b")
    ]
    assert loaded.capability is not None
    rows = {row.capability for row in loaded.capability.rows}
    assert not rows & set(RETIRED)
    assert {"generate", "decide", "pages", "tts"} <= rows
    assert loaded.upstream("anthropic").key == ANTHROPIC_KEY
    assert loaded.enable_llm is True and loaded.name == "crucible@test"

    said = capsys.readouterr().err
    assert said.count("retired on 2026-10-25") == 1, said
    for drop in EXPECTED_DROPS:
        assert drop in said
    assert "dropped on load" in said and "the apps (BookForge, Foundry)" in said
    load_config(home)
    assert "retired" not in capsys.readouterr().err, "said once per file, not per read"


def test_the_file_is_untouched_by_a_load_and_rewritten_once_by_a_starting_server(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_an_old_config(home)
    path = home / "config.toml"
    before = path.read_text(encoding="utf-8")
    loaded = load_config(home)
    assert path.read_text(encoding="utf-8") == before, "a load writes nothing"

    with TestClient(create_app(loaded, FAKE_BACKEND)):
        pass
    after = tomllib.loads(path.read_text(encoding="utf-8"))
    assert after["routes"] == {"generate": "anthropic/claude-sonnet-5"}
    assert after["local_models"] == {"decide": "qwen3.5-4b"}
    assert not {row["capability"] for row in after["capability"]["classes"]} & set(RETIRED)
    assert after["upstreams"] == {"anthropic": {"key": ANTHROPIC_KEY}}
    assert loaded.retired_dropped == (), "the running server adopted the rewritten file"
    assert f"rewrote {path} without " in capsys.readouterr().err

    again = load_config(home)
    assert again.retired_dropped == ()
    with TestClient(create_app(again, FAKE_BACKEND)):
        pass
    assert "rewrote" not in capsys.readouterr().err


def test_a_name_that_was_never_a_class_is_still_refused(home: Path) -> None:
    configure_box(home)
    path = home / "config.toml"
    plain = path.read_text(encoding="utf-8")
    path.write_text(plain + '\n[local_models]\ncleanup = "qwen3.5-9b"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="local_model_not_selectable 'cleanup'"):
        load_config(home)
    path.write_text(plain + '\n[routes]\ntranslation = "anthropic/x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="route_not_routable 'translation'"):
        load_config(home)


def test_dropping_leaves_the_callers_document_alone() -> None:
    document = {"routes": {"clean": "anthropic/x", "generate": "anthropic/y"}}
    out, dropped = config.drop_retired_classes(document)
    assert out["routes"] == {"generate": "anthropic/y"}
    assert dropped == ("[routes] clean",)
    assert document["routes"] == {"clean": "anthropic/x", "generate": "anthropic/y"}
    assert config.drop_retired_classes({"routes": {"generate": "anthropic/y"}})[1] == ()


@pytest.fixture
def upgraded(home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with FakeUpstream() as upstream:
        monkeypatch.setattr(upstreams, "ANTHROPIC_BASE", upstream.url)
        write_an_old_config(home)
        with TestClient(create_app(load_config(home), FAKE_BACKEND)) as client:
            yield client


def test_settings_and_capability_no_longer_show_them(
    upgraded: TestClient, auth: dict[str, str]
) -> None:
    settings = upgraded.get("/v1/settings", headers=auth).json()
    assert set(settings["routes"]) == {"generate"}
    assert settings["routes"]["generate"] == {
        "route": "upstream", "model": "anthropic/claude-sonnet-5",
    }
    assert not set(settings["local_models"]) & set(RETIRED)
    assert not set(settings["local_model_choices"]) & set(RETIRED)
    assert settings["local_models"]["decide"] == "qwen3.5-4b"
    assert settings["upstreams"]["anthropic"]["configured"] is True

    record = upgraded.get("/v1/capability", headers=auth).json()
    names = {row["capability"] for row in record["classes"]}
    assert not names & set(RETIRED)
    assert "generate" in names and "decide" in names
    classed = {name for row in record["job_types"] for name in row["classes"]}
    assert not classed & set(RETIRED)

    info = upgraded.get("/v1/info", headers=auth).json()
    assert not set(info["verbs"]) & set(RETIRED)


@pytest.mark.parametrize("verb", RETIRED)
def test_a_chat_naming_a_retired_verb_is_told_it_was_retired(
    upgraded: TestClient, auth: dict[str, str], verb: str
) -> None:
    response = upgraded.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"verb": verb, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "unknown_verb"
    assert f"{verb!r} was a capability class until 2026-10-25" in error["message"]
    assert "retired to the apps" in error["message"]
    assert error["details"] == {"verb": verb, "verbs": ["generate"], "retired": True}
    preview = upgraded.get("/v1/resolve", headers=auth, params={"verb": verb}).json()
    assert preview["refused"]["code"] == "unknown_verb"
    assert preview["refused"]["message"] == error["message"]


def test_a_session_naming_a_retired_verb_is_told_too(
    upgraded: TestClient, auth: dict[str, str]
) -> None:
    response = upgraded.post(
        "/v1/queue/sessions", headers=auth, json={"act": "clean", "verb": "clean"}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "unknown_verb"
    assert "retired to the apps" in response.json()["error"]["message"]


def test_settings_writes_naming_a_retired_class_are_told_it_was_retired(
    upgraded: TestClient, auth: dict[str, str]
) -> None:
    for patch, code, field in (
        ({"routes": {"translate": "anthropic/claude-sonnet-5"}}, "route_not_routable",
         "routes.translate"),
        ({"local_models": {"clean": "qwen3.5-9b"}}, "local_model_not_selectable",
         "local_models.clean"),
    ):
        response = upgraded.put("/v1/settings", headers=auth, json=patch)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == code
        assert "retired to the apps" in error["message"]
        assert error["details"]["field"] == field and error["details"]["retired"] is True


def test_a_capability_query_and_a_module_need_naming_one_are_told(
    upgraded: TestClient, auth: dict[str, str]
) -> None:
    queried = upgraded.get("/v1/capability", headers=auth, params={"class": "analysis"})
    assert queried.status_code == 400
    assert queried.json()["error"]["code"] == "unknown_capability"
    assert "retired to the apps" in queried.json()["error"]["message"]

    module = {
        "name": "bookforge", "version": "0.0.0+test", "job_types": [],
        "needs": [{"class": "clean"}], "subjects": [],
    }
    response = upgraded.post("/v1/tasks", headers=auth, json={"type": "module", "module": module})
    assert response.status_code == 400, response.text
    assert "'clean' was a capability class until 2026-10-25" in response.text
    assert "needs `generate` or `decide` instead" in response.text


def test_the_old_act_labels_still_work_on_chat_and_on_a_session(
    upgraded: TestClient, auth: dict[str, str]
) -> None:
    """BookForge sends `X-Crucible-Act: clean` on its chats and `act: "translate"` on its
    sessions, and keeps doing so: an act is the app's label (2.0.32), not a class."""
    chatted = upgraded.post(
        "/v1/openai/chat/completions",
        headers={**auth, "X-Crucible-Act": "clean", "User-Agent": "bookforge/1"},
        json={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert chatted.status_code == 200, chatted.text

    opened = upgraded.post("/v1/queue/sessions", headers=auth, json={"act": "translate"})
    assert opened.status_code == 202, opened.text
    session_id = opened.json()["session_id"]
    state = upgraded.get(f"/v1/queue/sessions/{session_id}", headers=auth).json()
    assert state["act"] == "translate"
    closed = upgraded.delete(f"/v1/queue/sessions/{session_id}", headers=auth)
    assert closed.status_code == 200, closed.text
