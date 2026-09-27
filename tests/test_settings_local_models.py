from __future__ import annotations

from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible.config import load_config

from .test_settings_api import decided, put


@pytest.fixture
def settings_client(make_client: Callable[..., TestClient]):
    with make_client(enable_llm=True, capability=decided()) as instance:
        yield instance


def document(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    response = client.get("/v1/settings", headers=auth)
    assert response.status_code == 200
    return response.json()


def test_the_document_offers_every_selectable_class_and_null_for_automatic(settings_client, auth):
    body = document(settings_client, auth)
    assert body["local_models"]["translate"] is None
    assert body["local_models"]["tts"] is None
    assert "echo" not in body["local_models"]
    assert "echo" not in body["local_model_choices"]

    offered = body["local_model_choices"]["translate"]
    assert [row["id"] for row in offered] == [
        "qwen3.8-27b-4bit",
        "qwen3.5-9b",
    ]
    assert [row["fits"] for row in offered] == [True, True]
    assert offered[0]["memory_bytes_estimate"] > offered[1]["memory_bytes_estimate"]
    assert all(isinstance(row["installed"], bool) for row in offered)


def test_a_choice_is_kept_written_and_decided_on(settings_client, auth, home):
    client = settings_client
    automatic = client.get("/v1/capability", headers=auth).json()
    was = next(r for r in automatic["classes"] if r["capability"] == "tts")["selected"]
    assert was == "deathstalker", "best-first, alphabetical among equals"

    response = put(client, auth, {"local_models": {"tts": "mistborn"}})
    assert response.status_code == 200
    assert response.json()["local_models"]["tts"] == "mistborn"

    assert load_config(home).local_model("tts") == "mistborn"
    rows = client.get("/v1/capability", headers=auth).json()["classes"]
    row = next(r for r in rows if r["capability"] == "tts")
    assert row["selected"] == "mistborn"
    assert row["enabled"] is True
    assert "was chosen" in row["reason"]

    recorded = load_config(home).capability
    assert recorded is not None
    assert recorded.row("tts").selected == "mistborn"


def test_a_model_that_is_not_installed_may_still_be_chosen(settings_client, auth, home):
    client = settings_client
    offered = document(client, auth)["local_model_choices"]["translate"]
    fitting = next(row for row in offered if row["fits"])
    assert fitting["installed"] is False, "this fixture has no weights on disk"

    response = put(client, auth, {"local_models": {"translate": fitting["id"]}})
    assert response.status_code == 200
    assert load_config(home).local_model("translate") == fitting["id"]


def test_a_choice_that_does_not_fit_is_refused_with_the_arithmetic(make_client, auth, home):
    with make_client(enable_llm=True, capability=decided(total=16 * 1024 ** 3)) as client:
        response = put(client, auth, {"local_models": {"translate": "qwen3.8-27b-4bit"}})
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "local_model_does_not_fit"
        details = error["details"]
        assert details["field"] == "local_models.translate"
        assert details["capability"] == "translate"
        assert details["model"] == "qwen3.8-27b-4bit"
        assert details["shortfall_bytes"] == (
            details["memory_bytes_estimate"] - details["available_bytes"]
        )
        assert details["shortfall_bytes"] > 0
        assert "20.1 GiB" in error["message"] and "13.0 GiB" in error["message"]
        assert load_config(home).local_model("translate") is None
        assert document(client, auth)["local_models"]["translate"] is None


def test_the_8bit_27b_is_not_a_choice_on_cuda_linux(settings_client, auth, home):
    client = settings_client
    response = put(client, auth, {"local_models": {"translate": "qwen3.8-27b-8bit"}})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "local_model_unknown"
    assert error["details"]["model"] == "qwen3.8-27b-8bit"
    assert error["details"]["choices"] == ["qwen3.8-27b-4bit", "qwen3.5-9b"]
    assert load_config(home).local_model("translate") is None


def test_null_restores_the_automatic_decision(settings_client, auth, home):
    client = settings_client
    assert put(client, auth, {"local_models": {"tts": "mistborn"}}).status_code == 200
    assert load_config(home).local_model("tts") == "mistborn"

    response = put(client, auth, {"local_models": {"tts": None}})
    assert response.status_code == 200
    assert response.json()["local_models"]["tts"] is None
    assert load_config(home).local_model("tts") is None
    assert "local_models" not in (home / "config.toml").read_text(encoding="utf-8")
    rows = client.get("/v1/capability", headers=auth).json()["classes"]
    assert next(r for r in rows if r["capability"] == "tts")["selected"] == "deathstalker"


def test_a_class_with_nothing_to_choose_between_is_refused_by_name(settings_client, auth):
    response = put(settings_client, auth, {"local_models": {"echo": "anything"}})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "local_model_not_selectable"
    assert error["details"]["field"] == "local_models.echo"
    assert "tts" in error["details"]["selectable"]
    assert "echo" not in error["details"]["selectable"]


def test_an_unknown_model_is_refused_and_names_what_there_was(settings_client, auth):
    response = put(settings_client, auth, {"local_models": {"translate": "gpt-9"}})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "local_model_unknown"
    assert error["details"]["choices"] == [
        "qwen3.8-27b-4bit",
        "qwen3.5-9b",
    ]
    assert error["details"]["field"] == "local_models.translate"


def test_a_selection_is_measured_against_the_allowance_in_the_same_patch(settings_client, auth, home):
    client = settings_client
    response = put(client, auth, {
        "desktop_allowance_bytes": 20 * 1024 ** 3,
        "local_models": {"translate": "qwen3.8-27b-4bit"},
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "local_model_does_not_fit"
    assert load_config(home).desktop_allowance_bytes == 3 * 1024 ** 3
