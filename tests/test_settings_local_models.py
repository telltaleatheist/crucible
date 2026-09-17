"""An app's own choice of local model, and the two ways it is refused.

INTENT.md: *"BookForge chooses its voice models and other required
capabilities. Foundry chooses its input-processing and language models."* The
engine runs them and owns the arithmetic; this door is where the choice is
made, so the two never have to guess at each other.

The numbers here are the real ones a `cuda-linux` build ships: `qwen3.8-27b`
at 52.5 GiB and `qwen3.8-27b-4bit` at 20.1 GiB, against the 23 GiB budget the
shared fixture decides (26 GiB card, 3 GiB desktop allowance). That pair is not
a convenience — it is the Mac's own configuration, where the default names a
full-size 27B that is not installed while the 4-bit variant beside it runs.
"""

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
    # `echo` has nothing to choose between, so it is absent rather than null:
    # null is "nobody chose", and that is a different claim from "there is no
    # choice to make here".
    assert "echo" not in body["local_models"]
    assert "echo" not in body["local_model_choices"]

    offered = body["local_model_choices"]["translate"]
    # THE 9B IS IN THIS LIST SINCE 2026-09-16, and this row is where Owen's
    # ruling becomes a thing a person can click: *"bookforge/foundry should give
    # the user the option of using 3.5:9b or 3.8:27b IF their system can manage
    # it."* Foundry's settings card renders exactly this list, so its picker
    # gains the option with no app change.
    #
    # Ordered best-first by declared size, which is why the 9B is LAST: the
    # default is still the largest thing that fits, and preferring a 9B at bf16
    # over a 27B at 4-bit for speed is a trade no arithmetic here can rank.
    assert [row["id"] for row in offered] == [
        "qwen3.8-27b",
        "qwen3.8-27b-4bit",
        "qwen3.5-9b",
    ]
    assert [row["fits"] for row in offered] == [False, True, True]
    assert offered[0]["memory_bytes_estimate"] > offered[1]["memory_bytes_estimate"]
    # `installed` is a fact about this disk and every row carries it, so a
    # chooser never has to infer "probably not" from a missing key.
    assert all(isinstance(row["installed"], bool) for row in offered)


def test_a_choice_is_kept_written_and_decided_on(settings_client, auth, home):
    client = settings_client
    automatic = client.get("/v1/capability", headers=auth).json()
    was = next(r for r in automatic["classes"] if r["capability"] == "tts")["selected"]
    assert was == "deathstalker", "best-first, alphabetical among equals"

    response = put(client, auth, {"local_models": {"tts": "mistborn"}})
    assert response.status_code == 200
    assert response.json()["local_models"]["tts"] == "mistborn"

    # Written, so it survives a restart...
    assert load_config(home).local_model("tts") == "mistborn"
    # ...and DECIDED ON, so the capability row names the chosen voice rather
    # than the one the best-first walk would have taken.
    rows = client.get("/v1/capability", headers=auth).json()["classes"]
    row = next(r for r in rows if r["capability"] == "tts")
    assert row["selected"] == "mistborn"
    assert row["enabled"] is True
    assert "was chosen" in row["reason"]

    # AND THE CHOICE IS WHAT PREPARATION WILL FETCH. `tasks.py`'s `_resolve_need`
    # reads exactly one thing to decide what a module owes — the RECORDED
    # capability row's `selected` — so this assertion is the join between "an app
    # chose a model" and "Crucible downloads that model". Owen, 2026-09-16: a
    # model that is not installed may still be chosen, because the choice is the
    # demand. Read off the file rather than the response, because a restarted
    # server and the task runner both read the file.
    recorded = load_config(home).capability
    assert recorded is not None
    assert recorded.row("tts").selected == "mistborn"


def test_a_model_that_is_not_installed_may_still_be_chosen(settings_client, auth, home):
    """Owen, 2026-09-16: accept it; preparation is what downloads the weights.

    Refusing here would force an app to install a model before it was allowed
    to say it wanted one, which is backwards — the choice is the demand that
    preparation reads.
    """
    client = settings_client
    offered = document(client, auth)["local_model_choices"]["translate"]
    fitting = next(row for row in offered if row["fits"])
    assert fitting["installed"] is False, "this fixture has no weights on disk"

    response = put(client, auth, {"local_models": {"translate": fitting["id"]}})
    assert response.status_code == 200
    assert load_config(home).local_model("translate") == fitting["id"]


def test_a_choice_that_does_not_fit_is_refused_with_the_arithmetic(settings_client, auth, home):
    client = settings_client
    response = put(client, auth, {"local_models": {"translate": "qwen3.8-27b"}})
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "local_model_does_not_fit"
    details = error["details"]
    assert details["field"] == "local_models.translate"
    assert details["capability"] == "translate"
    assert details["model"] == "qwen3.8-27b"
    assert details["shortfall_bytes"] == (
        details["memory_bytes_estimate"] - details["available_bytes"]
    )
    assert details["shortfall_bytes"] > 0
    # The numbers are in the sentence too, because the person who chose is the
    # one who has to act on them.
    assert "52.5 GiB" in error["message"] and "23.0 GiB" in error["message"]
    # A REFUSAL APPLIES NOTHING.
    assert load_config(home).local_model("translate") is None
    assert document(client, auth)["local_models"]["translate"] is None


def test_null_restores_the_automatic_decision(settings_client, auth, home):
    client = settings_client
    assert put(client, auth, {"local_models": {"tts": "mistborn"}}).status_code == 200
    assert load_config(home).local_model("tts") == "mistborn"

    response = put(client, auth, {"local_models": {"tts": None}})
    assert response.status_code == 200
    assert response.json()["local_models"]["tts"] is None
    assert load_config(home).local_model("tts") is None
    # The table is gone from the file entirely, not left behind empty.
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
        "qwen3.8-27b",
        "qwen3.8-27b-4bit",
        "qwen3.5-9b",
    ]
    assert error["details"]["field"] == "local_models.translate"


def test_a_selection_is_measured_against_the_allowance_in_the_same_patch(settings_client, auth, home):
    """One patch, applied in order: the allowance moves the budget the choice is checked against."""
    client = settings_client
    # 20.1 GiB fits the 23 GiB budget, but not once 20 GiB is reserved for the desktop.
    response = put(client, auth, {
        "desktop_allowance_bytes": 20 * 1024 ** 3,
        "local_models": {"translate": "qwen3.8-27b-4bit"},
    })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "local_model_does_not_fit"
    # Neither half landed: the allowance is unchanged too.
    assert load_config(home).desktop_allowance_bytes == 3 * 1024 ** 3
