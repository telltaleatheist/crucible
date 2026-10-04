from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

PATH = "/v1/playground/presets/yue2-3b"
LOFI = {"tags": "lo-fi, jazz, light drums, 70 BPM", "instrumental": True, "cfg": 1.0}


def test_a_preset_is_saved_listed_replaced_and_deleted(client: TestClient, auth: dict[str, str]) -> None:
    assert client.get(PATH, headers=auth).json()["presets"] == []
    saved = client.put(f"{PATH}/Late night", headers=auth, json={"params": LOFI})
    assert saved.status_code == 200, saved.text
    rows = client.get(PATH, headers=auth).json()["presets"]
    assert [(r["name"], r["params"]) for r in rows] == [("Late night", LOFI)]
    client.put(f"{PATH}/Late night", headers=auth, json={"params": {**LOFI, "cfg": 1.5}})
    assert client.get(PATH, headers=auth).json()["presets"][0]["params"]["cfg"] == 1.5
    assert client.delete(f"{PATH}/Late night", headers=auth).status_code == 200
    assert client.get(PATH, headers=auth).json()["presets"] == []
    assert client.delete(f"{PATH}/Late night", headers=auth).status_code == 404


def test_a_preset_keeps_a_sound_not_a_take(client: TestClient, auth: dict[str, str]) -> None:
    refused = client.put(f"{PATH}/x", headers=auth, json={"params": {**LOFI, "seed": 7}})
    assert refused.status_code == 400
    error: dict[str, Any] = refused.json()["error"]
    assert error["code"] == "preset_params_invalid" and "seed" in error["message"]


def test_presets_survive_a_new_server_on_the_same_home(make_client, auth: dict[str, str]) -> None:
    with make_client() as first:
        first.put(f"{PATH}/Keep", headers=auth, json={"params": LOFI})
    with make_client() as second:
        assert [r["name"] for r in second.get(PATH, headers=auth).json()["presets"]] == ["Keep"]


def test_presets_need_the_token(client: TestClient) -> None:
    assert client.get(PATH).status_code == 401
