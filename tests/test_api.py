"""The API v1 surface: ping, auth, api version, info, health."""

from __future__ import annotations

from typing import Callable

from fastapi.testclient import TestClient

from crucible import API_VERSION, VERSION

from .conftest import TOKEN


def test_ping_needs_no_auth_and_no_version_header(client: TestClient) -> None:
    response = client.get("/v1/ping")
    assert response.status_code == 200
    body = response.json()
    assert body == {"crucible": True, "name": "crucible@test", "api_version": API_VERSION}


def test_missing_authorization_is_401(client: TestClient) -> None:
    response = client.get("/v1/info", headers={"X-Crucible-Api": str(API_VERSION)})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_wrong_token_is_401(client: TestClient) -> None:
    response = client.get(
        "/v1/info",
        headers={"Authorization": "Bearer nope", "X-Crucible-Api": str(API_VERSION)},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_non_bearer_scheme_is_401(client: TestClient) -> None:
    response = client.get(
        "/v1/info",
        headers={"Authorization": f"Basic {TOKEN}", "X-Crucible-Api": str(API_VERSION)},
    )
    assert response.status_code == 401


def test_auth_is_checked_before_the_api_version(client: TestClient) -> None:
    """No token and no version header answers 401, not 426."""
    response = client.get("/v1/info")
    assert response.status_code == 401


def test_missing_api_version_header_is_426(client: TestClient) -> None:
    response = client.get("/v1/info", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 426
    error = response.json()["error"]
    assert error["code"] == "api_version_required"
    assert error["details"]["server_api_version"] == API_VERSION


def test_api_version_mismatch_is_426_naming_both(client: TestClient) -> None:
    response = client.get(
        "/v1/info",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Crucible-Api": "99"},
    )
    assert response.status_code == 426
    error = response.json()["error"]
    assert error["code"] == "api_version_mismatch"
    assert error["details"] == {
        "server_api_version": API_VERSION,
        "client_api_version": 99,
    }
    assert "99" in error["message"] and str(API_VERSION) in error["message"]


def test_unreadable_api_version_is_426(client: TestClient) -> None:
    response = client.get(
        "/v1/info",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Crucible-Api": "one"},
    )
    assert response.status_code == 426
    assert response.json()["error"]["code"] == "api_version_unreadable"


def test_info_shape(client: TestClient, auth: dict[str, str]) -> None:
    body = client.get("/v1/info", headers=auth).json()
    assert body["server"] == {
        "name": "crucible@test",
        "version": VERSION,
        "api_version": API_VERSION,
    }
    assert body["host"] == {
        "platform": "linux",
        "arch": "x86_64",
        "backend": "cuda-linux",
        "gpu": {
            "vendor": "nvidia",
            "name": "NVIDIA GeForce RTX 3090 Ti",
            "vram_bytes": 25_757_220_864,
        },
    }
    assert body["capabilities"] == [{"job_type": "echo", "models": []}]


def test_info_without_echo_advertises_nothing(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_echo=False) as client:
        body = client.get("/v1/info", headers=auth).json()
        assert body["capabilities"] == []


def test_health(client: TestClient, auth: dict[str, str]) -> None:
    body = client.get("/v1/health", headers=auth).json()
    assert body == {"status": "ok", "queue_depth": 0, "resident_models": []}


def test_unknown_route_is_json_error(client: TestClient, auth: dict[str, str]) -> None:
    response = client.get("/v1/nope", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
