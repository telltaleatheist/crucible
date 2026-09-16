"""Public discovery does not grant access; an authenticated matching-code approval does."""
import pytest

from crucible.connect import PairingRequests, MAX_REQUESTS
from crucible.errors import ApiError
from .conftest import TOKEN

HEADERS = {"X-Crucible-Api": "1"}
AUTH = {**HEADERS, "Authorization": "Bearer " + TOKEN}


def test_pairing_requires_operator_approval_and_device_secret(make_client):
    with make_client() as client:
        start = client.post("/v1/pairing/start", headers=HEADERS, json={"client_name": "BookForge"})
        assert start.status_code == 200
        assert start.headers["cache-control"] == "no-store"
        request = start.json()
        assert "token" not in request
        credentials = {"id": request["id"], "device_code": request["device_code"]}
        assert client.post("/v1/pairing/poll", headers=HEADERS, json=credentials).json() == {"status": "pending"}
        assert client.get("/v1/pairing/requests", headers=HEADERS).status_code == 401
        decision = {"id": request["id"], "user_code": request["user_code"], "allow": True}
        assert client.post("/v1/pairing/decision", headers=HEADERS, json=decision).status_code == 401
        pending = client.get("/v1/pairing/requests", headers=AUTH).json()["requests"]
        assert pending[0]["user_code"] == request["user_code"]
        assert "device_code" not in pending[0] and "secret_hash" not in pending[0]
        wrong = {**credentials, "device_code": "x" * 43}
        assert client.post("/v1/pairing/poll", headers=HEADERS, json=wrong).status_code == 403
        assert client.post("/v1/pairing/decision", headers=AUTH, json=decision).json() == {"status": "approved"}
        # Advance the fixture's clock rather than sleeping or touching any service.
        store = client.app.state.pairing_requests
        store.entries[request["id"]].last_poll -= 2
        result = client.post("/v1/pairing/poll", headers=HEADERS, json=credentials)
        assert result.headers["cache-control"] == "no-store"
        assert result.json() == {"status": "approved", "name": "crucible@test", "token": TOKEN}
        assert client.get("/v1/pairing/requests", headers=AUTH).json() == {"requests": []}


def test_pairing_denial_expiry_and_poll_throttle():
    now = [0.0]
    store = PairingRequests(lambda: now[0])
    row = store.start("Foundry", "fixture")
    assert store.poll(row["id"], row["device_code"]) == "pending"
    with pytest.raises(ApiError) as busy:
        store.poll(row["id"], row["device_code"])
    assert busy.value.code == "pairing_slow_down"
    with pytest.raises(ApiError):
        store.decide(row["id"], "WRNG-CODE", True)
    store.decide(row["id"], row["user_code"], False)
    now[0] = 3
    assert store.poll(row["id"], row["device_code"]) == "denied"
    with pytest.raises(ApiError):
        store.decide(row["id"], row["user_code"], True)
    now[0] = 301
    assert store.poll(row["id"], row["device_code"]) == "expired"
    assert store.pending() == []


def test_pairing_requests_are_bounded_and_expire():
    now = [0.0]
    store = PairingRequests(lambda: now[0])
    store.start("one", "same-ip")
    with pytest.raises(ApiError):
        store.start("two", "same-ip")
    for index in range(MAX_REQUESTS - 1):
        store.start("fixture", str(index))
    with pytest.raises(ApiError):
        store.start("too many", "another-ip")
    now[0] = 301
    assert store.start("again", "same-ip")
    assert len(store.pending()) == 1


def test_pairing_rejects_cross_origin_simple_posts_and_bad_names(make_client):
    with make_client() as client:
        assert client.post("/v1/pairing/start", json={"client_name": "browser"}).status_code == 426
        for name in ("", "   ", "bad\nname", "x" * 81):
            assert client.post("/v1/pairing/start", headers=HEADERS, json={"client_name": name}).status_code == 400
        assert client.post("/v1/pairing/decision", headers=AUTH, json={
            "id": "fixture", "user_code": "éééé-éééé", "allow": True,
        }).status_code == 400
