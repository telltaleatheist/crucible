import pytest

from crucible.connect import MAX_REQUESTS, PairingRequests
from crucible.errors import ApiError

from .conftest import TOKEN

HEADERS = {"X-Crucible-Api": "1"}
AUTH = {**HEADERS, "Authorization": "Bearer " + TOKEN}


def test_pairing_requires_operator_approval_and_device_secret(make_client):
    with make_client(open_pairing=False) as client:
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
        store = client.app.state.pairing_requests
        store.entries[request["id"]].last_poll -= 2
        result = client.post("/v1/pairing/poll", headers=HEADERS, json=credentials)
        assert result.headers["cache-control"] == "no-store"
        assert result.json() == {"status": "approved", "name": "crucible@test", "token": TOKEN}
        assert client.get("/v1/pairing/requests", headers=AUTH).json() == {"requests": []}


def test_pairing_denial_expiry_and_poll_throttle():
    now = [0.0]
    store = PairingRequests(lambda: now[0], open_pairing=False)
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
    store = PairingRequests(lambda: now[0], open_pairing=False)
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


def test_reaching_the_engine_is_the_whole_of_the_authorisation(make_client):
    with make_client() as client:
        start = client.post(
            "/v1/pairing/start", headers=HEADERS, json={"client_name": "BookForge"}
        )
        assert start.status_code == 200
        request = start.json()
        assert "token" not in request
        assert request["approval_required"] is False

        answer = client.post("/v1/pairing/poll", headers=HEADERS, json={
            "id": request["id"], "device_code": request["device_code"],
        }).json()
        assert answer == {"status": "approved", "name": "crucible@test", "token": TOKEN}

        assert client.get("/v1/pairing/requests", headers=AUTH).json() == {"requests": []}


def test_an_open_engine_still_refuses_the_wrong_device_secret(make_client):
    with make_client() as client:
        request = client.post(
            "/v1/pairing/start", headers=HEADERS, json={"client_name": "BookForge"}
        ).json()
        stolen = {"id": request["id"], "device_code": "x" * 43}
        assert client.post("/v1/pairing/poll", headers=HEADERS, json=stolen).status_code == 403


def test_the_approval_step_is_kept_and_a_config_can_ask_for_it(make_client):
    with make_client(open_pairing=False) as client:
        request = client.post(
            "/v1/pairing/start", headers=HEADERS, json={"client_name": "BookForge"}
        ).json()
        assert request["approval_required"] is True
        assert client.post("/v1/pairing/poll", headers=HEADERS, json={
            "id": request["id"], "device_code": request["device_code"],
        }).json() == {"status": "pending"}
        assert len(client.get("/v1/pairing/requests", headers=AUTH).json()["requests"]) == 1


def test_open_pairing_survives_a_config_round_trip(home):
    from crucible.config import load_config, write_config

    for chosen in (True, False):
        write_config(
            home, name="crucible@test", host="127.0.0.1", port=7100, token=TOKEN,
            backend_kind="cuda-linux", desktop_allowance_bytes=1,
            retention_days=7,
            desktop_allowance_basis="stated",
            desktop_allowance_note="",
            enable_echo=True, enable_llm=False, enable_asr=False, enable_tts=False,
            enable_align=False, enable_rvc=False, enable_denoise=False,
            open_pairing=chosen,
        )
        assert load_config(home).open_pairing is chosen


def test_a_quoted_boolean_is_refused_rather_than_read_as_true():
    from crucible.config import ConfigError, _open_pairing

    assert _open_pairing({"auth": {}}) is True, "absent means the ruled default"
    assert _open_pairing({}) is True
    assert _open_pairing({"auth": {"open_pairing": False}}) is False
    for bad in ("false", "no", 0, 1, None, []):
        with pytest.raises(ConfigError, match="true or false"):
            _open_pairing({"auth": {"open_pairing": bad}})
