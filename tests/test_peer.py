from __future__ import annotations

from typing import Any, Callable

from fastapi.testclient import TestClient

from crucible import API_VERSION, peer

from .conftest import TOKEN

ORCHESTRATOR = {
    "name": "crucible-orchestrator@owens-pc",
    "url": "http://127.0.0.1:7101",
    "version": "0.6.0",
}
OTHER = {
    "name": "crucible-orchestrator@somewhere-else",
    "url": "http://127.0.0.1:7999",
    "version": "0.6.0",
}


def claim(
    client: TestClient, auth: dict[str, str], who: dict[str, str] = ORCHESTRATOR, **extra: Any
) -> Any:
    return client.post(
        "/v1/peer/claim", headers=auth, json={"orchestrator": who, **extra}
    )


def test_an_engine_records_who_claimed_it_and_answers_info_with_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    before = client.get("/v1/info", headers=auth).json()
    assert before["role"] == "engine"
    assert before["managed_by"] is None, "an unclaimed engine is a whole Crucible"

    answer = claim(client, auth)
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["role"] == "engine"
    assert body["managed_by"] == {
        "name": ORCHESTRATOR["name"],
        "url": ORCHESTRATOR["url"],
    }
    assert body["claimed"], "a claim is stamped"

    after = client.get("/v1/info", headers=auth).json()
    assert after["managed_by"] == body["managed_by"]


def test_managed_by_carries_the_name_and_the_url_and_NOT_the_version(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    managed = client.get("/v1/info", headers=auth).json()["managed_by"]
    assert sorted(managed) == ["name", "url"]


def test_the_same_orchestrator_re_claiming_is_SUCCESS_and_not_a_conflict(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = claim(client, auth).json()
    second = claim(client, auth)
    assert second.status_code == 200, second.text
    assert second.json()["managed_by"] == first["managed_by"]


def test_a_trailing_slash_is_the_SAME_orchestrator(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    again = claim(client, auth, {**ORCHESTRATOR, "url": ORCHESTRATOR["url"] + "/"})
    assert again.status_code == 200, again.text


def test_a_DIFFERENT_orchestrator_is_refused_by_name_and_told_who_holds_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    refused = claim(client, auth, OTHER)
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == peer.PEER_ALREADY_MANAGED
    assert error["details"]["managed_by"]["url"] == ORCHESTRATOR["url"]
    assert error["details"]["claimant"]["url"] == OTHER["url"]
    assert (
        client.get("/v1/info", headers=auth).json()["managed_by"]["url"]
        == ORCHESTRATOR["url"]
    )


def test_force_takes_it_and_is_a_persons_act(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    taken = claim(client, auth, OTHER, force=True)
    assert taken.status_code == 200, taken.text
    assert taken.json()["managed_by"]["url"] == OTHER["url"]


def test_force_must_be_a_boolean(client: TestClient, auth: dict[str, str]) -> None:
    refused = claim(client, auth, force="yes")
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "invalid_request"


def test_an_orchestrator_that_cannot_say_who_it_is_is_refused(
    client: TestClient, auth: dict[str, str]
) -> None:
    for missing in ("name", "url", "version"):
        body = {key: value for key, value in ORCHESTRATOR.items() if key != missing}
        refused = claim(client, auth, body)
        assert refused.status_code == 400, missing
        assert refused.json()["error"]["code"] == "invalid_request"
    assert client.post("/v1/peer/claim", headers=auth, json={}).status_code == 400


def test_the_holder_releases_and_the_engine_is_unmanaged_again(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    released = client.request(
        "DELETE", "/v1/peer/claim", headers=auth, json={"orchestrator": ORCHESTRATOR}
    )
    assert released.status_code == 200, released.text
    assert released.json() == {"role": "engine", "managed_by": None}
    assert client.get("/v1/info", headers=auth).json()["managed_by"] is None


def test_releasing_when_nothing_is_claimed_is_NOT_a_refusal(
    client: TestClient, auth: dict[str, str]
) -> None:
    answer = client.request("DELETE", "/v1/peer/claim", headers=auth)
    assert answer.status_code == 200
    assert answer.json()["managed_by"] is None


def test_you_do_not_release_somebody_elses_claim_by_accident(
    client: TestClient, auth: dict[str, str]
) -> None:
    claim(client, auth)
    refused = client.request(
        "DELETE", "/v1/peer/claim", headers=auth, json={"orchestrator": OTHER}
    )
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == peer.PEER_ALREADY_MANAGED
    assert client.get("/v1/info", headers=auth).json()["managed_by"] is not None


def test_a_wrong_or_missing_bearer_is_peer_token_mismatch_and_not_unauthorized(
    client: TestClient
) -> None:
    version = {"X-Crucible-Api": str(API_VERSION)}
    for headers in (
        version,
        {**version, "Authorization": "Bearer wrong"},
        {**version, "Authorization": TOKEN},
        {**version, "Authorization": "Bearer "},
    ):
        answer = client.post(
            "/v1/peer/claim", headers=headers, json={"orchestrator": ORCHESTRATOR}
        )
        assert answer.status_code == 401, headers
        assert answer.json()["error"]["code"] == peer.PEER_TOKEN_MISMATCH


def test_a_different_api_version_is_peer_version_incompatible(
    client: TestClient
) -> None:
    for presented in ("2", "99", "not-a-version", None):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        if presented is not None:
            headers["X-Crucible-Api"] = presented
        answer = client.post(
            "/v1/peer/claim", headers=headers, json={"orchestrator": ORCHESTRATOR}
        )
        assert answer.status_code == 426, presented
        assert answer.json()["error"]["code"] == peer.PEER_VERSION_INCOMPATIBLE


def test_the_token_is_checked_BEFORE_the_version(client: TestClient) -> None:
    answer = client.post("/v1/peer/claim", json={"orchestrator": ORCHESTRATOR})
    assert answer.json()["error"]["code"] == peer.PEER_TOKEN_MISMATCH


def test_the_relation_read_carries_the_role_the_claim_and_a_monotonic_uptime(
    client: TestClient, auth: dict[str, str]
) -> None:
    first = client.get("/v1/peer", headers=auth)
    assert first.status_code == 200, first.text
    assert first.json()["role"] == "engine"
    assert first.json()["managed_by"] is None
    assert first.json()["uptime_s"] >= 0.0

    claim(client, auth)
    second = client.get("/v1/peer", headers=auth).json()
    assert second["managed_by"]["name"] == ORCHESTRATOR["name"]
    assert second["uptime_s"] >= first.json()["uptime_s"]


def test_there_is_no_engine_to_orchestrator_callback_anywhere_on_the_wire() -> None:
    source = (peer.__file__ and open(peer.__file__, encoding="utf-8").read()) or ""
    assert "def claim_engine" in source
    assert "def release_engine" in source
    assert "def read_peer" in source
    assert "def read_info" in source
    state_source = source[source.index("class PeerState") : source.index("class PeerCallFailed")]
    assert "urlopen" not in state_source
    assert "_call(" not in state_source


def test_a_pre_phase_17_document_reads_as_an_engine_with_no_orchestrator() -> None:
    old = {"server": {"name": "n", "version": "0.5.0", "api_version": 1}}
    assert "role" not in old
    assert old.get("role", peer.ROLE_ENGINE) == peer.ROLE_ENGINE
    assert old.get("managed_by") is None


def test_the_roles_and_the_owners_are_exactly_these_words() -> None:
    assert peer.ROLES == ("engine", "orchestrator")
    assert peer.OWNERS == ("wsl-unit", "child", "found")
    assert peer.BACKEND_ORCHESTRATOR == "orchestrator"


def test_orchestrator_is_a_backend_kind_on_the_wire_and_NOWHERE_else() -> None:
    from crucible.backend import BACKEND_KINDS

    assert peer.BACKEND_ORCHESTRATOR not in BACKEND_KINDS


def test_a_claim_dies_with_the_process_and_is_written_NOWHERE(
    make_client: Callable[..., TestClient], auth: dict[str, str], tmp_path: Any
) -> None:
    with make_client() as first:
        home = first.app.state.config.path.parent
        claim(first, auth)
        assert first.get("/v1/info", headers=auth).json()["managed_by"] is not None
        written = sorted(p.name for p in home.rglob("*") if p.is_file())
    with make_client() as second:
        assert second.get("/v1/info", headers=auth).json()["managed_by"] is None
    for name in written:
        assert "peer" not in name and "claim" not in name, name


def test_two_servers_do_not_share_a_claim(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client() as one, make_client() as two:
        claim(one, auth)
        assert one.get("/v1/info", headers=auth).json()["managed_by"] is not None
        assert two.get("/v1/info", headers=auth).json()["managed_by"] is None


def test_the_state_object_refuses_and_releases_without_a_server() -> None:
    state = peer.PeerState()
    assert state.managed_by() is None
    mine = peer.Orchestrator("a", "http://127.0.0.1:7101", "0.6.0")
    theirs = peer.Orchestrator("b", "http://127.0.0.1:7999", "0.6.0")
    state.claim(mine)
    assert state.managed_by() == {"name": "a", "url": "http://127.0.0.1:7101"}
    state.claim(mine)
    try:
        state.claim(theirs)
        raise AssertionError("a second orchestrator must be refused")
    except Exception as exc:
        assert getattr(exc, "code", "") == peer.PEER_ALREADY_MANAGED
    state.claim(theirs, force=True)
    assert state.managed_by()["name"] == "b"
    state.release(theirs)
    assert state.managed_by() is None
    state.claim(mine)
    state.release(None)
    assert state.managed_by() is None


def test_url_normalisation_touches_the_trailing_slash_and_nothing_else() -> None:
    assert peer.normalise_url("http://127.0.0.1:7101/") == "http://127.0.0.1:7101"
    assert peer.normalise_url("  http://127.0.0.1:7101  ") == "http://127.0.0.1:7101"
    assert peer.normalise_url("http://LocalHost:7101") == "http://LocalHost:7101"


def test_an_engines_refusal_code_survives_the_trip_back_to_the_orchestrator() -> None:
    assert (
        peer._refusal_code('{"error": {"code": "peer_already_managed"}}')
        == peer.PEER_ALREADY_MANAGED
    )
    assert peer._refusal_code("<html>nginx</html>") == "peer_unreadable"
    assert peer._refusal_code('{"error": {}}') == "peer_unreadable"


def test_an_unreachable_engine_is_a_named_failure_and_never_an_exception_to_crash_on() -> None:
    try:
        peer.claim_engine(
            "http://127.0.0.1:1",
            "tok",
            peer.Orchestrator("a", "http://127.0.0.1:7101", "0.6.0"),
            api_version=API_VERSION,
            timeout_s=2.0,
        )
        raise AssertionError("an unreachable engine must be reported")
    except peer.PeerCallFailed as exc:
        assert exc.code == "peer_unreachable"
