"""The orchestrator/engine relation, from the ENGINE's side. PHASE17 2 and 3.

NOTHING HERE STARTS A DISTRO, A SERVER, A TRAY OR A MODEL. The orchestrator's
own half is `tests/test_host.py`; what is pinned here is the wire an
orchestrator talks to — the claim, the release, the three refusals, and the
two fields `/v1/info` gained.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi.testclient import TestClient

from crucible import API_VERSION, peer
from crucible.api import create_app

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


# ------------------------------------------------------------------- claiming


def test_an_engine_records_who_claimed_it_and_answers_info_with_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    """The whole of what a claim buys: `/v1/info` can answer "who manages this"."""
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
    """The orchestrator's version is a fact about the ORCHESTRATOR.

    `managed_by` answers "who manages this and where do I reach them". Two
    copies of a version string in two documents is two things to keep in step
    across an upgrade that changes exactly one of them; `GET /v1/info` on the
    orchestrator's own door is where its version lives.
    """
    claim(client, auth)
    managed = client.get("/v1/info", headers=auth).json()["managed_by"]
    assert sorted(managed) == ["name", "url"]


def test_the_same_orchestrator_re_claiming_is_SUCCESS_and_not_a_conflict(
    client: TestClient, auth: dict[str, str]
) -> None:
    """2.1, and it must be: the orchestrator re-claims on every down-to-up edge."""
    first = claim(client, auth).json()
    second = claim(client, auth)
    assert second.status_code == 200, second.text
    assert second.json()["managed_by"] == first["managed_by"]


def test_a_trailing_slash_is_the_SAME_orchestrator(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Or an engine would refuse the very orchestrator holding the claim.

    On its own watch tick. Forever.
    """
    claim(client, auth)
    again = claim(client, auth, {**ORCHESTRATOR, "url": ORCHESTRATOR["url"] + "/"})
    assert again.status_code == 200, again.text


def test_a_DIFFERENT_orchestrator_is_refused_by_name_and_told_who_holds_it(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Two orchestrators on one engine is a machine misconfigured.

    The right first answer names the other one, so a person can see both,
    rather than stealing it silently.
    """
    claim(client, auth)
    refused = claim(client, auth, OTHER)
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == peer.PEER_ALREADY_MANAGED
    assert error["details"]["managed_by"]["url"] == ORCHESTRATOR["url"]
    assert error["details"]["claimant"]["url"] == OTHER["url"]
    # And the standing claim is UNCHANGED by the attempt.
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
    """No default name and no default url: it would be recorded as something
    no operator could find and no release could match."""
    for missing in ("name", "url", "version"):
        body = {key: value for key, value in ORCHESTRATOR.items() if key != missing}
        refused = claim(client, auth, body)
        assert refused.status_code == 400, missing
        assert refused.json()["error"]["code"] == "invalid_request"
    assert client.post("/v1/peer/claim", headers=auth, json={}).status_code == 400


# ------------------------------------------------------------------ releasing


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
    """A release is a release: "there is no claim" is the state asked for."""
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


# ------------------------------------------------------------- the refusals


def test_a_wrong_or_missing_bearer_is_peer_token_mismatch_and_not_unauthorized(
    client: TestClient
) -> None:
    """One name per relation, so a log line says WHICH handshake failed.

    An orchestrator told `unauthorized` cannot tell "the token I copied out of
    the guest's pairing line is stale" — its own bug — from "some app's token
    is wrong", which is not its business.
    """
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
    """A caller with neither is told about the credential first.

    The same order every other door uses: what is wrong with WHO YOU ARE
    before what is wrong with WHAT YOU SPEAK.
    """
    answer = client.post("/v1/peer/claim", json={"orchestrator": ORCHESTRATOR})
    assert answer.json()["error"]["code"] == peer.PEER_TOKEN_MISMATCH


# --------------------------------------------------------------- GET /v1/peer


def test_the_relation_read_carries_the_role_the_claim_and_a_monotonic_uptime(
    client: TestClient, auth: dict[str, str]
) -> None:
    """2.4: health flows ONE way, and `uptime_s` is the re-claim signal.

    It is what tells an orchestrator that an engine answering again is a NEW
    process rather than the one it claimed.
    """
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
    """2.4, pinned as an ABSENCE because that is what it is.

    An engine that phoned home would need to know its orchestrator's address,
    keep it fresh across restarts, and behave when it is wrong — three facts
    to own for a push a 15-second poll already delivers. So the engine's whole
    knowledge of its orchestrator is the two strings it was handed, and there
    is no code path that reaches back.
    """
    source = (peer.__file__ and open(peer.__file__, encoding="utf-8").read()) or ""
    # The only outbound calls in the module are the ORCHESTRATOR's three, and
    # they are made by a function the engine never calls.
    assert "def claim_engine" in source
    assert "def release_engine" in source
    assert "def read_peer" in source
    assert "def read_info" in source
    # `PeerState` — the engine's half — makes no call at all.
    state_source = source[source.index("class PeerState") : source.index("class PeerCallFailed")]
    assert "urlopen" not in state_source
    assert "_call(" not in state_source


# -------------------------------------------------------------- the vintage


def test_a_pre_phase_17_document_reads_as_an_engine_with_no_orchestrator() -> None:
    """3.3's all-or-nothing rule, from the READER's side.

    A `/v1/info` with no `role` comes from a server that predates this phase,
    and such a server IS an engine with `managed_by: null` — a fact the
    document states by its vintage, not a default a client fills. This pins
    the shape the SDK's rule is written against; `sdk/ts/test/
    unit-phase17-orchestrator.test.ts` pins the reader itself.
    """
    old = {"server": {"name": "n", "version": "0.5.0", "api_version": 1}}
    assert "role" not in old
    assert old.get("role", peer.ROLE_ENGINE) == peer.ROLE_ENGINE
    assert old.get("managed_by") is None


def test_the_roles_and_the_owners_are_exactly_these_words() -> None:
    """A word this build does not have is a word a client cannot be sent."""
    assert peer.ROLES == ("engine", "orchestrator")
    assert peer.OWNERS == ("wsl-unit", "child", "found")
    assert peer.BACKEND_ORCHESTRATOR == "orchestrator"


def test_orchestrator_is_a_backend_kind_on_the_wire_and_NOWHERE_else() -> None:
    """`detect_backend()` never returns it and `init --backend` never takes it.

    It is a literal in the orchestrator's own `/v1/info`, because a client
    reading `host.backend` must get an answer that is true.
    """
    from crucible.backend import BACKEND_KINDS

    assert peer.BACKEND_ORCHESTRATOR not in BACKEND_KINDS


# ------------------------------------------------------------ not persisted


def test_a_claim_dies_with_the_process_and_is_written_NOWHERE(
    make_client: Callable[..., TestClient], auth: dict[str, str], tmp_path: Any
) -> None:
    """2.3. A claim on disk outlives the orchestrator that made it.

    Uninstall the tray, reboot, and the engine still names a door that will
    never answer again — ARCHITECTURE.md's one shape. So a second server built
    on the same home comes up unmanaged, and the orchestrator re-asserts.
    """
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
    """The state is per PROCESS, which is what `role` is too."""
    with make_client() as one, make_client() as two:
        claim(one, auth)
        assert one.get("/v1/info", headers=auth).json()["managed_by"] is not None
        assert two.get("/v1/info", headers=auth).json()["managed_by"] is None


# ------------------------------------------------------------- the unit itself


def test_the_state_object_refuses_and_releases_without_a_server() -> None:
    state = peer.PeerState()
    assert state.managed_by() is None
    mine = peer.Orchestrator("a", "http://127.0.0.1:7101", "0.6.0")
    theirs = peer.Orchestrator("b", "http://127.0.0.1:7999", "0.6.0")
    state.claim(mine)
    assert state.managed_by() == {"name": "a", "url": "http://127.0.0.1:7101"}
    # Idempotent for the holder, refused for anybody else, taken with force.
    state.claim(mine)
    try:
        state.claim(theirs)
        raise AssertionError("a second orchestrator must be refused")
    except Exception as exc:  # ApiError
        assert getattr(exc, "code", "") == peer.PEER_ALREADY_MANAGED
    state.claim(theirs, force=True)
    assert state.managed_by()["name"] == "b"
    state.release(theirs)
    assert state.managed_by() is None
    # A release with nobody named drops whatever is held — the orchestrator's
    # Quit, which knows it is the one that claimed.
    state.claim(mine)
    state.release(None)
    assert state.managed_by() is None


def test_url_normalisation_touches_the_trailing_slash_and_nothing_else() -> None:
    """Lowercasing a host or dropping a default port would be this module
    inventing an opinion about addresses."""
    assert peer.normalise_url("http://127.0.0.1:7101/") == "http://127.0.0.1:7101"
    assert peer.normalise_url("  http://127.0.0.1:7101  ") == "http://127.0.0.1:7101"
    assert peer.normalise_url("http://LocalHost:7101") == "http://LocalHost:7101"


def test_an_engines_refusal_code_survives_the_trip_back_to_the_orchestrator() -> None:
    """One owner per name on the wire (ARCHITECTURE.md R1).

    A body this side cannot parse becomes `peer_unreadable`, which is the
    honest answer: something refused, and it did not say what in a shape this
    understands.
    """
    assert (
        peer._refusal_code('{"error": {"code": "peer_already_managed"}}')
        == peer.PEER_ALREADY_MANAGED
    )
    assert peer._refusal_code("<html>nginx</html>") == "peer_unreadable"
    assert peer._refusal_code('{"error": {}}') == "peer_unreadable"


def test_an_unreachable_engine_is_a_named_failure_and_never_an_exception_to_crash_on() -> None:
    """A tray that died telling an engine who manages it would take the watch
    with it. Port 1 answers nothing on any machine."""
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
