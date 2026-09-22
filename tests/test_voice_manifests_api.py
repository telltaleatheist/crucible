"""`PUT /v1/voices/{id}` and `DELETE /v1/voices/{id}` — the overlay's doors.

`<CRUCIBLE_HOME>/voices/*.toml` has been read since 2026-09-16 and until now
there was no way to FILL it except by putting a file on the machine by hand.
That works for the engine you are sitting at and not at all for one across the
room, which is the ordinary case: the Mac is a backend, and nobody wants to ssh
in to add a voice they trained last night.

Nothing here touches HuggingFace or the network. `resolve_revision` is
monkeypatched, because what these tests are about is the door — what it
validates, what it refuses, where it writes, and what a reader sees afterwards —
and a test that reached the Hub would be a test that fails when the Hub is slow.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import api as api_module
from crucible import voices as voices_module
from crucible.residency import KIND_TTS
from crucible.voices import load_all_voices, load_voice

#: A full 40-character sha, which is the only thing a manifest accepts.
SHA = "a" * 40
OTHER_SHA = "b" * 40

#: The packaged voice these tests override, and one they invent.
SHIPPED = "deathstalker"
CUSTOM = "nightingale"


@pytest.fixture
def tts_client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client(enable_tts=True) as instance:
        yield instance


@pytest.fixture
def no_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """`resolve_revision` answers a fixed sha and records what it was asked.

    The list is the assertion surface: a test that expects NO resolution proves
    it by finding this empty, which is stronger than asserting the sha that
    happened to be written.
    """
    asked: list[str] = []

    def fake(_config: Any, repo: str) -> str:
        asked.append(repo)
        return SHA

    monkeypatch.setattr(api_module, "resolve_revision", fake)
    return asked


def a_voice(
    *,
    display: str = "Nightingale",
    repo: str = "owenmorgan/nightingale-higgs-v3",
    revision: str | None = None,
    max_chars: int = 800,
) -> dict[str, Any]:
    """A manifest document that validates, with one backend block.

    Built from the fields `crucible/voices.py` actually requires rather than
    copied from a shipped file, so that a field becoming required shows up here
    as a failure to construct rather than as a mysterious 400.
    """
    backend: dict[str, Any] = {
        "hf_repo": repo,
        "memory_bytes_estimate": 19_000_000_000,
        # `estimate_basis` "declared" REQUIRES an `estimate_note`, and the
        # validator refuses without one by name: a declared number came from
        # somewhere and a reader has to be able to find out where. That rule is
        # what the app-side form has to surface, so the fixture obeys it rather
        # than choosing a basis that dodges it.
        "estimate_basis": "declared",
        "estimate_note": "SGLang-Omni's configured reservation, as the deathstalker manifest measured it.",
        "max_chars": max_chars,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
    }
    if revision is not None:
        backend["revision"] = revision
    return {
        "voice": {
            "id": CUSTOM,
            "display": display,
            "kind": "checkpoint",
            "narrator_engine": "higgs-v3",
            "language": "en",
            "sample_rate": 24000,
            "pace": {
                "pace_chars_per_sec": 15.91,
                "max_chars_per_sec": 20.68,
                "min_chars_per_sec": 12.24,
            },
            "serving": {
                "max_num_seqs": 16,
                "max_num_seqs_note": "vllm-omni's own stage-0 value.",
            },
            # `cuda-linux`, because the validator refuses a backend Crucible
            # does not have and the fixture server IS one. A made-up backend
            # name was the first draft's mistake, said by name.
            "backends": {"cuda-linux": backend},
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────


def test_a_new_voice_is_written_to_the_home_overlay_and_reads_back(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """The file lands in the overlay, and the READER sees it — not just the door."""
    answer = tts_client.put(
        f"/v1/voices/{CUSTOM}", json=a_voice(revision=OTHER_SHA), headers=auth
    )
    assert answer.status_code == 200, answer.text

    path = home / "voices" / f"{CUSTOM}.toml"
    assert path.is_file(), "the manifest was not written where the overlay is read from"
    assert answer.json()["path"] == str(path)

    # The point of the door is the READER, so the assertion is the reader's.
    manifest = load_voice(CUSTOM)
    assert manifest.display == "Nightingale"
    assert manifest.backends["cuda-linux"].revision == OTHER_SHA
    assert CUSTOM in load_all_voices()


def test_an_absent_revision_is_pinned_from_the_repo_head(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """Paste a repo, get a pin — the whole reason a person can add a voice at all.

    A manifest refuses a branch name, so without this the field is unfillable by
    anybody who does not already know their own repo's head sha.
    """
    answer = tts_client.put(f"/v1/voices/{CUSTOM}", json=a_voice(), headers=auth)
    assert answer.status_code == 200, answer.text
    assert no_hub == ["owenmorgan/nightingale-higgs-v3"]

    written = tomllib.loads((home / "voices" / f"{CUSTOM}.toml").read_text("utf-8"))
    assert written["voice"]["backends"]["cuda-linux"]["revision"] == SHA


def test_a_named_revision_is_obeyed_and_the_hub_is_never_asked(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    """Asking for an older commit is a real thing to want; this door does not argue."""
    answer = tts_client.put(
        f"/v1/voices/{CUSTOM}", json=a_voice(revision=OTHER_SHA), headers=auth
    )
    assert answer.status_code == 200, answer.text
    assert no_hub == [], "a caller's own revision was second-guessed"
    assert load_voice(CUSTOM).backends["cuda-linux"].revision == OTHER_SHA


def test_an_overlay_shadows_a_packaged_voice_and_removing_it_restores_the_packaged_one(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    """The override, and its undo. This is what makes trying one safe."""
    shipped = load_voice(SHIPPED)
    document = a_voice(display="Overridden", revision=OTHER_SHA)
    document["voice"]["id"] = SHIPPED

    answer = tts_client.put(f"/v1/voices/{SHIPPED}", json=document, headers=auth)
    assert answer.status_code == 200, answer.text
    assert load_voice(SHIPPED).display == "Overridden"

    gone = tts_client.delete(f"/v1/voices/{SHIPPED}", headers=auth)
    assert gone.status_code == 204
    assert load_voice(SHIPPED).display == shipped.display, (
        "removing the overlay did not bring the packaged voice back, so an "
        "override nobody wanted is permanent"
    )


def test_the_answer_is_the_readers_row_not_an_echo_of_the_request(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    """A door that echoed its input would report a save the file may have reshaped."""
    answer = tts_client.put(
        f"/v1/voices/{CUSTOM}", json=a_voice(revision=OTHER_SHA), headers=auth
    )
    row = answer.json()["voice"]
    assert row is not None and row["id"] == CUSTOM
    listed = [r for r in tts_client.get("/v1/voices", headers=auth).json()
              if r["id"] == CUSTOM]
    assert listed == [row], "the write answered something GET /v1/voices does not say"


# ─────────────────────────────────────────────────────────────────────────────
# Refusing
# ─────────────────────────────────────────────────────────────────────────────


def test_an_invalid_manifest_is_refused_by_the_readers_own_sentence_and_nothing_is_written(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """The validator is the one that already exists; the door invents no rules."""
    document = a_voice(revision=OTHER_SHA)
    del document["voice"]["pace"]["min_chars_per_sec"]

    answer = tts_client.put(f"/v1/voices/{CUSTOM}", json=document, headers=auth)
    assert answer.status_code == 400
    assert answer.json()["error"]["code"] == "voice_invalid"
    assert "min_chars_per_sec" in answer.json()["error"]["message"]
    assert not (home / "voices" / f"{CUSTOM}.toml").exists(), (
        "a manifest that does not validate reached the disk, where the next "
        "load_all_voices() will raise for every caller"
    )


def test_a_branch_name_is_refused_because_a_pull_must_be_reproducible(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    answer = tts_client.put(
        f"/v1/voices/{CUSTOM}", json=a_voice(revision="main"), headers=auth
    )
    assert answer.status_code == 400
    assert answer.json()["error"]["code"] == "voice_invalid"
    assert "40-character" in answer.json()["error"]["message"]


def test_an_id_that_is_a_path_is_refused_before_it_can_become_one(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """The id becomes a FILENAME. Anything with a separator in it is a write elsewhere."""
    answer = tts_client.put(
        "/v1/voices/..%2F..%2Fescaped", json=a_voice(revision=OTHER_SHA), headers=auth
    )
    assert answer.status_code in (400, 404), answer.text
    assert not (home.parent / "escaped.toml").exists()
    assert not (home / "escaped.toml").exists()


def test_a_body_that_is_not_a_manifest_document_is_refused(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    answer = tts_client.put(f"/v1/voices/{CUSTOM}", json=[1, 2, 3], headers=auth)
    assert answer.status_code == 400
    assert answer.json()["error"]["code"] == "voice_invalid"


def test_a_resident_voice_cannot_be_rewritten_underneath_itself(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    """Its pace and cap are in use by a render that is already running."""
    residency = tts_client.app.state.residency
    loaded = type("R", (), {"id": CUSTOM, "kind": KIND_TTS})()
    residency._resident = loaded

    answer = tts_client.put(
        f"/v1/voices/{CUSTOM}", json=a_voice(revision=OTHER_SHA), headers=auth
    )
    assert answer.status_code == 409
    assert answer.json()["error"]["code"] == "voice_in_use"


# ─────────────────────────────────────────────────────────────────────────────
# Removing
# ─────────────────────────────────────────────────────────────────────────────


def test_removing_a_voice_this_host_never_added_is_refused_by_name(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    """The packaged set is the install and is not deletable through this door."""
    answer = tts_client.delete(f"/v1/voices/{SHIPPED}", headers=auth)
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == "voice_not_custom"
    assert load_voice(SHIPPED) is not None, "the packaged manifest was touched"


def test_removing_the_manifest_leaves_the_weights_alone(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """Two decisions, two doors: `DELETE /v1/catalog/voice/{id}` is the weights."""
    tts_client.put(f"/v1/voices/{CUSTOM}", json=a_voice(revision=OTHER_SHA), headers=auth)
    weights = home / "weights" / "voices" / CUSTOM
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "keep-me").write_text("bytes", encoding="utf-8")

    assert tts_client.delete(f"/v1/voices/{CUSTOM}", headers=auth).status_code == 204
    assert (weights / "keep-me").is_file(), (
        "removing a manifest deleted the download it described"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The overlay writer itself
# ─────────────────────────────────────────────────────────────────────────────


def test_a_half_written_manifest_never_appears(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is atomic, and the reason is what one bad file does.

    `load_all_voices` reads the whole directory, so an unparseable file raises
    for EVERY caller — a torn write does not break one voice, it breaks the
    server. The temporary is in the same directory (a rename across filesystems
    is not atomic) and is cleaned up when the replace fails.
    """
    (home / "voices").mkdir(parents=True, exist_ok=True)
    real_replace = Path.replace

    def fail(self: Path, target: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(voices_module.VoiceError, match="could not write"):
        voices_module.write_home_voice(
            CUSTOM, a_voice(revision=OTHER_SHA)
        )
    monkeypatch.setattr(Path, "replace", real_replace)

    assert not (home / "voices" / f"{CUSTOM}.toml").exists()
    assert list((home / "voices").glob("*")) == [], (
        "the temporary file was left behind, so the next load_all_voices() sees it"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The `pin` body — PHASE21-VOICES-FROM-HF.md section 2.4
#
# `PUT /v1/voices/{id}` went back to being an override, and gained the body a
# deploy actually sends: which published revision THIS machine serves a voice
# from. Its facts then come down with its weights, out of a `crucible-voice.toml`
# committed in the same commit as the bytes.
#
# Nothing here reaches the Hub: the manifest is written where a fetch would have
# left it (the home's content-addressed cache), which is the seam the loader
# already reads before it asks the network.
# ─────────────────────────────────────────────────────────────────────────────

REPO_MANIFEST = """
schema = 1

[voice]
display         = "Nightingale"
kind            = "checkpoint"
narrator_engine = "higgs-v3"
language        = "en"
sample_rate     = 24000

[voice.pace]
basis              = "measured"
pace_chars_per_sec = 13.76
max_chars_per_sec  = 17.89
min_chars_per_sec  = 10.58
safe_min_chars     = 500
safe_max_chars     = 800
measured_from      = "ng_v1 ckpt-4257, n=51 in the 500-800 band"

[voice.arms.cuda-linux]
max_chars       = 800
max_chars_basis = "measured"
sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }
"""

PIN_REPO = "owenmorgan/nightingale-higgs-v3"


def a_cached_repo_manifest(
    home: Path, revision: str = SHA, text: str = REPO_MANIFEST
) -> Path:
    path = (
        home
        / "voice-manifests"
        / PIN_REPO.replace("/", "--")
        / revision
        / "crucible-voice.toml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_pin_body_writes_the_home_pins_row(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    a_cached_repo_manifest(home)
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": SHA}},
        headers=auth,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["path"].endswith("pins.toml")
    assert no_hub == [], "a stated revision was resolved anyway"
    row = body["voice"]
    assert row["id"] == CUSTOM
    assert row["display"] == "Nightingale"
    # THE ROW'S THREE NEW FACTS (section 6).
    assert row["manifest"] == "repo"
    assert row["pace_basis"] == "measured"
    assert row["max_chars_basis"] == "measured"
    assert row["revision"] == SHA
    # And the machine facts came from THIS BOX rather than from the manifest.
    assert row["memory_bytes_estimate"] == 19_000_000_000
    assert row["estimate_basis"] == "declared"


def test_a_pin_with_no_revision_resolves_the_head(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """Nobody knows their own repo's head sha, and the engine is what holds the
    HuggingFace credential and does the fetching."""
    a_cached_repo_manifest(home)
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": None}},
        headers=auth,
    )
    assert response.status_code == 200, response.text
    assert no_hub == [PIN_REPO]
    assert response.json()["voice"]["revision"] == SHA


def test_a_pin_to_a_revision_with_no_manifest_writes_nothing(
    tts_client: TestClient,
    auth: dict[str, str],
    home: Path,
    no_hub: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A door that wrote first would leave a server holding a pin nothing can
    load, and would report that by breaking the catalog."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    def missing(**_kwargs: Any) -> str:
        raise EntryNotFoundError("404")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": SHA}},
        headers=auth,
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "voice_invalid"
    assert "voice_manifest_missing" in response.json()["error"]["message"]
    assert not (home / "voices" / "pins.toml").exists()


def test_a_body_with_both_a_pin_and_a_voice_is_refused(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": SHA}, "voice": a_voice()},
        headers=auth,
    )
    assert response.status_code == 400
    assert "both a `pin` and a `voice`" in response.json()["error"]["message"]


def test_a_body_with_neither_says_what_the_two_shapes_are(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}", json={"something": 1}, headers=auth
    )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "pin" in message and "voice" in message


def test_an_unknown_key_in_a_pin_is_refused(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    response = tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": SHA, "pace": 13.3}},
        headers=auth,
    )
    assert response.status_code == 400
    assert "unknown key(s) [\'pace\']" in response.json()["error"]["message"]


def test_delete_removes_a_pin_row_as_well_as_an_override(
    tts_client: TestClient, auth: dict[str, str], home: Path, no_hub: list[str]
) -> None:
    """Section 2.4: whichever the id has. A person deleting a voice knows only
    that they added it."""
    a_cached_repo_manifest(home)
    assert tts_client.put(
        f"/v1/voices/{CUSTOM}",
        json={"pin": {"hf_repo": PIN_REPO, "revision": SHA}},
        headers=auth,
    ).status_code == 200
    assert CUSTOM in load_all_voices()
    assert tts_client.delete(f"/v1/voices/{CUSTOM}", headers=auth).status_code == 204
    assert CUSTOM not in load_all_voices()
    # And a second delete is 204 (2026-09-21, the ladder's ask): the state
    # asked for is the state there is, so a retry after a lost answer is not a
    # failure. 404 is kept for a SHIPPED voice, which this door cannot remove.
    assert tts_client.delete(f"/v1/voices/{CUSTOM}", headers=auth).status_code == 204


def test_an_override_row_still_says_it_is_one(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    """The `voice` body is unchanged, and the row now says which kind of file
    its facts came out of."""
    # `a_voice()` IS the whole document — a table with a `voice` key, exactly as
    # voices/<id>.toml holds it — which is what this door has always taken.
    response = tts_client.put(f"/v1/voices/{CUSTOM}", json=a_voice(), headers=auth)
    assert response.status_code == 200, response.text
    row = response.json()["voice"]
    assert row["manifest"] == "override"
    # A `voices/<id>.toml` has no basis keys, and null means THIS SCHEMA CANNOT
    # SAY rather than a third word for "measured".
    assert row["pace_basis"] is None
    assert row["max_chars_basis"] is None


def test_a_packaged_voice_row_says_packaged(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    rows = tts_client.get("/v1/voices", headers=auth).json()
    by_id = {row["id"]: row for row in rows}
    assert by_id["mistborn"]["manifest"] == "packaged"
    assert by_id["zeroshot"]["manifest"] == "engine"


# ------------------------------------------------------- the ladder's lifecycle
#
# 2026-09-21: a DELETE of the screening voice raced a restart and `ladder-screen`
# stayed registered for three hours. Two changes, both pinned here: DELETE is
# idempotent on the state reached, and a local voice nothing holds says so.


def test_removing_a_voice_that_is_simply_absent_is_204_and_repeatable(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    """The state asked for is the state there is. A retry after a lost answer
    must read as success, not as a second failure."""
    for _ in range(2):
        answer = tts_client.delete("/v1/voices/never-registered-here", headers=auth)
        assert answer.status_code == 204, answer.text


def test_removing_an_added_voice_twice_is_204_both_times(
    tts_client: TestClient, auth: dict[str, str], no_hub: list[str]
) -> None:
    put = tts_client.put(f"/v1/voices/{CUSTOM}", json=a_voice(revision=SHA), headers=auth)
    assert put.status_code == 200, put.text
    assert CUSTOM in {row["id"] for row in tts_client.get("/v1/voices", headers=auth).json()}

    first = tts_client.delete(f"/v1/voices/{CUSTOM}", headers=auth)
    assert first.status_code == 204, first.text
    second = tts_client.delete(f"/v1/voices/{CUSTOM}", headers=auth)
    assert second.status_code == 204, second.text
    assert CUSTOM not in {row["id"] for row in tts_client.get("/v1/voices", headers=auth).json()}


def test_a_shipped_voice_is_still_refused_by_name_not_silently_204(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    """Idempotency is about ABSENT voices. A voice that is here and is not this
    server's own cannot be removed by this door, and 204 would say it was."""
    answer = tts_client.delete(f"/v1/voices/{SHIPPED}", headers=auth)
    assert answer.status_code == 404
    assert answer.json()["error"]["code"] == "voice_not_custom"
    assert SHIPPED in {row["id"] for row in tts_client.get("/v1/voices", headers=auth).json()}


def a_local_voice(directory: Path) -> dict[str, Any]:
    """`a_voice`, registered from a directory rather than a pin: what a
    screening ladder PUTs for a checkpoint that has no repo."""
    document = a_voice()
    block = document["voice"]["backends"]["cuda-linux"]
    del block["hf_repo"]
    block["path"] = str(directory)
    block["identity"] = "ds_v9_recut1_3510, asserted by the ladder"
    return document


def test_a_local_voice_nothing_holds_reads_orphan_true(
    tts_client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    """After a restart every screening voice whose ladder ended without its
    DELETE looks like this. The row says so; nothing deletes it."""
    put = tts_client.put(f"/v1/voices/{CUSTOM}", json=a_local_voice(tmp_path), headers=auth)
    assert put.status_code == 200, put.text
    rows = {row["id"]: row for row in tts_client.get("/v1/voices", headers=auth).json()}
    assert rows[CUSTOM]["source"] == "local"
    assert rows[CUSTOM]["orphan"] is True
    # A pinned voice is never an orphan: the pin owns it.
    assert rows[SHIPPED]["orphan"] is False
    # And the same rows verbatim on /v1/info (PHASE3-TTS.md section 8).
    info = tts_client.get("/v1/info", headers=auth).json()
    tts_rows = next(c["models"] for c in info["capabilities"] if c["job_type"] == "tts")
    assert {row["id"]: row["orphan"] for row in tts_rows}[CUSTOM] is True
    # Still there: the server garbage-collects nothing on its own judgment.
    assert CUSTOM in {row["id"] for row in tts_client.get("/v1/voices", headers=auth).json()}


def test_a_lease_or_a_queued_job_on_a_local_voice_is_not_an_orphan(
    tts_client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    """The three holders, each one enough: a lease naming the voice, a job
    naming it on the lane or in the queue, or residency."""
    put = tts_client.put(f"/v1/voices/{CUSTOM}", json=a_local_voice(tmp_path), headers=auth)
    assert put.status_code == 200, put.text

    def orphan() -> bool | None:
        rows = {row["id"]: row for row in tts_client.get("/v1/voices", headers=auth).json()}
        return rows[CUSTOM]["orphan"]

    assert orphan() is True

    class _Lease:
        subject = CUSTOM

    class _Leases:
        def current(self) -> Any:
            return _Lease()

    real_leases = tts_client.app.state.leases
    tts_client.app.state.leases = _Leases()
    try:
        assert orphan() is False, "a lease naming the voice holds it"
    finally:
        tts_client.app.state.leases = real_leases
    assert orphan() is True

    store = tts_client.app.state.store
    job = store.create("load-voice", CUSTOM, {})
    # ON THE LANE'S OWN QUEUE, not dispatched: `enqueue` would hand the load to
    # the engine, and what this test is about is the READ of a job that names
    # the voice while waiting. `_pending` is the structure `queued()` snapshots.
    store._pending.append(job.id)
    try:
        assert job in store.queued()
        assert orphan() is False, "a job naming the voice holds it"
    finally:
        store._pending.remove(job.id)
        store._jobs.pop(job.id, None)
    assert orphan() is True
