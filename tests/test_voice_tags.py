from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import huggingface_hub
import pytest
from fastapi.testclient import TestClient

from crucible import voicecatalog, voicerefs, voicerepo
from crucible.api.routes import catalog as catalog_routes
from crucible.cli import voices as cli_voices
from crucible.narratorengines import declared_tts_footprints
from crucible.voicecard import export_manifest
from crucible.voicerepo import REPO_MANIFEST_NAME, merge, parse_repo_manifest
from crucible.voices import VoiceError, voice_document

from .conftest import (
    FAKE_BACKEND,
    HUB_REFS,
    PINNED_VOICE_MANIFESTS,
    PINNED_VOICE_TAG,
    configure_box,
    parse_sse,
)
from .fake_hub import FakeHub

VOICE = "mistborn"
REPO = "owenmorgan/mistborn-higgs-v3"
OLD = "8a8d1bf375c7c4aa097a481d332b94e835a8c6f4"
NEW = "b" * 40
OLD_MANIFEST = (
    PINNED_VOICE_MANIFESTS / REPO.replace("/", "--") / OLD / REPO_MANIFEST_NAME
).read_text(encoding="utf-8")

ARM_FACTS = """edge_fade_ms          = { in = 10, out = 25 }
reference_seconds_cap = 30
allowed_controls      = []
"""

CHUNK_GAP = """
[voice.chunk_gap]
inject_s              = 0.27
target_join_s         = 0.53
model_self_tail_s     = 0.26
reader_sentence_gap_s = 0.53
model_internal_gap_s  = 0.5
rule                  = "match-reader"
method                = "silence = -40 dB relative to each clip's own peak, 20 ms hop"
source                = "pause_match.py over the ow_v7 length ladder"
measured_on           = "2026-09-11"
"""


def with_facts(text: str) -> str:
    arm_line = 'sampling        = { temperature = 0.8, top_p = 0.95, top_k = 50 }\n'
    assert text.count(arm_line) == 2
    text = text.replace(arm_line, arm_line + ARM_FACTS)
    return text.replace("\n[voice.arms.cuda-linux]", CHUNK_GAP + "\n[voice.arms.cuda-linux]", 1)


NEW_MANIFEST = with_facts(OLD_MANIFEST)


def file_only_cache(home: Path, hf_repo: str, ref: str) -> Any:
    return voicerefs.read_checks(home).get(f"{hf_repo}@{ref}")


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    fake = FakeHub()

    def manifest_or_fake(**kwargs: Any) -> str:
        if kwargs.get("revision") == NEW and kwargs.get("filename") == REPO_MANIFEST_NAME:
            target = Path(kwargs["local_dir"]) / REPO_MANIFEST_NAME
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(NEW_MANIFEST, encoding="utf-8")
            return str(target)
        return fake.hf_hub_download(**kwargs)

    def snapshot_with_manifest(**kwargs: Any) -> str:
        where = fake.snapshot_download(**kwargs)
        text = NEW_MANIFEST if kwargs["revision"] == NEW else OLD_MANIFEST
        (Path(where) / REPO_MANIFEST_NAME).write_text(text, encoding="utf-8")
        return where

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", manifest_or_fake)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_with_manifest)
    return fake


@pytest.fixture
def tts_client(make_client: Callable[..., TestClient]) -> Iterator[TestClient]:
    with make_client(enable_tts=True) as instance:
        yield instance


def stamp_pulled(home: Path, revision: str = OLD) -> Path:
    directory = home / "voices" / VOICE / FAKE_BACKEND.kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REPO_MANIFEST_NAME).write_text(OLD_MANIFEST, encoding="utf-8")
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "family": "voices", "id": VOICE, "backend": FAKE_BACKEND.kind,
                "hf_repo": REPO, "revision": revision, "files": [], "bytes": 1234,
                "seconds": 1.0, "pulled": "2026-09-28T10:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return directory


def stamped_revision(home: Path) -> str:
    stamp = home / "voices" / VOICE / FAKE_BACKEND.kind / "crucible-pull.json"
    return json.loads(stamp.read_text(encoding="utf-8"))["revision"]


def voice_row(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    rows = client.get("/v1/voices", headers=auth).json()
    return next(row for row in rows if row["id"] == VOICE)


def pull(client: TestClient, auth: dict[str, str]) -> list[dict]:
    answer = client.post("/v1/tasks", headers=auth, json={"type": "pull", "kind": "voice", "id": VOICE})
    assert answer.status_code == 202, answer.text
    with client.stream("GET", f"/v1/tasks/{answer.json()['task_id']}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def the_tag_moves_to_new() -> None:
    HUB_REFS[(REPO, PINNED_VOICE_TAG)] = NEW


def test_the_packaged_pins_follow_the_crucible_tag_and_name_no_sha() -> None:
    pins = voicerepo.packaged_pins()
    assert pins
    assert {pin.ref for pin in pins.values()} == {PINNED_VOICE_TAG}
    assert {pin.revision for pin in pins.values()} == {None}


def test_a_pin_with_a_sha_in_ref_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(VoiceError, match="belongs in `revision`"):
        voicerepo.parse_pins(f'[{VOICE}]\nhf_repo = "{REPO}"\nref = "{OLD}"\n', tmp_path / "pins.toml")
    with pytest.raises(VoiceError, match="both revision and ref"):
        voicerepo.parse_pins(
            f'[{VOICE}]\nhf_repo = "{REPO}"\nref = "crucible"\nrevision = "{OLD}"\n',
            tmp_path / "pins.toml",
        )


def test_a_ref_pin_resolves_at_pull_and_the_row_shows_the_recorded_sha(
    tts_client: TestClient, auth: dict[str, str], home: Path, hub: FakeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(voicerefs, "cached", file_only_cache)
    events = pull(tts_client, auth)
    assert events[-1]["event"] == "done", events
    assert (REPO, OLD) in hub.asked
    assert stamped_revision(home) == OLD
    assert voicerefs.read_checks(home)[f"{REPO}@{PINNED_VOICE_TAG}"].revision == OLD
    row = voice_row(tts_client, auth)
    assert row["revision"] == OLD
    assert row["ref"] == PINNED_VOICE_TAG
    assert row["update_available"] is False


def test_an_unresolved_tag_is_an_unserved_row_that_names_the_command(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(voicerefs, "cached", file_only_cache)
    reason = voicecatalog.unserved_pins()[VOICE][1]
    assert "voice_ref_unresolved" in reason
    assert "crucible voices check-updates" in reason


def test_a_get_never_resolves_a_tag(
    tts_client: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp_pulled(home)

    def no_lookups(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("a read looked a tag up")

    monkeypatch.setattr(voicerefs, "resolve_ref", no_lookups)
    the_tag_moves_to_new()
    row = voice_row(tts_client, auth)
    assert row["revision"] == OLD
    assert row["update_available"] is False


def test_check_updates_says_a_pull_would_move_and_the_row_carries_it(
    tts_client: TestClient, auth: dict[str, str], home: Path, hub: FakeHub
) -> None:
    stamp_pulled(home)
    the_tag_moves_to_new()
    answer = tts_client.post("/v1/voices/updates", headers=auth)
    assert answer.status_code == 200, answer.text
    checked = next(row for row in answer.json()["voices"] if row["id"] == VOICE)
    assert checked["revision"] == OLD
    assert checked["latest_revision"] == NEW
    assert checked["update_available"] is True
    row = voice_row(tts_client, auth)
    assert (row["revision"], row["latest_revision"], row["update_available"]) == (OLD, NEW, True)


def test_pull_moves_the_box_to_the_tag_s_new_sha(
    tts_client: TestClient, auth: dict[str, str], home: Path, hub: FakeHub
) -> None:
    stamp_pulled(home)
    the_tag_moves_to_new()
    voicecatalog.check_updates(home)
    events = pull(tts_client, auth)
    assert events[-1]["event"] == "done", events
    assert (REPO, NEW) in hub.asked
    assert stamped_revision(home) == NEW
    row = voice_row(tts_client, auth)
    assert row["revision"] == NEW
    assert row["update_available"] is False
    assert row["chunk_gap"]["inject_s"] == 0.27
    assert row["edge_fade_ms"] == {"in": 10, "out": 25}


def test_a_resident_voice_is_not_swapped_underneath_it(
    tts_client: TestClient, auth: dict[str, str], home: Path, hub: FakeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamp_pulled(home)
    the_tag_moves_to_new()
    voicecatalog.check_updates(home)
    monkeypatch.setattr(
        catalog_routes,
        "held_on_card",
        lambda residency, subject: {
            "kind": subject.kind, "id": subject.id, "fact": "resident",
            "who": "it is the voice on the card right now; unload it first",
        },
    )
    events = pull(tts_client, auth)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["code"] == "subject_in_use"
    assert (REPO, NEW) not in hub.asked
    assert stamped_revision(home) == OLD


def test_the_card_guard_names_a_resident_voice() -> None:
    subject = SimpleNamespace(kind="voice", id=VOICE)
    resident = SimpleNamespace(resident=SimpleNamespace(id=VOICE, kind="tts"))
    held = catalog_routes.held_on_card(resident, subject)
    assert held is not None and held["fact"] == "resident"
    empty = SimpleNamespace(resident=None)
    assert catalog_routes.held_on_card(empty, subject) is None


def test_the_cli_does_not_move_a_voice_the_server_has_loaded(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = SimpleNamespace(home=home)
    server = SimpleNamespace(url="http://127.0.0.1:8765")
    monkeypatch.setattr(cli_voices.common, "server_here", lambda config, backend: server)
    monkeypatch.setattr(
        cli_voices.transport, "call", lambda *args, **kwargs: {"resident": {"id": VOICE}}
    )
    assert cli_voices._refuse_if_loaded(config, FAKE_BACKEND, VOICE) != 0
    assert "unload-voice" in capsys.readouterr().err


def test_an_unreachable_hub_leaves_the_box_where_it_is_and_says_so(
    tts_client: TestClient, auth: dict[str, str], home: Path
) -> None:
    stamp_pulled(home)
    HUB_REFS.clear()
    checked = next(row for row in voicecatalog.check_updates(home) if row["id"] == VOICE)
    assert checked["revision"] == OLD
    assert checked["update_available"] is False
    assert "crucible voices check-updates" in checked["update_error"]
    row = voice_row(tts_client, auth)
    assert row["revision"] == OLD
    assert row["update_error"] is not None


def test_an_exact_sha_pin_wins_over_the_tag(home: Path) -> None:
    configure_box(home, enable_tts=True)
    voicerepo.write_home_pin(VOICE, REPO, OLD)
    voicerefs.record(
        home,
        voicerefs.RefCheck(hf_repo=REPO, ref=PINNED_VOICE_TAG, revision=NEW, checked_at="now", error=None),
    )
    voice = voicecatalog.load_all_voices()[VOICE]
    assert voicecatalog.served_revision_of(voice) == OLD
    assert voicecatalog.moves_to(home, voice) is None
    assert voicecatalog.ref_state(home, voice) == {}


def test_a_corrupt_tag_record_is_quarantined_and_read_as_empty(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    voicerefs.refs_path(home).write_text("{not json", encoding="utf-8")
    assert voicerefs.read_checks(home) == {}
    assert list(home.glob(f"{voicerefs.REFS_FILE}.bad-*"))


def test_the_new_voice_facts_round_trip_from_a_repo_manifest() -> None:
    repo = parse_repo_manifest(NEW_MANIFEST, Path(REPO_MANIFEST_NAME))
    footprint = declared_tts_footprints(FAKE_BACKEND.kind)[0]
    voice = merge(repo, voicerepo.Pin(id=VOICE, hf_repo=REPO, revision=NEW, path=Path("pins.toml")), footprint)
    spec = voice.spec(FAKE_BACKEND.kind)
    assert spec.facts.edge_fade_ms == {"in": 10.0, "out": 25.0}
    assert spec.facts.reference_seconds_cap == 30.0
    assert spec.facts.allowed_controls == ()
    assert voice.chunk_gap is not None and voice.chunk_gap.target_join_s == 0.53
    document, _ = voice_document(voice)
    assert document["voice"]["chunk_gap"]["rule"] == "match-reader"
    assert document["voice"]["backends"][FAKE_BACKEND.kind]["edge_fade_ms"] == {"in": 10.0, "out": 25.0}


def test_export_writes_the_new_facts_and_they_read_back(home: Path) -> None:
    repo = parse_repo_manifest(NEW_MANIFEST, Path(REPO_MANIFEST_NAME))
    footprint = declared_tts_footprints(FAKE_BACKEND.kind)[0]
    voice = merge(repo, voicerepo.Pin(id=VOICE, hf_repo=REPO, revision=NEW, path=Path("pins.toml")), footprint)
    text, _ = export_manifest(
        voice, pace_basis="measured", measured_from="the ladder", inherited_from=None,
        max_chars_basis="measured", uncertified=False,
    )
    again = parse_repo_manifest(text, Path(REPO_MANIFEST_NAME))
    assert again.chunk_gap == repo.chunk_gap
    assert again.arms[FAKE_BACKEND.kind]["edge_fade_ms"] == {"in": 10, "out": 25}
    assert again.arms[FAKE_BACKEND.kind]["allowed_controls"] == []


@pytest.mark.parametrize(
    ("old", "new", "said"),
    [
        ("inject_s              = 0.27", "inject_s              = 0.53", "is not its target_join_s"),
        ("allowed_controls      = []", 'allowed_controls      = ["long pause"]', "is not a control token"),
        ("edge_fade_ms          = { in = 10, out = 25 }", "edge_fade_ms          = { in = 10 }", "missing required key(s) ['out']"),
        ("reference_seconds_cap = 30", "reference_seconds_cap = 31", "at most 30"),
        ('rule                  = "match-reader"\n', ""  , "missing required key(s) ['rule']"),
    ],
)
def test_a_malformed_voice_fact_is_refused_by_name(old: str, new: str, said: str) -> None:
    assert old in NEW_MANIFEST
    with pytest.raises(VoiceError) as caught:
        parse_repo_manifest(NEW_MANIFEST.replace(old, new, 1), Path(REPO_MANIFEST_NAME))
    assert said in str(caught.value)


def load_publish() -> Any:
    path = Path(__file__).resolve().parent.parent / "scripts" / "publish-voice.py"
    spec = importlib.util.spec_from_file_location("publish_voice", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublishHub:

    def __init__(self, *, commits: dict[str, str], tag: str | None) -> None:
        self.commits = commits
        self.tag = tag
        self.calls: list[tuple[str, ...]] = []

    def commit_of(self, repo: str, revision: str) -> str:
        if revision not in self.commits:
            raise RuntimeError(f"{repo} has no revision {revision}")
        return revision

    def manifest_at(self, repo: str, revision: str, into: Path) -> Path:
        path = into / REPO_MANIFEST_NAME
        path.write_text(self.commits[revision], encoding="utf-8")
        return path

    def tag_commit(self, repo: str, tag: str) -> str | None:
        return self.tag

    def delete_tag(self, repo: str, tag: str) -> None:
        self.calls.append(("delete", tag))

    def create_tag(self, repo: str, tag: str, revision: str) -> None:
        self.calls.append(("create", tag, revision))


def test_publish_moves_the_tag_to_a_commit_whose_manifest_parses(
    capsys: pytest.CaptureFixture[str]
) -> None:
    publish = load_publish()
    hub = PublishHub(commits={OLD: OLD_MANIFEST, NEW: NEW_MANIFEST}, tag=OLD)
    assert publish.main([VOICE, NEW], hub=hub) == 0
    assert hub.calls == [("delete", PINNED_VOICE_TAG), ("create", PINNED_VOICE_TAG, NEW)]
    assert "crucible voices pull mistborn" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("sha", "said"),
    [("main", "not a full 40-character commit sha"), ("c" * 40, "has no commit")],
)
def test_publish_refuses_a_sha_the_repo_does_not_have(
    sha: str, said: str, capsys: pytest.CaptureFixture[str]
) -> None:
    publish = load_publish()
    hub = PublishHub(commits={OLD: OLD_MANIFEST}, tag=OLD)
    assert publish.main([VOICE, sha], hub=hub) == 1
    assert said in capsys.readouterr().err
    assert hub.calls == []


def test_publish_refuses_a_commit_whose_manifest_does_not_parse(
    capsys: pytest.CaptureFixture[str]
) -> None:
    publish = load_publish()
    broken = NEW_MANIFEST.replace("inject_s              = 0.27", "inject_s              = 0.9")
    hub = PublishHub(commits={OLD: OLD_MANIFEST, NEW: broken}, tag=OLD)
    assert publish.main([VOICE, NEW], hub=hub) == 1
    assert "does not parse" in capsys.readouterr().err
    assert hub.calls == []


def test_publish_create_seeds_the_tag_at_the_shipped_revision_once(
    capsys: pytest.CaptureFixture[str]
) -> None:
    publish = load_publish()
    hub = PublishHub(commits={OLD: OLD_MANIFEST}, tag=None)
    assert publish.main([VOICE, "--create"], hub=hub) == 0
    assert hub.calls == [("create", PINNED_VOICE_TAG, OLD)]
    tagged = PublishHub(commits={OLD: OLD_MANIFEST}, tag=OLD)
    assert publish.main([VOICE, "--create"], hub=tagged) == 1
    assert "already has a 'crucible' tag" in capsys.readouterr().err
    assert tagged.calls == []


def test_publish_seeds_every_shipped_voice() -> None:
    publish = load_publish()
    assert set(publish.SEED_REVISIONS) == set(voicerepo.packaged_pins())
