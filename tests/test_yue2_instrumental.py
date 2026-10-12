"""yue2-3b-instrumental: yue2-3b's weights (FROM: `weights_of`) plus an adapter, pulled as
a companion into yue2-3b's folder, folded in by the worker only while composing
(tests/test_yue2_instrumental_fold.py), for instrumentals only (docs/AUDIO.md
"yue2-3b-instrumental")."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import audioweights, catalog, playground, weights
from crucible.audiomodels import AudioManifestError, load_all_audio_manifests, load_audio_manifest
from crucible.capabilityclasses import BY_NAME
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take

from .conftest import (
    FAKE_BACKEND,
    FAKE_MAC_BACKEND,
    close_queue_session,
    configure_box,
    open_queue_session,
)
from .test_audio_api import (
    LYRICS,
    SONG,
    TAGS,
    _envs,
    _weights,
    idle_card,  # noqa: F401 (a fixture)
    loads,
    run_job,
    submit,
    transcript,  # noqa: F401 (a fixture)
)

INSTRUMENTAL = "yue2-3b-instrumental"
ADAPTER_REPO = "Mothersuperior/YuE2-instrumental-cot-full-loras"
ADAPTER_REVISION = "947f2f4b28978b2b6c3e316e6a87925c76bf3c4b"
ADAPTER_FILE = "ar_lora_inst_v3abc.bf16.safetensors"
ADAPTER_SHA = "e408fd3148b75b1165f7ddbf63db575d83bb6402a0b5f876fcb767dbcb2c5414"
ADAPTER_BYTES = 139502088
INSTRUMENTAL_TAGS = "Instrumental, warm piano, soft strings, 72 BPM"


# --- the manifest ---------------------------------------------------------------------------


def test_the_manifest_is_yue2_3b_plus_a_pinned_adapter_folded_around_compose() -> None:
    manifests = load_all_audio_manifests()
    model, base = manifests[INSTRUMENTAL], manifests[SONG]
    assert (model.weights_of, model.weights_base.id, model.instrumental_only) == (SONG, SONG, True)
    assert sorted(model.backends) == ["cuda-linux"]
    spec, base_spec = model.spec("cuda-linux"), base.spec("cuda-linux")
    assert (spec.hf_repo, spec.revision, spec.files) == (base_spec.hf_repo, base_spec.revision, base_spec.files)
    assert [c.name for c in model.own_companions("cuda-linux")] == ["instrumental-lora"]
    adapter = spec.adapter
    assert adapter.identity() == {
        "name": "instrumental-lora", "hf_repo": ADAPTER_REPO, "revision": ADAPTER_REVISION,
        "file": ADAPTER_FILE, "sha256": ADAPTER_SHA, "bytes": ADAPTER_BYTES, "rank": 64, "scale": 1.0,
    }
    assert (adapter.layers, adapter.applies_to, len(adapter.targets)) == (28, ("compose",), 7)
    assert "CC BY-NC 4.0" in model.licence and "non-commercial" in model.commercial_use
    # yue2-3b is untouched: no adapter, owns its download, sings.
    assert base_spec.adapter is None and base.weights_of is None and not base.instrumental_only
    assert [a.id for a in base.aliases()] == [INSTRUMENTAL]
    assert model.to_dict()["backends"]["cuda-linux"]["adapter"]["applies_to"] == ["compose"]


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    for name in (SONG, INSTRUMENTAL):
        text = load_audio_manifest(name).path.read_text(encoding="utf-8")
        if name == INSTRUMENTAL:
            assert old in text, old
            text = text.replace(old, new, 1)
        (tmp_path / f"{name}.toml").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("old", "new", "said"),
    [
        ('applies_to = ["compose"]', 'applies_to = ["plan", "compose"]', "is not built"),
        ('companion = "instrumental-lora"', 'companion = "lora"', "is not one of this arm's companions"),
        ('rank = 64', 'rank = 0', "must be positive"),
        ('revision = "c044757a011169583f363168348ae380946efff8"',
         'revision = "0000000000000000000000000000000000000000"', "revision differ"),
        ('name = "vae"', 'name = "decoder"', "lacks 'yue2-3b''s companion(s) ['vae']"),
        ('weights_of = "yue2-3b"', 'weights_of = "yue2-3b-instrumental"', "must name another"),
        ('[backends.cuda-linux.adapter]', '[backends.cuda-linux.adapterx]', "unknown key(s) ['adapterx']"),
    ],
)
def test_a_manifest_that_misstates_the_adapter_or_its_base_is_refused(
    tmp_path: Path, old: str, new: str, said: str
) -> None:
    with pytest.raises(AudioManifestError) as caught:
        load_audio_manifest(INSTRUMENTAL, _variant(tmp_path, old, new))
    assert said in str(caught.value)


def test_an_adapter_is_refused_on_the_mac(tmp_path: Path) -> None:
    text = load_audio_manifest(INSTRUMENTAL).path.read_text(encoding="utf-8")
    mac = text.replace("backends.cuda-linux", "backends.mlx-darwin")
    (tmp_path / f"{SONG}.toml").write_text(load_audio_manifest(SONG).path.read_text(encoding="utf-8"),
                                           encoding="utf-8")
    (tmp_path / f"{INSTRUMENTAL}.toml").write_text(mac, encoding="utf-8")
    with pytest.raises(AudioManifestError, match="folded through the YuE2 worker's host homes"):
        load_audio_manifest(INSTRUMENTAL, tmp_path)


def test_it_is_never_the_servers_song_pick() -> None:
    """It is reached by name (B-Sides asks for it); the song verb keeps yue2-3b."""
    ids = [candidate.id for candidate in BY_NAME["song"].candidates("cuda-linux")]
    assert SONG in ids and INSTRUMENTAL not in ids


# --- what it takes ----------------------------------------------------------------------------


def _check(model: str, **params: Any) -> None:
    manifest = load_all_audio_manifests()[model]
    refuse_what_the_model_cannot_take(AudioParams(**params), manifest, manifest.spec("cuda-linux"))


@pytest.mark.parametrize(
    "params",
    [
        {"tags": TAGS, "lyrics": LYRICS},
        {"tags": TAGS, "lyrics": LYRICS, "instrumental": False},
        {"tags": INSTRUMENTAL_TAGS},
        {"tags": TAGS, "lyrics": LYRICS, "score": "melody.abc"},
    ],
    ids=["sung", "instrumental-false", "instrumental-unsent", "sung-cover"],
)
def test_anything_but_an_instrumental_is_refused_by_name(params: dict) -> None:
    with pytest.raises(ApiError) as caught:
        _check(INSTRUMENTAL, **params)
    error = caught.value
    assert (error.status_code, error.code) == (400, "instrumental_only")
    assert error.details["param"] == "instrumental" and error.details["sung_model"] == SONG
    assert "Send `instrumental: true`, or send a sung song to yue2-3b" in error.message
    # The same request is yue2-3b's to sing, as before.
    if params.get("score") is None and "lyrics" in params:
        _check(SONG, **params)


def test_instrumentals_are_taken_as_yue2_3b_takes_them() -> None:
    _check(INSTRUMENTAL, tags=INSTRUMENTAL_TAGS, instrumental=True)
    _check(INSTRUMENTAL, tags=INSTRUMENTAL_TAGS, instrumental=True, planning_set="harbor")
    _check(INSTRUMENTAL, tags=INSTRUMENTAL_TAGS, instrumental=True, lyrics="[Intro]\n\n[Verse]\n")
    _check(INSTRUMENTAL, tags=INSTRUMENTAL_TAGS, instrumental=True, score="melody.abc")
    _check(INSTRUMENTAL, tags=INSTRUMENTAL_TAGS, instrumental=True, min_duration_s=60, max_duration_s=120)


def test_the_playground_page_has_yue2_3bs_shape_with_the_switch_fixed_on() -> None:
    manifests = load_all_audio_manifests()
    mine = playground.audio_fields(manifests[INSTRUMENTAL], manifests[INSTRUMENTAL].spec("cuda-linux"))
    theirs = playground.audio_fields(manifests[SONG], manifests[SONG].spec("cuda-linux"))
    assert [f["name"] for f in mine] == [f["name"] for f in theirs]
    switch = next(f for f in mine if f["name"] == "instrumental")
    assert (switch["default"], switch["fixed"]) == (True, True)
    lyrics = next(f for f in mine if f["name"] == "lyrics")
    assert not lyrics["required"] and "[Verse]" in lyrics["placeholder"]
    assert "fixed" not in next(f for f in theirs if f["name"] == "instrumental")


# --- the catalogue, the pull and the removal ----------------------------------------------------


def _subject(config: Any, model: str) -> catalog.Subject:
    found = catalog.find(config, FAKE_BACKEND, "model", model)
    assert found is not None
    return found


def _stamp_adapter(home: Path) -> Path:
    spec = load_audio_manifest(INSTRUMENTAL).spec("cuda-linux")
    (companion,) = load_audio_manifest(INSTRUMENTAL).own_companions("cuda-linux")
    part = home / "models" / SONG / "cuda-linux" / companion.name
    part.mkdir(parents=True, exist_ok=True)
    for entry in companion.files:
        (part / entry.target).write_bytes(b"lora")
    (part / weights.STAMP_NAME).write_text(
        json.dumps({"hf_repo": companion.hf_repo, "revision": companion.revision,
                    "bytes": companion.bytes, "pulled": "2026-10-12T02:00:00+0000",
                    "files": [{"target": e.target} for e in companion.files]}),
        encoding="utf-8",
    )
    assert spec.adapter.companion == companion
    return part


def test_its_row_says_whose_weights_it_shares_and_downloads_only_the_adapter_beside_them(
    home: Path,
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    row = _subject(config, INSTRUMENTAL)
    assert (row.job_type, row.shares_weights_of, row.name) == ("audio", SONG, "YuE2 3B Instrumental")
    assert row.installed() is None and row.expected_bytes is None, "the base's size is not pinned"
    assert INSTRUMENTAL not in {s.id for s in catalog.subjects(config, FAKE_MAC_BACKEND)}

    _weights(home, SONG, "cuda-linux")
    row = _subject(config, INSTRUMENTAL)
    assert row.installed() is None and row.expected_bytes == ADAPTER_BYTES
    base = _subject(config, SONG).installed()
    assert base is not None, "yue2-3b is installed as before"

    _stamp_adapter(home)
    found = _subject(config, INSTRUMENTAL).installed()
    assert found is not None and found.path == home / "models" / SONG / "cuda-linux"
    assert found.bytes == ADAPTER_BYTES, "only the adapter is its own"
    assert _subject(config, SONG).installed() == base, "the adapter is not yue2-3b's"


def test_a_pull_with_yue2_3b_present_fetches_only_the_adapter(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    _weights(home, SONG, "cuda-linux")
    fetched = _fake_hub(monkeypatch)
    manifest = load_audio_manifest(INSTRUMENTAL)
    found = audioweights.pull(config, manifest, manifest.spec("cuda-linux"))
    assert fetched == [(ADAPTER_REPO, ADAPTER_REVISION, [ADAPTER_FILE])]
    assert found.bytes == len(b"fetched") and found.path == home / "models" / SONG / "cuda-linux"
    record = home / "models" / SONG / "cuda-linux" / f"{weights.ALIAS_RECORD_PREFIX}{INSTRUMENTAL}.json"
    assert json.loads(record.read_text(encoding="utf-8"))["weights_of"] == SONG
    assert not (home / "models" / INSTRUMENTAL).exists(), "no second folder, no second copy"


def test_a_pull_without_yue2_3b_fetches_it_once_into_its_own_folder_then_the_adapter(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    fetched = _fake_hub(monkeypatch)
    manifest = load_audio_manifest(INSTRUMENTAL)
    audioweights.pull(config, manifest, manifest.spec("cuda-linux"))
    assert [repo for repo, _, _ in fetched] == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae", ADAPTER_REPO]
    base = load_audio_manifest(SONG)
    assert audioweights.installed(config, base, base.spec("cuda-linux")) is not None
    fetched.clear()
    audioweights.pull(config, manifest, manifest.spec("cuda-linux"))
    assert fetched == [], "pulled again, nothing is fetched"


def _fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, list[str]]]:
    """The two doors a pull fetches through: the main repo's snapshot and a companion's
    pinned files. Each records what it was asked for and writes what a fetch writes."""
    fetched: list[tuple[str, str, list[str]]] = []

    def snapshot(config: Any, name: str, spec: Any, target: Path, *, patterns: Any, **_: Any) -> None:
        fetched.append((spec.hf_repo, spec.revision, list(patterns)))
        for pattern in patterns:
            (target / pattern).parent.mkdir(parents=True, exist_ok=True)
            (target / pattern).write_bytes(b"weights")

    def files(config: Any, *, hf_repo: str, revision: str, files: Any, target_root: Path, **_: Any) -> Any:
        if weights.files_installed(target_root, hf_repo, revision) is not None:
            return None
        fetched.append((hf_repo, revision, [entry.source for entry in files]))
        target_root.mkdir(parents=True, exist_ok=True)
        for entry in files:
            (target_root / entry.target).write_bytes(b"fetched")
        (target_root / weights.STAMP_NAME).write_text(
            json.dumps({"hf_repo": hf_repo, "revision": revision, "bytes": len(b"fetched") * len(files),
                        "pulled": "now", "files": [{"target": e.target} for e in files]}),
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(weights, "_snapshot", snapshot)
    monkeypatch.setattr(weights, "pull_files", files)
    monkeypatch.setattr(weights, "adopt_hub_cache", lambda *args, **kwargs: 0)
    return fetched


def test_removing_it_takes_only_the_adapter_and_yue2_3b_waits_for_it(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_box(home, enable_audio=True)
    config = load_config(home)
    _weights(home, SONG, "cuda-linux")
    _fake_hub(monkeypatch)
    manifest, base = load_audio_manifest(INSTRUMENTAL), load_audio_manifest(SONG)
    audioweights.pull(config, manifest, manifest.spec("cuda-linux"))
    with pytest.raises(weights.WeightsShared) as caught:
        audioweights.remove(config, base, base.spec("cuda-linux"))
    assert caught.value.aliases == (INSTRUMENTAL,)
    folder = home / "models" / SONG / "cuda-linux"
    audioweights.remove(config, manifest, manifest.spec("cuda-linux"))
    assert not (folder / "instrumental-lora").exists()
    assert audioweights.installed(config, manifest, manifest.spec("cuda-linux")) is None
    assert audioweights.installed(config, base, base.spec("cuda-linux")) is not None
    assert (folder / "vae" / "model.safetensors").is_file() and (folder / "model.safetensors").is_file()
    audioweights.remove(config, base, base.spec("cuda-linux"))
    assert not folder.exists()


# --- a song through the server, on the fake worker -----------------------------------------------


@pytest.fixture
def instrumental_ready(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,  # noqa: F811
    transcript: Path,  # noqa: F811
) -> Iterator[TestClient]:
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SONG, FAKE_BACKEND.kind)
    _stamp_adapter(home)
    with make_client(enable_audio=True) as client:
        yield client


def _instrumental(**extra: Any) -> dict:
    return {"tags": INSTRUMENTAL_TAGS, "instrumental": True, "seed": 1297391292, **extra}


def test_an_instrumental_is_made_with_the_adapter_folded_around_compose_and_says_so(
    instrumental_ready: TestClient, auth: dict[str, str], transcript: Path, home: Path  # noqa: F811
) -> None:
    job_id, events = run_job(instrumental_ready, auth, model=INSTRUMENTAL, params=_instrumental())
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert (audio["model"], audio["weights_of"], audio["instrumental"]) == (INSTRUMENTAL, SONG, True)
    assert audio["adapter"] == {
        "name": "instrumental-lora", "hf_repo": ADAPTER_REPO, "revision": ADAPTER_REVISION,
        "file": ADAPTER_FILE, "sha256": ADAPTER_SHA, "bytes": ADAPTER_BYTES, "rank": 64, "scale": 1.0,
        "applied_to": ["compose"], "linears": 196,
    }
    assert audio["hf_repo"] == "m-a-p/YuE2-3B", "the base it was folded into"
    (load,) = loads(transcript)
    folder = home / "models" / SONG / "cuda-linux"
    assert load["model_dir"] == str(folder) and load["parts"]["vae"] == str(folder / "vae")
    assert load["adapter"]["path"] == str(folder / "instrumental-lora" / ADAPTER_FILE)
    assert (load["adapter"]["sha256"], load["adapter"]["applies_to"]) == (ADAPTER_SHA, ["compose"])
    record = json.loads((home / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))
    assert record["done_extra"]["audio"]["adapter"]["applied_to"] == ["compose"]


@pytest.mark.parametrize(
    ("variable", "value", "differs"),
    [("CRUCIBLE_FAKE_AUDIO_ADAPTER_SHA256", "0" * 64, "['sha256']"),
     ("CRUCIBLE_FAKE_AUDIO_ADAPTER_STAGES", "plan,compose", "['applied_to']")],
)
def test_a_worker_whose_account_of_the_adapter_differs_fails_the_song(
    instrumental_ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch,
    variable: str, value: str, differs: str,
) -> None:
    monkeypatch.setenv(variable, value)
    _, events = run_job(instrumental_ready, auth, model=INSTRUMENTAL, params=_instrumental())
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "worker_failed" and differs in error["message"]


def test_a_sung_request_is_refused_at_submit_and_nothing_loads(
    instrumental_ready: TestClient, auth: dict[str, str], transcript: Path  # noqa: F811
) -> None:
    response = submit(instrumental_ready, auth, model=INSTRUMENTAL, params={"tags": TAGS, "lyrics": LYRICS})
    assert response.status_code == 400 and response.json()["error"]["code"] == "instrumental_only"
    assert loads(transcript) == []


def test_yue2_3b_loads_and_records_no_adapter_and_switching_loads_the_other(
    instrumental_ready: TestClient, auth: dict[str, str], transcript: Path  # noqa: F811
) -> None:
    # A queue session keeps the resident between its songs, as an album's run does.
    session_id = open_queue_session(instrumental_ready, auth, act="song")
    _, events = run_job(instrumental_ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    audio = events[-1]["data"]["audio"]
    assert (audio["model"], audio["adapter"], audio["weights_of"]) == (SONG, None, None)
    _, events = run_job(instrumental_ready, auth, model=INSTRUMENTAL, params=_instrumental())
    assert events[-1]["data"]["audio"]["adapter"]["applied_to"] == ["compose"]
    # The card holds one model id at a time: the instrumental is its own resident, so the
    # switch is a second load of the same base weights (docs/AUDIO.md says why).
    first, second = loads(transcript)
    assert first["model_dir"] == second["model_dir"]
    assert first["adapter"] is None and second["adapter"]["sha256"] == ADAPTER_SHA
    _, events = run_job(instrumental_ready, auth, model=INSTRUMENTAL, params=_instrumental(seed=5))
    assert events[-1]["event"] == "done" and len(loads(transcript)) == 2, "resident: no reload"
    close_queue_session(instrumental_ready, auth, session_id)
