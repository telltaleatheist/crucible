"""B-Sides' instrumental chain (Owen, 2026-10-11): a yue2-3b song is rendered and held,
and a denoise job takes it by reference as its input - the song never leaves the server -
and returns the instrumental in the format the app keeps. The workers are the fakes; what
is under test is the wire: the held artifact, the reference, the stems' names."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, hosttools, jobenv
from crucible.denoisemodels import load_denoise_manifest
from crucible.jobs import audio as audio_job
from crucible.jobs import denoise as denoise_job
from crucible.memorybudget import GIB

from .conftest import FAKE_BACKEND, parse_sse, stamp_env
from .test_audio_api import LYRICS, SONG, TAGS, _envs, _weights

HERE = Path(__file__).resolve().parent
VOCALS_MODEL = "vocals-roformer"

# vocals-roformer's stems as audio-separator 0.31.1 names them (read off the records of
# real runs on owens-pc): `<input name>_(vocals)_<checkpoint>` and `..._(other)_...`.
VOCALS_STEMS = json.dumps(
    [
        {"name": "{base}_(vocals)_vocals_mel_band_roformer.{ext}"},
        {"name": "{base}_(other)_vocals_mel_band_roformer.{ext}"},
    ]
)


@pytest.fixture
def ready(
    make_client: Callable[..., TestClient],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: "/opt/crucible/tools/bin/ffmpeg")
    monkeypatch.setattr(hosttools, "ffprobe_path", lambda: "/opt/crucible/tools/bin/ffprobe")

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_TRANSCRIPT", str(tmp_path / "audio.jsonl"))
    for engine in list(audio_job.WORKER_SCRIPTS):
        monkeypatch.setitem(audio_job.WORKER_SCRIPTS, engine, HERE / "fake_audio_worker.py")
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, SONG, FAKE_BACKEND.kind)

    monkeypatch.setattr(denoise_job, "WORKER_SCRIPT", HERE / "fake_denoise_worker.py")
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(tmp_path / "denoise.jsonl"))
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_STEMS", VOCALS_STEMS)
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_INPUT",
        json.dumps({
            "sample_rate": 48000, "frames": 48000 * 150, "channels": 2,
            "codec": "flac", "container": "flac",
        }),
    )
    stamp_env(
        home, jobenv.worker_env("rvc", FAKE_BACKEND.kind), FAKE_BACKEND.kind,
        monkeypatch, python=Path(sys.executable),
    )
    manifest = load_denoise_manifest(VOCALS_MODEL)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"not a checkpoint")
    (root / manifest.config_filename).write_bytes(b"not a config")

    with make_client(enable_audio=True, enable_denoise=True) as client:
        yield client


def run(client: TestClient, auth: dict[str, str], body: dict[str, Any]) -> tuple[str, dict]:
    response = client.post("/v1/jobs", headers=auth, json=body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done", events[-1]
    return job_id, events[-1]["data"]


def test_a_held_song_is_denoised_by_reference_and_only_the_instrumental_is_fetched(
    ready: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    # 1. The song, rendered lossless and held so its artifact outlives the next job.
    song_id, song = run(ready, auth, {
        "type": "audio", "model": SONG, "hold": True,
        "params": {"tags": TAGS, "lyrics": LYRICS, "format": "flac"},
    })
    assert "audio.flac" in song["artifacts"]
    rendered = ready.app.state.store.get(song_id).artifacts_dir / "audio.flac"

    # 2. The denoise job takes it by reference: no upload, the same bytes.
    denoise_id, stems = run(ready, auth, {
        "type": "denoise", "model": VOCALS_MODEL,
        "params": {"stems": "all", "format": "mp3"},
        "inputs": {"song.flac": {"artifact": {"job_id": song_id, "name": "audio.flac"}}},
    })
    given = ready.app.state.store.get(denoise_id).inputs_dir / "song.flac"
    assert given.read_bytes() == rendered.read_bytes()
    assert stems["artifacts"] == [
        "song_(vocals)_vocals_mel_band_roformer.mp3",
        "song_(other)_vocals_mel_band_roformer.mp3",
    ]
    assert stems["primary_stem"] == "song_(vocals)_vocals_mel_band_roformer.mp3"
    assert stems["format"] == "mp3"
    assert (stems["sample_rate"], stems["frames"]) == (48000, 48000 * 150)
    assert stems["input"]["codec"] == "flac"
    assert stems["separated"]["sample_rate"] == 44100
    assert stems["separated"]["resampled"] is True

    # The separator was handed the lossless song; mp3 is asked of the stems only.
    lines = (tmp_path / "denoise.jsonl").read_text(encoding="utf-8").splitlines()
    request = next(json.loads(line) for line in lines if '"op": "separate"' in line)
    assert request["input"].endswith("song.flac")
    assert (request["separator_format"], request["output_format"]) == ("WAV", "mp3")

    # 3. The app fetches the instrumental alone, then lets the song go.
    instrumental = ready.get(
        f"/v1/jobs/{denoise_id}/artifacts/song_(other)_vocals_mel_band_roformer.mp3",
        headers=auth,
    )
    assert instrumental.status_code == 200
    assert b"(other)" in instrumental.content
    assert ready.delete(f"/v1/jobs/{song_id}/hold", headers=auth).status_code == 204
    assert not rendered.exists()
    assert given.read_bytes()


def test_a_song_that_was_not_held_and_was_fetched_cannot_be_referenced(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """Without `hold`, fetching all of the song's artifacts is what lets it go; a
    reference then is refused by name before any job exists."""
    song_id, song = run(ready, auth, {
        "type": "audio", "model": SONG,
        "params": {"tags": TAGS, "lyrics": LYRICS, "format": "flac"},
    })
    for name in song["artifacts"]:
        for path in (name, f"{name}.provenance.json"):
            got = ready.get(f"/v1/jobs/{song_id}/artifacts/{path}", headers=auth)
            assert got.status_code == 200, (path, got.text)
    ready.app.state.store.reap()
    response = ready.post("/v1/jobs", headers=auth, json={
        "type": "denoise", "model": VOCALS_MODEL, "params": {"stems": "all"},
        "inputs": {"song.flac": {"artifact": {"job_id": song_id, "name": "audio.flac"}}},
    })
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "artifact_expired"
