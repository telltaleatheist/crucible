from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, tasks, weights
from crucible.audiomodels import load_audio_manifest
from crucible.jobs import audio as audio_job
from crucible.memorybudget import GIB

from .conftest import (
    FAKE_BACKEND,
    FAKE_MAC_BACKEND,
    close_queue_session,
    open_queue_session,
    parse_sse,
    stamp_env,
)

SFX = "stable-audio-3-small-sfx"
MUSIC = "stable-audio-3-medium"
SONG = "yue2-3b"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_audio_worker.py"
PROMPT = "TrackType: SFX. A heavy wooden door creaks open slowly in a stone hallway, close mic"
TAGS = "English, warm piano pop, expressive female voice, 88 BPM"
LYRICS = "[Verse]\nThe kettle sings the morning in\n\n[Chorus]\nStay, stay a while\n"


def _envs(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in jobenv.audio_envs(backend_kind):
        stamp_env(home, spec, backend_kind, monkeypatch, python=Path(sys.executable))


def _weights(home: Path, model: str, backend_kind: str) -> Path:
    spec = load_audio_manifest(model).spec(backend_kind)
    directory = home / "models" / model / backend_kind
    for name in spec.files:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"weights")
    (directory / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 3_450_000_000,
                    "pulled": "2026-09-29T02:00:00+0000"}),
        encoding="utf-8",
    )
    for companion in spec.companions:
        part = directory / companion.name
        part.mkdir(parents=True, exist_ok=True)
        for entry in companion.files:
            (part / entry.target).write_bytes(b"part")
        (part / "crucible-pull.json").write_text(
            json.dumps({"hf_repo": companion.hf_repo, "revision": companion.revision,
                        "bytes": companion.bytes, "pulled": "2026-09-29T02:00:00+0000",
                        "files": [{"target": e.target} for e in companion.files]}),
            encoding="utf-8",
        )
    return directory


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "audio-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_TRANSCRIPT", str(path))
    for engine in list(audio_job.WORKER_SCRIPTS):
        monkeypatch.setitem(audio_job.WORKER_SCRIPTS, engine, FAKE_WORKER)
    return path


@pytest.fixture
def ready(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    transcript: Path,
) -> Iterator[TestClient]:
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    for model in (SFX, MUSIC, SONG):
        _weights(home, model, FAKE_BACKEND.kind)
    with make_client(enable_audio=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "audio")
    body.setdefault("model", SFX)
    body.setdefault("params", {"prompt": PROMPT, "duration_s": 3, "steps": 4})
    return client.post("/v1/jobs", headers=auth, json=body)


def events_of(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict]:
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> tuple[str, list[dict]]:
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    return job_id, events_of(client, auth, job_id)


def refusal(response: Any) -> dict:
    assert response.status_code >= 400, response.text
    return response.json()["error"]


def loads(transcript: Path) -> list[dict]:
    if not transcript.is_file():
        return []
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "load"]


def streaminfo(flac: bytes) -> tuple[int, int, int]:
    assert flac[:4] == b"fLaC"
    packed = int.from_bytes(flac[18:26], "big")
    return packed >> 44, ((packed >> 41) & 0x7) + 1, packed & ((1 << 36) - 1)


def wait_until_running(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            return
        time.sleep(0.01)
    raise AssertionError("the job never started running")


@pytest.mark.parametrize(
    ("model", "params", "code", "words"),
    [
        (SFX, {"prompt": PROMPT, "lyrics": LYRICS}, "audio_param_unsupported", "does not take 'lyrics'"),
        (SFX, {"tags": TAGS}, "audio_param_unsupported", "not 'tags'"),
        (SFX, {"duration_s": 3}, "audio_param_missing", "needs 'prompt'"),
        (SFX, {"prompt": PROMPT, "duration_s": 121}, "audio_too_long", "at most 120 s"),
        (MUSIC, {"prompt": "house, 124 BPM", "duration_s": 381}, "audio_too_long", "at most 380 s"),
        (SFX, {"prompt": PROMPT, "negative_prompt": "hiss"}, "audio_param_unsupported", "post-trained"),
        (SFX, {"prompt": PROMPT, "cfg": 3}, "audio_param_unsupported", "cfg"),
        (SFX, {"prompt": PROMPT, "steps": 51}, "audio_param_out_of_range", "ceiling of 50"),
        (SONG, {"prompt": TAGS, "lyrics": LYRICS}, "audio_param_unsupported", "reads its description from 'tags'"),
        (SONG, {"tags": TAGS}, "audio_param_missing", "needs 'lyrics'"),
        (SONG, {"tags": TAGS, "lyrics": LYRICS, "duration_s": 60}, "audio_param_unsupported", "as long as its lyrics"),
        (SONG, {"tags": TAGS, "lyrics": LYRICS, "steps": 8}, "audio_param_unsupported", "32-step"),
        (SONG, {"tags": TAGS, "lyrics": LYRICS, "cfg": 21}, "audio_param_out_of_range", "ceiling of 20"),
        (SFX, {"prompt": PROMPT, "format": "ogg"}, "invalid_params", "format"),
        (SFX, {"prompt": PROMPT, "quality": 9}, "invalid_params", "Extra inputs are not permitted"),
        (SFX, {"prompt": "  "}, "invalid_params", "is empty"),
    ],
)
def test_params_are_refused_by_name_per_model(
    ready: TestClient, auth: dict[str, str], transcript: Path,
    model: str, params: dict, code: str, words: str,
) -> None:
    error = refusal(submit(ready, auth, model=model, params=params))
    assert error["code"] == code, error
    assert words in error["message"]
    assert loads(transcript) == []


def test_a_sound_effect_is_made_with_step_progress_and_a_flac(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(ready, auth, params={"prompt": PROMPT, "duration_s": 3, "steps": 4, "seed": 7})
    assert events[-1]["event"] == "done", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert steps[-4:] == [1, 2, 3, 4]
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    done = events[-1]["data"]
    assert done["artifacts"] == ["audio.flac"]
    audio = done["audio"]
    spec = load_audio_manifest(SFX).spec(FAKE_BACKEND.kind)
    assert (audio["seed"], audio["steps"], audio["duration_s"], audio["cfg"]) == (7, 4, 3.0, None)
    assert (audio["revision"], audio["engine"], audio["kind"], audio["score"]) == (spec.revision, "stable-audio-3", "sfx", None)
    assert (audio["sample_rate"], audio["channels"]) == (44100, 2)
    assert audio["peak_bytes"] == 9 and audio["memory_basis"] == "declared"
    assert (audio["decode_stages"], audio["stages_at_cap"]) == (None, None), "diffusion decodes no tokens"
    flac = ready.get(f"/v1/jobs/{job_id}/artifacts/audio.flac", headers=auth).content
    rate, channels, frames = streaminfo(flac)
    assert (rate, channels) == (44100, 2) and frames == audio["audio_seconds"] * 44100
    load = loads(transcript)[0]
    assert (load["engine"], load["device"], load["memory_cap_bytes"], load["parts"]) == (
        "stable-audio-3", "cuda", spec.memory_bytes_estimate, {}
    )


def test_the_flac_the_fake_writes_is_one_a_real_decoder_reads(
    ready: TestClient, auth: dict[str, str]
) -> None:
    soundfile = pytest.importorskip("soundfile")
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "done"
    path = Path(events[-1]["data"]["audio"]["artifact"])
    body = ready.get(f"/v1/jobs/{job_id}/artifacts/{path.name}", headers=auth).content
    target = Path(pytest.importorskip("tempfile").mkdtemp()) / "fake.flac"
    target.write_bytes(body)
    samples, rate = soundfile.read(str(target))
    assert rate == 44100 and samples.shape[1] == 2


def test_defaults_come_from_the_arm_and_wav_is_offered(
    ready: TestClient, auth: dict[str, str]
) -> None:
    job_id, events = run_job(ready, auth, model=MUSIC, params={"prompt": "house, 124 BPM", "format": "wav"})
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert (audio["duration_s"], audio["steps"], audio["format"]) == (60.0, 8, "wav")
    body = ready.get(f"/v1/jobs/{job_id}/artifacts/audio.wav", headers=auth).content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"


def test_a_song_can_be_an_mp3_at_192_kbps_cbr_served_as_audio_mpeg_with_ranges(
    ready: TestClient, auth: dict[str, str]
) -> None:
    pytest.importorskip("soundfile")
    job_id, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "format": "mp3"})
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert (audio["format"], audio["artifact"]) == ("mp3", "audio.mp3")
    whole = ready.get(f"/v1/jobs/{job_id}/artifacts/audio.mp3", headers=auth)
    assert whole.headers["content-type"] == "audio/mpeg"
    body = whole.content
    assert body[:2] == bytes([0xFF, 0xFB])
    # MPEG-1 Layer III header: bitrate index 1011 is 192 kbps, sample-rate index 00 is 44.1 kHz.
    assert (body[2] >> 4, (body[2] >> 2) & 0x3) == (0b1011, 0b00)
    # Every frame says the same: constant bitrate.
    frame = 144 * 192_000 // 44100
    assert body[frame] == 0xFF and (body[frame + 2] >> 4) == 0b1011
    part = ready.get(f"/v1/jobs/{job_id}/artifacts/audio.mp3", headers={**auth, "Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 0-99/{len(body)}"
    assert part.content == body[:100]


def test_a_flac_artifact_is_served_as_audio_flac(ready: TestClient, auth: dict[str, str]) -> None:
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "done", events[-1]
    answer = ready.get(f"/v1/jobs/{job_id}/artifacts/audio.flac", headers=auth)
    assert answer.headers["content-type"] == "audio/flac"


def test_a_song_publishes_its_score_beside_the_audio(
    ready: TestClient, auth: dict[str, str], transcript: Path, home: Path
) -> None:
    job_id, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    assert events[-1]["event"] == "done", events[-1]
    done = events[-1]["data"]
    assert done["artifacts"] == ["audio.flac", "score.abc"]
    audio = done["audio"]
    assert (audio["score"], audio["tags"], audio["lyrics"], audio["cfg"], audio["duration_s"]) == (
        "score.abc", TAGS, LYRICS, 1.0, None
    )
    assert streaminfo(ready.get(f"/v1/jobs/{job_id}/artifacts/audio.flac", headers=auth).content)[0] == 48000
    score = ready.get(f"/v1/jobs/{job_id}/artifacts/score.abc", headers=auth).text
    assert score.startswith("X:1") and "K:C" in score
    load = loads(transcript)[0]
    model_dir = home / "models" / SONG / FAKE_BACKEND.kind
    assert (load["engine"], load["parts"]) == ("yue2", {"vae": str(model_dir / "vae")})
    assert load["memory_budget_bytes"] == load_audio_manifest(SONG).spec(FAKE_BACKEND.kind).memory_bytes_estimate


def test_a_song_says_how_each_token_stage_ended(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    audio = events[-1]["data"]["audio"]
    assert list(audio["decode_stages"]) == ["scoring", "composing"]
    assert {stage["ended"] for stage in audio["decode_stages"].values()} == {"eos"}
    assert audio["stages_at_cap"] == []


def test_a_stage_that_ran_to_its_cap_is_on_the_done_record_and_the_job_still_succeeds(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch, home: Path
) -> None:
    """Victoria's 6-minute song (RTX 3070, 2026-10-09): only a re-run found the stage that
    never ended. The job is not failed and nothing re-runs it (Owen: a retry "seems like a
    band aid"); the record says it, for a client and for whoever reads job.json after."""
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_CAPPED", "composing")
    job_id, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    assert events[-1]["event"] == "done", events[-1]
    audio = events[-1]["data"]["audio"]
    assert audio["stages_at_cap"] == ["composing"]
    composing = audio["decode_stages"]["composing"]
    assert (composing["tokens"], composing["cap"], composing["ended"]) == (9000, 9000, "cap")
    last_words = [e["data"]["message"] for e in events if e["event"] == "progress"][-1]
    assert "composing at its 9000-token cap without ending" in last_words
    record = json.loads((home / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))
    kept = record["done_extra"]["audio"]
    assert kept["stages_at_cap"] == ["composing"] and kept["decode_stages"] == audio["decode_stages"]
    # What it ran with is on the same record: enough to run the seed again.
    assert (kept["tags"], kept["lyrics"], kept["seed"], kept["cfg"], kept["instrumental"], kept["low_vram"]) == (
        TAGS, LYRICS, 3, 1.0, False, False
    )


def test_a_seed_left_out_is_chosen_and_reported(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth)
    seed = events[-1]["data"]["audio"]["seed"]
    assert isinstance(seed, int) and 0 <= seed <= 2**32 - 1


def test_input_files_are_refused(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth, inputs={"x.wav": {"inline_base64": "UklGRg=="}})
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "invalid_inputs"


def test_load_audio_in_a_queue_session_and_the_batch_reuses_it(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    session_id = open_queue_session(ready, auth, act="sfx")
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-audio", "model": SFX})
    assert loaded.status_code == 202, loaded.json()
    done = events_of(ready, auth, loaded.json()["job_id"])[-1]
    assert done["event"] == "done", done
    assert done["data"]["resident"] == SFX
    for _ in range(2):
        _, events = run_job(ready, auth, params={"prompt": PROMPT, "duration_s": 2, "steps": 2})
        assert events[-1]["event"] == "done"
    assert len(loads(transcript)) == 1
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "audio"
    close_queue_session(ready, auth, session_id)
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_cancel_stops_between_steps_and_keeps_the_model(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_STEP_S", "0.3")
    open_queue_session(ready, auth, act="sfx")
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-audio", "model": SFX})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    job_id = submit(ready, auth, params={"prompt": PROMPT, "steps": 40}).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    time.sleep(1.0)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "cancelled", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert 0 < max(steps) < 40
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "audio"
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_STEP_S", "0")
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(loads(transcript)) == 1


def test_unload_audio_takes_the_generator_off_the_card(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(ready.post("/v1/jobs", headers=auth, json={"type": "unload-audio", "model": SFX}))
    assert error["code"] == "audio_generator_not_resident"
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-audio", "model": SFX})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "audio"
    response = ready.post("/v1/jobs", headers=auth, json={"type": "unload-audio", "model": SFX})
    assert response.status_code == 202, response.json()
    assert events_of(ready, auth, response.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_card_without_room_is_refused_before_the_worker_starts(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (13 * GIB, 16 * GIB))
    error = refusal(submit(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS}))
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 16_000_000_000
    assert loads(transcript) == []


def test_the_mac_runs_stable_audio_on_metal_and_takes_songs_too(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, transcript: Path,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _envs(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    _weights(home, SFX, FAKE_MAC_BACKEND.kind)
    with make_client(enable_audio=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        _, events = run_job(client, auth)
        refused = refusal(submit(client, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS}))
    assert events[-1]["event"] == "done", events[-1]
    load = loads(transcript)[0]
    assert (load["device"], load["dtype"], load["memory_cap_bytes"]) == ("mps", "float32", None)
    # A song is no longer refused by backend (YuE2 runs on the Mac's Metal with a torch that
    # fixed bfloat16 causal attention, 2026-10-03): the missing yue2 env starts installing.
    assert refused["code"] == "installing", refused


def test_a_gated_model_without_a_token_is_refused_with_the_page_to_accept(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_audio=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "model_gated"
    accept = "https://huggingface.co/stabilityai/stable-audio-3-small-sfx"
    assert error["details"]["accept_url"] == accept
    assert accept in error["message"] and "https://huggingface.co/settings/tokens" in error["message"]
    assert f"`crucible models pull {SFX}`" in error["message"]


def test_the_pull_itself_refuses_a_gated_repo_before_downloading(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from crucible import audioweights
    from crucible.config import load_config

    from .conftest import configure_box

    monkeypatch.delenv("HF_TOKEN", raising=False)
    configure_box(home)
    config = load_config(home)
    manifest = load_audio_manifest(MUSIC)
    fetched: list[Any] = []
    monkeypatch.setattr(weights, "_snapshot", lambda *a, **k: fetched.append(a))
    with pytest.raises(weights.WeightsError) as caught:
        audioweights.pull(config, manifest, manifest.spec(FAKE_BACKEND.kind))
    assert "is gated" in str(caught.value)
    assert "https://huggingface.co/stabilityai/stable-audio-3-medium" in str(caught.value)
    assert fetched == []


def test_a_gated_refusal_from_the_hub_names_the_page_and_the_retry(home: Path) -> None:
    from crucible.config import load_config

    from .conftest import configure_box

    configure_box(home)
    said = weights.gated_message(
        "stabilityai/stable-audio-3-medium", load_config(home), "crucible models pull stable-audio-3-medium",
        RuntimeError("403 Client Error"),
    )
    assert "accept the licence" in said and "403 Client Error" in said
    assert "Run `crucible models pull stable-audio-3-medium` again" in said


def test_missing_weights_that_need_no_licence_are_pulled_on_submit(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_audio=True) as client:
        error = refusal(submit(client, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS}))
    assert error["code"] == "installing"
    assert f"pulling the model '{SONG}'" in error["message"]


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient], auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_audio=True) as client:
        error = refusal(submit(client, auth))
        assert error["code"] == "installing"
        assert "installing the audio environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "audio"}]


def test_a_song_reports_the_workers_host_memory_before_and_after(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """Victoria's album (2026-10-10): the worker grew song after song until the OOM killer
    took it on track 12, and nothing on any job said so. Each song now carries the
    worker's host memory as it began and once it was saved, and the peak between."""
    _, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 3})
    host = events[-1]["data"]["audio"]["host_memory"]
    for reading in (host["before"], host["after"]):
        assert set(reading) == {"rss_bytes", "anon_bytes", "file_bytes"}
        assert reading["rss_bytes"] >= reading["anon_bytes"] > 0
    assert host["peak_rss_bytes"] >= max(host["before"]["rss_bytes"], host["after"]["rss_bytes"])
    assert host["host_homes_bytes"] is None, "the fake engine keeps nothing in host memory"


def _kept_request(home: Path, job_id: str) -> Path:
    return home / "jobs" / job_id / "request.json"


def test_a_song_keeps_its_request_and_seed_while_it_runs_and_drops_them_when_done(
    ready: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owen, 2026-10-10: "1 is fine. but it should clear once the job finishes." A song's
    params and the seed the server chose are on disk from its start; `done` clears them,
    and done_extra.audio is the record."""
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_STEP_S", "0.2")
    response = submit(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS})
    job_id = response.json()["job_id"]
    wait_until_running(ready, auth, job_id)
    deadline = time.monotonic() + 20.0
    while not _kept_request(home, job_id).is_file():
        assert time.monotonic() < deadline, "the song never kept its request"
        time.sleep(0.01)
    kept = json.loads(_kept_request(home, job_id).read_text(encoding="utf-8"))
    seed = kept["seed"]
    assert kept["seed_chosen_by"] == "server" and isinstance(seed, int)
    assert (kept["type"], kept["model"]) == ("audio", SONG)
    assert kept["params"] == {"tags": TAGS, "lyrics": LYRICS, "seed": seed}
    assert kept["settled"]["cfg"] == 1.0 and kept["low_vram"] is False
    assert f"seed {seed}" in kept["reproduce"]
    running = ready.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert running["request"]["seed"] == seed

    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["audio"]["seed"] == seed
    assert not _kept_request(home, job_id).exists()
    assert ready.get(f"/v1/jobs/{job_id}", headers=auth).json()["request"] is None
    record = json.loads((home / "jobs" / job_id / "job.json").read_text(encoding="utf-8"))
    assert record["done_extra"]["audio"]["seed"] == seed and "params" not in record


def test_a_song_that_fails_keeps_what_reproduces_it(
    ready: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Victoria's 1f3da14c: OOM in synthesizing, and nothing on disk said its seed."""
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "1")
    job_id, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "cfg": 1.5})
    assert events[-1]["event"] == "failed", events[-1]
    kept = json.loads(_kept_request(home, job_id).read_text(encoding="utf-8"))
    assert kept["params"] == {"tags": TAGS, "lyrics": LYRICS, "cfg": 1.5, "seed": kept["seed"]}
    state = ready.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert state["status"] == "failed" and state["request"] == kept

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "0")
    again, events = run_job(ready, auth, model=kept["model"], params=kept["params"])
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["audio"]["seed"] == kept["seed"]


def _generates(transcript: Path) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "generate"]


def test_an_instrumental_is_planned_from_the_pool_set_its_seed_picks_and_says_which(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch,
    home: Path,
) -> None:
    """Victoria's laptop, 2026-10-10: 4 of 13 instrumentals planned from empty sections ran
    the score to its cap. The server now hands the worker a pool set picked by the seed,
    and the record names it, so the kept request and seed make the same song again."""
    from crucible.jobs.audio import planning

    pool = planning.load_pool("yue2")
    params = {"tags": "Instrumental, piano, no vocals", "instrumental": True, "seed": 13}
    _, events = run_job(ready, auth, model=SONG, params=params)
    assert events[-1]["event"] == "done", events[-1]
    chosen = pool[13 % len(pool)]
    (generate,) = _generates(transcript)
    assert (generate["planning_lyrics"], generate["lyrics"]) == (chosen.lyrics, None)
    assert events[-1]["data"]["audio"]["planning_lyrics"] == {
        "source": "pool", "id": chosen.id, "requested": False, "lyrics": chosen.lyrics,
        "resized": False,
    }

    own = "[Verse]\nStone on stone the wall goes up\nMoss along the northern side\n"
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "1")
    job_id, events = run_job(ready, auth, model=SONG, params={**params, "planning_lyrics": own})
    assert events[-1]["event"] == "failed"
    assert _generates(transcript)[-1]["planning_lyrics"] == own
    kept = json.loads(_kept_request(home, job_id).read_text(encoding="utf-8"))
    assert kept["settled"]["planning_lyrics"] == {
        "source": "request", "id": None, "requested": None, "lyrics": own}

    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_GENERATE_FAIL", "0")
    _, events = run_job(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 13})
    assert events[-1]["event"] == "done" and events[-1]["data"]["audio"]["planning_lyrics"] is None
    assert _generates(transcript)[-1]["planning_lyrics"] is None


def test_planning_lyrics_are_refused_on_a_sung_song_before_anything_loads(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": TAGS, "lyrics": LYRICS, "planning_lyrics": LYRICS,
    }))
    assert error["code"] == "audio_param_conflict" and "instrumental: true" in error["message"]
    error = refusal(submit(ready, auth, model=SONG, params={
        "tags": TAGS, "instrumental": True, "planning_lyrics": "[Hook]\nla la\n",
    }))
    assert error["code"] == "invalid_params" and "[Hook]" in error["message"]
    assert loads(transcript) == []


def test_a_cancelled_song_keeps_its_request(
    ready: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_AUDIO_STEP_S", "0.3")
    job_id = submit(ready, auth, model=SONG, params={"tags": TAGS, "lyrics": LYRICS, "seed": 9}).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    time.sleep(1.0)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    assert events_of(ready, auth, job_id)[-1]["event"] == "cancelled"
    kept = json.loads(_kept_request(home, job_id).read_text(encoding="utf-8"))
    assert (kept["seed"], kept["seed_chosen_by"]) == (9, "client")


def test_a_sound_effect_that_ends_done_keeps_nothing_and_other_types_never_had_it(
    ready: TestClient, auth: dict[str, str], home: Path
) -> None:
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "done"
    assert not _kept_request(home, job_id).exists()
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-audio", "model": SFX})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    assert ready.get(f"/v1/jobs/{loaded.json()['job_id']}", headers=auth).json()["request"] is None
