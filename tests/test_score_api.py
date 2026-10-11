from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, hosttools, jobenv, tasks
from crucible.cardkinds import KIND_AUDIO, KIND_SCORE
from crucible.config import load_config
from crucible.errors import ApiError
from crucible.jobs import score as score_job
from crucible.memorybudget import GIB
from crucible.residency import Occupant, Residency, ResidentAudio
from crucible.scoremodels import load_score_manifest

from .conftest import (
    FAKE_BACKEND,
    FAKE_MAC_BACKEND,
    close_queue_session,
    configure_box,
    open_queue_session,
    parse_sse,
    stamp_env,
)
from .score_fixtures import MELODY_ONLY_SONG, SHEETSAGE2_SONG, SILENT_SONG, UNKNOWN_LABEL_SONG

MODEL = "sheetsage2"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_score_worker.py"

# A stand-in for ffmpeg: a file that begins `FAKEAUDIO <seconds>` decodes to that many
# seconds of silence at the rate and format it is asked for, from `-ss`, for at most `-t`;
# anything else is refused the way ffmpeg refuses a file that is not media.
FFMPEG_STUB = r'''
import struct, sys
args = sys.argv[1:]
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
source, target = value("-i"), args[-1]
with open(source, "rb") as handle:
    head = handle.read(64)
if not head.startswith(b"FAKEAUDIO "):
    sys.stderr.write(source + ": Invalid data found when processing input\n")
    sys.exit(1)
seconds = float(head.split()[1])
start = float(value("-ss", "0"))
length = min(max(0.0, seconds - start), float(value("-t", "1e9")))
rate = int(value("-ar"))
assert value("-ac") == "1" and value("-f") == "f32le", args
with open(target, "wb") as out:
    out.write(b"\0\0\0\0" * int(round(length * rate)))
'''


def recording_bytes(seconds: float) -> bytes:
    return f"FAKEAUDIO {seconds}\n".encode("ascii")


def inline(data: bytes) -> dict[str, str]:
    return {"inline_base64": base64.b64encode(data).decode("ascii")}


def _env(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    return stamp_env(
        home, jobenv.worker_env("score", backend_kind), backend_kind, monkeypatch,
        python=Path(sys.executable),
    )


def _weights(home: Path, backend_kind: str) -> Path:
    spec = load_score_manifest(MODEL).spec(backend_kind)
    directory = home / "models" / MODEL / backend_kind
    directory.mkdir(parents=True, exist_ok=True)
    for name in spec.files:
        (directory / name).write_bytes(b"weights")
    (directory / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 229_000_000,
                    "pulled": "2026-10-11T02:00:00+0000"}),
        encoding="utf-8",
    )
    part = directory / spec.encoder.name
    part.mkdir(parents=True, exist_ok=True)
    for entry in spec.encoder.files:
        (part / entry.target).write_bytes(b"part")
    (part / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": spec.encoder.hf_repo, "revision": spec.encoder.revision,
                    "bytes": spec.encoder.bytes, "pulled": "2026-10-11T02:00:00+0000",
                    "files": [{"target": e.target} for e in spec.encoder.files]}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    launcher = tmp_path / "bin" / "ffmpeg"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(f"#!{sys.executable}\n" + FFMPEG_STUB, encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: str(launcher))
    return str(launcher)


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "score-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_TRANSCRIPT", str(path))
    monkeypatch.setattr(score_job, "WORKER_SCRIPT", FAKE_WORKER)
    return path


@pytest.fixture
def ready(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    transcript: Path,
    ffmpeg: str,
) -> Iterator[TestClient]:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_BACKEND.kind)
    with make_client(enable_score=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "score")
    body.setdefault("model", MODEL)
    body.setdefault("params", {})
    body.setdefault("inputs", {"song.mp3": inline(recording_bytes(200))})
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


def rows(transcript: Path, op: str) -> list[dict]:
    if not transcript.is_file():
        return []
    found = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in found if row.get("op") == op]


def artifact(client: TestClient, auth: dict[str, str], job_id: str, name: str) -> bytes:
    response = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert response.status_code == 200, response.text
    return response.content


def labels_of(text: str) -> list[str]:
    return [line[2:] for line in text.splitlines() if line.startswith("% ")]


# --- a transcription -------------------------------------------------------------------


def test_a_song_becomes_a_yue2_native_score_with_its_evidence_and_report(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "done", events[-1]
    done = events[-1]["data"]
    assert done["artifacts"] == [
        "score.abc", "instrumental.abc", "sheetsage2.abc", "transcription.mid", "score.json"
    ]
    report = done["score"]
    spec = load_score_manifest(MODEL).spec(FAKE_BACKEND.kind)
    assert (report["model"], report["revision"], report["engine"], report["device"]) == (
        MODEL, spec.revision, "sheetsage2", "cuda"
    )
    assert report["encoder"] == {
        "hf_repo": "m-a-p/MERT-v2-FullSong", "revision": spec.encoder.revision, "companion": "mert"
    }
    assert (report["task"], report["keep_harmony"], report["dtype"]) == ("melody-full", False, "bf16")
    assert report["recording"] == {
        "input": "song.mp3", "start_s": 0.0, "max_seconds": None, "seconds": 200.0,
        "sample_rate": 24000, "channels": 1,
    }
    assert [w["window"] for w in report["windows"]] == [1]
    assert report["windows"][0]["token_cap"] == 5120 and report["windows_at_token_limit"] == []
    assert report["warnings"] == [] and report["peak_gpu_mib"] == 3071.5
    assert report["elapsed_seconds"] == 3.25 and report["peak_bytes"] == 11
    assert report["memory_basis"] == "declared" and report["verified"] is False
    assert report["licence"].startswith("CC BY-NC 4.0")
    native = report["yue2_native"]
    assert native["ok"] is True and native["error"] is None
    assert [c["check"] for c in native["checks"]] == ["labels", "parse_abc", "convert_score", "validate_score"]
    assert [(r["sheetsage2"], r["yue2"], r["kept"]) for r in native["labels"]][:3] == [
        ("silence", "intro", True), ("verse", "verse", True), ("verse", "verse", False)
    ]
    score = artifact(ready, auth, job_id, "score.abc").decode("utf-8")
    assert labels_of(score) == ["intro", "verse", "chorus", "verse", "outro"]
    assert artifact(ready, auth, job_id, "sheetsage2.abc").decode("utf-8") == MELODY_ONLY_SONG
    instrumental = artifact(ready, auth, job_id, "instrumental.abc").decode("utf-8")
    assert labels_of(instrumental) == labels_of(score)
    assert artifact(ready, auth, job_id, "transcription.mid").startswith(b"MThd")
    assert json.loads(artifact(ready, auth, job_id, "score.json")) == report
    lengths = report["lengths"]
    assert lengths["score"]["seconds"] == 30.0 and lengths["score"]["bars"] == 15
    assert lengths["score"]["chars"] == len(score)
    assert lengths["score"]["yue2_tokens"] is None and "not counted" in lengths["score"]["yue2_tokens_note"]
    assert lengths["instrumental"]["seconds"] == 30.0
    sent = rows(transcript, "score")[0]
    assert sent["samples"] == 200 * 24000 and sent["melody_only"] is True
    assert sent["prompts"] == ["timestamp", "downbeat_meter", "structure", "key", "melody_full"]
    load = rows(transcript, "load")[0]
    assert (load["engine"], load["device"], load["autocast"]) == ("sheetsage2", "cuda", "bfloat16")
    assert load["memory_cap_bytes"] == spec.memory_bytes_estimate
    assert load["encoder_dir"].endswith(f"{MODEL}/{FAKE_BACKEND.kind}/mert")
    assert (load["hf_offline"], load["mps_fallback"]) == ("1", None)
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    stages = {e["data"].get("stage") for e in events if e["event"] == "progress"}
    assert {"audio", "reading", "encoding", "decoding", "notation", "normalising", "done"} <= stages


def test_keep_harmony_asks_for_the_chords_and_keeps_them(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(ready, auth, params={"keep_harmony": True})
    assert events[-1]["event"] == "done", events[-1]
    report = events[-1]["data"]["score"]
    assert (report["task"], report["keep_harmony"]) == ("full", True)
    assert rows(transcript, "score")[0]["prompts"][-2:] == ["chord_full", "melody_full"]
    assert rows(transcript, "score")[0]["melody_only"] is False
    assert report["yue2_native"]["instrumental_transfer"]["chords"] == "preserved"
    assert artifact(ready, auth, job_id, "sheetsage2.abc").decode("utf-8") == SHEETSAGE2_SONG
    assert '"C"' in artifact(ready, auth, job_id, "instrumental.abc").decode("utf-8")


def test_a_long_song_is_two_windows_and_a_window_at_its_limit_is_reported(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_TOKENS", "5121")
    _, events = run_job(ready, auth, inputs={"long.wav": inline(recording_bytes(360))})
    assert events[-1]["event"] == "done", events[-1]
    report = events[-1]["data"]["score"]
    assert [(w["start_s"], w["end_s"]) for w in report["windows"]] == [(0.0, 300.0), (60.0, 360.0)]
    assert report["windows_at_token_limit"] == [1, 2]
    assert report["warnings"] == [
        "Window 1 reached the token limit; inspect its token coverage",
        "Window 2 reached the token limit; inspect its token coverage",
    ]


def test_a_crop_transcribes_only_the_part_asked_for(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        ready, auth, params={"start_s": 600, "max_seconds": 90},
        inputs={"concert.flac": inline(recording_bytes(3600))},
    )
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["score"]["recording"]["seconds"] == 90.0
    assert rows(transcript, "score")[0]["samples"] == 90 * 24000


# --- refused by name ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "params", "code", "words"),
    [
        ({}, {}, "invalid_inputs", "exactly one recording"),
        ({"a.mp3": inline(recording_bytes(60)), "b.mp3": inline(recording_bytes(60))}, {},
         "invalid_inputs", "exactly one recording"),
        ({"notes.pdf": inline(b"%PDF-1.7 not a song")}, {}, "audio_undecodable",
         "Invalid data found when processing input"),
        ({"long.mp3": inline(recording_bytes(361))}, {}, "audio_too_long", "runs past 360 s"),
        ({"song.mp3": inline(recording_bytes(60))}, {"start_s": 75}, "audio_too_short",
         "start_s may be past the recording's end"),
    ],
)
def test_a_recording_it_cannot_score_is_refused_before_the_model_loads(
    ready: TestClient, auth: dict[str, str], transcript: Path,
    inputs: dict, params: dict, code: str, words: str,
) -> None:
    _, events = run_job(ready, auth, inputs=inputs, params=params)
    assert events[-1]["event"] == "failed", events[-1]
    error = events[-1]["data"]["error"]
    assert error["code"] == code and words in error["message"], error
    assert rows(transcript, "load") == []


@pytest.mark.parametrize(
    ("params", "code", "words"),
    [
        ({"max_seconds": 400}, "audio_too_long", "past sheetsage2's ceiling of 360 s"),
        ({"max_seconds": 0}, "invalid_params", "max_seconds"),
        ({"start_s": -1}, "invalid_params", "start_s"),
        ({"harmony": True}, "invalid_params", "Extra inputs are not permitted"),
    ],
)
def test_params_are_refused_at_submit(
    ready: TestClient, auth: dict[str, str], params: dict, code: str, words: str
) -> None:
    error = refusal(submit(ready, auth, params=params))
    assert error["code"] == code and words in error["message"], error


def test_no_ffmpeg_is_refused_at_submit(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: None)
    assert refusal(submit(ready, auth))["code"] == "ffmpeg_missing"


# --- what cannot be made native -------------------------------------------------------------


@pytest.mark.parametrize(
    ("abc", "code", "words"),
    [
        (UNKNOWN_LABEL_SONG, "score_label_unknown", "'kazoo break'"),
        (SILENT_SONG, "score_not_yue2_native", "The score contains no sounding notes"),
    ],
)
def test_a_score_that_cannot_be_made_native_fails_by_name_and_keeps_the_raw_score(
    ready: TestClient, auth: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    abc: str, code: str, words: str,
) -> None:
    named = tmp_path / "raw.abc"
    named.write_text(abc, encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_ABC", str(named))
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "failed", events[-1]
    error = events[-1]["data"]["error"]
    assert error["code"] == code and words in error["message"], error
    assert "sheetsage2.abc" in error["message"]
    state = ready.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert state["artifacts"] == ["sheetsage2.abc", "transcription.mid", "score.json"]
    assert artifact(ready, auth, job_id, "sheetsage2.abc").decode("utf-8") == abc
    report = json.loads(artifact(ready, auth, job_id, "score.json"))
    assert report["yue2_native"]["ok"] is False and report["yue2_native"]["error"]["code"] == code
    assert report["artifacts"]["score"] is None and report["lengths"]["score"] is None


def test_no_score_from_sheetsage2_fails_with_its_reason(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_NO_ABC", "1")
    job_id, events = run_job(ready, auth)
    error = events[-1]["data"]["error"]
    assert error["code"] == "score_not_transcribed" and "No key was decoded" in error["message"]
    state = ready.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert state["artifacts"] == ["transcription.mid", "score.json"]


# --- residency -----------------------------------------------------------------------------


def test_outside_a_session_the_model_is_unloaded_when_the_job_ends(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 2


def test_load_score_in_a_queue_session_and_every_song_reuses_it(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    session_id = open_queue_session(ready, auth, act="score")
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-score", "model": MODEL})
    assert loaded.status_code == 202, loaded.json()
    done = events_of(ready, auth, loaded.json()["job_id"])[-1]
    assert done["event"] == "done", done
    assert done["data"]["resident"] == MODEL
    for _ in range(2):
        assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 1
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "score"
    close_queue_session(ready, auth, session_id)
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_unload_score_takes_the_transcriber_off_the_card(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(ready.post("/v1/jobs", headers=auth, json={"type": "unload-score", "model": MODEL}))
    assert error["code"] == "score_transcriber_not_resident"
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-score", "model": MODEL})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    response = ready.post("/v1/jobs", headers=auth, json={"type": "unload-score", "model": MODEL})
    assert response.status_code == 202, response.json()
    assert events_of(ready, auth, response.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_cancel_stops_the_transcription_and_keeps_the_model(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_PAUSE_S", "20")
    open_queue_session(ready, auth, act="score")
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-score", "model": MODEL})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    job_id = submit(ready, auth).json()["job_id"]
    deadline = time.monotonic() + 20.0
    while not rows(transcript, "score") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "cancelled", events[-1]
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "score"
    monkeypatch.setenv("CRUCIBLE_FAKE_SCORE_PAUSE_S", "0")
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 1


def test_loading_it_takes_a_resident_yue2_off_the_card_first(
    home: Path, monkeypatch: pytest.MonkeyPatch, transcript: Path, idle_card: None
) -> None:
    """One resident engine at a time: a YuE2 kept for a cover's song is unloaded before
    SheetSage2 loads (Residency.occupy's eviction), and said so."""
    configure_box(home, enable_score=True, enable_audio=True)
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    weights_dir = _weights(home, FAKE_BACKEND.kind)
    residency = Residency(load_config(home))
    stopped: list[str] = []

    class Session:
        pids: frozenset[int] = frozenset()
        alive = True

        def stop(self) -> None:
            stopped.append("yue2-3b")

    yue2 = ResidentAudio(
        model_id="yue2-3b", backend=FAKE_BACKEND.kind, engine="yue2", revision="c" * 40,
        fingerprint="f", device="cuda", dtype="bfloat16", versions={},
        memory_bytes_estimate=16_000_000_000, log_path=home / "yue2.log", loaded_at="now",
    )
    residency.occupy(KIND_AUDIO, "yue2-3b", lambda: Occupant(yue2, session=Session()), say=lambda _: None)
    assert residency.resident_kind == KIND_AUDIO
    manifest = load_score_manifest(MODEL)
    needs = score_job.Needs(manifest, manifest.spec(FAKE_BACKEND.kind), Path(sys.executable), weights_dir)
    said: list[str] = []
    resident = score_job.occupy_scorer(residency, needs, on_progress=said.append)
    try:
        assert stopped == ["yue2-3b"]
        assert any("unloading yue2-3b" in line for line in said), said
        assert (resident.id, residency.resident_kind) == (MODEL, KIND_SCORE)
        assert residency.score_session is not None and residency.score_session.alive
    finally:
        residency.unload(MODEL)


def test_a_card_without_room_is_refused_before_the_worker_starts(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (4 * GIB, 7 * GIB))
    error = refusal(submit(ready, auth))
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 6_000_000_000
    assert rows(transcript, "load") == []


# --- the Mac, and installing ---------------------------------------------------------------


def test_the_mac_runs_it_on_metal_in_float32_with_the_cpu_fallback(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, transcript: Path, ffmpeg: str,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _env(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_MAC_BACKEND.kind)
    with make_client(enable_score=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        _, events = run_job(client, auth)
    assert events[-1]["event"] == "done", events[-1]
    assert (events[-1]["data"]["score"]["device"], events[-1]["data"]["score"]["dtype"]) == ("mps", "fp32")
    load = rows(transcript, "load")[0]
    assert (load["device"], load["autocast"], load["memory_cap_bytes"]) == ("mps", None, None)
    assert load["mps_fallback"] == "1"
    assert rows(transcript, "score")[0]["dtype"] == "fp32"


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient], auth: dict[str, str],
    idle_card: None, ffmpeg: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_score=True) as client:
        error = refusal(submit(client, auth))
        assert error["code"] == "installing"
        assert "installing the score environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "score"}]


def test_missing_weights_are_pulled_on_submit(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, ffmpeg: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_score=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "installing"
    assert f"pulling the model '{MODEL}'" in error["message"]


def test_a_partly_pulled_model_names_the_missing_encoder(
    home: Path, monkeypatch: pytest.MonkeyPatch, ffmpeg: str
) -> None:
    configure_box(home, enable_score=True)
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    weights_dir = _weights(home, FAKE_BACKEND.kind)
    (weights_dir / "mert" / "crucible-pull.json").unlink()
    config = load_config(home)
    job_type = score_job.ScoreJobType(config, FAKE_BACKEND, Residency(config))
    with pytest.raises(ApiError) as refused:
        job_type.loadable(MODEL)
    assert refused.value.code == "model_not_installed"
    assert "mert" in refused.value.message and "m-a-p/MERT-v2-FullSong" in refused.value.message
