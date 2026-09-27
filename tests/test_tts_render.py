from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, residency as residency_module
from crucible.accelerator import GIB
from crucible.errors import ApiError
from crucible.jobs import asr as asr_jobs
from crucible.jobs.tts import render as render_jobs
from crucible.residency import KIND_TTS
from crucible.voicerepo import REPO_MANIFEST_NAME
from crucible.voices import load_voice

from . import fake_narrator_engine
from .conftest import (
    FAKE_BACKEND,
    configure_box,
    holding_the_card,
    parse_sse,
    wav_base64,
    wav_bytes,
)
from .test_tts_api import (
    fake_env,
    fake_weights,
    idle_card,
    tts_client,
    tts_recipes,
)
from .test_tts_api import run_job, submit

VOICE = "deathstalker"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="these tests encode real FLACs; install ffmpeg to run them",
)

CHARS_PER_SEC = 15.0

CHUNKS = [
    {"index": 41, "text": "He had been walking for some time."},
    {"index": 42, "text": "The road did not appear to end, not that day."},
    {"index": 43, "text": "Rain."},
]


@pytest.fixture(autouse=True)
def narrator(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    return fake_narrator_engine.install(monkeypatch)


@pytest.fixture(autouse=True)
def quick_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("crucible.engines.narrator.QUIT_GRACE_SECONDS", 1.0)
    monkeypatch.setattr("crucible.engines.base.READY_POLL_SECONDS", 0.05)


@pytest.fixture
def rendered(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> Callable[..., list[dict[str, Any]]]:

    def go(**params: Any) -> list[dict[str, Any]]:
        fake_weights(VOICE)
        body = {"language": "en", "take": 0, "chunks": CHUNKS}
        body.update(params)
        response = submit(tts_client, auth, type="tts", model=VOICE, params=body)
        assert response.status_code == 202, response.json()
        go.job_id = response.json()["job_id"]
        with tts_client.stream(
            "GET", f"/v1/jobs/{go.job_id}/events", headers=auth
        ) as stream:
            return parse_sse(line for line in stream.iter_lines())

    go.job_id = ""
    return go


def events_of(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event["data"] for event in events if event["event"] == kind]


def terminal(events: list[dict[str, Any]]) -> dict[str, Any]:
    return events[-1]


def test_a_render_publishes_one_flac_per_chunk(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    events = rendered()
    assert terminal(events)["event"] == "done", terminal(events)
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    assert names == ["41.flac", "42.flac", "43.flac"]
    assert terminal(events)["data"]["rendered"] == 3
    assert terminal(events)["data"]["failed"] == []


def test_the_bytes_are_a_real_flac_at_the_voices_sample_rate(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    rendered()
    body = tts_client.get(
        f"/v1/jobs/{rendered.job_id}/artifacts/41.flac", headers=auth
    ).content
    assert body[:4] == b"fLaC"
    streaminfo = body[8:8 + 34]
    packed = int.from_bytes(streaminfo[10:13], "big")
    assert packed >> 4 == 24_000
    assert ((packed >> 1) & 0b111) + 1 == 1
    assert streaminfo[12] & 0b1 or True
    depth = (((streaminfo[12] & 0b1) << 4) | (streaminfo[13] >> 4)) + 1
    assert depth == 16


def test_the_chunk_event_carries_the_measurements_and_the_verdict(
    rendered: Callable[..., list[dict[str, Any]]]
) -> None:
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert sorted(chunks) == [41, 42, 43]
    row = chunks[41]
    assert set(row) == {
        "index", "seconds", "chars", "chars_per_sec", "tokens", "capped", "take",
        "guard",
    }
    assert row["guard"] is None
    assert row["chars"] == len(CHUNKS[0]["text"])
    assert row["seconds"] == pytest.approx(row["chars"] / CHARS_PER_SEC, abs=1e-4)
    assert row["chars_per_sec"] == pytest.approx(CHARS_PER_SEC, abs=1e-3)
    assert row["take"] == 0
    assert row["capped"] is False


def test_a_longer_chunk_is_longer_and_a_shorter_one_shorter(
    rendered: Callable[..., list[dict[str, Any]]]
) -> None:
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[42]["seconds"] > chunks[41]["seconds"] > chunks[43]["seconds"]


def test_capped_is_reported_when_narrator_says_so(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_CAP_CHARS", "10")
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["capped"] is True
    assert chunks[42]["capped"] is True
    assert chunks[43]["capped"] is False


def test_a_narrator_that_reports_no_cap_publishes_null_and_not_false(
    tmp_path: Path,
) -> None:
    from crucible.jobs.tts.render import _optional_bool, _optional_int

    bare = {"i": 41, "format": "pcm16", "data": "", "duration": 1.0}
    assert _optional_bool(bare, "capped") is None
    assert _optional_int(bare, "tokens") is None
    assert _optional_bool({**bare, "capped": False}, "capped") is False


GUARD = {
    "verdict": "rerolled",
    "clean": True,
    "parts": 1,
    "band": {
        "max_chars_per_sec": 20.0,
        "min_chars_per_sec": 14.5,
        "reference": 17.03,
        "observed": 4,
        "warm": False,
    },
    "takes": [
        {
            "index": 41,
            "depth": 0,
            "side": "short",
            "chars": 34,
            "seconds": 1.2,
            "chars_per_second": 28.33,
            "max_chars_per_sec": 20.0,
            "min_chars_per_sec": 14.5,
            "hole_seconds": 0.0,
            "max_hole_seconds": 5.0,
            "pace": 17.03,
            "pace_source": "recorded",
            "action": "short",
            "rung": "reroll",
        }
    ],
}


def test_a_guard_verdict_survives_the_round_trip_byte_for_byte(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": GUARD}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["guard"] == GUARD


def test_a_row_with_no_guard_publishes_null_and_not_a_missing_key(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": GUARD}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[41]["guard"] == GUARD
    for index in (42, 43):
        assert "guard" in chunks[index], chunks[index]
        assert chunks[index]["guard"] is None


def test_crucible_reads_nothing_inside_the_guard(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    future = {
        "verdict": "a-rung-invented-next-year",
        "clean": False,
        "parts": 3,
        "band": None,
        "takes": [{"whatever": ["the", "ladder", "wanted"]}],
        "a_key_this_server_has_never_seen": {"nested": 1},
    }
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"42": future}))
    chunks = {row["index"]: row for row in events_of(rendered(), "chunk")}
    assert chunks[42]["guard"] == future


def test_a_guard_that_is_not_an_object_fails_its_row_and_not_the_batch(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_GUARD", json.dumps({"41": "clean"}))
    events = rendered()
    assert terminal(events)["event"] == "done"
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    assert names == ["42.flac", "43.flac"]
    failed = terminal(events)["data"]["failed"]
    assert [row["index"] for row in failed] == [41]
    assert "guard='clean', which is not an object" in failed[0]["message"]
    assert sorted(row["index"] for row in events_of(events, "chunk")) == [42, 43]


def test_a_guard_reaches_the_event_without_being_rebuilt() -> None:
    from crucible.jobs.tts.render import _guard_of

    assert _guard_of({"i": 41}) is None
    assert _guard_of({"i": 41, "guard": None}) is None
    verdict = {"verdict": "clean", "takes": []}
    assert _guard_of({"i": 41, "guard": verdict}) is verdict


def test_a_failed_chunk_is_reported_and_its_neighbours_still_land(
    rendered: Callable[..., list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_FAIL_ROW", "42")
    events = rendered()
    assert terminal(events)["event"] == "done"
    names = sorted(event["name"] for event in events_of(events, "artifact"))
    assert names == ["41.flac", "43.flac"]
    failed = terminal(events)["data"]["failed"]
    assert [row["index"] for row in failed] == [42]
    assert "told to fail row 42" in failed[0]["message"]
    assert any(
        "chunk 42 failed" in row["message"] for row in events_of(events, "progress")
    )
    assert sorted(row["index"] for row in events_of(events, "chunk")) == [41, 43]


@pytest.fixture
def batch_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], list[dict[str, Any]]]:
    path = tmp_path / "batch.jsonl"
    fake_narrator_engine.steer(monkeypatch, batch_log=str(path))

    def read() -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return read


@pytest.fixture
def sampling_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[], list[dict[str, Any]]]:
    path = tmp_path / "sampling.jsonl"
    fake_narrator_engine.steer(monkeypatch, sampling_log=str(path))

    def read() -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return read


def test_take_zero_sends_no_sampling_key_at_all(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered(take=0)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["sampling"] is None for row in rows), rows
    assert all(row["take"] == 0 for row in rows), rows


def test_take_one_sends_that_rungs_numbers_on_every_item(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered(take=1)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["sampling"] == {"temperature": 0.7} for row in rows), rows
    assert all(row["take"] == 1 for row in rows), rows
    assert {row["take"] for row in events_of(events, "chunk")} == {1}


def test_a_rung_narrator_cannot_honour_fails_that_row_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_narrator_engine.steer(monkeypatch, sampling_levers="topP")
    events = rendered(take=1)
    done = terminal(events)["data"]
    assert done["rendered"] == 0
    assert sorted(row["index"] for row in done["failed"]) == [41, 42, 43]
    for row in done["failed"]:
        assert row["message"].startswith("sampling_not_supported:")
        assert "temperature" in row["message"]


def test_a_narrator_without_the_channel_refuses_the_rung_instead_of_rendering_take_zero(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_narrator_engine.steer(monkeypatch, no_item_take=1)
    events = rendered(take=1)

    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "sampling_not_wired"
    assert "{'temperature': 0.7}" in error["message"]
    assert "did not announce `itemTake`" in error["message"]
    assert sampling_log() == []


def test_take_zero_still_renders_on_a_narrator_without_the_channel(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_narrator_engine.steer(monkeypatch, no_item_take=1)
    events = rendered(take=0)

    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert all(row["sampling"] is None for row in sampling_log())


def test_a_zeroshot_voice_loads_with_its_clip_and_the_wav_is_on_disk(
    tts_client: TestClient,
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    weights = fake_weights("zeroshot")
    events = run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(8.4), "transcript": "He had been walking.",
            "name": "the stranger",
        }},
    )
    done = terminal(events)
    assert done["event"] == "done", done
    assert done["data"]["resident"] == "zeroshot"
    reference = done["data"]["reference"]
    assert reference["name"] == "the stranger"
    assert reference["seconds"] == pytest.approx(8.4)
    assert reference["sha256"] == hashlib.sha256(wav_bytes(8.4)).hexdigest()

    document = json.loads(
        (home / "narrator-higgs-voices.json").read_text(encoding="utf-8")
    )
    entry = document["zeroshot"]
    assert entry["kind"] == "clips"
    assert entry["checkpointDir"] == str(weights)
    clip = entry["clips"][0]
    assert clip["transcript"] == "He had been walking."
    assert Path(clip["path"]).read_bytes() == wav_bytes(8.4)


def test_the_resident_report_says_which_clip_is_loaded(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(4.0), "transcript": "Rain.", "name": "rain-01",
        }},
    )
    resident = tts_client.get("/v1/activity", headers=auth).json()["resident"]
    assert resident["kind"] == "tts"
    assert resident["id"] == "zeroshot"
    assert resident["reference"]["name"] == "rain-01"
    assert resident["reference"]["seconds"] == pytest.approx(4.0)
    assert resident["reference"]["sha256"] == hashlib.sha256(
        wav_bytes(4.0)
    ).hexdigest()


def test_a_resident_checkpoint_voice_reports_a_null_reference(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    resident = tts_client.get("/v1/activity", headers=auth).json()["resident"]
    assert resident["id"] == VOICE
    assert resident["reference"] is None


def test_a_resident_zeroshot_voice_renders_like_any_other(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {
            "data": wav_base64(4.0), "transcript": "Rain.",
        }},
    )
    response = submit(
        tts_client, auth, type="tts", model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 202, response.json()
    with tts_client.stream(
        "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)


def wsl2_card_holding_our_own_engine(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    looks: list[int] = []

    def compute_apps() -> list[Any]:
        looks.append(1)
        return []

    monkeypatch.setattr(accelerator, "probe_compute_apps", compute_apps)
    monkeypatch.setattr(
        accelerator, "probe_vram", lambda: (24 * GIB - 18_100 * 1024 ** 2, 24 * GIB)
    )
    return looks


def test_a_render_on_the_resident_voice_never_asks_the_card_for_room(
    tts_client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights("zeroshot")
    run_job(
        tts_client, auth, type="load-voice", model="zeroshot",
        params={"reference": {"data": wav_base64(4.0), "transcript": "Rain."}},
    )
    looks = wsl2_card_holding_our_own_engine(monkeypatch)

    response = submit(
        tts_client, auth, type="tts", model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 202, response.json()
    with tts_client.stream(
        "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert looks == []


def test_a_render_of_a_voice_that_is_not_resident_still_asks(
    tts_client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    wsl2_card_holding_our_own_engine(monkeypatch)

    response = submit(
        tts_client, auth, type="tts", model=VOICE,
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 409, response.json()
    assert response.json()["error"]["code"] == "accelerator_busy"


def test_a_render_with_no_flag_asks_for_the_bare_arm_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered()
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    batches = batch_log()
    assert len(batches) == 1, batches
    assert batches[0]["retake"] is False
    assert batches[0]["band"] is None
    assert all(row["guard"] is None for row in events_of(events, "chunk"))


def test_retake_true_with_a_band_sends_both_in_narrators_spelling(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered(
        retake=True,
        band={
            "pace_chars_per_sec": 15.91,
            "max_chars_per_sec": 20.68,
            "min_chars_per_sec": 12.24,
        },
    )
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    batches = batch_log()
    assert len(batches) == 1, batches
    assert batches[0]["retake"] is True
    assert batches[0]["band"] == {
        "paceCharsPerSec": 15.91,
        "maxCharsPerSec": 20.68,
        "minCharsPerSec": 12.24,
    }


def test_a_band_on_a_bare_render_travels_and_is_not_acted_on(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered(
        band={
            "pace_chars_per_sec": 15.91,
            "max_chars_per_sec": 20.68,
            "min_chars_per_sec": 12.24,
        },
    )
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert batch_log()[0]["retake"] is False
    assert batch_log()[0]["band"]["paceCharsPerSec"] == 15.91


def test_a_job_that_states_no_width_sends_none_and_the_engine_keeps_its_own(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered()
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert "width" not in batch_log()[0]["keys"], batch_log()[0]["keys"]
    assert terminal(events)["data"]["width"] is None


def test_a_narrower_width_is_forwarded_and_nothing_is_restarted(
    rendered: Callable[..., list[dict[str, Any]]],
    batch_log: Callable[[], list[dict[str, Any]]],
    narrator: list[Any],
) -> None:
    events = rendered(width=4)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    assert batch_log()[0]["width"] == 4
    assert terminal(events)["data"]["width"] == 4
    assert len(narrator) == 1, "narrowing a job restarted the engine"


def test_on_the_mlx_arm_a_stated_width_travels_and_this_door_refuses_none(
    home: Path,
) -> None:
    configure_box(home)
    manifest = load_voice(VOICE)
    wide = render_jobs.TtsParams(language="en", take=0, chunks=CHUNKS, width=32)

    mlx = manifest.spec("mlx-darwin")
    assert jobenv.tts_env(manifest.narrator_engine, mlx.backend).serving_stack is None
    assert render_jobs._require_width(manifest, mlx, wide) == 32

    served = manifest.spec(FAKE_BACKEND.kind)
    assert (
        jobenv.tts_env(manifest.narrator_engine, served.backend).serving_stack
        is not None
    )
    with pytest.raises(ApiError) as refusal:
        render_jobs._require_width(manifest, served, wide)
    assert refusal.value.code == "width_over_serving"


def test_the_result_names_the_full_sampling_and_the_weights_that_ran(
    rendered: Callable[..., list[dict[str, Any]]],
) -> None:
    take_zero = terminal(rendered(take=0))["data"]
    assert take_zero["sampling"] == {
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    }
    take_one = terminal(rendered(take=1))["data"]
    assert take_one["sampling"] == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 50,
    }
    assert take_one["voice"]["id"] == VOICE
    assert take_one["voice"]["identity_basis"] == "verified"
    assert len(take_one["voice"]["identity"]) == 40


def _refuse(
    client: TestClient, auth: dict[str, str], **params: Any
) -> dict[str, Any]:
    body = {"language": "en", "take": 0, "chunks": CHUNKS}
    body.update(params)
    response = submit(client, auth, type="tts", model=VOICE, params=body)
    assert response.status_code >= 400, response.json()
    return response.json()["error"]


def test_a_take_past_the_end_of_the_ladder_renders_in_its_own_seed_lane(
    rendered: Callable[..., list[dict[str, Any]]],
    sampling_log: Callable[[], list[dict[str, Any]]],
) -> None:
    events = rendered(take=4)
    assert terminal(events)["data"]["rendered"] == len(CHUNKS)
    rows = sampling_log()
    assert sorted(row["i"] for row in rows) == [41, 42, 43]
    assert all(row["take"] == 4 for row in rows), rows
    assert all(row["sampling"] is None for row in rows), rows
    assert {row["take"] for row in events_of(events, "chunk")} == {4}


def test_a_chunk_over_the_cap_renders_instead_of_being_refused(
    rendered: Callable[..., list[dict[str, Any]]],
) -> None:
    events = rendered(chunks=[{"index": 0, "text": "x" * 900}])
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["rendered"] == 1
    assert terminal(events)["data"]["failed"] == []
    row = events_of(events, "chunk")[0]
    assert row["chars"] == 900


def test_retake_true_with_no_band_is_refused_by_name(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, retake=True)
    assert error["code"] == "retake_without_band"
    assert "states no band" in error["message"]
    assert "pace_chars_per_sec" in error["message"]


@pytest.mark.parametrize(
    "band, names",
    [
        ({"pace_chars_per_sec": 15.9, "max_chars_per_sec": 20.7}, "min_chars_per_sec"),
        (
            {
                "pace_chars_per_sec": 15.9,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
                "safe_max_chars": 800,
            },
            "safe_max_chars",
        ),
        (
            {
                "pace_chars_per_sec": "fast",
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
            },
            "not a rate",
        ),
        (
            {
                "pace_chars_per_sec": 15.9,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 0,
            },
            "positive",
        ),
        (
            {
                "pace_chars_per_sec": 25.0,
                "max_chars_per_sec": 20.7,
                "min_chars_per_sec": 12.2,
            },
            "out of order",
        ),
    ],
)
def test_every_way_a_band_can_be_wrong_is_one_refusal(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    band: dict[str, Any],
    names: str,
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, band=band)
    assert error["code"] == "band_malformed", error
    assert names in error["message"], error


def test_a_width_above_the_voices_serving_width_is_refused_never_clamped(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, width=32)
    assert error["code"] == "width_over_serving"
    assert error["details"] == {"width": 32, "max_num_seqs": 16}
    assert "Never clamped" in error["message"]


def test_a_zeroshot_voice_that_is_not_resident_is_refused_because_this_door_cannot_load_it(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights("zeroshot")
    response = submit(
        tts_client,
        auth,
        type="tts",
        model="zeroshot",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 400, response.json()
    error = response.json()["error"]
    assert error["code"] == "voice_kind_unsupported"
    assert "is not resident" in error["message"]
    assert "params.reference" in error["message"]
    assert error["details"]["resident"] is False


def test_a_blank_chunk_is_refused_before_it_ends_the_batch(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, chunks=[{"index": 0, "text": "   "}])
    assert error["code"] == "invalid_params"
    assert "must not be blank" in error["message"]


def test_two_chunks_with_one_index_are_refused(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    error = _refuse(
        tts_client,
        auth,
        chunks=[{"index": 7, "text": "one"}, {"index": 7, "text": "two"}],
    )
    assert error["code"] == "invalid_params"
    assert "[7] appear more than once" in error["message"]


def test_an_unknown_param_is_refused_rather_than_ignored(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(VOICE)
    error = _refuse(tts_client, auth, temperature=0.9)
    assert error["code"] == "invalid_params"


def test_the_wire_word_for_the_voice_is_model(
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    response = submit(
        tts_client,
        auth,
        type="tts",
        model="qwen3.5-9b",
        params={"language": "en", "take": 0, "chunks": CHUNKS},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


def test_a_render_loads_its_own_voice_and_says_it_is_warming(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    with holding_the_card(tts_client):
        events = rendered()
        health = tts_client.get("/v1/health", headers=auth).json()
    warmings = [row["message"] for row in events_of(events, "warming")]
    assert any("checking the accelerator for deathstalker" in m for m in warmings)
    assert any("starting narrator (higgs-v3)" in m for m in warmings)
    assert any("narrator loaded deathstalker" in m for m in warmings)
    assert health["resident_models"] == [VOICE]
    assert health["resident_kind"] == KIND_TTS


def test_a_render_writes_the_voices_document_narrator_reads(
    rendered: Callable[..., list[dict[str, Any]]],
    home: Path,
    narrator: list[Any],
) -> None:
    rendered()
    document = home / "narrator-higgs-voices.json"
    assert document.is_file()
    written = json.loads(document.read_text(encoding="utf-8"))
    assert list(written) == [VOICE]
    entry = written[VOICE]
    assert entry["kind"] == "checkpoint"
    assert entry["checkpointDir"] == str(home / "voices" / VOICE / "cuda-linux")
    assert entry["maxChars"] == 800
    assert entry["safeMinChars"] == 400
    assert entry["safeMaxChars"] == 700
    assert entry["sampling"] == {"temperature": 0.8, "topP": 0.95, "topK": 50}
    assert entry["paceCharsPerSec"] == 16.14
    assert narrator[0].environment()["NARRATOR_HIGGS_VOICES"] == str(document)


def test_a_second_render_does_not_restart_narrator(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    narrator: list[Any],
) -> None:
    with holding_the_card(tts_client):
        rendered()
        assert len(narrator) == 1
        second = rendered()
        assert terminal(second)["event"] == "done"
        assert len(narrator) == 1
        assert not any(
            "starting narrator" in row["message"]
            for row in events_of(second, "warming")
        )


def test_an_unheld_render_clears_the_card_and_the_next_one_reloads(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
    narrator: list[Any],
) -> None:
    events = rendered()
    note = [row for row in events_of(events, "note")]
    assert note, "the unload must be said on the job that triggered it"
    assert note[-1]["unloaded"] == VOICE
    assert "nothing holds it" in note[-1]["message"]
    kinds = [row["event"] for row in events]
    assert kinds.index("note") < kinds.index("done")
    assert tts_client.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert len(narrator) == 1

    rendered()
    assert len(narrator) == 2


def test_a_voice_lease_turns_a_book_rendered_chapter_by_chapter_into_one_load(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    narrator: list[Any],
) -> None:
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert len(narrator) == 1

    opened = tts_client.post(
        f"/v1/models/{VOICE}/lease",
        headers=auth,
        json={"act": "tts", "ttl_seconds": 60},
    )
    assert opened.status_code == 201, opened.text
    lease = opened.json()
    assert lease["subject"] == VOICE
    assert lease["kind"] == KIND_TTS
    assert tts_client.get("/v1/activity", headers=auth).json()["lease"]["kind"] == (
        KIND_TTS
    )

    for _ in range(3):
        assert terminal(rendered())["event"] == "done"
        assert len(narrator) == 1, "a leased voice is rendered against, not reloaded"
        assert (
            tts_client.get("/v1/health", headers=auth).json()["resident_kind"]
            == KIND_TTS
        )

    released = tts_client.delete(f"/v1/leases/{lease['lease_id']}", headers=auth)
    assert released.status_code == 204
    assert tts_client.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert len(narrator) == 1


def test_a_voice_lease_refuses_the_jobs_that_would_evict_it(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(VOICE)
    run_job(tts_client, auth, type="load-voice", model=VOICE)
    opened = tts_client.post(
        f"/v1/models/{VOICE}/lease",
        headers=auth,
        json={"act": "tts", "ttl_seconds": 60},
    )
    assert opened.status_code == 201, opened.text

    for body in (
        {"type": "load-voice", "model": "mistborn"},
        {"type": "load-voice", "model": VOICE},
        {"type": "unload-voice", "model": VOICE},
        {
            "type": "tts",
            "model": "mistborn",
            "params": {"language": "en", "take": 0, "chunks": CHUNKS},
        },
    ):
        response = tts_client.post("/v1/jobs", headers=auth, json=body)
        assert response.status_code == 409, (body, response.text)
        error = response.json()["error"]
        assert error["code"] == "leased", body
        assert error["details"]["kind"] == KIND_TTS
        assert "the resident voice" in error["message"]

    echoed = tts_client.post(
        "/v1/jobs",
        headers=auth,
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"x.bin": {"inline_base64": "YQ=="}},
        },
    )
    assert echoed.json().get("error", {}).get("code") != "leased"


def test_the_load_is_part_of_the_load(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(VOICE)
    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "done", terminal(events)
    warmings = [row["message"] for row in events_of(events, "warming")]
    assert any("narrator loaded deathstalker" in m for m in warmings)
    assert any("24000 Hz" in m for m in warmings)


def test_a_worker_that_dies_during_a_load_leaves_nothing_resident(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_EXIT_CODE", "3")
    fake_weights(VOICE)
    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "engine_failed"
    assert "exited 3 before it was ready" in terminal(events)["data"]["error"]["message"]
    health = tts_client.get("/v1/health", headers=auth).json()
    assert health["resident_models"] == []
    assert health["resident_kind"] is None


def test_a_sample_rate_the_engine_disagrees_with_is_refused_not_resampled(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    pulled = fake_weights(VOICE) / REPO_MANIFEST_NAME
    text = pulled.read_text(encoding="utf-8")
    assert "sample_rate     = 24000" in text
    pulled.write_text(
        text.replace("sample_rate     = 24000", "sample_rate     = 48000"),
        encoding="utf-8",
    )

    events = run_job(tts_client, auth, type="load-voice", model=VOICE)
    assert terminal(events)["event"] == "failed"
    message = terminal(events)["data"]["error"]["message"]
    assert "renders deathstalker at 24000 Hz" in message
    assert "declares 48000" in message
    assert "refuses rather than resampling" in message


def test_one_holder_still_serves_both_kinds(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    with holding_the_card(tts_client):
        rendered()
        assert tts_client.app.state.residency.resident_kind == KIND_TTS
        assert tts_client.app.state.residency.resident_model is None
        assert tts_client.app.state.residency.voice_engine is not None


def test_the_provenance_sidecar_names_the_merge_that_rendered_it(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    rendered()
    sidecar = json.loads(
        tts_client.get(
            f"/v1/jobs/{rendered.job_id}/artifacts/41.flac.provenance.json",
            headers=auth,
        ).content
    )
    assert sidecar["model"]["id"] == VOICE
    assert sidecar["model"]["fingerprint"].startswith(f"{VOICE}@")
    assert len(sidecar["model"]["revision"]) == 40
    assert [chunk["index"] for chunk in sidecar["params"]["chunks"]] == [41]
    assert sidecar["job_id"] == rendered.job_id
    assert len(sidecar["params_sha256"]) == 64


def test_the_residency_is_torn_down_when_the_server_stops(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    narrator: list[Any],
) -> None:
    with holding_the_card(tts_client):
        rendered()
        engine = narrator[0]
        assert engine.pids
    tts_client.app.state.residency.shutdown()
    assert engine.pids == frozenset()


def _wait_for(
    tts_client: TestClient,
    auth: dict[str, str],
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    what: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = tts_client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if predicate(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never {what}; last state {state}")


def _start_a_slow_render(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> str:
    fake_weights(VOICE)
    chunks = [
        {"index": index, "text": f"Sentence number {index} of the chapter."}
        for index in range(12)
    ]
    response = submit(
        tts_client,
        auth,
        type="tts",
        model=VOICE,
        params={"language": "en", "take": 0, "chunks": chunks},
    )
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    _wait_for(
        tts_client, auth, job_id,
        lambda state: bool(state["artifacts"]),
        "rendered a first chunk",
    )
    return job_id


def test_a_cancelled_render_stops_within_one_chunk(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_narrator_engine.steer(monkeypatch, row_delay_ms=120)
    job_id = _start_a_slow_render(tts_client, auth, fake_weights)

    cancelled = tts_client.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert cancelled.status_code == 200, cancelled.json()
    assert cancelled.json()["status"] == "cancelling"

    state = _wait_for(
        tts_client, auth, job_id,
        lambda state: state["status"] in ("done", "failed", "cancelled"),
        "reached a terminal state",
    )
    assert state["status"] == "cancelled", state.get("error") or state
    assert 0 < len(state["artifacts"]) < 12, state["artifacts"]

    residency = tts_client.app.state.residency
    assert residency.claimed_by is None, "the claim outlived the job"
    assert residency.resident is None, "the voice is still on the card"

    name = sorted(n for n in state["artifacts"] if n.endswith(".flac"))[0]
    fetched = tts_client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert fetched.status_code == 200
    assert fetched.content[:4] == b"fLaC", name


def test_a_narrator_that_ignores_the_cancel_is_taken_off_the_card(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("crucible.engines.narrator.CANCEL_GRACE_SECONDS", 1.0)
    fake_narrator_engine.steer(monkeypatch, row_delay_ms=120, ignore_cancel=1)
    job_id = _start_a_slow_render(tts_client, auth, fake_weights)

    assert tts_client.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200

    state = _wait_for(
        tts_client, auth, job_id,
        lambda state: state["status"] in ("done", "failed", "cancelled"),
        "reached a terminal state",
    )
    assert state["status"] == "cancelled", state.get("error") or state

    residency = tts_client.app.state.residency
    assert residency.claimed_by is None
    assert residency.resident is None, (
        "an engine that would not stop must not be left resident: nothing else "
        "is ever going to take it off the card"
    )

    events = parse_sse(
        line
        for line in tts_client.get(
            f"/v1/jobs/{job_id}/events", headers=auth
        ).text.splitlines()
    )
    notes = [e["data"]["message"] for e in events if e["event"] == "note"]
    assert any("was sent a cancel" in note for note in notes), events


def test_cancelling_a_finished_render_is_refused_by_name(
    rendered: Callable[..., list[dict[str, Any]]],
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    assert terminal(rendered())["event"] == "done"
    response = tts_client.delete(f"/v1/jobs/{rendered.job_id}", headers=auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_cancellable"
    assert "already done" in response.json()["error"]["message"]


def test_cancelling_an_unknown_job_is_refused_by_name(
    tts_client: TestClient,
    auth: dict[str, str],
) -> None:
    response = tts_client.delete("/v1/jobs/notajob", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_job"


def test_nothing_in_this_module_touched_a_real_engine_module() -> None:
    assert residency_module.build_voice_engine.__module__ == (
        "tests.fake_narrator_engine"
    )
