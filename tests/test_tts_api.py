from __future__ import annotations

import base64
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, residency as residency_module, tasks, weights
from crucible.accelerator import GIB, ComputeApp
from crucible.jobs import ALL_JOB_TYPES
from crucible.residency import KIND_LLM, KIND_TTS, ResidentVoice
from crucible.settle import SETTLEMENT_HOLDER
from crucible.voices import NARRATOR_ENGINE_SAMPLING, load_voice

from .conftest import (
    FAKE_BACKEND,
    a_clearance_to_hold,
    parse_sse,
    wav_base64,
    write_env_stamp,
)
from .fake_engine import FakeEngine

VOICE = "deathstalker"
OTHER_VOICE = "thirdreich"

RECIPE_PINS = {"narrator": "0.1.0", "torch": "2.13.0"}


@pytest.fixture
def tts_recipes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "recipes"
    (root / "tts").mkdir(parents=True)
    for engine in sorted(NARRATOR_ENGINE_SAMPLING):
        (root / "tts" / f"{engine}-{FAKE_BACKEND.kind}.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in RECIPE_PINS.items()),
            encoding="utf-8",
        )
    monkeypatch.setenv("CRUCIBLE_RECIPES_DIR", str(root))
    return root


@pytest.fixture
def fake_env(
    home: Path, tts_recipes: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    spec = jobenv.tts_env("higgs-v3", FAKE_BACKEND.kind)
    directory = jobenv.env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    write_env_stamp(home, jobenv.tts_env("higgs-v3", FAKE_BACKEND.kind), FAKE_BACKEND.kind)
    monkeypatch.setattr(
        jobenv, "installed_packages", lambda _home, _spec: dict(RECIPE_PINS)
    )
    return directory


@pytest.fixture
def fake_weights(home: Path) -> Callable[[str], Path]:

    def stamp(voice_id: str) -> Path:
        spec = load_voice(voice_id).spec(FAKE_BACKEND.kind)
        directory = home / "voices" / voice_id / FAKE_BACKEND.kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "family": "voices",
                    "id": voice_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 8_500_000_000,
                    "seconds": 400.0,
                    "pulled": "2026-09-13T02:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        return directory

    return stamp


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (23 * GIB, 24 * GIB))


@pytest.fixture
def tts_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_tts=True, enable_echo=False) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    return client.post("/v1/jobs", headers=auth, json=body)


def run_job(client: TestClient, auth: dict[str, str], **body: Any):
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def rows(client: TestClient, auth: dict[str, str]) -> dict[str, dict[str, Any]]:
    response = client.get("/v1/voices", headers=auth)
    assert response.status_code == 200, response.json()
    return {row["id"]: row for row in response.json()}


def test_voices_is_refused_when_the_type_is_off(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/voices", headers=auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"
    assert "enable_tts" in response.json()["error"]["message"]


def test_a_voice_with_everything_in_place_is_loadable(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(VOICE)
    row = rows(tts_client, auth)[VOICE]
    assert row["loadable"] is True
    assert row["reason"] is None
    assert row["installed"] is True
    assert row["resident"] is False
    assert row["backend_supported"] is True
    assert row["kind"] == "checkpoint"
    assert row["language"] == "en"
    assert row["sample_rate"] == 24000
    assert row["max_chars"] == 800
    assert row["takes"] == 2
    assert row["needs_reference"] is False
    assert row["estimate_basis"] == "declared"
    assert row["fingerprint"] == f"{VOICE}@{row['revision']}"
    assert row["pace"]["pace_chars_per_sec"] == 15.91
    assert row["pace"]["safe_min_chars"] == 500


def test_a_row_never_carries_the_sampling(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(VOICE)
    for row in rows(tts_client, auth).values():
        assert "sampling" not in row
        assert "sampling_reason" not in row
        assert "estimate_note" not in row
        assert "narrator_engine" in row


def test_an_unpulled_voice_says_which_command_pulls_it(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    row = rows(tts_client, auth)[VOICE]
    assert row["loadable"] is False
    assert row["installed"] is False
    assert "crucible voices pull deathstalker" in row["reason"]


def test_a_missing_env_is_the_reason_before_the_weights_are(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tts_recipes: Path,
) -> None:
    with make_client(enable_tts=True) as client:
        row = rows(client, auth)[VOICE]
        assert row["loadable"] is False
        assert "the tts env for higgs-v3 is not ready" in row["reason"]
        assert "crucible install tts" in row["reason"]


def test_a_voice_too_big_for_the_card_is_refused_for_being_too_big(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    tiny = replace(
        FAKE_BACKEND,
        gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA T4", vram_bytes=16 * GIB),
    )
    with make_client(enable_tts=True, backend=tiny) as client:
        row = rows(client, auth)[VOICE]
        assert row["loadable"] is False
        assert "NVIDIA T4 has 16.0 GiB in total" in row["reason"]


def test_a_backend_this_host_is_not_gets_nulls_and_not_zeroes(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "voices"
    directory.mkdir()
    (directory / "cuda-only.toml").write_text(
        (Path(__file__).resolve().parent.parent / "crucible" / "voices" / "deathstalker.toml")
        .read_text(encoding="utf-8")
        .replace('id = "deathstalker"', 'id = "cuda-only"')
        .split("[voice.backends.mlx-darwin]")[0],
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(directory))
    from .conftest import FAKE_MAC_BACKEND

    with make_client(enable_tts=True, backend=FAKE_MAC_BACKEND) as client:
        row = rows(client, auth)["cuda-only"]
        assert row["backend_supported"] is False
        assert row["revision"] is None
        assert row["fingerprint"] is None
        assert row["memory_bytes_estimate"] is None
        assert row["estimate_basis"] is None
        assert row["max_chars"] is None
        assert "has no mlx-darwin block" in row["reason"]


def test_info_carries_the_voice_rows_verbatim(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(VOICE)
    info = tts_client.get("/v1/info", headers=auth).json()
    capability = [c for c in info["capabilities"] if c["job_type"] == "tts"]
    assert len(capability) == 1
    assert capability[0]["models"] == tts_client.get("/v1/voices", headers=auth).json()


def test_the_voice_types_describe_installed_as_the_voices_route_does(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    store = tts_client.app.state.store
    names = sorted(name for name, capability in ALL_JOB_TYPES.items() if capability == "tts")
    assert names == ["load-voice", "tts", "unload-voice"]
    for name in names:
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[VOICE]["installed"] is False
        assert rows["sigma"]["installed"] is False
    fake_weights(VOICE)
    served = {row["id"]: row for row in tts_client.get("/v1/voices", headers=auth).json()}
    for name in names:
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[VOICE]["installed"] is True
        assert rows[VOICE]["resident"] is False
        assert rows["sigma"]["installed"] is False
        for voice_id, row in rows.items():
            assert row["installed"] is served[voice_id]["installed"], voice_id
            assert row["resident"] is served[voice_id]["resident"], voice_id


def test_the_voices_are_listed_in_id_order(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    listed = [row["id"] for row in tts_client.get("/v1/voices", headers=auth).json()]
    assert listed == sorted(listed)


def test_an_unknown_voice_is_refused_before_the_job_exists(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(tts_client, auth, type="load-voice", model="gandalf")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert "gandalf" in error["message"]


def test_a_voice_with_no_block_for_this_backend_is_refused(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "voices"
    directory.mkdir()
    (directory / "cuda-only.toml").write_text(
        (Path(__file__).resolve().parent.parent / "crucible" / "voices" / "deathstalker.toml")
        .read_text(encoding="utf-8")
        .replace('id = "deathstalker"', 'id = "cuda-only"')
        .split("[voice.backends.mlx-darwin]")[0],
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_VOICES_DIR", str(directory))
    from .conftest import FAKE_MAC_BACKEND

    with make_client(enable_tts=True, backend=FAKE_MAC_BACKEND) as client:
        response = submit(client, auth, type="load-voice", model="cuda-only")
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "backend_unsupported"
        assert error["details"]["declared"] == ["cuda-linux"]


def test_a_missing_env_is_installed_on_submit_for_the_voice_s_engine(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tts_recipes: Path,
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_tts=True) as client:
        response = submit(client, auth, type="load-voice", model=VOICE)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "installing"
        assert "installing the tts environment for higgs-v3" in error["message"]
        assert error["details"]["steps"][0] == "install tts (higgs-v3)"
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [
            {"type": "tts", "narrator_engine": "higgs-v3"}
        ]


def test_unpulled_weights_are_refused_by_name(
    tts_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    response = submit(tts_client, auth, type="load-voice", model=VOICE)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "installing"
    assert "pulling the voice 'deathstalker'" in error["message"]


def test_a_busy_accelerator_is_refused_by_name(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(VOICE)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=4321, name="python3", used_bytes=9 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (23 * GIB, 24 * GIB))
    response = submit(tts_client, auth, type="load-voice", model=VOICE)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "accelerator_busy"


def test_a_full_card_is_refused_by_name(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(VOICE)
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (10 * GIB, 12 * GIB))
    response = submit(tts_client, auth, type="load-voice", model=VOICE)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "insufficient_memory"


def test_unknown_params_are_refused(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(VOICE)
    response = submit(
        tts_client, auth, type="load-voice", model=VOICE,
        params={"temperature": 0.7},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


ZEROSHOT = "zeroshot"


def load_zeroshot(
    client: TestClient, auth: dict[str, str], **reference: Any
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if reference:
        params["reference"] = reference
    return submit(client, auth, type="load-voice", model=ZEROSHOT, params=params)


def test_the_zeroshot_row_says_it_needs_a_reference(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(ZEROSHOT)
    listed = rows(tts_client, auth)
    assert listed[ZEROSHOT]["needs_reference"] is True
    assert listed[ZEROSHOT]["kind"] == "zeroshot"
    for voice_id, row in listed.items():
        assert row["needs_reference"] is (row["kind"] == "zeroshot"), voice_id


def test_a_zeroshot_load_with_no_reference_is_refused_by_name(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(ZEROSHOT)
    response = load_zeroshot(tts_client, auth)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "reference_required"
    assert "params.reference" in error["message"]
    assert error["details"]["kind"] == "zeroshot"


def test_a_reference_on_a_checkpoint_voice_is_refused_by_name(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(VOICE)
    response = submit(
        tts_client, auth, type="load-voice", model=VOICE,
        params={"reference": {"data": wav_base64(3.0), "transcript": "Rain."}},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "reference_not_allowed"
    assert error["details"]["kind"] == "checkpoint"


@pytest.mark.parametrize(
    "reference, expected",
    [
        pytest.param(
            {"data": "not base64 at all!!", "transcript": "Rain."},
            "not base64",
            id="not-base64",
        ),
        pytest.param(
            {"data": base64.b64encode(b"nothing like a wav").decode("ascii"),
             "transcript": "Rain."},
            "not a readable WAV",
            id="not-a-wav",
        ),
        pytest.param(
            {"data": wav_base64(31.0), "transcript": "Rain."},
            "caps a reference at 30 s",
            id="over-the-budget",
        ),
        pytest.param(
            {"data": wav_base64(3.0), "transcript": "   "},
            "BOOK-EXACT text",
            id="no-transcript",
        ),
    ],
)
def test_a_reference_that_is_not_one_is_refused_by_name(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path],
    idle_card: None, reference: dict[str, Any], expected: str,
) -> None:
    fake_weights(ZEROSHOT)
    response = load_zeroshot(tts_client, auth, **reference)
    assert response.status_code == 400, response.json()
    error = response.json()["error"]
    assert error["code"] == "reference_malformed"
    assert expected in error["message"]


def test_a_reference_carrying_a_key_this_door_does_not_know_is_refused(
    tts_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(ZEROSHOT)
    response = load_zeroshot(
        tts_client, auth, data=wav_base64(3.0), transcript="Rain.", seconds=3.0
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


def test_unloading_a_voice_that_is_not_resident_is_refused(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(tts_client, auth, type="unload-voice", model=VOICE)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "voice_not_resident"
    assert "no voice is" in error["message"]
    assert error["details"] == {"requested": VOICE, "resident": None}


def test_unloading_the_voice_the_settlement_is_clearing_is_the_same_intent(
    tts_client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    residency = tts_client.app.state.residency
    engine = FakeEngine(Path("python"), tmp_path / "engine-narrator.log")
    residency._resident = resident_voice(VOICE)
    residency._engine = engine
    reached, release = a_clearance_to_hold(engine)

    settlement = tts_client.app.state.settlement
    settled: list[Any] = []
    clearing = threading.Thread(
        target=lambda: settled.append(
            settlement.settle_quietly("the render job finished")
        ),
        name="the-settlement",
        daemon=True,
    )
    clearing.start()
    assert reached.wait(timeout=10), "the settlement never reached the engine"
    assert residency.claimed_by == SETTLEMENT_HOLDER

    response = submit(tts_client, auth, type="unload-voice", model=VOICE)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]

    release.set()
    clearing.join(timeout=30)
    assert not clearing.is_alive()
    assert settled[0] is not None and settled[0].subject_id == VOICE

    with tts_client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["resident"] is None
    assert tts_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert engine.stopped is True


def test_unloading_a_voice_under_a_holder_using_the_card_is_still_engine_in_use(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    residency = tts_client.app.state.residency
    residency._resident = resident_voice(VOICE)
    residency.claim("tts stream abc123", may_mutate=False)
    try:
        response = submit(tts_client, auth, type="unload-voice", model=VOICE)
    finally:
        residency.release("tts stream abc123")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "engine_in_use"
    assert error["details"]["held_by"] == "tts stream abc123"
    assert residency.resident_voice is not None


def resident_voice(voice_id: str) -> ResidentVoice:
    manifest = load_voice(voice_id)
    spec = manifest.spec(FAKE_BACKEND.kind)
    return ResidentVoice(
        voice_id=manifest.id,
        backend=spec.backend,
        narrator_engine=manifest.narrator_engine,
        revision=spec.revision,
        fingerprint=manifest.fingerprint(spec.backend),
        sample_rate=manifest.sample_rate,
        max_chars=spec.max_chars,
        memory_bytes_estimate=spec.memory_bytes_estimate,
        log_path=Path("/tmp/engine-test.log"),
        loaded_at="2026-09-13T02:00:00+00:00",
    )


def test_health_names_the_kind_that_holds_the_card(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    residency = tts_client.app.state.residency
    residency._resident = resident_voice(VOICE)
    health = tts_client.get("/v1/health", headers=auth).json()
    assert health["resident_models"] == [VOICE]
    assert health["resident_kind"] == "tts"


def test_a_resident_voice_lights_up_its_own_row_only(
    tts_client: TestClient, auth: dict[str, str]
) -> None:
    residency = tts_client.app.state.residency
    residency._resident = resident_voice(VOICE)
    listed = rows(tts_client, auth)
    assert listed[VOICE]["resident"] is True
    assert listed[OTHER_VOICE]["resident"] is False


def test_unload_voice_will_not_take_a_model_off_the_card(
    tts_client: TestClient, auth: dict[str, str], home: Path
) -> None:
    from crucible.residency import ResidentModel

    residency = tts_client.app.state.residency
    residency._resident = ResidentModel(
        model_id=VOICE,
        backend=FAKE_BACKEND.kind,
        engine="vllm",
        engine_model_name=VOICE,
        base_url="http://127.0.0.1:1",
        port=1,
        revision="0" * 40,
        max_model_len=12288,
        memory_bytes_estimate=1,
        log_path=Path("/tmp/x.log"),
        loaded_at="2026-09-13T02:00:00+00:00",
        engine_args=(),
    )
    response = submit(tts_client, auth, type="unload-voice", model=VOICE)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "voice_not_resident"
    assert "the resident model is 'deathstalker'" in error["message"]


def test_a_resident_voice_is_what_a_model_load_would_reclaim(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible.config import load_config, write_config

    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=False,
        enable_llm=True,
        enable_asr=False,
        enable_tts=True,
        enable_align=False,
        enable_rvc=False,
        desktop_allowance_bytes=3 * GIB,
        enable_denoise=False,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    holder = residency_module.Residency(load_config(home))
    assert holder.reclaimable_bytes() == 0
    holder._resident = resident_voice(VOICE)
    assert holder.resident_kind == KIND_TTS
    assert holder.reclaimable_bytes() == 19_000_000_000
    assert holder.reclaimable_bytes(excluding=VOICE) == 0
    assert holder.is_resident(KIND_TTS, VOICE) is True
    assert holder.is_resident(KIND_LLM, VOICE) is False


def test_the_openai_proxy_does_not_see_a_voice_as_a_model(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    with make_client(enable_llm=True, enable_tts=True) as client:
        client.app.state.residency._resident = resident_voice(VOICE)
        listed = client.get("/v1/openai/models", headers=auth).json()
        assert listed == {"object": "list", "data": []}
        response = client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": VOICE, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "model_not_resident"


def test_unload_model_will_not_take_a_voice_off_the_card(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    with make_client(enable_llm=True, enable_tts=True) as client:
        client.app.state.residency._resident = resident_voice(VOICE)
        response = client.post(
            "/v1/jobs", headers=auth,
            json={"type": "unload-model", "model": "qwen3.5-9b"},
        )
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "model_not_resident"
        assert "the resident voice is 'deathstalker'" in error["message"]


def test_one_holder_serves_both_job_types(
    make_client: Callable[..., TestClient], auth: dict[str, str], fake_env: Path
) -> None:
    with make_client(enable_llm=True, enable_tts=True) as client:
        store = client.app.state.store
        holder = client.app.state.residency
        for name in ("load-model", "unload-model", "load-voice", "unload-voice"):
            assert store.registry[name].residency is holder


def test_a_render_with_no_ffmpeg_is_refused_before_it_is_queued(
    tts_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible import hosttools

    fake_weights("deathstalker")
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: None)
    response = submit(
        tts_client,
        auth,
        type="tts",
        model="deathstalker",
        params={
            "language": "en",
            "take": 0,
            "chunks": [{"index": 41, "text": "He had been walking for some time."}],
        },
    )
    assert response.status_code == 409, response.json()
    assert response.json()["error"]["code"] == "ffmpeg_missing"
    assert "libsndfile" in response.json()["error"]["message"]
