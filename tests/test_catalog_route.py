from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from crucible import catalog, denoisemodels, lineup, rvcbase
from crucible.alignmodels import load_all_align_manifests
from crucible.audiomodels import load_all_audio_manifests
from crucible.videomodels import load_all_video_manifests
from crucible.imagemodels import load_all_image_manifests
from crucible.asrmodels import load_all_asr_manifests
from crucible.errors import CrucibleError
from crucible.manifests import load_all_manifests
from crucible.rvcmodels import load_all_rvc_manifests
from crucible.segmentmodels import load_all_segment_manifests
from crucible.voicecatalog import load_all_voices

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND


def fetch(client: TestClient, auth: dict[str, str]) -> list[dict[str, Any]]:
    response = client.get("/v1/catalog", headers=auth)
    assert response.status_code == 200, response.text
    return response.json()["rows"]


def by_id(rows: list[dict[str, Any]], kind: str, subject_id: str) -> dict[str, Any]:
    found = [r for r in rows if r["kind"] == kind and r["id"] == subject_id]
    assert len(found) == 1, f"{kind}/{subject_id} appears {len(found)} times"
    return found[0]


def test_every_kind_is_present(client: TestClient, auth: dict[str, str]) -> None:
    rows = fetch(client, auth)
    assert {row["kind"] for row in rows} == {
        "model",
        "voice",
        "rvc",
        "rvc-base",
        "denoise",
    }


def test_rvc_base_has_exactly_one_row_and_its_id_is_base(
    client: TestClient, auth: dict[str, str]
) -> None:
    rows = [row for row in fetch(client, auth) if row["kind"] == "rvc-base"]
    assert [row["id"] for row in rows] == [catalog.RVC_BASE_ID] == ["base"]


def test_a_row_carries_exactly_the_contract_s_fields(
    client: TestClient, auth: dict[str, str]
) -> None:
    for row in fetch(client, auth):
        assert set(row) == {
            "kind",
            "id",
            "name",
            "job_type",
            "installed",
            "installed_bytes",
            "expected_bytes",
            "shares_weights_of",
            "missing_files",
            "floors",
            "license",
            "source",
            "resident",
        }


def test_the_catalog_needs_the_token(client: TestClient) -> None:
    assert client.get("/v1/catalog").status_code == 401


def test_the_model_rows_are_the_three_manifest_directories(
    client: TestClient, auth: dict[str, str]
) -> None:
    rows = [row for row in fetch(client, auth) if row["kind"] == "model"]
    expected = {
        **{m.id: "llm" for m in load_all_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "asr" for m in load_all_asr_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "align" for m in load_all_align_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "image" for m in load_all_image_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "audio" for m in load_all_audio_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "segment" for m in load_all_segment_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
        **{m.id: "video" for m in load_all_video_manifests().values()
           if m.supports(FAKE_BACKEND.kind)},
    }
    assert {row["id"]: row["job_type"] for row in rows} == expected


def test_the_voice_rows_are_the_voice_manifests(
    client: TestClient, auth: dict[str, str]
) -> None:
    rows = [row for row in fetch(client, auth) if row["kind"] == "voice"]
    manifests = [v for v in load_all_voices().values() if v.supports(FAKE_BACKEND.kind)]
    assert [row["id"] for row in rows] == [v.id for v in manifests]
    assert [row["name"] for row in rows] == [v.display for v in manifests]
    assert {row["job_type"] for row in rows} == {"tts"}


def test_source_is_the_backend_block_s_repo(
    client: TestClient, auth: dict[str, str]
) -> None:
    row = by_id(fetch(client, auth), "model", "qwen3.5-9b")
    spec = load_all_manifests()["qwen3.5-9b"].spec(FAKE_BACKEND.kind)
    assert row["source"] == f"hf:{spec.hf_repo}"


def test_floors_come_from_the_lineup_and_nowhere_else(
    client: TestClient, auth: dict[str, str]
) -> None:
    rows = fetch(client, auth)
    declared = lineup.floors(lineup.build()[0])
    for capability_class, model_id in declared.items():
        assert capability_class in by_id(rows, "model", model_id)["floors"]
    floored = {
        row["id"] for row in rows if row["kind"] == "model" and row["floors"]
    }
    assert floored == set(declared.values())


def test_nothing_but_a_model_ever_carries_a_floor(
    client: TestClient, auth: dict[str, str]
) -> None:
    for row in fetch(client, auth):
        if row["kind"] != "model":
            assert row["floors"] == []


def test_expected_bytes_is_declared_where_a_file_is_named_and_null_otherwise(
    client: TestClient, auth: dict[str, str]
) -> None:
    rows = fetch(client, auth)
    for row in rows:
        if row["kind"] in ("model", "voice") and row["shares_weights_of"] is None:
            assert row["expected_bytes"] is None, row["id"]
        elif row["kind"] == "model":
            assert row["expected_bytes"] == 0, row["id"]
        else:
            assert isinstance(row["expected_bytes"], int) and row["expected_bytes"] > 0

    rvc_id = next(iter(load_all_rvc_manifests()))
    assert by_id(rows, "rvc", rvc_id)["expected_bytes"] == (
        load_all_rvc_manifests()[rvc_id].spec(FAKE_BACKEND.kind).archive_bytes
    )
    assert by_id(rows, "rvc-base", "base")["expected_bytes"] == (
        rvcbase.load_rvc_base().total_bytes
    )
    denoise_id = next(iter(denoisemodels.load_all_denoise_manifests()))
    assert by_id(rows, "denoise", denoise_id)["expected_bytes"] == (
        denoisemodels.load_all_denoise_manifests()[denoise_id]
        .spec(FAKE_BACKEND.kind)
        .total_bytes
    )


def test_license_is_null_everywhere_because_no_manifest_declares_one(
    client: TestClient, auth: dict[str, str]
) -> None:
    assert all(row["license"] is None for row in fetch(client, auth))


def test_a_subject_with_no_block_for_this_backend_is_absent(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(backend=FAKE_BACKEND) as client:
        on_pc = {(r["kind"], r["id"]) for r in fetch(client, auth)}
    with make_client(backend=FAKE_MAC_BACKEND) as client:
        on_mac = {(r["kind"], r["id"]) for r in fetch(client, auth)}
    for kind, subject_id in on_pc - on_mac:
        if kind == "rvc-base":
            continue
        assert not _supports(kind, subject_id, FAKE_MAC_BACKEND.kind)
    for kind, subject_id in on_mac - on_pc:
        if kind == "rvc-base":
            continue
        assert not _supports(kind, subject_id, FAKE_BACKEND.kind)


def _supports(kind: str, subject_id: str, backend_kind: str) -> bool:
    loaders = {
        "voice": load_all_voices,
        "rvc": load_all_rvc_manifests,
        "denoise": denoisemodels.load_all_denoise_manifests,
    }
    if kind == "model":
        for loader in (load_all_manifests, load_all_asr_manifests,
                       load_all_align_manifests, load_all_image_manifests,
                       load_all_audio_manifests, load_all_segment_manifests,
                       load_all_video_manifests):
            found = loader().get(subject_id)
            if found is not None:
                return found.supports(backend_kind)
        raise AssertionError(f"no manifest for model {subject_id!r}")
    return loaders[kind]()[subject_id].supports(backend_kind)


def test_installed_follows_the_stamp_weights_py_writes(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    fake_weights: Callable[[str], Path],
) -> None:
    with make_client() as client:
        before = by_id(fetch(client, auth), "model", "qwen3.5-9b")
    assert before["installed"] is False
    assert before["installed_bytes"] is None

    fake_weights("qwen3.5-9b")
    with make_client() as client:
        after = by_id(fetch(client, auth), "model", "qwen3.5-9b")
    assert after["installed"] is True
    assert after["installed_bytes"] == 19_306_310_880


def test_a_stamp_at_another_revision_does_not_read_as_installed(
    make_client: Callable[..., TestClient], auth: dict[str, str], home: Path
) -> None:
    directory = home / "models" / "qwen3.5-9b" / FAKE_BACKEND.kind
    directory.mkdir(parents=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "hf_repo": load_all_manifests()["qwen3.5-9b"]
                .spec(FAKE_BACKEND.kind)
                .hf_repo,
                "revision": "0" * 40,
                "bytes": 1,
                "pulled": "2026-01-01T00:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    with make_client() as client:
        assert by_id(fetch(client, auth), "model", "qwen3.5-9b")["installed"] is False


def test_resident_is_the_residency_s_own_answer(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=True) as client:
        assert all(row["resident"] is False for row in fetch(client, auth))

        class _Resident:
            kind = "llm"
            id = "qwen3.5-9b"
            log_path = Path("a-resident-no-engine-served.log")

        client.app.state.residency._resident = _Resident()
        rows = fetch(client, auth)
        assert by_id(rows, "model", "qwen3.5-9b")["resident"] is True
        assert sum(1 for row in rows if row["resident"]) == 1


def test_a_voice_sharing_a_model_s_id_is_not_made_resident_by_it(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_tts=True) as client:

        class _Resident:
            kind = "tts"
            id = "sigma"
            log_path = Path("a-resident-no-engine-served.log")

        client.app.state.residency._resident = _Resident()
        rows = fetch(client, auth)
        assert by_id(rows, "voice", "sigma")["resident"] is True
        assert by_id(rows, "rvc", "sigma")["resident"] is False


def test_a_catalog_that_cannot_be_read_refuses_whole(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> None:
        raise CrucibleError("voices/deathstalker.toml: not valid TOML")

    monkeypatch.setattr(catalog, "load_all_voices", broken)
    with make_client() as client:
        response = client.get("/v1/catalog", headers=auth)
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "catalog_unreadable"
    assert "deathstalker.toml" in body["message"]
