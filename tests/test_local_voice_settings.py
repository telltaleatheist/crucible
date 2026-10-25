"""A local voice (a home manifest whose arm names a `path`) has a catalog row, so the settings
document the pickers draw from stays whole; and a candidate the catalog has no row for is
confined to its own class.

2026-10-25: training's screening voice omv10b-869 on the Mac was a local voice. The catalog
skipped it, tts still offered it, and GET /v1/settings failed 500 catalog_incomplete, which
blanked every client's pickers (Briefcase's LLM picker: "No models available")."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from crucible import catalog, settings
from crucible.config import load_config

from .conftest import FAKE_BACKEND
from .test_settings_api import decided
from .test_voices_local_weights import LOCAL


def home_local_voice(home: Path, weights_dir: Path, *, with_files: bool = True) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    if with_files:
        (weights_dir / "model.safetensors").write_bytes(b"x" * 64)
    voices = home / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    (voices / "screening.toml").write_text(LOCAL.format(path=weights_dir.as_posix()), encoding="utf-8")


def test_a_local_voice_has_a_catalog_row_that_reads_its_directory(
    make_client: Callable[..., TestClient], home: Path, tmp_path: Path
) -> None:
    with make_client(enable_tts=True, capability=decided()):
        home_local_voice(home, tmp_path / "merged")
        config = load_config(home)
        rows = {s.id: s for s in catalog.subjects(config, FAKE_BACKEND) if s.kind == "voice"}
    row = rows["screening"]
    assert row.source == f"local:{(tmp_path / 'merged').as_posix()}" or row.source.startswith("local:")
    found = row.installed()
    assert found is not None and found.source == "local"


def test_settings_stay_whole_with_a_local_voice_on_the_box(
    make_client: Callable[..., TestClient], home: Path, tmp_path: Path, auth: dict[str, str]
) -> None:
    home_local_voice(home, tmp_path / "merged")
    with make_client(enable_llm=True, enable_tts=True, capability=decided()) as client:
        response = client.get("/v1/settings", headers=auth)
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["choice_problems"] == {}
    assert body["local_model_choices"]["translate"], "the text pickers still draw"


def test_one_class_whose_candidate_has_no_row_does_not_blank_the_others(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str]
) -> None:
    with make_client(enable_llm=True, enable_tts=True, capability=decided()):
        config = load_config(home)
    full, _ = settings._choices(config, _every_candidate(config))
    tts_ids = [row["id"] for row in full["tts"]]
    assert tts_ids, "the decided fixture offers tts candidates"
    missing = tts_ids[0]
    installed = {key: value for key, value in _every_candidate(config).items() if key != missing}

    choices, problems = settings._choices(config, installed)

    assert missing not in [row["id"] for row in choices["tts"]]
    assert problems["tts"][0]["code"] == "catalog_incomplete"
    assert problems["tts"][0]["details"] == {"capability": "tts", "model": missing}
    assert choices["translate"] == full["translate"], "every other class draws as before"
    assert set(problems) == {"tts"}


def _every_candidate(config) -> dict[str, bool]:
    from crucible import capabilityclasses, classnames

    found: dict[str, bool] = {}
    for name in classnames.SELECTABLE_CLASSES:
        entry = capabilityclasses.BY_NAME[name]
        for candidate in entry.candidates(config.capability.backend_kind):
            found[candidate.id] = False
    return found
