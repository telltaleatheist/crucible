from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take, settle

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent / "crucible" / "jobs" / "audio" / "yue2_worker.py"
SCORE = (HERE / "fixtures" / "yue2-score-ballad.abc").read_text(encoding="utf-8")
TAGS = "Instrumental, slow somber piano ballad, strings, no vocals, 66 BPM"


def _check(model: str, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


def test_an_instrumental_song_needs_no_lyrics_and_a_sung_one_still_does() -> None:
    parsed = _check("yue2-3b", tags=TAGS, instrumental=True)
    assert settle(parsed, load_all_audio_manifests()["yue2-3b"].spec("cuda-linux"), 1).instrumental is True
    with pytest.raises(ApiError) as caught:
        _check("yue2-3b", tags=TAGS)
    assert caught.value.code == "audio_param_missing" and "'lyrics'" in caught.value.message


def test_stable_audio_has_no_instrumental_switch() -> None:
    with pytest.raises(ApiError) as caught:
        _check("stable-audio-3-medium", prompt="a sad piano piece", instrumental=True)
    assert caught.value.code == "audio_param_unsupported"


def _load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    import types

    workerio = types.ModuleType("workerio")
    workerio.claim_stdout = lambda: None
    workerio.load_sibling = lambda name, _file: types.SimpleNamespace(
        Throttled=lambda progress, every: SimpleNamespace(tick=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "workerio", workerio)
    spec = importlib.util.spec_from_file_location("yue2_worker_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePipe:
    """YuE2Pipeline.plan as the worker calls it: a model plan first, then a fixed score."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def plan(self, tags, lyrics, **kwargs):
        self.calls.append({"tags": tags, "lyrics": lyrics, **kwargs})
        abc = kwargs.get("abc", SCORE)
        return SimpleNamespace(abc=abc, truncated=False)


def test_an_instrumental_moves_the_vocal_melody_and_renders_that_score_unsung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = FakePipe()
    job = SimpleNamespace(tags=TAGS, lyrics=None, seed=7, cfg=1.0, instrumental=True)
    first = engine._pipe.plan(job.tags, module.INSTRUMENTAL_SECTIONS, seed=7, cfg_scale=1.0)
    fixed = engine._instrumental_plan(job, first)

    final_call = engine._pipe.calls[-1]
    assert final_call["lyrics"] == "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n", "only section tags"
    assert final_call["cot"] == "full" and final_call["seed"] == 7
    sys.path.insert(0, str(WORKER.parent / "yue2music"))
    try:
        from abc_tools import parse_abc
    finally:
        sys.path.pop(0)
    before, after = parse_abc(SCORE), parse_abc(fixed.abc)
    assert after.voices["Vocal"].notes == [], "nothing left for a voice to sing"
    assert all(note in after.voices["Ins"].notes for note in before.voices["Vocal"].notes)
    transfer = engine.notes["instrumental_transfer"]
    assert transfer["vocal_notes_before"] == len(before.voices["Vocal"].notes) > 0
    assert engine.notes["planned_score"] == SCORE


def test_an_empty_or_truncated_plan_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = FakePipe()
    job = SimpleNamespace(tags=TAGS, lyrics=None, seed=7, cfg=1.0, instrumental=True)
    with pytest.raises(RuntimeError, match="another seed"):
        engine._instrumental_plan(job, SimpleNamespace(abc=SCORE, truncated=True))


def test_an_instrumental_with_words_in_its_lyrics_is_refused_not_silently_unsung() -> None:
    """Owen ticked Instrumental over a full set of lyrics: the words were dropped without a
    word and YuE2 sang nonsense to the vocal tags (PC, 2026-10-03)."""
    _check("yue2-3b", tags=TAGS, instrumental=True, lyrics="[Intro]\n\n[Verse]\n\n[Chorus]\n")
    with pytest.raises(ApiError) as caught:
        _check("yue2-3b", tags=TAGS, instrumental=True,
               lyrics="[Verse]\nDarla married Dwayne in the summer of '92\n")
    assert caught.value.code == "audio_param_conflict"
    assert "Darla married Dwayne" in caught.value.message and "Untick instrumental" in caught.value.message


def test_the_song_tag_suggestions_are_grouped_phrases_without_commas() -> None:
    from crucible import playground

    groups = playground.song_tag_suggestions()
    names = [group["group"] for group in groups]
    assert {"Genre", "Mood", "Vocal", "Instruments", "Language"} <= set(names)
    assert all(tag and "," not in tag for group in groups for tag in group["tags"])
    manifest = load_all_audio_manifests()["yue2-3b"]
    tags = playground.audio_fields(manifest, manifest.spec("cuda-linux"))[0]
    assert (tags["name"], tags["kind"], tags["required"]) == ("tags", "tags", True)
    assert tags["suggestions"] == groups
    music = load_all_audio_manifests()["stable-audio-3-medium"]
    assert playground.audio_fields(music, music.spec("cuda-linux"))[0]["kind"] == "text"
