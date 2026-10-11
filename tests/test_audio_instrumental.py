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
PLANNING = "[Verse]\nA lantern burning by the gate\nThe evening settles on the stone\n"


def _check(model: str, **params: Any) -> AudioParams:
    manifest = load_all_audio_manifests()[model]
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, manifest.spec("cuda-linux"))
    return parsed


def test_an_instrumental_song_needs_no_lyrics_and_a_sung_one_still_does() -> None:
    parsed = _check("yue2-3b", tags=TAGS, instrumental=True)
    settled = settle(parsed, load_all_audio_manifests()["yue2-3b"].spec("cuda-linux"), 1)
    assert settled.instrumental is True and settled.planning_lyrics["source"] == "pool"
    with pytest.raises(ApiError) as caught:
        _check("yue2-3b", tags=TAGS)
    assert caught.value.code == "audio_param_missing" and "'lyrics'" in caught.value.message


def test_stable_audio_has_no_instrumental_switch() -> None:
    with pytest.raises(ApiError) as caught:
        _check("stable-audio-3-medium", prompt="a sad piano piece", instrumental=True)
    assert caught.value.code == "audio_param_unsupported"


class _Refused(Exception):
    """audiocore.Refused as the worker raises it: a code, a sentence and the details."""

    def __init__(self, code: str, message: str, details: dict) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details


def _load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    import types

    from crucible.jobs.audio import planning, scorelength

    workerio = types.ModuleType("workerio")
    workerio.claim_stdout = lambda: None
    # The worker's pure siblings are the real ones; audiocore (stdout, the protocol) is
    # stood in for by what the stages call.
    siblings = {"planning": planning, "scorelength": scorelength}
    workerio.load_sibling = lambda name, _file: siblings.get(name) or types.SimpleNamespace(
        Throttled=lambda progress, every: SimpleNamespace(tick=lambda: None),
        Refused=_Refused,
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
    job = SimpleNamespace(tags=TAGS, lyrics=None, planning_lyrics=PLANNING, seed=7, cfg=1.0,
                          instrumental=True)
    first = engine._pipe.plan(job.tags, module.planned_from(job), seed=7, cfg_scale=1.0)
    fixed = engine._instrumental_plan(job, first)

    assert engine._pipe.calls[0]["lyrics"] == PLANNING, "the score is planned from the words"
    final_call = engine._pipe.calls[-1]
    assert final_call["lyrics"] == "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n", "only section tags"
    assert "lantern" not in final_call["lyrics"].lower(), "the planning words are never sung"
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


class FailedPlan(SimpleNamespace):
    """A SymbolicPlan that came back unusable; `save` is SymbolicPlan.save's door."""

    def save(self, directory) -> None:
        Path(directory).mkdir(parents=True)
        (Path(directory) / "plan.json").write_text("{}", encoding="utf-8")
        self.saved_to = Path(directory)


@pytest.mark.parametrize(
    ("plan", "said"),
    [
        (dict(abc=SCORE, truncated=True, timing={"output_tokens": 4096}),
         "truncated: it ran to its 4096-token cap without ending"),
        (dict(abc="", truncated=False, timing={"output_tokens": 1}),
         "empty: the model ended it after 1 tokens"),
    ],
)
def test_an_empty_or_truncated_plan_is_refused_by_name_and_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan: dict, said: str
) -> None:
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = FakePipe()
    engine._pipe.generation_config = SimpleNamespace(abc=SimpleNamespace(max_tokens=4096))
    job = SimpleNamespace(tags=TAGS, lyrics=None, planning_lyrics=PLANNING, seed=7, cfg=1.0,
                          instrumental=True, output_path=str(tmp_path / "audio.flac"))
    failed = FailedPlan(**plan)
    with pytest.raises(RuntimeError, match="another seed") as raised:
        engine._instrumental_plan(job, failed)
    assert said in str(raised.value)
    assert failed.saved_to == tmp_path / module.FAILED_PLAN_DIR
    assert (failed.saved_to / "plan.json").is_file()


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


def test_tag_conflicts_are_symmetric_and_name_only_offered_tags() -> None:
    """Owen: picked tags that contradict each other light up red."""
    from crucible import playground

    conflicts = playground.song_tag_conflicts()
    offered = {tag.casefold() for g in playground.song_tag_suggestions() for tag in g["tags"]}
    assert set(conflicts) <= offered
    for tag, rules in conflicts.items():
        for rule in rules:
            assert rule["tag"].casefold() in offered and rule["why"]
            assert any(back["tag"].casefold() == tag for back in conflicts[rule["tag"].casefold()])
    assert {r["tag"] for r in conflicts["light drums"]} == {"double-kick drums"}
    assert "male vocal" in {r["tag"] for r in conflicts["instrumental"]}
    assert "jazz" not in conflicts, "genre blends are allowed"


def test_a_score_the_transfer_refuses_is_kept_and_said(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A score that ended but has a bar longer than its meter (the first planning-lyrics
    song on the PC, 2026-10-10): the refusal names the skill's own error and keeps the plan."""
    module = _load_worker(monkeypatch)
    engine = module.YuE2Engine.__new__(module.YuE2Engine)
    engine._pipe = FakePipe()
    job = SimpleNamespace(tags=TAGS, lyrics=None, seed=7, cfg=1.0, instrumental=True,
                          output_path=str(tmp_path / "audio.flac"))
    broken = FailedPlan(abc="not a score", truncated=False, timing={"output_tokens": 2000})
    with pytest.raises(RuntimeError, match="cannot be moved to the instrument") as raised:
        engine._instrumental_plan(job, broken)
    assert "another seed" in str(raised.value)
    assert broken.saved_to == tmp_path / module.FAILED_PLAN_DIR
