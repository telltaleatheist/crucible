"""Planning lyrics for YuE2 instrumentals (crucible/jobs/audio/planning.py).

Victoria's laptop, 2026-10-10: planned from empty sections, 4 of 13 instrumentals ran the
score to its 4096-token cap. An instrumental is now planned from words it never sings: the
client's `planning_lyrics`, or a set from the server's pool picked by the seed."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from crucible.audiomodels import load_all_audio_manifests
from crucible.errors import ApiError
from crucible.jobs.audio import planning
from crucible.jobs.audio.params import AudioParams, refuse_what_the_model_cannot_take, settle

HERE = Path(__file__).resolve().parent
WORKER = HERE.parent / "crucible" / "jobs" / "audio" / "yue2_worker.py"
TAGS = "Instrumental, slow somber piano ballad, strings, no vocals, 66 BPM"
OWN = "[Verse]\nStone on stone the wall goes up\nMoss along the northern side\n\n[Chorus]\nSlow and steady\n"
SONG = "yue2-3b"


def _spec(model: str = SONG):
    manifest = load_all_audio_manifests()[model]
    return manifest, manifest.spec("cuda-linux")


def _check(model: str = SONG, **params: Any) -> AudioParams:
    manifest, spec = _spec(model)
    parsed = AudioParams(**params)
    refuse_what_the_model_cannot_take(parsed, manifest, spec)
    return parsed


# --- the pool ---------------------------------------------------------------------------


def _syllables(word: str) -> int:
    """A rough count (vowel groups, a silent final e dropped): enough to hold every set in
    the sung band, not to scan verse."""
    word = re.sub(r"[^a-z]", "", word.lower())
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def _shape(entry: planning.PlanningSet) -> dict[str, Any]:
    sections = planning.check(entry.lyrics)
    lines = [line for _, words in sections for line in words]
    return {
        "order": tuple(label for label, _ in sections),
        "lines_per_section": [len(words) for _, words in sections if words],
        "lines": len(lines),
        "syllables": sum(_syllables(w) for line in lines for w in line.split()),
        "per_line": [sum(_syllables(w) for w in line.split()) for line in lines],
    }


def test_the_pool_is_ten_sets_each_one_a_valid_plan() -> None:
    pool = planning.load_pool("yue2")
    assert len(pool) == 10
    assert len({entry.id for entry in pool}) == 10
    assert len({entry.lyrics for entry in pool}) == 10
    for entry in pool:
        assert entry.shape and entry.lyrics.endswith("\n")


def test_every_set_is_sized_like_a_sung_song() -> None:
    """B-Sides' sung shape - [Verse] [Chorus] [Verse] [Chorus], four lines of 6 to 9
    syllables, about 100 to 140 syllables - scores 1800 to 2600 tokens. Every set sits
    in that band, with 3 to 6 lines in each section that has words."""
    for entry in planning.load_pool("yue2"):
        shape = _shape(entry)
        assert all(3 <= n <= 6 for n in shape["lines_per_section"]), (entry.id, shape)
        assert 13 <= shape["lines"] <= 22, (entry.id, shape["lines"])
        assert 90 <= shape["syllables"] <= 150, (entry.id, shape["syllables"])


def test_the_sets_differ_in_structure_not_only_in_words() -> None:
    shapes = {entry.id: _shape(entry) for entry in planning.load_pool("yue2")}
    orders = {shape["order"] for shape in shapes.values()}
    assert len(orders) >= 9, "nearly every set has its own section order"
    assert len({len(shape["order"]) for shape in shapes.values()}) >= 4, "section counts vary"
    assert any(shape["order"][0] == "intro" for shape in shapes.values())
    assert any(shape["order"][0] != "intro" for shape in shapes.values())
    assert any(shape["order"][-1] == "outro" for shape in shapes.values())
    assert any(shape["order"][0] == "chorus" for shape in shapes.values()), "a chorus-first set"
    assert any(len(set(shape["order"])) == len(shape["order"]) for shape in shapes.values()), (
        "a through-composed set, nothing repeated"
    )
    short = [i for i, s in shapes.items() if max(s["per_line"]) <= 7]
    long = [i for i, s in shapes.items() if sorted(s["per_line"])[len(s["per_line"]) // 2] >= 9]
    assert short and long, (short, long)
    assert {n for s in shapes.values() for n in s["lines_per_section"]} == {3, 4, 5, 6}


def test_the_section_tags_are_yue2s_own() -> None:
    """One fact, one owner: the vocabulary is the vendored skill's; this is a checked copy."""
    sys.path.insert(0, str(WORKER.parent / "yue2music"))
    try:
        from compile_score import SECTIONS
    finally:
        sys.path.pop(0)
    assert planning.SECTIONS == SECTIONS


def test_the_seed_picks_the_set_and_the_same_seed_picks_it_again() -> None:
    pool = planning.load_pool("yue2")
    assert [planning.pick(pool, seed).id for seed in range(10)] == [entry.id for entry in pool]
    assert planning.pick(pool, 4_294_967_295) == planning.pick(pool, 4_294_967_295)
    assert planning.pick(pool, 13) == pool[3]


# --- the param --------------------------------------------------------------------------


def test_an_instrumental_without_lyrics_plans_from_the_pool_set_its_seed_picks() -> None:
    manifest, spec = _spec()
    pool = planning.load_pool("yue2")
    parsed = _check(tags=TAGS, instrumental=True)
    for seed in (0, 7, 2_771_032_915):
        settled = settle(parsed, spec, seed)
        chosen = pool[seed % len(pool)]
        assert settled.planning_lyrics == {"source": "pool", "id": chosen.id, "lyrics": chosen.lyrics}
        assert settle(parsed, spec, seed) == settled, "the same seed, the same set"


def test_a_clients_own_planning_lyrics_are_used_and_recorded_as_theirs() -> None:
    _, spec = _spec()
    parsed = _check(tags=TAGS, instrumental=True, planning_lyrics=OWN)
    assert settle(parsed, spec, 5).planning_lyrics == {"source": "request", "id": None, "lyrics": OWN}


def test_no_planning_lyrics_for_a_sung_song_or_an_instrumental_shaped_by_its_tags() -> None:
    _, spec = _spec()
    sung = _check(tags=TAGS, lyrics=OWN)
    assert settle(sung, spec, 1).planning_lyrics is None
    tagged = _check(tags=TAGS, instrumental=True, lyrics="[Intro]\n\n[Verse]\n\n[Outro]\n")
    assert settle(tagged, spec, 1).planning_lyrics is None
    music = _check("stable-audio-3-medium", prompt="a slow piano piece")
    assert settle(music, load_all_audio_manifests()["stable-audio-3-medium"].spec("cuda-linux"), 1).planning_lyrics is None


@pytest.mark.parametrize(
    ("params", "code", "words"),
    [
        (dict(tags=TAGS, planning_lyrics=OWN), "audio_param_conflict", "instrumental: true"),
        (dict(tags=TAGS, lyrics=OWN, planning_lyrics=OWN), "audio_param_conflict", "instrumental: true"),
        (dict(tags=TAGS, instrumental=True, lyrics="[Verse]\n\n[Chorus]\n", planning_lyrics=OWN),
         "audio_param_conflict", "not both"),
    ],
)
def test_planning_lyrics_belong_to_an_instrumental_alone(params: dict, code: str, words: str) -> None:
    with pytest.raises(ApiError) as caught:
        _check(**params)
    assert caught.value.code == code and words in caught.value.message
    assert caught.value.details["param"] == "planning_lyrics"


def test_a_model_without_instrumentals_refuses_planning_lyrics_by_name() -> None:
    with pytest.raises(ApiError) as caught:
        _check("stable-audio-3-medium", prompt="a slow piano piece", planning_lyrics=OWN)
    assert caught.value.code == "audio_param_unsupported"
    assert "planning_lyrics" in caught.value.message


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("   ", "is empty"),
        ("Stone on stone\n", "before any section tag"),
        ("[Hook]\nStone on stone\n", "not one of YuE2's section tags"),
        ("[Verse]\nStone on stone\n\n[Verse]\nMoss on the wall\n", "twice in a row"),
        ("[Verse] Stone on stone\n", "line of its own"),
        ("[Intro]\n\n[Verse]\n\n[Outro]\n", "send the tags as `lyrics`"),
        ("[Verse]\n" + "Stone on stone\n" * 37, "at most 36"),
        ("[Verse]\n" + ("Stone on stone and stone on stone again " * 3 + "\n") * 18, "at most 2000"),
    ],
)
def test_planning_lyrics_that_cannot_plan_are_refused_saying_why(text: str, words: str) -> None:
    with pytest.raises(ValidationError) as caught:
        AudioParams(tags=TAGS, instrumental=True, planning_lyrics=text)
    assert words in str(caught.value)


def test_section_tags_are_read_in_any_case() -> None:
    planning.check("[pre-chorus]\nStone on stone\n\n[CHORUS]\nMoss on the wall\n")


# --- the worker -------------------------------------------------------------------------


def _load_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    import types

    workerio = types.ModuleType("workerio")
    workerio.claim_stdout = lambda: None
    workerio.load_sibling = lambda name, _file: types.SimpleNamespace(
        Throttled=lambda progress, every: SimpleNamespace(tick=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "workerio", workerio)
    spec = importlib.util.spec_from_file_location("yue2_worker_planning_under_test", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**{"lyrics": None, "planning_lyrics": None, "instrumental": False, **fields})


def test_the_worker_plans_the_score_from_what_the_server_settled(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_worker(monkeypatch)
    assert module.planned_from(_job(instrumental=True, planning_lyrics=OWN)) == OWN
    assert module.planned_from(_job(lyrics=OWN)) == OWN
    tags = "[Intro]\n\n[Verse]\n"
    assert module.planned_from(_job(instrumental=True, lyrics=tags)) == tags


@pytest.mark.parametrize(
    "job",
    [
        _job(instrumental=True),
        _job(planning_lyrics=OWN),
        _job(instrumental=True, lyrics="[Verse]\n", planning_lyrics=OWN),
    ],
)
def test_the_worker_refuses_a_request_the_server_never_sends(monkeypatch: pytest.MonkeyPatch, job) -> None:
    module = _load_worker(monkeypatch)
    with pytest.raises(RuntimeError):
        module.planned_from(job)


def test_the_check_script_is_valid_python() -> None:
    script = HERE.parent / "scripts" / "check-instrumental-planning.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")
